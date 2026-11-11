//! A map of the source tree, so agents navigate by module instead of searching blindly: every
//! search round is a full model request that re-sends the whole context.
//!
//! - Level 1 (`overview`): directories with source files, each with a one-line description, file
//!   count and size. Goes into the lane agents' system prompts once per session (stable, so the
//!   provider caches it).
//! - Level 2 (`detail`): a directory's files, or one file, with their top-level definitions and
//!   line numbers. Served on demand by the `repo_map` tool.
//!
//! Everything is extracted from the code itself, with no model calls and no external tools:
//! module doc comments, README / Cargo.toml / package.json descriptions, and per-language patterns
//! for top-level items. It points agents at the right file; it does not need to be exact.

use ignore::WalkBuilder;
use regex::Regex;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

/// Files larger than this are counted but not scanned.
const MAX_SCAN_BYTES: u64 = 512 * 1024;
/// Definitions listed per file in a detail view.
const MAX_ITEMS_PER_FILE: usize = 60;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Item {
    pub line: usize,
    pub sig: String,
    /// The defined name (`pick_task_fair`, `Config`), for find_definition.
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct FileEntry {
    /// Path relative to the root, with `/` separators.
    pub path: String,
    /// Modification time when scanned (to re-scan only what changed).
    pub mtime: Option<std::time::SystemTime>,
    pub lines: usize,
    pub doc: Option<String>,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RepoMap {
    pub files: Vec<FileEntry>,
    /// Directory (relative, "" = root) → description from a README or a package manifest.
    dir_docs: BTreeMap<String, String>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Lang {
    Rust,
    Python,
    Js,
    Go,
    Lua,
    Shell,
    Java,
    Ruby,
    C,
}

fn lang_of(path: &str) -> Option<Lang> {
    let ext = path.rsplit_once('.').map(|(_, e)| e)?;
    Some(match ext {
        "rs" => Lang::Rust,
        "py" => Lang::Python,
        "js" | "jsx" | "ts" | "tsx" | "mjs" | "cjs" => Lang::Js,
        "go" => Lang::Go,
        "lua" => Lang::Lua,
        "sh" | "bash" => Lang::Shell,
        "java" | "kt" | "kts" | "scala" | "cs" => Lang::Java,
        "rb" => Lang::Ruby,
        "c" | "h" | "cc" | "cpp" | "hpp" | "cxx" => Lang::C,
        _ => return None,
    })
}

fn item_pattern(lang: Lang) -> &'static Regex {
    static RES: OnceLock<Vec<(Lang, Regex)>> = OnceLock::new();
    let all = RES.get_or_init(|| {
        let r = |s: &str| Regex::new(s).expect("valid pattern");
        vec![
            // Top-level items, and `pub fn` methods one level in (inside impl blocks).
            (
                Lang::Rust,
                r(r"^(?:(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?(?:unsafe\s+)?(?:fn|struct|enum|trait|type|const|static|mod|union)\s+\w|impl\b|macro_rules!)|^    pub(?:\([^)]*\))?\s+(?:async\s+)?(?:unsafe\s+)?fn\s+\w"),
            ),
            (Lang::Python, r(r"^(?:async\s+)?(?:def|class)\s+\w|^    (?:async\s+)?def\s+[a-zA-Z]\w*")),
            (
                Lang::Js,
                r(r"^(?:export\s+(?:default\s+)?)?(?:async\s+)?(?:function\*?|class|interface|enum|abstract\s+class)\s+\w|^export\s+(?:type|const|let)\s+\w|^export\s+default\b"),
            ),
            (Lang::Go, r(r"^(?:func|type)\s")),
            (Lang::Lua, r(r"^(?:local\s+)?function\s+[\w.:]+|^[\w.]+\s*=\s*function\b|^local\s+\w+\s*=\s*\{\s*\}")),
            (Lang::Shell, r(r"^(?:function\s+)?[\w-]+\s*\(\)\s*\{?")),
            (
                Lang::Java,
                r(r"^\s{0,4}(?:public\s+|private\s+|protected\s+|internal\s+)?(?:abstract\s+|final\s+|static\s+|sealed\s+|data\s+|open\s+)*(?:class|interface|enum|record|object|fun)\s+\w"),
            ),
            (Lang::Ruby, r(r"^\s{0,2}(?:class|module|def)\s+\w")),
            (Lang::C, r(r"^(?:typedef\s+)?(?:struct|enum|union|class|namespace)\s+\w+\s*\{?$|^[A-Za-z_][\w\s\*&:<>,]*\s\**[A-Za-z_]\w*\([^;]*\)\s*\{?$")),
        ]
    });
    &all.iter().find(|(l, _)| *l == lang).expect("every language has a pattern").1
}

/// A definition line shortened to its signature: up to the body, at most 120 columns.
fn signature(line: &str) -> String {
    let mut s = line.trim();
    for stop in [" {", "{", " where ", " =>", " = "] {
        if let Some(i) = s.find(stop) {
            if i > 0 {
                s = &s[..i];
            }
        }
    }
    let s = s.trim_end_matches([':', '=', ' ']).trim();
    if s.chars().count() > 120 {
        format!("{}…", s.chars().take(119).collect::<String>())
    } else {
        s.to_string()
    }
}

fn is_boilerplate(line: &str) -> bool {
    let l = line.to_lowercase();
    l.contains("copyright")
        || l.contains("spdx")
        || l.contains("license")
        || l.starts_with('!')
        || l.starts_with("vim:")
        || l.contains("-*-")
}

/// The first sentence of a text block, at most 140 characters.
fn first_sentence(text: &str) -> Option<String> {
    let t = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let t = t.trim();
    if t.is_empty() {
        return None;
    }
    let end = t.find(". ").map(|i| i + 1).unwrap_or(t.len());
    let s = &t[..end];
    Some(if s.chars().count() > 140 { format!("{}…", s.chars().take(139).collect::<String>()) } else { s.to_string() })
}

/// The file's own summary: a module doc comment, docstring or leading comment block.
fn file_doc(lang: Lang, text: &str) -> Option<String> {
    let mut block = Vec::new();
    let mut lines = text.lines().map(str::trim).skip_while(|l| l.is_empty() || l.starts_with("#!"));
    match lang {
        Lang::Rust => {
            for l in text.lines().map(str::trim) {
                if let Some(rest) = l.strip_prefix("//!") {
                    if rest.trim().is_empty() && !block.is_empty() {
                        break;
                    }
                    block.push(rest.trim().to_string());
                } else if !l.is_empty() && !l.starts_with("#![") && !l.starts_with("//") {
                    break;
                }
            }
        }
        Lang::Python => {
            let first = lines.find(|l| !l.starts_with('#'))?;
            let quote = ["\"\"\"", "'''"].into_iter().find(|q| first.starts_with(q))?;
            let rest = &first[3..];
            if let Some(end) = rest.find(quote) {
                block.push(rest[..end].to_string());
            } else {
                block.push(rest.to_string());
                for l in lines {
                    if let Some(end) = l.find(quote) {
                        block.push(l[..end].to_string());
                        break;
                    }
                    if l.is_empty() {
                        break;
                    }
                    block.push(l.to_string());
                }
            }
        }
        _ => {
            let prefixes: &[&str] = match lang {
                Lang::Lua => &["--"],
                Lang::Shell | Lang::Ruby => &["#"],
                _ => &["///", "//", "/**", "/*", "*"],
            };
            for l in lines {
                // Blank lines between a skipped license header and the real comment are fine.
                if l.is_empty() && block.is_empty() {
                    continue;
                }
                let Some(p) = prefixes.iter().find(|p| l.starts_with(**p)) else { break };
                let body = l[p.len()..].trim().trim_end_matches("*/").trim();
                if body.is_empty() {
                    if block.is_empty() {
                        continue;
                    }
                    break;
                }
                if is_boilerplate(body) {
                    block.clear();
                    continue;
                }
                block.push(body.to_string());
            }
        }
    }
    first_sentence(&block.join(" "))
}

/// A file's top-level definitions with line numbers (None for languages the map does not know).
pub fn outline(path: &str, text: &str) -> Option<Vec<Item>> {
    lang_of(path).map(|l| scan(l, text).1)
}

fn scan(lang: Lang, text: &str) -> (Option<String>, Vec<Item>) {
    let re = item_pattern(lang);
    let items = text
        .lines()
        .enumerate()
        .filter(|(_, l)| re.is_match(l))
        .map(|(i, l)| {
            let sig = signature(l);
            Item { line: i + 1, name: item_name(&sig).unwrap_or_default().to_string(), sig }
        })
        .filter(|it| !it.sig.is_empty())
        .collect();
    (file_doc(lang, text), items)
}

/// A README's first paragraph (skipping headings, badges and HTML).
fn readme_doc(text: &str) -> Option<String> {
    let mut para = Vec::new();
    for l in text.lines().map(str::trim) {
        // Underlined (setext) headings: the title line, then a line of = or -.
        if !l.is_empty() && l.chars().all(|c| c == '=' || c == '-') {
            para.pop();
            continue;
        }
        let skip = l.starts_with('#') || l.starts_with("[!") || l.starts_with("![") || l.starts_with('<') || l.starts_with("```");
        if l.is_empty() || skip {
            if !para.is_empty() {
                break;
            }
            continue;
        }
        para.push(l);
    }
    first_sentence(&para.join(" "))
}

/// `description = "…"` from a Cargo.toml, or `"description": "…"` from a package.json.
fn manifest_doc(name: &str, text: &str) -> Option<String> {
    let re = match name {
        "Cargo.toml" => Regex::new(r#"(?m)^description\s*=\s*"([^"]+)""#).ok()?,
        "package.json" => Regex::new(r#""description"\s*:\s*"([^"]+)""#).ok()?,
        "pyproject.toml" => Regex::new(r#"(?m)^description\s*=\s*"([^"]+)""#).ok()?,
        _ => return None,
    };
    re.captures(text).and_then(|c| first_sentence(&c[1]))
}

fn parent(path: &str) -> &str {
    path.rsplit_once('/').map_or("", |(d, _)| d)
}

/// Module entry files whose doc describes their directory.
const MODULE_FILES: &[&str] =
    &["mod.rs", "lib.rs", "main.rs", "__init__.py", "index.ts", "index.js", "init.lua", "doc.go", "main.go", "main.py"];

impl RepoMap {
    /// The map of `root` from the disk cache if there is one (possibly behind: call `refresh`), else
    /// a fresh scan, which is then cached. Returns (map, came from the cache).
    pub fn load_or_build(root: &Path) -> (Self, bool) {
        if let Some(map) = cache_file(root).and_then(|p| std::fs::read(p).ok()).and_then(|b| serde_json::from_slice::<RepoMap>(&b).ok()) {
            return (map, true);
        }
        let map = Self::build(root);
        map.save(root);
        (map, false)
    }

    /// Cache the map (best effort) in zilch's data directory, never in the project.
    pub fn save(&self, root: &Path) {
        let Some(path) = cache_file(root) else { return };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        if let Ok(bytes) = serde_json::to_vec(self) {
            let tmp = path.with_extension("tmp");
            if std::fs::write(&tmp, bytes).is_ok() {
                let _ = std::fs::rename(&tmp, &path);
            }
        }
    }

    /// Scan `root`, honouring .gitignore and skipping hidden files.
    pub fn build(root: &Path) -> Self {
        Self::scan_under(root, root, &[])
    }

    /// Re-scan what changed under `path` (relative; "" = everything) since the last scan: edited
    /// files, new and deleted ones, directory descriptions. Cheap: unchanged files are only stat-ed.
    pub fn refresh(&mut self, root: &Path, path: &str) {
        let path = path.trim().trim_start_matches("./").trim_end_matches('/');
        let path = if path == "." { "" } else { path };
        let prefix = if path.is_empty() { String::new() } else { format!("{path}/") };
        let under = |p: &str| path.is_empty() || p == path || p.starts_with(&prefix);
        let start = if path.is_empty() { root.to_path_buf() } else { root.join(path) };
        let old: Vec<FileEntry> = self.files.iter().filter(|f| under(&f.path)).cloned().collect();
        let fresh = if start.exists() { Self::scan_under(root, &start, &old) } else { RepoMap::default() };
        self.files.retain(|f| !under(&f.path));
        self.files.extend(fresh.files);
        self.files.sort_by(|a, b| a.path.cmp(&b.path));
        self.dir_docs.retain(|d, _| !under(d));
        self.dir_docs.extend(fresh.dir_docs);
    }

    /// Scan files under `start`; a file whose mtime matches its entry in `old` is reused as is.
    fn scan_under(root: &Path, start: &Path, old: &[FileEntry]) -> Self {
        let old: std::collections::HashMap<&str, &FileEntry> = old.iter().map(|f| (f.path.as_str(), f)).collect();
        // Walk in parallel (ripgrep's walker), then scan the files on every core.
        let found = std::sync::Mutex::new(Vec::<PathBuf>::new());
        WalkBuilder::new(start).hidden(true).git_ignore(true).require_git(false).build_parallel().run(|| {
            let found = &found;
            Box::new(move |e| {
                if let Ok(e) = e {
                    if e.file_type().is_some_and(|t| t.is_file()) {
                        found.lock().unwrap().push(e.into_path());
                    }
                }
                ignore::WalkState::Continue
            })
        });
        let mut paths = found.into_inner().unwrap();
        paths.sort();
        enum Found {
            File(FileEntry),
            /// A directory description: (directory, text, from a README).
            Doc(String, String, bool),
        }
        let one = |p: &PathBuf| -> Vec<Found> {
            let mut out = vec![];
            let Ok(rel) = p.strip_prefix(root) else { return out };
            let rel = rel.to_string_lossy().replace('\\', "/");
            let name = name_of(&rel).to_string();
            let meta = std::fs::metadata(p).ok();
            let small = meta.as_ref().is_some_and(|m| m.len() <= MAX_SCAN_BYTES);
            let mtime = meta.as_ref().and_then(|m| m.modified().ok());
            if matches!(name.as_str(), "README.md" | "README" | "readme.md" | "Cargo.toml" | "package.json" | "pyproject.toml") && small {
                if let Ok(text) = std::fs::read_to_string(p) {
                    let readme = name.to_lowercase().starts_with("readme");
                    let doc = if readme { readme_doc(&text) } else { manifest_doc(&name, &text) };
                    if let Some(d) = doc {
                        out.push(Found::Doc(parent(&rel).to_string(), d, readme));
                    }
                }
            }
            let Some(lang) = lang_of(&rel) else { return out };
            if let Some(prev) = old.get(rel.as_str()).filter(|f| f.mtime.is_some() && f.mtime == mtime) {
                out.push(Found::File((*prev).clone()));
                return out;
            }
            let (lines, doc, items) = match std::fs::read_to_string(p) {
                Ok(text) if small => {
                    let (doc, items) = scan(lang, &text);
                    (text.lines().count(), doc, items)
                }
                Ok(text) => (text.lines().count(), None, vec![]),
                Err(_) => return out, // not text
            };
            out.push(Found::File(FileEntry { path: rel, mtime, lines, doc, items }));
            out
        };
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get()).min(16);
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut results: Vec<(usize, Vec<Found>)> = std::thread::scope(|s| {
            let workers: Vec<_> = (0..threads)
                .map(|_| {
                    s.spawn(|| {
                        let mut mine = vec![];
                        loop {
                            let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(p) = paths.get(i) else { break };
                            mine.push((i, one(p)));
                        }
                        mine
                    })
                })
                .collect();
            workers.into_iter().flat_map(|w| w.join().unwrap_or_default()).collect()
        });
        results.sort_by_key(|(i, _)| *i);
        let mut map = RepoMap::default();
        for found in results.into_iter().flat_map(|(_, f)| f) {
            match found {
                Found::File(f) => map.files.push(f),
                // A README describes its directory better than a manifest does.
                Found::Doc(dir, d, readme) => {
                    if readme || !map.dir_docs.contains_key(&dir) {
                        map.dir_docs.insert(dir, d);
                    }
                }
            }
        }
        map
    }

    /// A directory's description: README or manifest, else its module file's doc comment.
    fn dir_doc(&self, dir: &str) -> Option<String> {
        if let Some(d) = self.dir_docs.get(dir) {
            return Some(d.clone());
        }
        // Files are sorted by path: binary search instead of a scan.
        MODULE_FILES.iter().find_map(|m| {
            let path = if dir.is_empty() { m.to_string() } else { format!("{dir}/{m}") };
            self.files.binary_search_by(|f| f.path.as_str().cmp(path.as_str())).ok().and_then(|i| self.files[i].doc.clone())
        })
    }

    /// Level 1: one line per directory holding source files, collapsed to fewer levels until it
    /// fits in about `max_chars`.
    pub fn overview(&self, max_chars: usize) -> String {
        if self.files.is_empty() {
            return "(no source files found)".into();
        }
        let deepest = self.files.iter().map(|f| parent(&f.path).matches('/').count() + 1).max().unwrap_or(1);
        let total_lines: usize = self.files.iter().map(|f| f.lines).sum();
        let project = self.dir_docs.get("").map(|d| format!("Project: {d}\n")).unwrap_or_default();
        let head = format!("{project}{} source files, {} lines.\n", self.files.len(), approx_lines(total_lines));
        // Richest first: every directory with a line per documented file; else directories only,
        // opened where the code is.
        // (A line per file cannot fit when there are more files than budget/20 characters each.)
        if self.files.len() <= max_chars / 20 {
            let rich = self.render_overview(deepest, true);
            if head.len() + rich.len() <= max_chars {
                return format!("{head}{rich}");
            }
        }
        format!("{head}{}", self.frontier_overview(max_chars.saturating_sub(head.len())))
    }

    /// Directories only, expanded where the code is: start from the top level and keep opening the
    /// directory with the most files while the listing fits in `budget` characters. A huge tree
    /// (`drivers/`) opens into its parts; a small one stays a single line.
    fn frontier_overview(&self, budget: usize) -> String {
        // Every directory with source files below it: (files, lines) in total, and its own files.
        let mut total: BTreeMap<String, (usize, usize)> = BTreeMap::new();
        let mut direct: BTreeMap<String, Vec<&FileEntry>> = BTreeMap::new();
        for f in &self.files {
            let d = parent(&f.path).to_string();
            direct.entry(d.clone()).or_default().push(f);
            let mut cur = Some(d);
            while let Some(dir) = cur {
                let e = total.entry(dir.clone()).or_default();
                e.0 += 1;
                e.1 += f.lines;
                cur = if dir.is_empty() { None } else { Some(parent(&dir).to_string()) };
            }
        }
        let mut kids: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for d in total.keys().filter(|d| !d.is_empty()) {
            kids.entry(parent(d).to_string()).or_default().push(d.clone());
        }
        // A listing entry: a whole subtree, or only the files directly in an opened directory.
        #[derive(Clone, PartialEq)]
        enum Entry {
            Tree(String),
            Here(String),
            /// The subdirectories of a partly opened directory that are not listed one by one.
            Rest(String, Vec<String>),
        }
        let line = |e: &Entry| -> (String, String) {
            if let Entry::Rest(d, rest) = e {
                let (files, lines) = rest.iter().fold((0, 0), |(f, l), r| (f + total[r].0, l + total[r].1));
                let some: Vec<&str> = rest.iter().take(6).map(|r| name_of(r)).collect();
                let more = if rest.len() > 6 { ", …" } else { "" };
                // Sorted right after the directory's listed children.
                return (
                    format!("{d}/\u{10ffff}"),
                    format!("{d}/… {} more: {}{more} ({files} files, {} lines)\n", rest.len(), some.join(", "), approx_lines(lines)),
                );
            }
            let (dir, files, lines, names): (&str, usize, usize, Vec<&str>) = match e {
                Entry::Tree(d) => {
                    let names = direct.get(d).map(|v| v.iter().map(|f| name_of(&f.path)).collect()).unwrap_or_default();
                    (d, total[d].0, total[d].1, names)
                }
                Entry::Here(d) => {
                    let v = &direct[d];
                    (d, v.len(), v.iter().map(|f| f.lines).sum(), v.iter().map(|f| name_of(&f.path)).collect())
                }
                Entry::Rest(..) => unreachable!("handled above"),
            };
            let shown = if dir.is_empty() { String::from("./") } else { format!("{dir}/") };
            let what = match self.dir_doc(dir) {
                Some(d) => format!(" — {d}"),
                None if !names.is_empty() => {
                    let mut n = names.iter().take(5).copied().collect::<Vec<_>>().join(", ");
                    if names.len() > 5 {
                        n.push_str(", …");
                    }
                    format!(" — {n}")
                }
                None => String::new(),
            };
            (dir.to_string(), format!("{shown}{what} ({files} files, {} lines)\n", approx_lines(lines)))
        };
        let open = |dir: &str| -> Vec<Entry> {
            let mut v: Vec<Entry> = kids.get(dir).into_iter().flatten().cloned().map(Entry::Tree).collect();
            if direct.contains_key(dir) {
                v.push(Entry::Here(dir.to_string()));
            }
            v
        };
        // Each entry's line is built once (a directory description looks through the files).
        let memo: std::cell::RefCell<std::collections::HashMap<(bool, String), (String, String)>> = Default::default();
        let cached = |e: &Entry| -> (String, String) {
            let key = match e {
                Entry::Tree(d) => (true, d.clone()),
                Entry::Here(d) => (false, d.clone()),
                Entry::Rest(..) => return line(e),
            };
            memo.borrow_mut().entry(key).or_insert_with(|| line(e)).clone()
        };
        let render = |entries: &[Entry]| -> String {
            let mut lines: Vec<(String, String)> = entries.iter().map(cached).collect();
            lines.sort();
            lines.into_iter().map(|(_, l)| l).collect()
        };
        // Lengths are tracked as entries come and go; the listing is rendered once, at the end
        // (re-rendering for every candidate took 44 s on the kernel).
        let len = |e: &Entry| cached(e).1.len();
        let mut entries = open("");
        let mut used: usize = entries.iter().map(len).sum();
        let mut stuck: std::collections::HashSet<String> = Default::default();
        loop {
            // The biggest subtree that can still be opened.
            let pick = entries
                .iter()
                .filter_map(|e| match e {
                    Entry::Tree(d) if !stuck.contains(d) && kids.contains_key(d) => Some(d.clone()),
                    _ => None,
                })
                .max_by_key(|d| total[d].0);
            let Some(dir) = pick else { break };
            let me = Entry::Tree(dir.clone());
            let base = used - len(&me);
            let here: Vec<Entry> = if direct.contains_key(&dir) { vec![Entry::Here(dir.clone())] } else { vec![] };
            let here_len: usize = here.iter().map(len).sum();
            let mut children: Vec<String> = kids[&dir].clone();
            children.sort_by_key(|c| std::cmp::Reverse(total[c].0));
            let child_lens: Vec<usize> = children.iter().map(|c| len(&Entry::Tree(c.clone()))).collect();
            let all = base + here_len + child_lens.iter().sum::<usize>();
            let replacement: Option<(Vec<Entry>, usize)> = if all <= budget {
                Some((children.iter().cloned().map(Entry::Tree).chain(here.clone()).collect(), all))
            } else {
                // Too many children to list them all: the biggest ones, and the rest in one line.
                let mut best = None;
                let mut prefix = 0;
                for k in 0..children.len().saturating_sub(1) {
                    prefix += child_lens[k];
                    if k + 1 < 2 {
                        continue;
                    }
                    let rest = Entry::Rest(dir.clone(), children[k + 1..].to_vec());
                    let total_len = base + here_len + prefix + len(&rest);
                    if total_len > budget {
                        break; // longer prefixes only grow
                    }
                    let mut v: Vec<Entry> = children[..=k].iter().cloned().map(Entry::Tree).collect();
                    v.push(rest);
                    v.extend(here.clone());
                    best = Some((v, total_len));
                }
                best
            };
            match replacement {
                Some((add, new_len)) => {
                    entries.retain(|e| *e != me);
                    entries.extend(add);
                    used = new_len;
                }
                None => {
                    stuck.insert(dir);
                }
            }
        }
        render(&entries)
    }

    fn render_overview(&self, depth: usize, with_files: bool) -> String {
        // Each directory's own files, grouped once.
        let mut by_dir: BTreeMap<&str, Vec<&FileEntry>> = BTreeMap::new();
        for f in &self.files {
            by_dir.entry(parent(&f.path)).or_default().push(f);
        }
        // Directory (cut at `depth` levels) → (files, lines, file names).
        let mut dirs: BTreeMap<String, (usize, usize, Vec<String>)> = BTreeMap::new();
        for f in &self.files {
            let d = parent(&f.path);
            let cut = d.split('/').take(depth).collect::<Vec<_>>().join("/");
            let e = dirs.entry(cut.clone()).or_default();
            e.0 += 1;
            e.1 += f.lines;
            if parent(&f.path) == cut {
                e.2.push(f.path.rsplit('/').next().unwrap_or(&f.path).to_string());
            }
        }
        let mut out = String::new();
        for (dir, (files, lines, names)) in &dirs {
            let shown = if dir.is_empty() { "./".to_string() } else { format!("{dir}/") };
            let what = match self.dir_doc(dir) {
                Some(d) => format!(" — {d}"),
                None if !names.is_empty() => {
                    let mut n = names.iter().take(5).cloned().collect::<Vec<_>>().join(", ");
                    if names.len() > 5 {
                        n.push_str(", …");
                    }
                    format!(" — {n}")
                }
                None => String::new(),
            };
            if with_files {
                // Documented files get their own line; the rest stay named in the directory line.
                let here: Vec<&FileEntry> = by_dir.get(dir.as_str()).cloned().unwrap_or_default();
                let undocumented: Vec<&str> =
                    here.iter().filter(|f| f.doc.is_none()).map(|f| f.path.rsplit('/').next().unwrap_or(&f.path)).collect();
                let dir_line = match self.dir_doc(dir) {
                    Some(d) => format!(" — {d}"),
                    None => String::new(),
                };
                out.push_str(&format!("{shown}{dir_line} ({files} files, {} lines)\n", approx_lines(*lines)));
                for f in here.iter().filter(|f| f.doc.is_some()) {
                    let name = f.path.rsplit('/').next().unwrap_or(&f.path);
                    out.push_str(&format!("  {name} — {}\n", f.doc.as_deref().unwrap_or_default()));
                }
                if !undocumented.is_empty() {
                    out.push_str(&format!("  also: {}\n", undocumented.join(", ")));
                }
                continue;
            }
            out.push_str(&format!("{shown}{what} ({files} files, {} lines)\n", approx_lines(*lines)));
        }
        out
    }

    /// Level 2: a directory's files (with their definitions) and subdirectories, or one file's
    /// definitions. `path` is relative to the root ("" or "." = the root).
    pub fn detail(&self, path: &str, max_chars: usize) -> String {
        let path = path.trim().trim_start_matches("./").trim_end_matches('/');
        let path = if path == "." { "" } else { path };
        if let Some(f) = self.files.iter().find(|f| f.path == path) {
            return clip(file_block(f, MAX_ITEMS_PER_FILE * 3), max_chars);
        }
        let prefix = if path.is_empty() { String::new() } else { format!("{path}/") };
        let inside: Vec<&FileEntry> = self.files.iter().filter(|f| f.path.starts_with(&prefix)).collect();
        if inside.is_empty() {
            return format!("No source files under '{path}'. Call repo_map without a path for the overview.");
        }
        let mut out = String::new();
        if let Some(d) = self.dir_doc(path) {
            out.push_str(&format!("{}: {d}\n", if path.is_empty() { "." } else { path }));
        }
        // Subdirectories first, summarized; then the files directly here, with their definitions.
        let mut subdirs: BTreeMap<&str, (usize, usize)> = BTreeMap::new();
        for f in &inside {
            let rest = &f.path[prefix.len()..];
            if let Some((sub, _)) = rest.split_once('/') {
                let e = subdirs.entry(sub).or_default();
                e.0 += 1;
                e.1 += f.lines;
            }
        }
        for (sub, (files, lines)) in &subdirs {
            let full = format!("{prefix}{sub}");
            let doc = self.dir_doc(&full).map(|d| format!(" — {d}")).unwrap_or_default();
            out.push_str(&format!("{sub}/{doc} ({files} files, {} lines)\n", fmt_lines(*lines)));
        }
        for f in inside.iter().filter(|f| !f.path[prefix.len()..].contains('/')) {
            out.push_str(&file_block(f, MAX_ITEMS_PER_FILE));
        }
        clip(out, max_chars)
    }
}

fn file_block(f: &FileEntry, max_items: usize) -> String {
    let name = f.path.rsplit('/').next().unwrap_or(&f.path);
    let doc = f.doc.as_ref().map(|d| format!(" — {d}")).unwrap_or_default();
    let mut s = format!("{name} ({} lines){doc}\n", f.lines);
    for it in f.items.iter().take(max_items) {
        s.push_str(&format!("  {:>5}  {}\n", it.line, it.sig));
    }
    if f.items.len() > max_items {
        s.push_str(&format!("  … {} more definitions (repo_map on the file lists them all)\n", f.items.len() - max_items));
    }
    s
}

fn clip(s: String, max: usize) -> String {
    if s.len() <= max {
        return s;
    }
    let mut cut = max;
    while !s.is_char_boundary(cut) {
        cut -= 1;
    }
    let end = s[..cut].rfind('\n').unwrap_or(cut);
    format!("{}\n… (cut; ask repo_map for a narrower path)", &s[..end])
}

/// Sizes in the overview (the lead agents' system prompt) are rounded, so ordinary edits leave the
/// prompt unchanged and the provider's cached prefix valid across sessions: one significant digit
/// below 10k (43 → "~40", 187 → "~200", 4 123 → "~4k"), two above (15 234 → "~15k").
fn approx_lines(n: usize) -> String {
    if n < 10 {
        return n.to_string();
    }
    let digits = n.to_string().len() as u32;
    let keep = if n < 10_000 { 1 } else { 2 };
    let unit = 10usize.pow(digits - keep);
    let rounded = (n + unit / 2) / unit * unit;
    format!("~{}", if rounded >= 1000 { format!("{}k", rounded / 1000) } else { rounded.to_string() })
}

fn fmt_lines(n: usize) -> String {
    if n >= 10_000 {
        format!("{}k", n / 1000)
    } else if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, text: &str) {
        let p = root.join(rel);
        std::fs::create_dir_all(p.parent().unwrap()).unwrap();
        std::fs::write(p, text).unwrap();
    }

    #[test]
    fn extracts_docs_and_definitions_per_language() {
        let rs = "//! Token checks for the API.\n//! More detail.\n\nuse x;\n\npub fn verify(t: &str) -> bool {\n    true\n}\nstruct Inner;\nimpl Inner {\n    pub fn new() -> Self { Inner }\n    fn private() {}\n}\n";
        let (doc, items) = scan(Lang::Rust, rs);
        assert_eq!(doc.as_deref(), Some("Token checks for the API."), "the first sentence");
        let sigs: Vec<&str> = items.iter().map(|i| i.sig.as_str()).collect();
        assert_eq!(sigs, ["pub fn verify(t: &str) -> bool", "struct Inner;", "impl Inner", "pub fn new() -> Self"]);
        assert_eq!(items[0].line, 6);

        let py = "\"\"\"Billing helpers.\n\nLonger text.\"\"\"\nimport os\n\nclass Invoice:\n    def total(self):\n        pass\n    def _hidden(self):\n        pass\n\ndef main():\n    pass\n";
        let (doc, items) = scan(Lang::Python, py);
        assert_eq!(doc.as_deref(), Some("Billing helpers."));
        let sigs: Vec<&str> = items.iter().map(|i| i.sig.as_str()).collect();
        assert_eq!(sigs, ["class Invoice", "def total(self)", "def main()"]);

        let ts = "// Copyright 2024 Acme\n// SPDX-License-Identifier: MIT\n\n// Routes for the web app.\nimport x from 'y';\nexport async function handler(req) {\n}\nexport const LIMIT = 5;\nfunction local() {}\nexport default class App {}\n";
        let (doc, items) = scan(Lang::Js, ts);
        assert_eq!(doc.as_deref(), Some("Routes for the web app."), "license headers are skipped");
        let sigs: Vec<&str> = items.iter().map(|i| i.sig.as_str()).collect();
        assert_eq!(sigs, ["export async function handler(req)", "export const LIMIT", "function local()", "export default class App"]);

        let lua = "-- zilch.nvim: talk to zilch.\nlocal M = {}\nfunction M.send(text)\nend\nlocal function helper() end\nreturn M\n";
        let (doc, items) = scan(Lang::Lua, lua);
        assert_eq!(doc.as_deref(), Some("zilch.nvim: talk to zilch."));
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn overview_and_detail_follow_the_tree_and_respect_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path();
        write(r, "README.md", "# Proj\n\n[![ci](x)](y)\n\nA tiny ledger service. It does things.\n");
        write(r, "crates/api/Cargo.toml", "[package]\nname = \"api\"\ndescription = \"HTTP API of the ledger\"\n");
        write(r, "crates/api/src/lib.rs", "//! API entry.\npub fn serve() {}\n");
        write(r, "crates/api/src/auth/mod.rs", "//! Authentication: tokens and sessions.\npub mod token;\n");
        write(r, "crates/api/src/auth/token.rs", "pub fn verify(t: &str) -> bool { true }\npub struct Claims;\n");
        write(r, "tools/gen.py", "def main():\n    pass\n");
        write(r, ".gitignore", "target/\n");
        write(r, "target/debug/junk.rs", "pub fn nope() {}\n");
        let map = RepoMap::build(r);
        assert!(map.files.iter().all(|f| !f.path.starts_with("target/")), "gitignored files are skipped");

        let o = map.overview(10_000);
        assert!(o.contains("  token.rs") || o.contains("also: token.rs"), "rich view lists files: {o}");
        let o = map.overview(300); // too small for the rich view: directories only
        assert!(o.starts_with("Project: A tiny ledger service.\n4 source files"), "{o}");
        assert!(o.contains("crates/api/src/auth/ — Authentication: tokens and sessions. (2 files"), "{o}");
        assert!(o.contains("crates/api/src/ — API entry."), "{o}");
        assert!(o.contains("tools/ — gen.py (1 files"), "{o}");
        // A tight budget collapses deeper levels into their parents.
        let tight = map.overview(120);
        assert!(tight.contains("crates/ (3 files"), "{tight}");
        assert!(!tight.contains("auth"), "{tight}");

        let d = map.detail("crates/api/src", 10_000);
        assert!(d.contains("auth/ — Authentication: tokens and sessions. (2 files"), "{d}");
        assert!(d.contains("lib.rs (2 lines) — API entry.") && d.contains("pub fn serve()"), "{d}");
        let f = map.detail("crates/api/src/auth/token.rs", 10_000);
        assert!(f.contains("      1  pub fn verify(t: &str) -> bool") && f.contains("pub struct Claims;"), "{f}");
        assert!(map.detail("nope/", 1000).starts_with("No source files under 'nope'"));
    }
}

#[cfg(test)]
mod refresh_tests {
    use super::*;

    #[test]
    fn refresh_picks_up_edits_new_and_deleted_files_under_the_path() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path();
        std::fs::create_dir_all(r.join("src/a")).unwrap();
        std::fs::write(r.join("src/a/x.rs"), "pub fn one() {}\n").unwrap();
        std::fs::write(r.join("src/a/gone.rs"), "pub fn gone() {}\n").unwrap();
        std::fs::write(r.join("top.rs"), "pub fn top() {}\n").unwrap();
        let mut map = RepoMap::build(r);
        // Edit (with a later mtime), add, delete — and change a file outside the path.
        std::thread::sleep(std::time::Duration::from_millis(20));
        std::fs::write(r.join("src/a/x.rs"), "pub fn one() {}\npub fn two() {}\n").unwrap();
        std::fs::write(r.join("src/a/new.rs"), "pub fn fresh() {}\n").unwrap();
        std::fs::remove_file(r.join("src/a/gone.rs")).unwrap();
        std::fs::write(r.join("top.rs"), "pub fn top() {}\npub fn later() {}\n").unwrap();
        map.refresh(r, "src/a/");
        let d = map.detail("src/a", 10_000);
        assert!(d.contains("pub fn two()") && d.contains("new.rs") && !d.contains("gone.rs"), "{d}");
        assert!(!map.detail("top.rs", 1000).contains("later"), "outside the refreshed path: untouched");
        map.refresh(r, "");
        assert!(map.detail("top.rs", 1000).contains("later"));
    }
}

fn name_of(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

/// `<data dir>/zilch/repomap/<workspace key>.json`.
fn cache_file(root: &Path) -> Option<PathBuf> {
    let abs = std::fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let key: String = abs.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    dirs::data_dir().map(|d| d.join("zilch").join("repomap").join(format!("{}.json", key.trim_matches('_'))))
}

/// The name a definition line defines: the identifier before `(` for functions (C, JS, …), else
/// the one after the keyword (`struct Config`, `class App`, `impl Foo`, `type T`).
pub fn item_name(sig: &str) -> Option<&str> {
    static AFTER: OnceLock<Regex> = OnceLock::new();
    static CALL: OnceLock<Regex> = OnceLock::new();
    let after = AFTER.get_or_init(|| {
        Regex::new(r"\b(?:fn|struct|enum|trait|type|const|static|mod|union|impl(?:<[^>]*>)?|macro_rules!|class|interface|def|function\*?|func(?:\s*\([^)]*\))?|module|record|object|fun|namespace|let|var)\s+([A-Za-z_][\w:.]*)").expect("valid")
    });
    let call = CALL.get_or_init(|| Regex::new(r"([A-Za-z_]\w*)\s*\(").expect("valid"));
    // A function: the identifier right before a `(` (skipping keywords: `pub(crate)`, Go's
    // `func (receiver)`).
    const NOT_NAMES: &[&str] = &["pub", "func", "fn", "if", "while", "for", "return", "sizeof", "static", "def", "function"];
    if let Some(n) = call.captures_iter(sig).filter_map(|c| c.get(1)).map(|m| m.as_str()).find(|n| !NOT_NAMES.contains(n)) {
        if !sig.contains("impl") {
            return Some(n);
        }
    }
    if let Some(c) = after.captures(sig) {
        let n = c.get(1)?.as_str();
        // `impl Trait for Type`: the type is what is defined.
        if sig.contains("impl") {
            if let Some((_, ty)) = sig.split_once(" for ") {
                return ty.split(|c: char| !(c.is_alphanumeric() || c == '_')).find(|s| !s.is_empty());
            }
        }
        return Some(n.rsplit(['.', ':']).next().unwrap_or(n));
    }
    call.captures(sig).and_then(|c| c.get(1)).map(|m| m.as_str())
}

/// One find_definition hit.
#[derive(Debug, Clone, PartialEq)]
pub struct Found {
    pub path: String,
    pub line: usize,
    pub sig: String,
}

impl RepoMap {
    /// Definitions named `name`: exact matches first, then the same name ignoring case, then names
    /// starting with it, then names containing it (each group in path order). Under `path` if given.
    pub fn find(&self, name: &str, path: &str, limit: usize) -> (Vec<Found>, usize) {
        let name = name.trim();
        if name.is_empty() {
            return (vec![], 0);
        }
        let path = path.trim().trim_start_matches("./").trim_end_matches('/');
        let path = if path == "." { "" } else { path };
        let prefix = if path.is_empty() { String::new() } else { format!("{path}/") };
        let lower = name.to_lowercase();
        let mut groups: [Vec<Found>; 4] = Default::default();
        for f in self.files.iter().filter(|f| path.is_empty() || f.path == path || f.path.starts_with(&prefix)) {
            for it in &f.items {
                let n = if it.name.is_empty() { item_name(&it.sig).unwrap_or("") } else { it.name.as_str() };
                if n.is_empty() {
                    continue;
                }
                let rank = if n == name {
                    0
                } else {
                    let nl = n.to_lowercase();
                    if nl == lower {
                        1
                    } else if nl.starts_with(&lower) {
                        2
                    } else if nl.contains(&lower) {
                        3
                    } else {
                        continue;
                    }
                };
                groups[rank].push(Found { path: f.path.clone(), line: it.line, sig: it.sig.clone() });
            }
        }
        let total = groups.iter().map(Vec::len).sum();
        (groups.into_iter().flatten().take(limit).collect(), total)
    }
}

#[cfg(test)]
mod find_tests {
    use super::*;

    #[test]
    fn names_are_taken_from_definitions_in_each_language() {
        let cases = [
            ("pub fn verify(t: &str) -> bool", "verify"),
            ("pub(crate) async fn run_task(&self)", "run_task"),
            ("struct Inner;", "Inner"),
            ("impl Inner", "Inner"),
            ("impl Display for Claims", "Claims"),
            ("impl<T> Tool for Wrapper<T>", "Wrapper"),
            ("class Invoice", "Invoice"),
            ("def total(self)", "total"),
            ("export async function handler(req)", "handler"),
            ("export const LIMIT", "LIMIT"),
            ("func (s *Server) Serve(l net.Listener) error", "Serve"),
            ("function M.send(text)", "send"),
            ("static struct task_struct *pick_task_fair(struct rq *rq)", "pick_task_fair"),
            ("int ext4_mb_new_blocks(handle_t *handle, struct ext4_allocation_request *ar, int *errp)", "ext4_mb_new_blocks"),
        ];
        for (sig, want) in cases {
            assert_eq!(item_name(sig), Some(want), "{sig}");
        }
    }

    #[test]
    fn find_ranks_exact_then_case_then_prefix_then_substring() {
        let dir = tempfile::tempdir().unwrap();
        let r = dir.path();
        std::fs::create_dir_all(r.join("kernel/sched")).unwrap();
        std::fs::create_dir_all(r.join("drivers")).unwrap();
        std::fs::write(
            r.join("kernel/sched/fair.c"),
            "static struct task_struct *pick_task_fair(struct rq *rq)\n{\n}\nvoid pick_task_fair_helper(void)\n{\n}\n",
        )
        .unwrap();
        std::fs::write(r.join("drivers/x.c"), "int do_pick_task_fair(void)\n{\n}\nint Pick_Task_Fair(void)\n{\n}\n").unwrap();
        let map = RepoMap::build(r);
        let (hits, total) = map.find("pick_task_fair", "", 50);
        let got: Vec<(&str, usize)> = hits.iter().map(|h| (h.path.as_str(), h.line)).collect();
        assert_eq!(total, 4);
        assert_eq!(got, [("kernel/sched/fair.c", 1), ("drivers/x.c", 4), ("kernel/sched/fair.c", 4), ("drivers/x.c", 1)]);
        let (hits, _) = map.find("pick_task_fair", "kernel", 50);
        assert!(hits.iter().all(|h| h.path.starts_with("kernel/")));
        assert_eq!(map.find("pick", "", 2).0.len(), 2, "capped");
    }
}

#[cfg(test)]
mod approx_tests {
    use super::*;

    #[test]
    fn overview_sizes_are_rounded_so_small_edits_keep_the_prompt() {
        assert_eq!(approx_lines(7), "7");
        assert_eq!(approx_lines(43), "~40");
        assert_eq!(approx_lines(46), "~50");
        assert_eq!(approx_lines(187), "~200");
        assert_eq!(approx_lines(4_123), "~4k");
        assert_eq!(approx_lines(15_234), "~15k");
        assert_eq!(approx_lines(2_217_000), "~2200k");
        // Adding a few lines to a 4k-line directory does not change its overview line.
        assert_eq!(approx_lines(4_123), approx_lines(4_190));
    }
}
