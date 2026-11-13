use super::{arg_bool, arg_opt_str, arg_str, Tool, ToolCtx};
use anyhow::Result;
use globset::Glob;
use serde_json::{json, Value};
use std::path::PathBuf;

const MAX_FILES: usize = 500;
const MAX_MATCHES: usize = 300;

fn walk(root: PathBuf) -> impl Iterator<Item = PathBuf> {
    crate::repomap::walker(&root)
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
        .map(|e| e.into_path())
}

pub struct GlobTool;

impl Tool for GlobTool {
    fn name(&self) -> &str {
        "glob"
    }
    fn description(&self) -> String {
        "Find files by glob pattern (e.g. `src/**/*.rs`). Respects .gitignore; hidden entries are skipped except CI and build config (.github, .gitlab, .cargo, …).".into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "pattern":{"type":"string"},
            "path":{"type":"string","description":"Directory to search (default: working directory)"}
        },"required":["pattern"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("pattern").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let pattern = arg_str(&args, "pattern")?;
        let matcher = Glob::new(pattern)?.compile_matcher();
        let root = ctx.resolve_checked(arg_opt_str(&args, "path").unwrap_or("."), false)?;
        let cwd = ctx.cwd.clone();
        {
            let mut hits: Vec<String> = walk(root.clone())
                .filter_map(|p| {
                    ctx.alive();
                    let rel = p.strip_prefix(&root).ok()?.to_string_lossy().into_owned();
                    matcher.is_match(&rel).then(|| p.strip_prefix(&cwd).unwrap_or(&p).to_string_lossy().into_owned())
                })
                .collect();
            hits.sort();
            let total = hits.len();
            hits.truncate(MAX_FILES);
            let mut out = hits.join("\n");
            if total == 0 {
                out = "No files found".into();
            } else if total > MAX_FILES {
                out.push_str(&format!("\n[{} more files not shown]", total - MAX_FILES));
            }
            Ok(super::repeat_note(ctx, "glob", &args, &out).unwrap_or(out))
        }
    }
}

pub struct GrepTool;

impl Tool for GrepTool {
    fn name(&self) -> &str {
        "grep"
    }
    fn description(&self) -> String {
        "Search file contents with a regex. Returns `path:line: text`. Respects .gitignore.".into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "pattern":{"type":"string","description":"Rust regex syntax"},
            "path":{"type":"string","description":"Directory or file to search (default: working directory)"},
            "glob":{"type":"string","description":"Only search files matching this glob, e.g. `*.rs`"},
            "case_insensitive":{"type":"boolean"}
        },"required":["pattern"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("pattern").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let re =
            regex::bytes::RegexBuilder::new(arg_str(&args, "pattern")?).case_insensitive(arg_bool(&args, "case_insensitive")).build()?;
        let filter = arg_opt_str(&args, "glob").map(Glob::new).transpose()?.map(|g| g.compile_matcher());
        let root = ctx.resolve_checked(arg_opt_str(&args, "path").unwrap_or("."), false)?;
        let cwd = ctx.cwd.clone();
        let alive = ctx.alive.clone();
        // Like ripgrep: walk in parallel, and test each whole file once (the regex engine's literal
        // prefilters make that fast) before splitting a matching file into lines.
        let found = std::sync::Mutex::new(Vec::<(String, usize, String)>::new());
        crate::repomap::walker(&root).build_parallel().run(|| {
            let (re, filter, cwd, found, alive) = (&re, &filter, &cwd, &found, &alive);
            Box::new(move |entry| {
                let Ok(e) = entry else { return ignore::WalkState::Continue };
                if !e.file_type().is_some_and(|t| t.is_file()) {
                    return ignore::WalkState::Continue;
                }
                if let Some(a) = alive {
                    a(); // heartbeat for the kernel (rate-limited there)
                }
                let p = e.path();
                let rel = p.strip_prefix(cwd).unwrap_or(p).to_string_lossy().into_owned();
                if let Some(f) = filter {
                    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    if !f.is_match(&name) && !f.is_match(&rel) {
                        return ignore::WalkState::Continue;
                    }
                }
                let Ok(bytes) = std::fs::read(p) else { return ignore::WalkState::Continue };
                if bytes.iter().take(8000).any(|b| *b == 0) || !re.is_match(&bytes) {
                    return ignore::WalkState::Continue; // binary, or no match anywhere in the file
                }
                let mut mine = Vec::new();
                for (i, line) in bytes.split(|b| *b == b'\n').enumerate() {
                    if re.is_match(line) {
                        let text = String::from_utf8_lossy(line);
                        let text = if text.len() > 300 { text[..text.floor_char_boundary(300)].to_string() } else { text.into_owned() };
                        mine.push((rel.clone(), i + 1, text.trim_end().to_string()));
                    }
                }
                found.lock().unwrap().extend(mine);
                ignore::WalkState::Continue
            })
        });
        let mut hits = found.into_inner().unwrap();
        hits.sort_by(|a, b| (&a.0, a.1).cmp(&(&b.0, b.1)));
        let total = hits.len();
        let mut s = hits.iter().take(MAX_MATCHES).map(|(p, n, t)| format!("{p}:{n}: {t}")).collect::<Vec<_>>().join("\n");
        if total == 0 {
            s = "No matches".into();
        } else if total > MAX_MATCHES {
            s.push_str(&format!("\n[{} more matches not shown]", total - MAX_MATCHES));
        }
        Ok(super::repeat_note(ctx, "grep", &args, &s).unwrap_or(s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ci_config_is_found_other_hidden_entries_are_not() {
        let dir = tempfile::tempdir().unwrap();
        let w = |p: &str, s: &str| {
            let p = dir.path().join(p);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, s).unwrap();
        };
        w(".github/workflows/ci.yml", "run: cargo test --workspace\n");
        w(".env", "SECRET=cargo test\n");
        w(".zilch/config.toml", "api_key = \"cargo test\"\n");
        w(".hidden/notes.txt", "cargo test\n");
        w("src/main.rs", "// cargo test\n");
        let ctx = ToolCtx::new("t", dir.path().to_path_buf());
        let found = GlobTool.call(json!({"pattern":"**/*"}), &ctx).unwrap();
        assert!(found.contains(".github/workflows/ci.yml") && found.contains("src/main.rs"), "{found}");
        assert!(!found.contains(".env") && !found.contains(".zilch") && !found.contains(".hidden"), "{found}");
        let hits = GrepTool.call(json!({"pattern":"cargo test"}), &ctx).unwrap();
        assert!(hits.contains(".github/workflows/ci.yml") && hits.contains("src/main.rs"), "{hits}");
        assert!(!hits.contains(".env") && !hits.contains(".zilch") && !hits.contains(".hidden"), "{hits}");
    }
}
