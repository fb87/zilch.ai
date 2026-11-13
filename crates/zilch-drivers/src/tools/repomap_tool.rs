//! `repo_map` and `find_definition`: the repository map's detail view and its definition index,
//! always current (whatever changed under the requested path is re-scanned first).

use super::{arg_opt_str, arg_str, Tool, ToolCtx};
use crate::repomap::RepoMap;
use anyhow::Result;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

/// Output limit of one repo_map call (characters).
const MAX_CHARS: usize = 12_000;
/// Results listed per find_definition call.
const MAX_DEFINITIONS: usize = 50;

/// Maps of fetched outside sources (they don't change, so each is built once).
fn outside_maps() -> &'static Mutex<HashMap<PathBuf, Arc<RepoMap>>> {
    static MAPS: OnceLock<Mutex<HashMap<PathBuf, Arc<RepoMap>>>> = OnceLock::new();
    MAPS.get_or_init(Mutex::default)
}

fn outside_map(root: &Path) -> Arc<RepoMap> {
    if let Some(m) = outside_maps().lock().unwrap().get(root) {
        return m.clone();
    }
    let m = Arc::new(RepoMap::build(root));
    outside_maps().lock().unwrap().insert(root.to_path_buf(), m.clone());
    m
}

/// A source was fetched again: its map is stale.
pub fn forget_outside_map(root: &Path) {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    outside_maps().lock().unwrap().remove(&root);
}

/// The overview of a fetched source (for fetch_source's answer).
pub fn outside_overview(root: &Path, max_chars: usize) -> String {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    outside_map(&root).overview(max_chars)
}

/// For a path inside a fetched source: the source's root (the directory with `<name>.source.json`
/// beside it, else the read root) and the path relative to it.
fn outside_root(ctx: &ToolCtx, path: &str) -> Option<(PathBuf, String)> {
    let (read_root, real) = ctx.read_root(path)?;
    let root = real
        .ancestors()
        .take_while(|a| a.starts_with(&read_root) && *a != read_root)
        .find(|a| a.file_name().is_some_and(|n| a.with_file_name(format!("{}.source.json", n.to_string_lossy())).exists()))
        .map(Path::to_path_buf)
        .unwrap_or(read_root);
    let rel = real.strip_prefix(&root).ok()?.to_string_lossy().into_owned();
    Some((root, rel))
}

/// Run `f` on the map holding `path` with the path relative to that map, and the root to prefix
/// results with: a fetched source's map, or the shared workspace map after re-scanning what
/// changed under `path` (a fresh scan when this context has no shared map).
fn with_map<T>(ctx: &ToolCtx, path: &str, f: impl FnOnce(&RepoMap, &str, Option<&Path>) -> T) -> T {
    if let Some((root, rel)) = outside_root(ctx, path) {
        return f(&outside_map(&root), &rel, Some(&root));
    }
    match &ctx.repo_map {
        Some(m) => {
            let mut map = m.lock().unwrap();
            map.refresh(&ctx.cwd, path);
            f(&map, path, None)
        }
        None => f(&RepoMap::build(&ctx.cwd), path, None),
    }
}

pub struct RepoMapTool;

impl Tool for RepoMapTool {
    fn name(&self) -> &str {
        "repo_map"
    }
    fn description(&self) -> String {
        "The repository map. Without `path`: every module with its purpose. With a directory: its files, each with \
         its summary and top-level definitions with line numbers. With a file: all its definitions."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{"path":{"type":"string","description":"Directory or file (default: overview)"}}})
    }
    fn summarize(&self, args: &Value) -> String {
        arg_opt_str(args, "path").unwrap_or(".").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = arg_opt_str(&args, "path").unwrap_or("").to_string();
        ctx.resolve_checked(if path.is_empty() { "." } else { &path }, false)?;
        Ok(with_map(ctx, &path, |map, path, root| {
            let s = if path.trim_matches(['.', '/']).is_empty() { map.overview(MAX_CHARS) } else { map.detail(path, MAX_CHARS) };
            match root {
                Some(r) => format!("[paths below are relative to {}]\n{s}", r.display()),
                None => s,
            }
        }))
    }
}

pub struct FindDefinition;

impl Tool for FindDefinition {
    fn name(&self) -> &str {
        "find_definition"
    }
    fn description(&self) -> String {
        "Where a function, type, class or constant is defined: `file:line  signature`, exact name first, then \
         names starting with or containing it. Faster and more precise than grep, which also returns comments, \
         docs and usages; use grep for text and usages."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "name":{"type":"string","description":"e.g. `pick_task_fair` or `Config` (part of a name works too)"},
            "path":{"type":"string","description":"Only under this directory"}
        },"required":["name"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("name").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let name = arg_str(&args, "name")?.to_string();
        let path = arg_opt_str(&args, "path").unwrap_or("").to_string();
        ctx.resolve_checked(if path.is_empty() { "." } else { &path }, false)?;
        Ok(with_map(ctx, &path, |map, path, root| {
            let (hits, total) = map.find(&name, path, MAX_DEFINITIONS);
            let prefix = root.map(|r| format!("{}/", r.display())).unwrap_or_default();
            if hits.is_empty() {
                return format!("No definition named like '{name}'. Try grep (it may be a macro, a generated name, or text).");
            }
            let mut s: String = hits.iter().map(|h| format!("{prefix}{}:{}  {}\n", h.path, h.line, h.sig)).collect();
            if total > hits.len() {
                s.push_str(&format!("[{} more; narrow with a longer name or `path`]\n", total - hits.len()));
            }
            s
        }))
    }
}
