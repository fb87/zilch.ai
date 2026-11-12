//! `repo_map` and `find_definition`: the repository map's detail view and its definition index,
//! always current (whatever changed under the requested path is re-scanned first).

use super::{arg_opt_str, arg_str, Tool, ToolCtx};
use crate::repomap::RepoMap;
use anyhow::Result;
use serde_json::{json, Value};

/// Output limit of one repo_map call (characters).
const MAX_CHARS: usize = 12_000;
/// Results listed per find_definition call.
const MAX_DEFINITIONS: usize = 50;

/// Run `f` on the shared map after re-scanning what changed under `path` (or on a fresh scan when
/// this context has no shared map).
fn with_map<T>(ctx: &ToolCtx, path: &str, f: impl FnOnce(&RepoMap) -> T) -> T {
    match &ctx.repo_map {
        Some(m) => {
            let mut map = m.lock().unwrap();
            map.refresh(&ctx.cwd, path);
            f(&map)
        }
        None => f(&RepoMap::build(&ctx.cwd)),
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
        Ok(with_map(ctx, &path, |map| {
            if path.trim_matches(['.', '/']).is_empty() {
                map.overview(MAX_CHARS)
            } else {
                map.detail(&path, MAX_CHARS)
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
        Ok(with_map(ctx, &path, |map| {
            let (hits, total) = map.find(&name, &path, MAX_DEFINITIONS);
            if hits.is_empty() {
                return format!("No definition named like '{name}'. Try grep (it may be a macro, a generated name, or text).");
            }
            let mut s: String = hits.iter().map(|h| format!("{}:{}  {}\n", h.path, h.line, h.sig)).collect();
            if total > hits.len() {
                s.push_str(&format!("[{} more; narrow with a longer name or `path`]\n", total - hits.len()));
            }
            s
        }))
    }
}
