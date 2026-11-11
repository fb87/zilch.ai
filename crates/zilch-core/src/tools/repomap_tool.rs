//! `repo_map`: the repository map's detail view (a module's files and their definitions), always
//! current: whatever changed under the requested path is re-scanned first.

use super::{arg_opt_str, Tool, ToolCtx};
use crate::repomap::RepoMap;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};

/// Output limit of one call (characters).
const MAX_CHARS: usize = 12_000;

pub struct RepoMapTool;

#[async_trait]
impl Tool for RepoMapTool {
    fn name(&self) -> &str {
        "repo_map"
    }
    fn description(&self) -> String {
        "The repository map. Without `path`: every module with its purpose. With a directory: its files, each with \
         its summary and top-level definitions (functions, types, classes) with line numbers. With a file: all its \
         definitions. Cheaper than glob + read_file to find where something lives; read_file the lines you need after."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string","description":"Directory or file, relative to the workspace (default: overview)"}
        }})
    }
    fn summarize(&self, args: &Value) -> String {
        arg_opt_str(args, "path").unwrap_or(".").to_string()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = arg_opt_str(&args, "path").unwrap_or("").to_string();
        // Policy check (no escaping the workspace); the map itself works on relative paths.
        ctx.resolve_checked(if path.is_empty() { "." } else { &path }, false)?;
        let shared = ctx.repo_map.clone();
        let cwd = ctx.cwd.clone();
        tokio::task::spawn_blocking(move || {
            let render = |map: &RepoMap| {
                if path.trim_matches(['.', '/']).is_empty() {
                    map.overview(MAX_CHARS)
                } else {
                    map.detail(&path, MAX_CHARS)
                }
            };
            match shared {
                Some(m) => {
                    let mut map = m.lock().unwrap();
                    map.refresh(&cwd, &path);
                    render(&map)
                }
                None => render(&RepoMap::build(&cwd)),
            }
        })
        .await
        .map_err(Into::into)
    }
}

/// Results listed per find_definition call.
const MAX_DEFINITIONS: usize = 50;

pub struct FindDefinition;

#[async_trait]
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
            "name":{"type":"string","description":"The name, e.g. `pick_task_fair` or `Config` (part of a name works too)"},
            "path":{"type":"string","description":"Only under this directory (default: the whole workspace)"}
        },"required":["name"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("name").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let name = super::arg_str(&args, "name")?.to_string();
        let path = arg_opt_str(&args, "path").unwrap_or("").to_string();
        ctx.resolve_checked(if path.is_empty() { "." } else { &path }, false)?;
        let shared = ctx.repo_map.clone();
        let cwd = ctx.cwd.clone();
        tokio::task::spawn_blocking(move || {
            let render = |map: &RepoMap| {
                let (hits, total) = map.find(&name, &path, MAX_DEFINITIONS);
                if hits.is_empty() {
                    return format!("No definition named like '{name}'. Try grep (it may be a macro, a generated name, or text).");
                }
                let mut s: String = hits.iter().map(|h| format!("{}:{}  {}\n", h.path, h.line, h.sig)).collect();
                if total > hits.len() {
                    s.push_str(&format!("[{} more; narrow with a longer name or `path`]\n", total - hits.len()));
                }
                s
            };
            match shared {
                Some(m) => {
                    let mut map = m.lock().unwrap();
                    map.refresh(&cwd, &path);
                    render(&map)
                }
                None => render(&RepoMap::build(&cwd)),
            }
        })
        .await
        .map_err(Into::into)
    }
}
