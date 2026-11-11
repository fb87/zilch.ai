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
