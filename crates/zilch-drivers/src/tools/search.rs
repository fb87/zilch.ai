use super::{arg_bool, arg_opt_str, arg_str, Tool, ToolCtx};
use anyhow::Result;
use globset::Glob;
use ignore::WalkBuilder;
use serde_json::{json, Value};
use std::path::PathBuf;

const MAX_FILES: usize = 500;
const MAX_MATCHES: usize = 300;

fn walk(root: PathBuf) -> impl Iterator<Item = PathBuf> {
    WalkBuilder::new(root)
        .hidden(true)
        .git_ignore(true)
        .require_git(false)
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
        "Find files by glob pattern (e.g. `src/**/*.rs`). Respects .gitignore.".into()
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
            Ok(out)
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
        let re = regex::RegexBuilder::new(arg_str(&args, "pattern")?).case_insensitive(arg_bool(&args, "case_insensitive")).build()?;
        let filter = arg_opt_str(&args, "glob").map(Glob::new).transpose()?.map(|g| g.compile_matcher());
        let root = ctx.resolve_checked(arg_opt_str(&args, "path").unwrap_or("."), false)?;
        let cwd = ctx.cwd.clone();
        {
            let mut out = Vec::new();
            let mut total = 0usize;
            for p in walk(root) {
                if let Some(f) = &filter {
                    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    let rel = p.strip_prefix(&cwd).unwrap_or(&p).to_string_lossy().into_owned();
                    if !f.is_match(&name) && !f.is_match(&rel) {
                        continue;
                    }
                }
                let Ok(text) = std::fs::read_to_string(&p) else { continue };
                let rel = p.strip_prefix(&cwd).unwrap_or(&p).to_string_lossy().into_owned();
                for (i, line) in text.lines().enumerate() {
                    if re.is_match(line) {
                        total += 1;
                        if out.len() < MAX_MATCHES {
                            let l = if line.len() > 300 { &line[..line.floor_char_boundary(300)] } else { line };
                            out.push(format!("{rel}:{}: {}", i + 1, l.trim_end()));
                        }
                    }
                }
            }
            let mut s = out.join("\n");
            if total == 0 {
                s = "No matches".into();
            } else if total > MAX_MATCHES {
                s.push_str(&format!("\n[{} more matches not shown]", total - MAX_MATCHES));
            }
            Ok(s)
        }
    }
}
