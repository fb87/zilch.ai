//! Tool trait, per-agent tool context, and the registry.

pub mod bash;
pub mod delegate;
pub mod edit;
pub mod fs;
pub mod search;
pub mod todo_tools;

use crate::agent::approval::ApprovalGate;
use crate::agent::events::EventSink;
use crate::team::file_locks::FileLocks;
use crate::llm::ToolSpec;
use anyhow::Result;
use async_trait::async_trait;
use globset::{Glob, GlobSet, GlobSetBuilder};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio_util::sync::CancellationToken;

/// Names of master-only coordination tools; never given to workers.
pub const COORDINATION_TOOLS: &[&str] = &["todo_write", "propose_plan", "run_todos", "delegate"];

#[async_trait]
pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> String;
    /// JSON Schema for the arguments object.
    fn schema(&self) -> Value;
    /// Mutating tools go through approval and are never given to the master.
    fn mutating(&self) -> bool {
        false
    }
    /// For file-writing tools: the path this call would modify (used for plan-scoped approval).
    fn target_path(&self, _args: &Value) -> Option<String> {
        None
    }
    /// One-line human summary of a call, for UI and approval prompts.
    fn summarize(&self, args: &Value) -> String {
        let s = args.to_string();
        if s.len() > 120 { format!("{}…", &s[..s.floor_char_boundary(120)]) } else { s }
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String>;
}

/// The files a worker is authorized to modify for its current task.
#[derive(Clone, Debug)]
pub struct Scope {
    pub patterns: Vec<String>,
    set: GlobSet,
}

impl Scope {
    pub fn new(patterns: &[String]) -> Result<Self> {
        let mut b = GlobSetBuilder::new();
        for p in patterns {
            b.add(Glob::new(p)?);
        }
        Ok(Self { patterns: patterns.to_vec(), set: b.build()? })
    }
    pub fn contains(&self, rel_path: &str) -> bool {
        let p = rel_path.trim_start_matches("./");
        self.set.is_match(p) || self.patterns.iter().any(|x| x == p)
    }
}

/// Per-agent execution context shared by all tools of that agent.
pub struct ToolCtx {
    pub agent_id: String,
    pub cwd: PathBuf,
    pub gate: Arc<ApprovalGate>,
    pub events: EventSink,
    pub cancel: CancelCell,
    pub file_locks: FileLocks,
    /// Content hash of every file this agent has read or written (stale-edit detection).
    pub read_hashes: Mutex<HashMap<PathBuf, String>>,
    /// Approved write scope for the task currently being executed (None = nothing pre-approved).
    pub scope: Mutex<Option<Scope>>,
}

impl ToolCtx {
    pub fn new(agent_id: impl Into<String>, cwd: PathBuf, gate: Arc<ApprovalGate>, events: EventSink, cancel: CancelCell, file_locks: FileLocks) -> Self {
        Self {
            agent_id: agent_id.into(),
            cwd,
            gate,
            events,
            cancel,
            file_locks,
            read_hashes: Mutex::new(HashMap::new()),
            scope: Mutex::new(None),
        }
    }

    pub fn resolve(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        let abs = if path.is_absolute() { path.to_path_buf() } else { self.cwd.join(path) };
        normalize(&abs)
    }

    pub fn relative(&self, abs: &Path) -> String {
        abs.strip_prefix(&self.cwd).unwrap_or(abs).to_string_lossy().into_owned()
    }

    pub fn in_scope(&self, path: &str) -> bool {
        let rel = self.relative(&self.resolve(path));
        self.scope.lock().unwrap().as_ref().is_some_and(|s| s.contains(&rel))
    }
}

/// Lexically normalize `.` and `..` (works for files that don't exist yet).
pub fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// Team-wide cancellation: one current token, replaced per user turn.
#[derive(Clone, Default)]
pub struct CancelCell(Arc<Mutex<CancellationToken>>);

impl CancelCell {
    pub fn current(&self) -> CancellationToken {
        self.0.lock().unwrap().clone()
    }
    pub fn cancel(&self) {
        self.0.lock().unwrap().cancel();
    }
    pub fn reset(&self) {
        *self.0.lock().unwrap() = CancellationToken::new();
    }
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, t: Arc<dyn Tool>) {
        self.tools.insert(t.name().to_string(), t);
    }
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
    pub fn specs(&self) -> Vec<ToolSpec> {
        self.tools.values().map(|t| ToolSpec::new(t.name(), t.description(), t.schema())).collect()
    }
    pub fn filter(&self, f: impl Fn(&dyn Tool) -> bool) -> Self {
        Self { tools: self.tools.iter().filter(|(_, t)| f(t.as_ref())).map(|(k, v)| (k.clone(), v.clone())).collect() }
    }
    /// Keep only the named tools (unknown names are ignored).
    pub fn subset(&self, names: &[String]) -> Self {
        self.filter(|t| names.iter().any(|n| n == t.name()))
    }
}

/// The standard worker toolset.
pub fn worker_tools() -> ToolRegistry {
    let mut r = ToolRegistry::default();
    r.register(Arc::new(fs::ReadFile));
    r.register(Arc::new(fs::WriteFile));
    r.register(Arc::new(edit::EditFile));
    r.register(Arc::new(search::GlobTool));
    r.register(Arc::new(search::GrepTool));
    r.register(Arc::new(bash::Bash));
    r
}

/// Keep the head and tail of long outputs.
pub fn truncate_output(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let head = s.floor_char_boundary(max * 2 / 3);
    let tail = s.ceil_char_boundary(s.len() - max / 3);
    format!("{}\n… [{} bytes truncated] …\n{}", &s[..head], tail - head, &s[tail..])
}

pub fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::digest(data);
    d.iter().map(|b| format!("{b:02x}")).collect()
}

/// Helpers for pulling typed args out of a JSON object.
pub(crate) fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| anyhow::anyhow!("missing string argument '{key}'"))
}
pub(crate) fn arg_opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}
pub(crate) fn arg_opt_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
}
pub(crate) fn arg_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_matching() {
        let s = Scope::new(&["src/**".into(), "Cargo.toml".into()]).unwrap();
        assert!(s.contains("src/main.rs"));
        assert!(s.contains("./Cargo.toml"));
        assert!(!s.contains("README.md"));
    }

    #[test]
    fn truncation_keeps_head_and_tail() {
        let s = "a".repeat(100) + &"b".repeat(100);
        let t = truncate_output(&s, 60);
        assert!(t.starts_with("aaaa") && t.ends_with("bbbb") && t.contains("truncated"));
    }
}
