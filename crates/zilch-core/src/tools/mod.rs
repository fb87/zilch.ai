//! Tool trait, per-agent tool context, and the registry.

pub mod bash;
pub mod delegate;
pub mod edit;
pub mod fs;
pub mod sandbox;
pub mod search;
pub mod todo_tools;

use crate::agent::approval::ApprovalGate;
use crate::agent::events::EventSink;
use crate::llm::ToolSpec;
use crate::team::file_locks::FileLocks;
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
        if s.len() > 120 {
            format!("{}…", &s[..s.floor_char_boundary(120)])
        } else {
            s
        }
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
    /// Team-wide safety policy (sandbox, environment scrubbing).
    pub policy: Arc<ToolPolicy>,
    /// Team-wide token budget shared by every agent (None = unlimited).
    pub budget: Option<Arc<TokenBudget>>,
}

/// Safety policy applied by the built-in tools.
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// Confine file tools to the workspace (no `..`, absolute paths or symlinks escaping it),
    /// and forbid writes inside `.git/`.
    pub sandbox: bool,
    /// Environment variables never passed to `bash` (in addition to the secret-name heuristic).
    pub secret_env: Vec<String>,
    /// Variables passed to `bash` even if they look secret (e.g. `GITHUB_TOKEN` for `gh`).
    pub env_passthrough: Vec<String>,
    /// OS-level sandbox for `bash` (bubblewrap).
    pub bash_sandbox: sandbox::BashSandbox,
    /// What sandboxed commands may reach (network, extra hidden/writable paths).
    pub bash: sandbox::SandboxSpec,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            sandbox: true,
            secret_env: vec![],
            env_passthrough: vec![],
            bash_sandbox: sandbox::BashSandbox::Auto,
            bash: sandbox::SandboxSpec { network: true, ..Default::default() },
        }
    }
}

impl ToolPolicy {
    /// Should `name` be removed from the environment of commands the model runs?
    pub fn is_secret_env(&self, name: &str) -> bool {
        if self.env_passthrough.iter().any(|p| p == name) {
            return false;
        }
        let up = name.to_ascii_uppercase();
        self.secret_env.iter().any(|s| s == name)
            || ["_API_KEY", "_APIKEY", "_TOKEN", "_SECRET", "_PASSWORD", "_PASSWD", "_CREDENTIALS", "_PRIVATE_KEY"]
                .iter()
                .any(|s| up.ends_with(s))
            || ["AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"].contains(&up.as_str())
    }
}

/// Token budget shared across the whole team; agents stop when it is exhausted.
#[derive(Debug)]
pub struct TokenBudget {
    pub limit: u64,
    used: std::sync::atomic::AtomicU64,
}

impl TokenBudget {
    pub fn new(limit: u64) -> Self {
        Self { limit, used: Default::default() }
    }
    pub fn add(&self, tokens: u64) {
        self.used.fetch_add(tokens, std::sync::atomic::Ordering::Relaxed);
    }
    pub fn used(&self) -> u64 {
        self.used.load(std::sync::atomic::Ordering::Relaxed)
    }
    pub fn exhausted(&self) -> bool {
        self.used() >= self.limit
    }
}

impl ToolCtx {
    pub fn new(
        agent_id: impl Into<String>,
        cwd: PathBuf,
        gate: Arc<ApprovalGate>,
        events: EventSink,
        cancel: CancelCell,
        file_locks: FileLocks,
    ) -> Self {
        Self {
            agent_id: agent_id.into(),
            cwd,
            gate,
            events,
            cancel,
            file_locks,
            read_hashes: Mutex::new(HashMap::new()),
            scope: Mutex::new(None),
            policy: Arc::new(ToolPolicy::default()),
            budget: None,
        }
    }

    pub fn with_policy(mut self, policy: Arc<ToolPolicy>, budget: Option<Arc<TokenBudget>>) -> Self {
        self.policy = policy;
        self.budget = budget;
        self
    }

    /// Lexical resolution against the workspace (no policy checks). Prefer [`Self::resolve_checked`].
    pub fn resolve(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        let abs = if path.is_absolute() { path.to_path_buf() } else { self.cwd.join(path) };
        normalize(&abs)
    }

    /// Resolve a model-supplied path and enforce the sandbox: the real location (after following
    /// symlinks of every existing ancestor) must be inside the workspace; writes into `.git/` are
    /// refused.
    pub fn resolve_checked(&self, p: &str, write: bool) -> Result<PathBuf> {
        let abs = self.resolve(p);
        if !self.policy.sandbox {
            return Ok(abs);
        }
        let root = self.cwd.canonicalize().unwrap_or_else(|_| self.cwd.clone());
        // Canonicalize the deepest existing ancestor, then re-append the not-yet-existing tail.
        let mut existing = abs.clone();
        let mut tail = Vec::new();
        while !existing.exists() {
            match (existing.file_name().map(|s| s.to_os_string()), existing.parent()) {
                (Some(name), Some(parent)) => {
                    tail.push(name);
                    existing = parent.to_path_buf();
                }
                _ => break,
            }
        }
        let mut real = existing.canonicalize().unwrap_or(existing);
        for part in tail.iter().rev() {
            real.push(part);
        }
        if !real.starts_with(&root) {
            anyhow::bail!(
                "access denied: '{p}' is outside the workspace ({}). Only files inside the project can be accessed.",
                root.display()
            );
        }
        if write && real.strip_prefix(&root).ok().and_then(|r| r.components().next()).is_some_and(|c| c.as_os_str() == ".git") {
            anyhow::bail!("access denied: writing inside .git/ is not allowed; use git commands via bash instead");
        }
        Ok(abs)
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
