//! zilch OS drivers: the equipment the kernel uses. LLM calls (real HTTP or scripted), file and
//! search tools, git, and commands in the OS sandbox. The kernel checks capabilities first; the
//! drivers then do the work on the kernel's driver-pool threads.

pub mod config;
pub mod llm;
pub mod session;
pub mod tools;

use llm::catalog::CostMeter;
use llm::Backend;
use serde_json::Value;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tools::{FileLocks, ToolCtx, ToolPolicy, ToolRegistry};
use zilch_ipc::{KernelEvent, Pid, PlanTask, Syscall, SyscallResult, TicketId, ToolSpec};
use zilch_kernel::{Access, DriverCtx, Drivers};

/// Tools of each group (the kernel's capabilities restrict their writes further).
pub const ARCHITECT_TOOLS: &[&str] = &["read_file", "write_file", "edit_file", "glob", "grep", "git"];
pub const DEVELOPER_TOOLS: &[&str] = &["read_file", "write_file", "edit_file", "glob", "grep", "git", "bash"];
/// The spec lane: read the code, write only the documents of the spec directory.
pub const SPEC_TOOLS: &[&str] = &["read_file", "glob", "grep", "git", "write_doc", "edit_doc", "release_spec"];
/// Explorers: read-only.
pub const EXPLORER_TOOLS: &[&str] = &["read_file", "glob", "grep", "git"];

/// Per-process content hashes of files read (stale-edit detection).
type ReadHashes = Arc<Mutex<HashMap<PathBuf, String>>>;

pub struct DriverSet {
    cwd: PathBuf,
    policy: Arc<ToolPolicy>,
    tools: ToolRegistry,
    locks: FileLocks,
    hashes: Mutex<HashMap<Pid, ReadHashes>>,
    backends: HashMap<String, Arc<dyn Backend>>,
    /// Where the main agent's conversation is saved (None = no sessions).
    session_file: Mutex<Option<PathBuf>>,
    /// Prices, spend and the dollar cap.
    meter: Arc<CostMeter>,
    /// Session token cap: (limit, used).
    token_cap: Option<(u64, std::sync::atomic::AtomicU64)>,
}

impl DriverSet {
    pub fn new(cwd: &Path, policy: ToolPolicy, backends: HashMap<String, Arc<dyn Backend>>) -> Self {
        Self {
            cwd: cwd.to_path_buf(),
            policy: Arc::new(policy),
            tools: tools::all_tools(),
            locks: FileLocks::default(),
            hashes: Mutex::default(),
            backends,
            session_file: Mutex::new(None),
            meter: Arc::new(CostMeter::new(None)),
            token_cap: None,
        }
    }

    /// Prices and the session caps (checked before each model call).
    pub fn with_costs(mut self, meter: Arc<CostMeter>, max_session_tokens: Option<u64>) -> Self {
        self.meter = meter;
        self.token_cap = max_session_tokens.map(|l| (l, std::sync::atomic::AtomicU64::new(0)));
        self
    }

    pub fn meter(&self) -> Arc<CostMeter> {
        self.meter.clone()
    }

    /// Providers this driver set can call.
    pub fn providers(&self) -> Vec<String> {
        let mut v: Vec<String> = self.backends.keys().cloned().collect();
        v.sort();
        v
    }

    /// The spec lane's documents live in `dir` (write_doc / edit_doc / release_spec).
    pub fn with_spec_dir(mut self, dir: PathBuf) -> Self {
        let store = Arc::new(tools::spec_tools::SpecStore::new(dir));
        self.tools.register(Arc::new(tools::spec_tools::WriteDoc(store.clone())));
        self.tools.register(Arc::new(tools::spec_tools::EditDoc(store.clone())));
        self.tools.register(Arc::new(tools::spec_tools::ReleaseSpec(store)));
        self
    }

    /// Save the main agent's conversation to this file after every served order.
    pub fn with_session_file(self, path: PathBuf) -> Self {
        *self.session_file.lock().unwrap() = Some(path);
        self
    }

    /// Save from now on to this file instead (/clear starts a new session).
    pub fn set_session_file(&self, path: Option<PathBuf>) {
        *self.session_file.lock().unwrap() = path;
    }

    pub fn specs(&self, names: &[&str]) -> Vec<ToolSpec> {
        self.tools.specs(names)
    }

    fn tool_ctx(&self, ctx: &DriverCtx) -> ToolCtx {
        let hashes = self.hashes.lock().unwrap().entry(ctx.pid).or_default().clone();
        ToolCtx {
            agent_id: ctx.pid.to_string(),
            cwd: self.cwd.clone(),
            policy: self.policy.clone(),
            cancel: ctx.cancel.clone(),
            locks: self.locks.clone(),
            read_hashes: hashes,
            progress: None,
            alive: Some(ctx.alive.clone()),
        }
    }
}

impl Drivers for DriverSet {
    fn classify(&self, name: &str, args: &Value) -> Access {
        self.tools.classify(&self.cwd, self.policy.sandbox, name, args)
    }

    fn run(&self, ctx: &DriverCtx, call: Syscall) -> SyscallResult {
        match call {
            Syscall::Complete(req) => {
                let Some(backend) = self.backends.get(&req.binding.provider) else {
                    return SyscallResult::Failed(format!("unknown provider '{}'", req.binding.provider));
                };
                // Session caps: refuse to send a request that would go over them.
                let key = req.binding.key();
                let estimate = (serde_json::to_string(&req.messages).map(|s| s.len()).unwrap_or(0) / 4) as u64;
                if let Some((limit, used)) = &self.token_cap {
                    let used = used.load(std::sync::atomic::Ordering::Relaxed);
                    if used + estimate > *limit {
                        return SyscallResult::Failed(format!(
                            "session token cap reached ({used} used + ~{estimate} for this request > {limit}); raise max_session_tokens to continue"
                        ));
                    }
                }
                if self.meter.would_exceed(&key, estimate) {
                    return SyscallResult::Failed(format!(
                        "session cost cap of ${:.4} would be exceeded (spent ${:.4}); raise max_session_cost to continue",
                        self.meter.limit_usd.unwrap_or(0.0),
                        self.meter.spent()
                    ));
                }
                let (pid, bus) = (ctx.pid, ctx.bus);
                match llm::complete(backend.as_ref(), &req, ctx.cancel, &*ctx.alive, |t| {
                    bus.publish(KernelEvent::TextDelta { pid, text: t.to_string() })
                }) {
                    Ok((message, usage, cut)) => {
                        self.meter.record(&key, &usage);
                        if let Some((_, used)) = &self.token_cap {
                            used.fetch_add(usage.prompt_tokens + usage.completion_tokens, std::sync::atomic::Ordering::Relaxed);
                        }
                        SyscallResult::Completion { message, usage, cut }
                    }
                    Err(e) => SyscallResult::Failed(format!("{e:#}")),
                }
            }
            Syscall::Tool { name, args, .. } => {
                let Some(tool) = self.tools.get(&name) else { return SyscallResult::Failed(format!("unknown tool '{name}'")) };
                let tctx = self.tool_ctx(ctx);
                match tool.call(args, &tctx) {
                    Ok(out) => SyscallResult::Output(out),
                    // Tool failures are for the model to read and react to.
                    Err(e) => SyscallResult::Output(format!("Error: {e:#}")),
                }
            }
            other => SyscallResult::Failed(format!("not a driver call: {}", other.summary())),
        }
    }

    fn save_plan(&self, _ticket: TicketId, tasks: &[PlanTask]) {
        let dir = self.cwd.join("docs");
        let _ = std::fs::create_dir_all(&dir);
        if let Ok(json) = serde_json::to_string_pretty(&serde_json::json!({ "tasks": tasks })) {
            let _ = std::fs::write(dir.join("plan.json"), json + "\n");
        }
    }

    fn release(&self, pid: Pid) {
        self.hashes.lock().unwrap().remove(&pid);
    }

    fn save_session(&self, lane: zilch_ipc::Group, snapshot: &Value) {
        if let Some(path) = self.session_file.lock().unwrap().clone() {
            let field = if lane == zilch_ipc::Group::Spec { "spec" } else { "snapshot" };
            let _ = session::save_field(&path, &self.cwd, field, snapshot);
        }
    }
}
