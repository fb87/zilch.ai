//! zilch OS drivers: the equipment the kernel uses. LLM calls (real HTTP or scripted), file and
//! search tools, git, and commands in the OS sandbox. The kernel checks capabilities first; the
//! drivers then do the work on the kernel's driver-pool threads.

pub mod config;
pub mod llm;
pub mod tools;

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

/// Per-process content hashes of files read (stale-edit detection).
type ReadHashes = Arc<Mutex<HashMap<PathBuf, String>>>;

pub struct DriverSet {
    cwd: PathBuf,
    policy: Arc<ToolPolicy>,
    tools: ToolRegistry,
    locks: FileLocks,
    hashes: Mutex<HashMap<Pid, ReadHashes>>,
    backends: HashMap<String, Arc<dyn Backend>>,
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
        }
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
                let (pid, bus) = (ctx.pid, ctx.bus);
                match llm::complete(backend.as_ref(), &req, ctx.cancel, |t| {
                    bus.publish(KernelEvent::TextDelta { pid, text: t.to_string() })
                }) {
                    Ok((message, usage, cut)) => SyscallResult::Completion { message, usage, cut },
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
}
