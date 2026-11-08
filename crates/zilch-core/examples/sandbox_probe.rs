//! Runs one command through the real `bash` tool with the default policy (bubblewrap if available).
//! cargo run -p zilch-core --example sandbox_probe -- <workspace> <command>
use std::sync::Arc;
use zilch_core::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
use zilch_core::team::file_locks::FileLocks;
use zilch_core::tools::{bash::Bash, sandbox::bwrap_available, CancelCell, Tool, ToolCtx};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
    let ctx = ToolCtx::new("probe", args[1].clone().into(), gate, EventSink::null(), CancelCell::default(), FileLocks::default());
    println!("bubblewrap available: {}", bwrap_available());
    println!("{}", Bash.call(serde_json::json!({"command": args[2], "timeout_secs": 300}), &ctx).await?);
    Ok(())
}
