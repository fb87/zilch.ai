//! Time the grep tool: `cargo run --release -p zilch-core --example grep_bench -- <dir> <pattern>...`.
use std::sync::Arc;
use zilch_core::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
use zilch_core::team::file_locks::FileLocks;
use zilch_core::tools::{search::GrepTool, CancelCell, Tool, ToolCtx};

#[tokio::main]
async fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let dir = std::path::PathBuf::from(&args[0]);
    let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
    let ctx = ToolCtx::new("bench", dir, gate, EventSink::null(), CancelCell::default(), FileLocks::default());
    for p in &args[1..] {
        let t = std::time::Instant::now();
        let out = GrepTool.call(serde_json::json!({ "pattern": p }), &ctx).await.unwrap();
        let first = out.lines().next().unwrap_or("").chars().take(90).collect::<String>();
        println!("{:>7.2}s  {p:<32} {} lines; first: {first}", t.elapsed().as_secs_f64(), out.lines().count());
    }
}
