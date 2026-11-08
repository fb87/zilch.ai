//! LIVE corner-case tests of the agent loop against a real FREE model, with READ-ONLY tools on a
//! temp directory (no writes, no shell). Opt-in:
//!   OPENCODE_API_KEY=... cargo test -p zilch-core --test live_corners -- --ignored --nocapture --test-threads=2
//! Optional: ZILCH_LIVE_MODEL (default "beam-free"), ZILCH_LIVE_BASE_URL.

use std::sync::Arc;
use std::time::{Duration, Instant};
use zilch_core::agent::run::new_session_id;
use zilch_core::agent::{Agent, ApprovalGate, ApprovalMode, AutoApprove, EventSink};
use zilch_core::llm::openai::OpenAiClient;
use zilch_core::llm::ModelBinding;
use zilch_core::team::file_locks::FileLocks;
use zilch_core::tools::{worker_tools, CancelCell, ToolCtx};

const SYSTEM: &str = "You are a precise assistant inspecting a project with read-only tools. \
Use tools when needed. Answer briefly and exactly.";

fn agent(dir: &std::path::Path, model: Option<&str>, max_steps: usize) -> Agent {
    let key = std::env::var("OPENCODE_API_KEY").expect("set OPENCODE_API_KEY");
    let base = std::env::var("ZILCH_LIVE_BASE_URL").unwrap_or_else(|_| "https://opencode.ai/inference/go/openai/v1".into());
    let model = model.map(str::to_string).unwrap_or_else(|| std::env::var("ZILCH_LIVE_MODEL").unwrap_or_else(|_| "beam-free".into()));
    let llm = Arc::new(OpenAiClient::new("live", base, Some(key), 2).with_session_header(Some("x-opencode-session".into())));
    let gate = Arc::new(ApprovalGate::new(ApprovalMode::Ask, Arc::new(AutoApprove)));
    let ctx = Arc::new(ToolCtx::new("tester", dir.to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default()));
    Agent {
        id: "tester".into(),
        profile: "explorer".into(),
        llm,
        binding: ModelBinding { provider: "live".into(), model, ..Default::default() },
        system: SYSTEM.into(),
        // Read-only: never give a live model write or shell tools in tests.
        tools: worker_tools().filter(|t| !t.mutating()),
        history: vec![],
        max_steps,
        context_tokens: 128_000,
        ctx,
        hooks: None,
        usage: Default::default(),
        session: new_session_id("tester"),
    }
}

#[tokio::test]
#[ignore = "live"]
async fn unicode_paths_and_content() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(dir.path().join("tài_liệu")).unwrap();
    std::fs::write(dir.path().join("tài_liệu/ghi_chú.md"), "Mật khẩu kho: 🦀-ĐẶC-BIỆT-42\n日本語の行\n").unwrap();
    let mut a = agent(dir.path(), None, 10);
    let out = a.run("Find the markdown file in this project and tell me the exact 'Mật khẩu kho' value. Reply with only the value.").await.unwrap();
    println!("unicode → {out}");
    assert!(out.contains("🦀-ĐẶC-BIỆT-42"), "got: {out}");
}

#[tokio::test]
#[ignore = "live"]
async fn huge_outputs_are_truncated_not_fatal() {
    let dir = tempfile::tempdir().unwrap();
    let big: String = (0..40_000).map(|i| format!("line {i}: NEEDLE filler filler filler\n")).collect();
    std::fs::write(dir.path().join("big.log"), big).unwrap();
    std::fs::write(dir.path().join("one_line.txt"), "x".repeat(500_000) + "END").unwrap();
    let mut a = agent(dir.path(), None, 12);
    let out = a
        .run("grep for NEEDLE in this project and read one_line.txt. How many matches did grep report (it may say some were not shown)? And is the file one_line.txt a single line? Answer in one sentence.")
        .await
        .unwrap();
    println!("huge → {out}");
    // The agent survives multi-MB inputs; tool outputs were bounded.
    let biggest = a.history.iter().filter_map(|m| m.content.as_ref()).map(String::len).max().unwrap();
    assert!(biggest < 60_000, "a tool result of {biggest} bytes reached the model");
    assert!(!out.trim().is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn missing_files_and_bad_tool_input_are_recoverable() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("real.txt"), "the real content is BANANA").unwrap();
    let mut a = agent(dir.path(), None, 10);
    let out = a.run("Read the file ghost.txt. If it does not exist, read real.txt instead and tell me the uppercase word in it.").await.unwrap();
    println!("missing → {out}");
    assert!(out.contains("BANANA"), "got: {out}");
    // The model may detect the missing file via glob or via a read_file error; both are fine.
}

#[tokio::test]
#[ignore = "live"]
async fn unknown_model_fails_fast_with_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = agent(dir.path(), Some("definitely-not-a-model-xyz"), 3);
    let t = Instant::now();
    let err = a.run("hi").await.unwrap_err().to_string();
    println!("bad model ({:?}) → {err}", t.elapsed());
    assert!(err.contains("returned 4"), "expected a 4xx surfaced to the user: {err}");
    assert!(t.elapsed() < Duration::from_secs(30), "4xx must not be retried");
    // The failed turn was rolled back: switching to a working model, the agent is fully usable.
    assert!(a.history.is_empty(), "failed request left {} messages in history", a.history.len());
    a.binding.model = std::env::var("ZILCH_LIVE_MODEL").unwrap_or_else(|_| "beam-free".into());
    let ok = a.run("Reply with exactly the word: PONG").await.unwrap();
    println!("after recovery → {ok}");
    assert!(ok.contains("PONG"));
}

#[tokio::test]
#[ignore = "live"]
async fn cancellation_is_prompt() {
    let dir = tempfile::tempdir().unwrap();
    for i in 0..30 {
        std::fs::write(dir.path().join(format!("f{i}.txt")), format!("file {i}")).unwrap();
    }
    let mut a = agent(dir.path(), None, 40);
    let cancel = a.ctx.cancel.clone();
    tokio::spawn(async move {
        tokio::time::sleep(Duration::from_millis(1500)).await;
        cancel.cancel();
    });
    let t = Instant::now();
    let res = a.run("Read every .txt file one at a time, one read_file call per turn, then list their contents.").await;
    println!("cancel ({:?}) → {res:?}", t.elapsed());
    assert!(res.unwrap_err().to_string().contains("cancelled"));
    assert!(t.elapsed() < Duration::from_secs(5), "cancellation took {:?}", t.elapsed());
}

#[tokio::test]
#[ignore = "live"]
async fn step_limit_stops_gracefully() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "alpha").unwrap();
    let mut a = agent(dir.path(), None, 1);
    let out = a.run("Read a.txt and tell me what it says.").await.unwrap();
    println!("max_steps=1 → {out}");
    // Either it answered without tools in one step, or it hit the limit — never an error/hang.
    assert!(out.contains("max_steps=1") || out.to_lowercase().contains("alpha") || !out.is_empty());
}

#[tokio::test]
#[ignore = "live"]
async fn blank_input_does_not_break_the_loop() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = agent(dir.path(), None, 3);
    let r1 = a.run("   ").await;
    assert!(r1.as_ref().unwrap_err().to_string().contains("empty input"), "blank input must be rejected locally");
    assert!(a.history.is_empty(), "rejected input must not enter history");
    println!("blank → {r1:?}");
    // Whatever the provider does with a blank message, the agent stays usable afterwards.
    let r2 = a.run("Reply with exactly the word: PONG").await.unwrap();
    println!("after blank → {r2}");
    assert!(r2.contains("PONG"));
}
