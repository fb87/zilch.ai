//! Offline corner cases of the agent loop: malformed model output a real model rarely produces
//! on demand (unknown tools, broken JSON, wrong argument shapes, provider failures).

use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use zilch_core::agent::run::new_session_id;
use zilch_core::agent::{Agent, ApprovalGate, ApprovalMode, AutoApprove, EventSink};
use zilch_core::llm::{ChatRequest, EventStream, LlmClient, LlmEvent, ModelBinding, Role};
use zilch_core::team::file_locks::FileLocks;
use zilch_core::tools::{worker_tools, CancelCell, ToolCtx};

/// Replays one scripted response per request; `None` = provider error.
struct Scripted(Mutex<Vec<Option<Vec<LlmEvent>>>>);

#[async_trait]
impl LlmClient for Scripted {
    async fn stream(&self, _req: &ChatRequest) -> Result<EventStream> {
        let next = self.0.lock().unwrap().remove(0);
        match next {
            Some(evs) => Ok(Box::pin(futures::stream::iter(evs.into_iter().map(Ok)))),
            None => anyhow::bail!("provider returned 503 Service Unavailable"),
        }
    }
}

fn call(index: usize, name: &str, args: &str) -> LlmEvent {
    LlmEvent::ToolCallDelta { index, id: Some(format!("c{index}")), name: Some(name.into()), args: args.into() }
}

fn agent(dir: &std::path::Path, script: Vec<Option<Vec<LlmEvent>>>) -> Agent {
    let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
    Agent {
        id: "t".into(),
        profile: "coder".into(),
        llm: Arc::new(Scripted(Mutex::new(script))),
        binding: ModelBinding::default(),
        system: "s".into(),
        tools: worker_tools(),
        history: vec![],
        max_steps: 10,
        loop_limit: 3,
        context_tokens: 128_000,
        ctx: Arc::new(ToolCtx::new("t", dir.to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default())),
        hooks: None,
        usage: Default::default(),
        session: new_session_id("t"),
    }
}

#[tokio::test]
async fn malformed_tool_calls_become_errors_for_the_model() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
    let mut a = agent(
        dir.path(),
        vec![
            Some(vec![
                call(0, "teleport", "{}"),                    // unknown tool
                call(1, "read_file", "{\"path\": \"a.txt\""), // truncated JSON
                call(2, "read_file", "[\"a.txt\"]"),          // not an object
                call(3, "read_file", "{\"pth\":\"a.txt\"}"),  // missing required arg
                call(4, "read_file", "{\"path\":\"a.txt\"}"), // valid, in the same batch
                call(5, "", "{}"),                            // empty name
            ]),
            Some(vec![LlmEvent::TextDelta("recovered".into())]),
        ],
    );
    assert_eq!(a.run("go").await.unwrap(), "recovered");
    let results: Vec<String> = a.history.iter().filter(|m| m.role == Role::Tool).map(|m| m.text().to_string()).collect();
    assert_eq!(results.len(), 6, "every call gets exactly one result, in order");
    assert!(results[0].contains("unknown tool 'teleport'"));
    assert!(results[1].contains("must be a JSON object"));
    assert!(results[2].contains("must be a JSON object"));
    assert!(results[3].contains("missing string argument 'path'"));
    assert!(results[4].contains("hello"), "a valid call in a bad batch still runs");
    assert!(results[5].contains("unknown tool"));
}

#[tokio::test]
async fn provider_failure_mid_task_keeps_completed_work_and_history_consistent() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
    let mut a = agent(dir.path(), vec![Some(vec![call(0, "read_file", "{\"path\":\"a.txt\"}")]), None]);
    let err = a.run("go").await.unwrap_err().to_string();
    assert!(err.contains("503"));
    // The model already acted, so history is kept (not rolled back) and stays well-formed:
    // the assistant tool call is followed by its result.
    let roles: Vec<Role> = a.history.iter().map(|m| m.role).collect();
    assert_eq!(roles, vec![Role::User, Role::Assistant, Role::Tool]);
}

#[tokio::test]
async fn provider_failure_before_any_reply_rolls_back() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = agent(dir.path(), vec![None, Some(vec![LlmEvent::TextDelta("ok".into())])]);
    assert!(a.run("first").await.is_err());
    assert!(a.history.is_empty(), "failed turn must not poison history");
    assert_eq!(a.run("second").await.unwrap(), "ok");
    assert_eq!(a.history.len(), 2);
}

#[tokio::test]
async fn empty_and_reasoning_only_replies_are_handled() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = agent(dir.path(), vec![Some(vec![LlmEvent::ReasoningDelta("thinking…".into()), LlmEvent::Finish("stop".into())])]);
    // A reply with no text and no tool calls ends the turn with an empty answer, not a hang or panic.
    assert_eq!(a.run("go").await.unwrap(), "");
    assert_eq!(a.history[1].reasoning_content.as_deref(), Some("thinking…"));
}

#[tokio::test]
async fn caps_refuse_before_sending_and_costs_are_recorded() {
    use zilch_core::llm::catalog::{CostMeter, ModelInfo};
    use zilch_core::tools::{TokenBudget, ToolPolicy};
    let dir = tempfile::tempdir().unwrap();

    // Token cap: the next request alone would exceed it → refused with NO provider call
    // (the script is empty, so any call would panic).
    let mut a = agent(dir.path(), vec![]);
    let ctx =
        ToolCtx::new("t", dir.path().to_path_buf(), a.ctx.gate.clone(), EventSink::null(), CancelCell::default(), FileLocks::default())
            .with_policy(Arc::new(ToolPolicy::default()), Some(Arc::new(TokenBudget::new(50))));
    a.ctx = Arc::new(ctx);
    let err = a.run("a request whose prompt (system prompt + tool schemas) is well over fifty tokens").await.unwrap_err().to_string();
    assert!(err.contains("token budget reached"), "{err}");

    // Cost: recorded per response; the cap refuses the next request once it would be exceeded.
    let meter = Arc::new(CostMeter::new(Some(0.02)));
    meter.catalog.write().unwrap().insert("/".into(), ModelInfo { context: None, input_per_mtok: Some(10.0), output_per_mtok: Some(10.0) });
    let mut b = agent(
        dir.path(),
        vec![Some(vec![
            LlmEvent::Usage(zilch_core::llm::Usage { prompt_tokens: 1500, completion_tokens: 10, ..Default::default() }),
            LlmEvent::TextDelta("ok".into()),
        ])],
    );
    let ctx =
        ToolCtx::new("t", dir.path().to_path_buf(), b.ctx.gate.clone(), EventSink::null(), CancelCell::default(), FileLocks::default())
            .with_meter(meter.clone());
    b.ctx = Arc::new(ctx);
    assert_eq!(b.run("first").await.unwrap(), "ok");
    assert!((meter.spent() - 0.0151).abs() < 1e-9, "spent {}", meter.spent());
    // First request: ~0.007 estimated < 0.02. Second: 0.0151 spent + ~0.007 > 0.02 → refused.
    assert!(b.history.len() == 2);
    let err = b.run("second").await.unwrap_err().to_string();
    assert!(err.contains("cost cap"), "{err}");
}

#[tokio::test]
async fn a_model_stuck_repeating_one_call_is_nudged_then_stopped() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("a.txt"), "hello").unwrap();
    // The model would make the same failing edit forever (10 scripted steps = max_steps).
    let edit = || Some(vec![call(0, "edit_file", r#"{"path":"a.txt","old_string":"nope","new_string":"x"}"#)]);
    let mut a = agent(dir.path(), (0..10).map(|_| edit()).collect());
    let out = a.run("go").await.unwrap();
    assert!(out.starts_with("[stopped by the supervisor:") && out.contains("STATUS: failed"), "{out}");
    let steps = a.history.iter().filter(|m| m.role == Role::Assistant && !m.tool_calls.is_empty()).count();
    assert_eq!(steps, 4, "nudged after 3 identical calls, stopped on the 4th instead of using all 10 steps");
    assert!(a.history.iter().any(|m| m.role == Role::User && m.text().starts_with("[supervisor]")), "the nudge is in history");
}

#[tokio::test]
async fn a_reply_that_keeps_repeating_while_streaming_is_cut() {
    let dir = tempfile::tempdir().unwrap();
    let spam: Vec<LlmEvent> = (0..500).map(|_| LlmEvent::TextDelta("Let me check that again. ".into())).collect();
    let mut a = agent(dir.path(), vec![Some(spam), Some(vec![LlmEvent::TextDelta("Done, sorry.".into())])]);
    assert_eq!(a.run("go").await.unwrap(), "Done, sorry.", "after the cut and the nudge the model answers normally");
    let cut = a.history.iter().find(|m| m.text().contains("cut by the supervisor")).expect("cut reply kept, shortened");
    assert!(cut.text().len() < 600, "only a short head of the runaway text stays in history");
}

/// Different read-only calls, one per model call (varied so the loop supervisor stays quiet).
fn varied_reads(n: usize) -> Vec<Option<Vec<LlmEvent>>> {
    (0..n).map(|i| Some(vec![call(0, "glob", &format!("{{\"pattern\":\"*{i}*\"}}"))])).collect()
}

#[tokio::test]
async fn out_of_budget_an_agent_is_asked_to_wrap_up_and_its_report_is_kept() {
    let dir = tempfile::tempdir().unwrap();
    // Same glob result every time ("No files found"): no new information, so no extension.
    let mut script = varied_reads(10);
    script.push(Some(vec![LlmEvent::TextDelta("parts 1-2 done, part 3 left\nSTATUS: failed".into())]));
    let mut a = agent(dir.path(), script);
    a.loop_limit = 0;
    let out = a.run("go").await.unwrap();
    assert_eq!(out, "parts 1-2 done, part 3 left\nSTATUS: failed", "the wrap-up report, not a stop note");
    assert!(a.history.iter().any(|m| m.role == Role::User && m.text().contains("used its budget of 10 model calls")));
}

#[tokio::test]
async fn an_agent_ignoring_the_wrap_up_is_stopped_after_the_grace() {
    let dir = tempfile::tempdir().unwrap();
    let mut a = agent(dir.path(), varied_reads(40));
    a.loop_limit = 0;
    let out = a.run("go").await.unwrap();
    assert!(out.starts_with("[stopped after reaching max_steps=10 (15 model calls"), "{out}");
}

#[tokio::test]
async fn an_agent_making_progress_gets_more_calls() {
    let dir = tempfile::tempdir().unwrap();
    // Writes a new file on every call: progress, so 14 calls are fine with a budget of 10.
    let mut script: Vec<_> =
        (0..14).map(|i| Some(vec![call(0, "write_file", &format!("{{\"path\":\"f{i}.txt\",\"content\":\"x\"}}"))])).collect();
    script.push(Some(vec![LlmEvent::TextDelta("all written".into())]));
    let mut a = agent(dir.path(), script);
    a.loop_limit = 0;
    assert_eq!(a.run("go").await.unwrap(), "all written");
    assert!(dir.path().join("f13.txt").exists());
}
