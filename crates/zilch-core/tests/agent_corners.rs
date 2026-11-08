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
