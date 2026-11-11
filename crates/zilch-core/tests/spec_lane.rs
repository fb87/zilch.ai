//! The /spec lane: requirements engineering in parallel with the code lane, versioned documents,
//! and the code master's awareness of plan/design versions.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zilch_core::llm::{ChatRequest, EventStream, LlmClient, LlmEvent, Providers, Role};
use zilch_core::team::session::SessionStore;
use zilch_core::*;

type Script = dyn Fn(&ChatRequest) -> Vec<LlmEvent> + Send + Sync;

struct Fake {
    script: Box<Script>,
    log: Mutex<Vec<ChatRequest>>,
    master_delay: Duration,
}

#[async_trait]
impl LlmClient for Fake {
    async fn stream(&self, req: &ChatRequest) -> Result<EventStream> {
        self.log.lock().unwrap().push(req.clone());
        if is_master(req) {
            tokio::time::sleep(self.master_delay).await; // a slow code lane
        }
        Ok(Box::pin(futures::stream::iter((self.script)(req).into_iter().map(Ok))))
    }
}

fn is_master(r: &ChatRequest) -> bool {
    r.messages[0].text().contains("You are the MASTER")
}
fn is_spec(r: &ChatRequest) -> bool {
    r.messages[0].text().contains("You are the SPEC MASTER")
}
fn call(i: usize, name: &str, args: serde_json::Value) -> LlmEvent {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    LlmEvent::ToolCallDelta {
        index: i,
        id: Some(format!("c{}", N.fetch_add(1, Ordering::Relaxed))),
        name: Some(name.into()),
        args: args.to_string(),
    }
}
fn text(s: &str) -> LlmEvent {
    LlmEvent::TextDelta(s.into())
}
/// Tool results since the last real user message.
fn turn_results(r: &ChatRequest) -> usize {
    let start = r.messages.iter().rposition(|m| m.role == Role::User && !m.text().starts_with("<system-context>")).unwrap_or(0);
    r.messages[start..].iter().filter(|m| m.role == Role::Tool).count()
}
fn last_user(r: &ChatRequest) -> String {
    r.messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User && !m.text().starts_with("<system-context>"))
        .map(|m| m.text().to_string())
        .unwrap_or_default()
}
fn context(r: &ChatRequest) -> String {
    r.messages.iter().rev().find(|m| m.text().starts_with("<system-context>")).map(|m| m.text().to_string()).unwrap_or_default()
}

fn team(dir: &std::path::Path, store: &SessionStore, delay: Duration, script: Box<Script>) -> (Arc<Team>, Arc<Fake>) {
    let env: BTreeMap<String, String> = [("OPENAI_API_KEY".to_string(), "k".to_string())].into();
    let ov = Overrides { approval: Some(ApprovalMode::Auto), spec_dir: Some("docs/spec".into()), ..Default::default() };
    let cfg = Config::load_from(dir, &[], &env, &ov).unwrap();
    let fake = Arc::new(Fake { script, log: Mutex::new(vec![]), master_delay: delay });
    let mut p = Providers::default();
    p.insert("openai", fake.clone());
    let t = Team::new(cfg, p, Arc::new(AutoApprove), Arc::new(AutoReviewer), EventSink::null()).unwrap().with_store(store.clone());
    (Arc::new(t), fake)
}

/// Spec master behaviour: on "release" write+release spec and derive design/plan; otherwise ask.
fn spec_script(r: &ChatRequest) -> Vec<LlmEvent> {
    let msg = last_user(r);
    let n = turn_results(r);
    if msg.contains("v2 change") {
        return match n {
            0 => vec![call(0, "edit_doc", json!({"doc":"spec","old_string":"FR-1 login","new_string":"FR-1 login with SSO"}))],
            1 => vec![call(0, "release_spec", json!({"summary":"SSO login"}))],
            2 => vec![call(0, "edit_doc", json!({"doc":"plan","old_string":"M1: FR-1","new_string":"M1: FR-1 (SSO)"}))],
            _ => vec![text("Released v2; plan updated.")],
        };
    }
    if msg.contains("looks good, release it") {
        return match n {
            0 => vec![call(0, "release_spec", json!({"summary":"initial requirements"}))],
            1 => vec![call(0, "write_doc", json!({"doc":"design","content":"# Design\n- auth module (FR-1)"}))],
            2 => vec![call(0, "write_doc", json!({"doc":"plan","content":"# Plan\n- M1: FR-1 login form"}))],
            _ => vec![text("Spec v1 released; design and plan written.")],
        };
    }
    match n {
        0 => vec![call(
            0,
            "write_doc",
            json!({"doc":"spec","content":"# App — Specification\n## 4. Functional requirements\n- FR-1 login\n## Changelog\n"}),
        )],
        _ => vec![text("Draft updated. Q1: Who are the users? Q2: Is SSO required?")],
    }
}

#[tokio::test]
async fn spec_lane_versions_documents_and_runs_in_parallel_with_code() {
    let ws = tempfile::tempdir().unwrap();
    let sdir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(sdir.path());
    let (t, fake) = team(
        ws.path(),
        &store,
        Duration::from_millis(700),
        Box::new(|r| {
            if is_spec(r) {
                return spec_script(r);
            }
            // Code master: one slow read-only step, then an answer.
            match turn_results(r) {
                0 => vec![call(0, "glob", json!({"pattern":"*"}))],
                _ => vec![text("code lane done")],
            }
        }),
    );

    // 1) Interview → draft.
    let a = t.run_spec("I want a small web app").await.unwrap();
    assert!(a.contains("Q1"), "{a}");
    let st = t.spec_status();
    assert!(st.spec_exists && st.released == 0 && st.unreleased_changes);
    assert!(ws.path().join("docs/spec/spec.md").exists(), "user-chosen spec dir is used");

    // 2) Release → v1, design + plan based on v1.
    t.run_spec("looks good, release it").await.unwrap();
    let st = t.spec_status();
    assert_eq!((st.released, st.design_based_on, st.plan_based_on), (1, Some(1), Some(1)));

    // 3) Parallel: a slow code turn runs while the spec lane releases v2.
    let code = {
        let t = t.clone();
        tokio::spawn(async move {
            let s = Instant::now();
            let out = t.run("implement the plan").await.unwrap();
            (out, s.elapsed(), Instant::now())
        })
    };
    tokio::time::sleep(Duration::from_millis(100)).await; // code lane is now mid-turn
    let spec_started = Instant::now();
    t.run_spec("v2 change: we need SSO").await.unwrap();
    let spec_done = Instant::now();
    let (out, code_took, code_done) = code.await.unwrap();
    assert_eq!(out, "code lane done");
    assert!(spec_done < code_done, "spec turn finished while the code turn was still running");
    assert!(spec_done - spec_started < Duration::from_millis(500), "spec lane was not blocked by the code lane");
    assert!(code_took >= Duration::from_millis(1400), "code lane really was slow ({code_took:?})");
    assert_eq!(t.spec_status().released, 2);

    // 4) The code master works from plan/design and sees versions + updates.
    let log = fake.log.lock().unwrap().clone();
    let master_ctx: Vec<String> = log.iter().filter(|r| is_master(r)).map(context).collect();
    assert!(
        master_ctx[0].contains("plan.md: based on spec v1") && master_ctx[0].contains("(never from spec.md); otherwise ignore them"),
        "{}",
        master_ctx[0]
    );
    assert!(master_ctx.last().unwrap().contains("plan.md was updated since you last looked"), "{:?}", master_ctx.last());
    // The code master never got spec-writing tools; the spec master never got coordination ones.
    let master_tools: Vec<String> = log.iter().find(|r| is_master(r)).unwrap().tools.iter().map(|t| t.function.name.clone()).collect();
    assert!(
        !master_tools.iter().any(|n| ["write_doc", "edit_doc", "release_spec", "write_file", "bash"].contains(&n.as_str())),
        "{master_tools:?}"
    );
    let spec_tools: Vec<String> = log.iter().find(|r| is_spec(r)).unwrap().tools.iter().map(|t| t.function.name.clone()).collect();
    assert!(!spec_tools.iter().any(|n| ["delegate", "run_todos", "write_file", "bash"].contains(&n.as_str())), "{spec_tools:?}");

    // 5) Stale detection when the spec moves ahead of the plan.
    std::fs::write(ws.path().join("docs/spec/spec.md"), "# App\n- FR-1 login with SSO\n- FR-2 audit log\n").unwrap();
    let store_dir = ws.path().join("docs/spec");
    let spec_store = zilch_core::tools::spec_tools::SpecStore::new(&store_dir);
    spec_store.release("audit log").unwrap();
    assert!(t.spec_status().plan_is_stale());
    t.run("continue").await.unwrap();
    let last_ctx = fake.log.lock().unwrap().iter().rev().find(|r| is_master(r)).map(context).unwrap();
    assert!(last_ctx.contains("STALE") && last_ctx.contains("WARNING"), "{last_ctx}");

    // 6) Both lanes' conversations persist and resume.
    let id = t.session_id();
    let saved = store.load(&id).unwrap();
    assert!(saved.spec.as_ref().is_some_and(|s| s.history.iter().any(|m| m.text().contains("v2 change"))), "latest spec turn saved");
    let (t2, fake2) =
        team(ws.path(), &store, Duration::ZERO, Box::new(|r| if is_spec(r) { vec![text("resumed")] } else { vec![text("ok")] }));
    t2.resume(saved).await.unwrap();
    t2.run_spec("where were we?").await.unwrap();
    let first_spec_req = fake2.log.lock().unwrap().iter().find(|r| is_spec(r)).cloned().unwrap();
    assert!(first_spec_req.messages.iter().any(|m| m.text() == "I want a small web app"), "spec history restored");
}
