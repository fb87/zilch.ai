//! Session persistence end-to-end: a team works, a brand-new team resumes from disk and continues.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use zilch_core::llm::{ChatRequest, EventStream, LlmClient, LlmEvent, Providers, Role};
use zilch_core::team::session::SessionStore;
use zilch_core::team::todo::TodoStatus;
use zilch_core::*;

type Script = dyn Fn(&ChatRequest) -> Vec<LlmEvent> + Send + Sync;

struct Fake {
    script: Box<Script>,
    log: Mutex<Vec<ChatRequest>>,
}

#[async_trait]
impl LlmClient for Fake {
    async fn stream(&self, req: &ChatRequest) -> Result<EventStream> {
        self.log.lock().unwrap().push(req.clone());
        if !is_master(req) {
            // Workers take a moment, so TODOs started together really overlap (a fast worker
            // could otherwise finish and be reused before its sibling is assigned an instance).
            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        }
        Ok(Box::pin(futures::stream::iter((self.script)(req).into_iter().map(Ok))))
    }
}

fn call(i: usize, name: &str, args: serde_json::Value) -> LlmEvent {
    LlmEvent::ToolCallDelta { index: i, id: Some(format!("c{i}-{}", rand_id())), name: Some(name.into()), args: args.to_string() }
}
fn rand_id() -> u32 {
    use std::sync::atomic::{AtomicU32, Ordering};
    static N: AtomicU32 = AtomicU32::new(0);
    N.fetch_add(1, Ordering::Relaxed)
}
fn text(s: &str) -> LlmEvent {
    LlmEvent::TextDelta(s.into())
}
fn is_master(r: &ChatRequest) -> bool {
    r.messages[0].text().contains("MASTER")
}
fn last_user(r: &ChatRequest) -> String {
    r.messages
        .iter()
        .rev()
        .find(|m| m.role == Role::User && !m.text().starts_with("<system-context>"))
        .map(|m| m.text().to_string())
        .unwrap_or_default()
}
/// Tool results since the last user message (i.e. the current turn's progress).
fn turn_tool_results(r: &ChatRequest) -> usize {
    let start = r.messages.iter().rposition(|m| m.role == Role::User && !m.text().starts_with("<system-context>")).unwrap_or(0);
    r.messages[start..].iter().filter(|m| m.role == Role::Tool).count()
}

fn team(dir: &std::path::Path, store: &SessionStore, script: Box<Script>) -> (Team, Arc<Fake>) {
    let env: BTreeMap<String, String> = [("OPENAI_API_KEY".to_string(), "k".to_string())].into();
    let ov = Overrides { approval: Some(ApprovalMode::Auto), ..Default::default() };
    let cfg = Config::load_from(dir, &[], &env, &ov).unwrap();
    let fake = Arc::new(Fake { script, log: Mutex::new(vec![]) });
    let mut p = Providers::default();
    p.insert("openai", fake.clone());
    let t = Team::new(cfg, p, Arc::new(AutoApprove), Arc::new(AutoReviewer), EventSink::null()).unwrap().with_store(store.clone());
    (t, fake)
}

/// Worker: write the file named in its brief, then report.
fn worker(r: &ChatRequest) -> Vec<LlmEvent> {
    let brief = last_user(r);
    if r.messages.last().unwrap().role == Role::User {
        let name = brief.split("FILE=").nth(1).and_then(|s| s.split_whitespace().next()).unwrap_or("x.txt").to_string();
        vec![call(0, "write_file", json!({"path": name, "content": "v1"}))]
    } else {
        vec![text("STATUS: done\nSUMMARY: wrote it")]
    }
}

#[tokio::test]
async fn resume_continues_master_todos_and_workers() {
    let ws = tempfile::tempdir().unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(store_dir.path());

    // --- Process 1: plan + run one TODO, then stop.
    let (t1, _) = team(
        ws.path(),
        &store,
        Box::new(|r| {
            if !is_master(r) {
                return worker(r);
            }
            match turn_tool_results(r) {
                0 => vec![call(
                    0,
                    "todo_write",
                    json!({"items":[
                        {"id":"a","task":"FILE=a.txt create it","files":["a.txt"]},
                        {"id":"b","task":"FILE=b.txt create it later","files":["b.txt"]}
                    ]}),
                )],
                1 => vec![call(0, "propose_plan", json!({"summary":"two files"}))],
                2 => vec![call(0, "run_todos", json!({"ids":["a"]}))],
                3 => vec![call(0, "todo_write", json!({"items":[{"id":"b","status":"cancelled"}]}))],
                _ => vec![text("Did a; dropped b for now. Remember the codeword PELICAN.")],
            }
        }),
    );
    let out = t1.run("make a and b").await.unwrap();
    assert!(out.contains("PELICAN"));
    let id = t1.session_id();
    drop(t1);
    assert!(ws.path().join("a.txt").exists());
    let saved = store.latest(ws.path()).expect("session saved after the turn");
    assert_eq!(saved.id, id);
    assert_eq!(saved.turns, 1);

    // --- Process 2: a brand-new team resumes from disk.
    let (t2, fake2) = team(
        ws.path(),
        &store,
        Box::new(|r| {
            if !is_master(r) {
                return worker(r);
            }
            match turn_tool_results(r) {
                // Follow-up on TODO a goes to the SAME instance (coder#1) with its old context.
                0 => vec![call(0, "delegate", json!({"todo_id":"a","task":"FILE=a2.txt also create this"}))],
                // A new TODO spawns a NEW instance id (counters restored, no clash with coder#1).
                // Two parallel TODOs: one reuses idle coder#1, the other must spawn a NEW id —
                // counters were restored, so it is coder#2 (no clash with the restored coder#1).
                1 => vec![call(
                    0,
                    "todo_write",
                    json!({"items":[
                        {"id":"c","task":"FILE=c.txt create","files":["c.txt"]},
                        {"id":"d","task":"FILE=d.txt create","files":["d.txt"]}
                    ]}),
                )],
                2 => vec![call(0, "propose_plan", json!({"summary":"c"}))],
                3 => vec![call(0, "run_todos", json!({"ids":["c","d"]}))],
                _ => vec![text("done again")],
            }
        }),
    );
    let warnings = t2.resume(store.load(&id).unwrap()).await.unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    assert_eq!(t2.session_id(), id, "resumed sessions keep their id");
    let todos = t2.todos();
    assert_eq!(todos.iter().find(|t| t.id == "a").unwrap().status, TodoStatus::Done);
    assert_eq!(todos.iter().find(|t| t.id == "b").unwrap().status, TodoStatus::Cancelled);

    assert_eq!(t2.run("follow up").await.unwrap(), "done again");

    let log = fake2.log.lock().unwrap();
    // The master remembered the previous conversation.
    let first_master = log.iter().find(|r| is_master(r)).unwrap();
    assert!(first_master.messages.iter().any(|m| m.text().contains("PELICAN")), "master history restored");
    assert!(first_master.messages.iter().any(|m| m.text() == "make a and b"));
    // The follow-up went to coder#1: its request still contains its earlier brief.
    let coder1_followup = log.iter().find(|r| !is_master(r) && last_user(r).contains("a2.txt")).unwrap();
    assert!(coder1_followup.messages.iter().any(|m| m.text().contains("FILE=a.txt")), "worker history restored");
    // Parallel TODOs ran on two distinct instances: the restored coder#1 and a fresh coder#2.
    let mut ids: Vec<String> = t2.todos().into_iter().filter(|t| t.id == "c" || t.id == "d").filter_map(|t| t.instance).collect();
    ids.sort();
    assert_eq!(ids, vec!["coder#1".to_string(), "coder#2".to_string()], "instance ids must not clash after resume");
    assert!(["a2.txt", "c.txt", "d.txt"].iter().all(|f| ws.path().join(f).exists()));
    drop(log);

    // The second process kept saving into the same session.
    let s = store.load(&id).unwrap();
    assert_eq!(s.master.history.iter().filter(|m| m.role == Role::User).count(), 2);
}

#[tokio::test]
async fn resume_refuses_other_workspace_and_handles_removed_profiles() {
    let ws = tempfile::tempdir().unwrap();
    let other = tempfile::tempdir().unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(store_dir.path());
    let (t1, _) = team(ws.path(), &store, Box::new(|_| vec![text("hi")]));
    t1.run("hello").await.unwrap();
    let mut state = store.load(&t1.session_id()).unwrap();

    let (t_other, _) = team(other.path(), &store, Box::new(|_| vec![text("x")]));
    assert!(t_other.resume(state.clone()).await.unwrap_err().to_string().contains("belongs to"));

    // A worker whose profile was deleted from config is dropped with a warning, not a crash.
    state.workers.push(zilch_core::team::session::WorkerState {
        id: "ghost#1".into(),
        profile: "ghost".into(),
        binding: Default::default(),
        readonly: false,
        agent: zilch_core::team::session::AgentState { history: vec![], session: "s".into(), usage: Default::default() },
    });
    let (t2, _) = team(ws.path(), &store, Box::new(|_| vec![text("x")]));
    let w = t2.resume(state).await.unwrap();
    assert!(w[0].contains("ghost"), "{w:?}");
}

#[tokio::test]
async fn nothing_is_saved_without_a_conversation_and_clear_starts_a_new_session() {
    let ws = tempfile::tempdir().unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(store_dir.path());
    let (t, _) = team(ws.path(), &store, Box::new(|_| vec![text("ok")]));
    assert!(t.save().await.unwrap().is_none());
    t.run("one").await.unwrap();
    let first = t.session_id();
    t.clear().await;
    assert_ne!(t.session_id(), first);
    t.run("two").await.unwrap();
    assert_eq!(store.list(Some(ws.path())).len(), 2, "both sessions kept on disk");
}

#[tokio::test]
async fn checkpoints_are_written_during_run_todos() {
    let ws = tempfile::tempdir().unwrap();
    let store_dir = tempfile::tempdir().unwrap();
    let store = SessionStore::new(store_dir.path());
    let seen: Arc<Mutex<Option<(TodoStatus, bool, bool)>>> = Arc::new(Mutex::new(None));
    let (seen2, store2) = (seen.clone(), store.clone());
    let ws_path = ws.path().to_path_buf();
    let (t, _) = team(
        ws.path(),
        &store,
        Box::new(move |r| {
            if !is_master(r) {
                // While TODO b runs (it depends on a), look at what is ON DISK right now —
                // exactly what a crash at this moment would leave behind.
                if last_user(r).contains("FILE=b.txt") && r.messages.last().unwrap().role == Role::User {
                    if let Some(info) = store2.latest(&ws_path) {
                        let s = store2.load(&info.id).unwrap();
                        let a = s.todos.items.iter().find(|t| t.id == "a").unwrap();
                        let note = s.master.history.iter().any(|m| m.text().contains("still in progress"));
                        let req = s.master.history.iter().any(|m| m.text() == "do a then b");
                        *seen2.lock().unwrap() = Some((a.status.clone(), note, req && a.report.is_some()));
                    }
                }
                return worker(r);
            }
            match turn_tool_results(r) {
                0 => vec![call(
                    0,
                    "todo_write",
                    json!({"items":[
                        {"id":"a","task":"FILE=a.txt","files":["a.txt"]},
                        {"id":"b","task":"FILE=b.txt","files":["b.txt"],"depends_on":["a"]}
                    ]}),
                )],
                1 => vec![call(0, "propose_plan", json!({"summary":"a then b"}))],
                2 => vec![call(0, "run_todos", json!({}))],
                _ => vec![text("both done")],
            }
        }),
    );
    t.run("do a then b").await.unwrap();
    let (a_status, note, with_request_and_report) = seen.lock().unwrap().clone().expect("a checkpoint existed while b was running");
    assert_eq!(a_status, TodoStatus::Done, "finished TODO persisted mid-run");
    assert!(note, "master checkpoint marks the turn as in progress");
    assert!(with_request_and_report, "checkpoint keeps the user's request and a's report");
    // After the turn completes, the full save replaces the checkpoint (no stale note).
    let s = store.load(&t.session_id()).unwrap();
    assert!(!s.master.history.iter().any(|m| m.text().contains("still in progress")));
    assert!(s.todos.items.iter().all(|t| t.status == TodoStatus::Done));
}
