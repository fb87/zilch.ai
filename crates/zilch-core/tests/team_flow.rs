//! End-to-end tests of the master → TODO → parallel workers flow with a scripted in-process LLM.

use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zilch_core::agent::{ApprovalRequest, Decision};
use zilch_core::llm::{ChatRequest, EventStream, LlmClient, LlmEvent, Providers, Role};
use zilch_core::team::todo::TodoStatus;
use zilch_core::*;

type Script = dyn Fn(&ChatRequest) -> Vec<LlmEvent> + Send + Sync;

/// Scripted LLM. Worker turns sleep to make concurrency measurable.
struct FakeLlm {
    script: Box<Script>,
    worker_delay: Duration,
    requests: Mutex<Vec<ChatRequest>>,
}

#[async_trait]
impl LlmClient for FakeLlm {
    async fn stream(&self, req: &ChatRequest) -> Result<EventStream> {
        self.requests.lock().unwrap().push(req.clone());
        let is_worker = !req.messages[0].text().contains("MASTER");
        if is_worker {
            tokio::time::sleep(self.worker_delay).await;
        }
        let evs = (self.script)(req);
        Ok(Box::pin(futures::stream::iter(evs.into_iter().map(Ok))))
    }
}

fn call(id: &str, name: &str, args: serde_json::Value) -> LlmEvent {
    LlmEvent::ToolCallDelta {
        index: id.trim_start_matches(|c: char| !c.is_ascii_digit()).parse().unwrap_or(0),
        id: Some(id.into()),
        name: Some(name.into()),
        args: args.to_string(),
    }
}
fn text(s: &str) -> LlmEvent {
    LlmEvent::TextDelta(s.into())
}
fn assistant_turns(req: &ChatRequest) -> usize {
    req.messages.iter().filter(|m| m.role == Role::Assistant).count()
}
fn last_user(req: &ChatRequest) -> String {
    req.messages.iter().rev().find(|m| m.role == Role::User).map(|m| m.text().to_string()).unwrap_or_default()
}

/// Denies everything: proves that plan-scoped writes need no prompt.
struct DenyAll(Mutex<Vec<String>>);
#[async_trait]
impl Approver for DenyAll {
    async fn approve(&self, req: &ApprovalRequest) -> Decision {
        self.0.lock().unwrap().push(format!("{}:{}", req.agent, req.tool));
        Decision::Deny
    }
}

fn setup(
    dir: &std::path::Path,
    max_workers: usize,
    script: Box<Script>,
) -> (Team, Arc<FakeLlm>, Arc<DenyAll>, tokio::sync::mpsc::UnboundedReceiver<AgentEvent>) {
    let env: BTreeMap<String, String> = [("OPENAI_API_KEY".to_string(), "test".to_string())].into();
    let ov = Overrides { max_workers: Some(max_workers), approval: Some(ApprovalMode::Plan), ..Default::default() };
    let cfg = Config::load_from(dir, &[], &env, &ov).unwrap();
    let fake = Arc::new(FakeLlm { script, worker_delay: Duration::from_millis(150), requests: Mutex::new(vec![]) });
    let mut providers = Providers::default();
    providers.insert("openai", fake.clone());
    let deny = Arc::new(DenyAll(Mutex::new(vec![])));
    let (sink, rx) = EventSink::new();
    let team = Team::new(cfg, providers, deny.clone(), Arc::new(AutoReviewer), sink).unwrap();
    (team, fake, deny, rx)
}

/// Worker behaviour: write `<id>.txt` for its TODO, then report.
fn worker_script(req: &ChatRequest) -> Vec<LlmEvent> {
    // Instances persist across TODOs, so the current brief is the latest user message.
    let brief = last_user(req);
    let id = brief.split('[').nth(1).and_then(|s| s.split(']').next()).unwrap_or("x").to_string();
    if req.messages.last().unwrap().role == Role::User {
        vec![call("w0", "write_file", json!({"path": format!("{id}.txt"), "content": format!("made by {id}")}))]
    } else {
        let last = req.messages.last().unwrap().text().to_string();
        let ok = last.starts_with("Wrote");
        vec![text(&format!("STATUS: {}\nSUMMARY: {id}: {last}\nFILES: {id}.txt\nNOTES: none", if ok { "done" } else { "failed" }))]
    }
}

#[tokio::test]
async fn master_plans_and_workers_run_in_parallel() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("README.md"), "hello").unwrap();
    let script: Box<Script> = Box::new(|req| {
        if !req.messages[0].text().contains("MASTER") {
            return worker_script(req);
        }
        match assistant_turns(req) {
            0 => vec![call("c0", "read_file", json!({"path":"README.md"})), call("c1", "glob", json!({"pattern":"*"}))],
            1 => vec![call(
                "c0",
                "todo_write",
                json!({"items":[
                    {"id":"a","title":"A","task":"make a","files":["a.txt"]},
                    {"id":"b","title":"B","task":"make b","files":["b.txt"]},
                    {"id":"c","title":"C","task":"make c","files":["c.txt"],"depends_on":["a","b"]}
                ]}),
            )],
            2 => vec![call("c0", "propose_plan", json!({"summary":"three files"}))],
            3 => vec![call("c0", "run_todos", json!({}))],
            _ => vec![text("All done.")],
        }
    });
    let (team, fake, deny, mut rx) = setup(dir.path(), 4, script);
    let out = team.run("create the files").await.unwrap();
    assert_eq!(out, "All done.");

    // Files written by workers.
    for id in ["a", "b", "c"] {
        assert_eq!(std::fs::read_to_string(dir.path().join(format!("{id}.txt"))).unwrap(), format!("made by {id}"));
    }
    // All TODOs done with reports; a and b on different instances.
    let todos = team.todos();
    assert!(todos.iter().all(|t| t.status == TodoStatus::Done && t.report.as_deref().unwrap().contains("STATUS: done")));
    let inst = |id: &str| todos.iter().find(|t| t.id == id).unwrap().instance.clone().unwrap();
    assert_ne!(inst("a"), inst("b"));
    // No approval prompts were needed (in-scope writes under an approved plan).
    assert!(deny.0.lock().unwrap().is_empty(), "unexpected prompts: {:?}", deny.0.lock().unwrap());

    // Timing: a and b overlapped, c started after both finished.
    let mut start: BTreeMap<String, Instant> = BTreeMap::new();
    let mut end: BTreeMap<String, Instant> = BTreeMap::new();
    while let Ok(ev) = rx.try_recv() {
        if let EventKind::TodoStatus { id, status, .. } = ev.kind {
            match status {
                TodoStatus::InProgress => {
                    start.entry(id).or_insert_with(Instant::now);
                }
                TodoStatus::Done => {
                    end.insert(id, Instant::now());
                }
                _ => {}
            }
        }
    }
    // Events were drained after the fact, so compare the order via the request log instead.
    let reqs = fake.requests.lock().unwrap().clone();
    let worker_firsts: Vec<String> = reqs
        .iter()
        .filter(|r| !r.messages[0].text().contains("MASTER") && r.messages.last().unwrap().role == Role::User)
        .map(|r| last_user(r).split('[').nth(1).unwrap().split(']').next().unwrap().to_string())
        .collect();
    assert_eq!(worker_firsts.len(), 3);
    assert_eq!(worker_firsts[2], "c", "c must start last: {worker_firsts:?}");
    assert_eq!(start.len(), 3);
    assert_eq!(end.len(), 3);

    // Master never had mutating tools and never issued one.
    let master_tools: Vec<String> = team.master_tool_names().await;
    assert!(master_tools.iter().all(|t| !["write_file", "edit_file", "bash"].contains(&t.as_str())));
    for r in reqs.iter().filter(|r| r.messages[0].text().contains("MASTER")) {
        assert!(r.tools.iter().all(|t| !["write_file", "edit_file", "bash"].contains(&t.function.name.as_str())));
        for m in &r.messages {
            assert!(m.tool_calls.iter().all(|c| !["write_file", "edit_file", "bash"].contains(&c.function.name.as_str())));
        }
    }
    // Each worker instance got its own composed system prompt (coder default + contract + its id).
    let sys: Vec<String> =
        reqs.iter().filter(|r| !r.messages[0].text().contains("MASTER")).map(|r| r.messages[0].text().to_string()).collect();
    assert!(sys.iter().all(|s| s.contains("You are a CODER") && s.contains("STATUS: done | failed")));
    assert!(sys.iter().any(|s| s.contains("coder#1")) && sys.iter().any(|s| s.contains("coder#2")));
}

#[tokio::test]
async fn parallelism_speedup_and_scope_conflicts() {
    async fn run(max_workers: usize, overlapping: bool) -> Duration {
        let dir = tempfile::tempdir().unwrap();
        let files = |id: &str| if overlapping { json!(["shared/**"]) } else { json!([format!("{id}.txt")]) };
        let items = json!([
            {"id":"p","task":"p","files":files("p")},
            {"id":"q","task":"q","files":files("q")},
            {"id":"r","task":"r","files":files("r")},
            {"id":"s","task":"s","files":files("s")}
        ]);
        let script: Box<Script> = Box::new(move |req| {
            if !req.messages[0].text().contains("MASTER") {
                return worker_script(req);
            }
            match assistant_turns(req) {
                0 => vec![call("c0", "todo_write", json!({"items": items.clone()}))],
                1 => vec![call("c0", "propose_plan", json!({"summary":"x"}))],
                2 => vec![call("c0", "run_todos", json!({}))],
                _ => vec![text("ok")],
            }
        });
        let (team, ..) = setup(dir.path(), max_workers, script);
        let t = Instant::now();
        team.run("go").await.unwrap();
        // With overlapping scopes, writes outside `shared/**` aren't pre-approved, so they get denied;
        // we only care about timing here.
        t.elapsed()
    }
    // 4 items × 2 worker turns × 150ms ≈ 1.2s sequential vs ≈ 0.3s fully parallel.
    let serial = run(1, false).await;
    let parallel = run(4, false).await;
    let conflicting = run(4, true).await;
    assert!(parallel * 2 < serial, "parallel {parallel:?} vs serial {serial:?}");
    assert!(conflicting * 2 > serial, "overlapping scopes must serialize: {conflicting:?} vs {serial:?}");
}

#[tokio::test]
async fn unapproved_todos_cannot_run_and_master_is_reminded() {
    let dir = tempfile::tempdir().unwrap();
    let script: Box<Script> = Box::new(|req| {
        if !req.messages[0].text().contains("MASTER") {
            return worker_script(req);
        }
        // Skip the ephemeral TODO context the runtime appends for the master.
        let last = req.messages.iter().rev().find(|m| !m.text().starts_with("<system-context>")).unwrap();
        match assistant_turns(req) {
            0 => vec![call("c0", "todo_write", json!({"items":[{"id":"a","task":"a","files":["a.txt"]}]}))],
            1 => vec![call("c0", "run_todos", json!({}))], // skipped propose_plan
            2 => {
                assert!(last.text().contains("not approved"), "got: {}", last.text());
                vec![text("I'm done.")] // tries to finish with a pending TODO
            }
            3 => {
                assert!(last.text().contains("not finished"), "expected reminder, got: {}", last.text());
                vec![call("c0", "todo_write", json!({"items":[{"id":"a","status":"cancelled"}]}))]
            }
            _ => vec![text("Cancelled the item.")],
        }
    });
    let (team, ..) = setup(dir.path(), 2, script);
    assert_eq!(team.run("go").await.unwrap(), "Cancelled the item.");
    assert_eq!(team.todos()[0].status, TodoStatus::Cancelled);
    assert!(!dir.path().join("a.txt").exists());
}

#[tokio::test]
async fn readonly_delegation_fans_out_in_parallel() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(dir.path().join("x.rs"), "fn main() {}").unwrap();
    let script: Box<Script> = Box::new(|req| {
        let sys = req.messages[0].text();
        if sys.contains("MASTER") {
            return match assistant_turns(req) {
                0 => vec![
                    call("c0", "delegate", json!({"task":"find main"})),
                    call("c1", "delegate", json!({"task":"find tests"})),
                    call("c2", "delegate", json!({"task":"try to write", "worker":"coder"})),
                ],
                _ => vec![text("explored")],
            };
        }
        if req.messages.last().unwrap().role == Role::User {
            if last_user(req).contains("try to write") {
                // Read-only coder instance must not have write tools.
                assert!(req.tools.iter().all(|t| t.function.name != "write_file"));
                return vec![text("STATUS: done\nSUMMARY: no write tools available")];
            }
            vec![call("w0", "grep", json!({"pattern":"main"}))]
        } else {
            vec![text("STATUS: done\nSUMMARY: found it in x.rs")]
        }
    });
    let (team, fake, ..) = setup(dir.path(), 4, script);
    let t = Instant::now();
    assert_eq!(team.run("look around").await.unwrap(), "explored");
    // 3 delegations × up to 2 turns × 150ms would be ~0.75s serial.
    assert!(t.elapsed() < Duration::from_millis(600), "took {:?}", t.elapsed());
    let insts = team.instances();
    assert_eq!(insts.iter().filter(|i| i.profile == "explorer").count(), 2);
    assert!(insts.iter().all(|i| i.readonly));
    let reqs = fake.requests.lock().unwrap();
    assert!(reqs.iter().any(|r| r.messages[0].text().contains("You are an EXPLORER")));
}
