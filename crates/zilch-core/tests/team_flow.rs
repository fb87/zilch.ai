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

#[tokio::test]
async fn a_task_that_keeps_failing_unchanged_is_not_run_a_third_time() {
    use zilch_core::team::todo::Todo;
    let dir = tempfile::tempdir().unwrap();
    // Every worker run fails.
    let (team, fake, _, _rx) = setup(dir.path(), 2, Box::new(|_| vec![text("Could not do it.\nSTATUS: failed")]));
    let todo = |id: &str, task: &str| Todo {
        id: id.into(),
        title: "fix the build".into(),
        task: task.into(),
        worker: "coder".into(),
        files: vec!["src/**".into()],
        depends_on: vec![],
        model: None,
        provider: None,
        status: TodoStatus::Pending,
        approved: true,
        instance: None,
        report: None,
    };
    let worker_runs = || fake.requests.lock().unwrap().len();
    let run = |t: Todo| {
        let shared = team.shared.clone();
        async move {
            shared.todos.lock().unwrap().items = vec![t];
            team::scheduler::run(&shared, None).await.unwrap();
            shared.todos.lock().unwrap().items[0].clone()
        }
    };
    assert_eq!(run(todo("t1", "Make cargo build pass.")).await.status, TodoStatus::Failed);
    // Retried under another id with only whitespace changed: still the same task.
    assert_eq!(run(todo("t2", "Make  cargo build\npass.")).await.status, TodoStatus::Failed);
    assert_eq!(worker_runs(), 2);
    let third = run(todo("t3", "Make cargo build pass.")).await;
    assert_eq!(third.status, TodoStatus::Failed);
    assert!(third.report.as_deref().unwrap_or("").contains("already failed 2 times"), "{:?}", third.report);
    assert_eq!(worker_runs(), 2, "no worker ran for the third identical attempt");
    // Changed instructions are a new task and run again.
    run(todo("t4", "Make cargo build pass: the error is a missing import in src/lib.rs.")).await;
    assert_eq!(worker_runs(), 3);
}

#[tokio::test]
async fn model_switches_apply_to_the_next_turn_and_switch_back() {
    let dir = tempfile::tempdir().unwrap();
    let (team, fake, _, _rx) = setup(dir.path(), 2, Box::new(|_| vec![text("ok")]));
    let configured = team.shared.cfg.master().binding.model.clone();
    let last_model = || fake.requests.lock().unwrap().last().unwrap().binding.model.clone();

    team.run("hi").await.unwrap();
    assert_eq!(last_model(), configured);
    let msg = team.set_model("master", Some("big-model"), None).unwrap();
    assert!(msg.contains("big-model") && msg.contains("next turn"), "{msg}");
    team.run("hi again").await.unwrap();
    assert_eq!(last_model(), "big-model", "the master uses the switched model from its next turn");
    assert!(team.models().iter().any(|(a, m)| a.starts_with("master") && m.contains("switched")));
    team.set_model("master", None, None).unwrap();
    team.run("and again").await.unwrap();
    assert_eq!(last_model(), configured, "back to the configured model");

    // Worker profiles switch for new tasks; instances and unknown names are refused.
    team.set_model("coder", Some("fast-model"), None).unwrap();
    let p = team.shared.cfg.profile("coder").unwrap();
    assert_eq!(team.shared.effective_binding(p).model, "fast-model");
    assert!(team.set_model("coder#1", Some("x"), None).is_err());
    assert!(team.set_model("nobody", Some("x"), None).is_err());
    assert!(team.set_model("coder", Some("x"), Some("no-such-provider")).is_err());
    // With a provider model list known, a model not in it is refused (it would fail mid-task).
    let info = zilch_core::llm::catalog::ModelInfo::default();
    team.set_catalog([("openai/listed".to_string(), info)].into());
    assert!(team.set_model("coder", Some("typo-model"), None).is_err());
    assert!(team.set_model("coder", Some("listed"), None).is_ok());
}

#[tokio::test]
async fn a_high_cost_model_in_use_too_long_alerts_until_acknowledged_or_switched_back() {
    use zilch_core::llm::catalog::ModelInfo;
    use zilch_core::llm::Usage;
    let dir = tempfile::tempdir().unwrap();
    let (team, _, _, _rx) = setup(dir.path(), 2, Box::new(|_| vec![text("ok")]));
    let price = |o: f64| ModelInfo { context: None, input_per_mtok: Some(1.0), output_per_mtok: Some(o) };
    team.set_catalog([("openai/opus-big".to_string(), price(75.0)), ("openai/cheap".to_string(), price(0.5))].into());
    let meter = &team.shared.meter;
    let small = Usage { prompt_tokens: 1000, completion_tokens: 100, ..Default::default() };

    // A cheap model never alerts, however long it is used.
    team.set_model("master", Some("cheap"), None).unwrap();
    meter.record("openai/cheap", &small);
    meter.backdate("openai/cheap", Duration::from_secs(5 * 3600));
    assert!(team.cost_alerts().is_empty());

    // The high-cost one: quiet at first, alerts after 30 minutes.
    team.set_model("master", Some("opus-big"), None).unwrap();
    meter.record("openai/opus-big", &small);
    assert!(team.cost_alerts().is_empty(), "just switched");
    meter.backdate("openai/opus-big", Duration::from_secs(31 * 60));
    let a = team.cost_alerts();
    assert_eq!(a.len(), 1);
    assert_eq!((a[0].model.as_str(), a[0].agents.clone(), a[0].minutes), ("openai/opus-big", vec!["master".to_string()], 31));

    // /ack silences it until the next step (another 30 minutes, or $2 spent).
    assert_eq!(team.ack_cost_alerts(), 1);
    assert!(team.cost_alerts().is_empty());
    meter.record("openai/opus-big", &Usage { prompt_tokens: 0, completion_tokens: 30_000, ..Default::default() }); // $2.25
    assert_eq!(team.cost_alerts().len(), 1, "spending past $2 alerts again");
    team.ack_cost_alerts();

    // Switching back ends it: no agent is bound to the high-cost model any more.
    meter.backdate("openai/opus-big", Duration::from_secs(3600));
    assert_eq!(team.cost_alerts().len(), 1);
    team.set_model("master", None, None).unwrap();
    assert!(team.cost_alerts().is_empty());
}
