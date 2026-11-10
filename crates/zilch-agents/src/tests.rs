//! End to end: kernel + real drivers (tools on a temp project) + a scripted model playing the
//! Architect and the Developers.

use super::*;
use std::collections::HashMap;
use std::io::Write as _;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zilch_drivers::llm::fake::ScriptedLlm;
use zilch_drivers::llm::{Backend, LlmEvent, Role};
use zilch_drivers::tools::ToolPolicy;
use zilch_drivers::{DriverSet, ARCHITECT_TOOLS, DEVELOPER_TOOLS};
use zilch_ipc::{EventBus, KernelEvent, Usage};
use zilch_kernel::{Cmd, KernelConfig};

fn call(i: usize, name: &str, args: Value) -> LlmEvent {
    LlmEvent::ToolCallDelta { index: i, id: Some(format!("c{i}")), name: Some(name.into()), args: args.to_string() }
}

fn last_tool_result(req: &ChatRequest) -> Option<String> {
    req.messages.last().filter(|m| m.role == Role::Tool).map(|m| m.text().to_string())
}

#[derive(Clone, Default)]
struct Buf(Arc<Mutex<Vec<u8>>>);
impl std::io::Write for Buf {
    fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(b);
        Ok(b.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn an_order_is_planned_approved_cooked_in_parallel_and_served() {
    let dir = tempfile::tempdir().unwrap();
    let ws = dir.path().to_path_buf();
    // When each developer's first request reached the model (after the scripted delay).
    let seen: Arc<Mutex<HashMap<String, Vec<Instant>>>> = Arc::default();
    let log = seen.clone();
    let llm = ScriptedLlm::new(move |req| {
        let system = req.messages[0].text().to_string();
        let first_user = req.messages.iter().find(|m| m.role == Role::User).map(|m| m.text().to_string()).unwrap_or_default();
        let usage = LlmEvent::Usage(Usage { prompt_tokens: 100, completion_tokens: 10 });
        if system.contains("ARCHITECT") {
            return match last_tool_result(req) {
                None => vec![
                    call(0, "write_file", json!({"path":"docs/requirements.md","content":"- a, b and c exist\n"})),
                    call(1, "write_file", json!({"path":"docs/design.md","content":"one file per task\n"})),
                    usage,
                ],
                Some(r) if r.starts_with("All tasks finished") => {
                    assert!(r.contains("[T3] Write c — Done"), "{r}");
                    vec![LlmEvent::TextDelta("Summary: a, b and c were written.".into()), usage]
                }
                Some(_) => vec![
                    call(0, "submit_plan", json!({"tasks":[
                        {"id":"T1","title":"Write a","task":"Create a.txt containing A.","owned_paths":["a.txt"],"acceptance":["a.txt exists"]},
                        {"id":"T2","title":"Write b","task":"Create b.txt containing B.","owned_paths":["b.txt"]},
                        {"id":"T3","title":"Write c","task":"Create c.txt from a and b.","owned_paths":["c.txt"],"depends_on":["T1","T2"]}
                    ]})),
                    usage,
                ],
            };
        }
        let id = first_user.split(']').next().unwrap_or("").trim_start_matches("Task [").to_string();
        log.lock().unwrap().entry(id.clone()).or_default().push(Instant::now());
        match (id.as_str(), last_tool_result(req)) {
            ("T1", None) => vec![
                call(0, "write_file", json!({"path":"a.txt","content":"A"})),
                // Not its file: the kernel must refuse this.
                call(1, "write_file", json!({"path":"b.txt","content":"T1 was here"})),
                usage,
            ],
            ("T2", None) => vec![call(0, "write_file", json!({"path":"b.txt","content":"B"})), usage],
            ("T3", None) => vec![call(0, "write_file", json!({"path":"c.txt","content":"AB"})), usage],
            (_, Some(_)) => {
                let results: Vec<&str> = req.messages.iter().filter(|m| m.role == Role::Tool).map(|m| m.text()).collect();
                if id == "T1" {
                    assert!(results.iter().any(|r| r.contains("permission denied") && r.contains("b.txt")), "{results:?}");
                }
                vec![LlmEvent::TextDelta(format!("{id} written.\nSTATUS: done")), usage]
            }
            _ => unreachable!(),
        }
    })
    .with_delay(Duration::from_millis(150));
    let llm: Arc<dyn Backend> = Arc::new(llm);
    let binding = ModelBinding { provider: "fake".into(), model: "scripted".into(), ..Default::default() };
    let drivers = DriverSet::new(&ws, ToolPolicy::default(), [("fake".to_string(), llm)].into());
    let book = Book::new(binding.clone(), binding, drivers.specs(ARCHITECT_TOOLS), drivers.specs(DEVELOPER_TOOLS), &ws);

    let bus = Arc::new(EventBus::new());
    let log_buf = Buf::default();
    bus.set_log(Box::new(log_buf.clone()));
    let events = bus.subscribe(100_000);
    let cfg = KernelConfig { tick: Duration::from_millis(10), auto_approve: true, ..Default::default() };
    let k = zilch_kernel::start(cfg, Arc::new(drivers), Arc::new(book), bus);
    k.send(Cmd::Order("write a, b and c".into()));

    let deadline = Instant::now() + Duration::from_secs(20);
    let mut all = Vec::new();
    let (ok, summary) = loop {
        let env = events.recv_timeout(deadline.saturating_duration_since(Instant::now())).unwrap_or_else(|_| panic!("timed out: {all:#?}"));
        all.push(env.event.clone());
        if let KernelEvent::TicketServed { ok, summary, .. } = &env.event {
            break (*ok, summary.clone());
        }
    };
    assert!(ok && summary.contains("Summary: a, b and c were written."), "{summary}");
    let read = |p: &str| std::fs::read_to_string(ws.join(p)).unwrap_or_default();
    assert_eq!(
        (read("a.txt"), read("b.txt"), read("c.txt")),
        ("A".into(), "B".into(), "AB".into()),
        "b.txt is T2's, T1's write was refused"
    );
    assert!(read("docs/requirements.md").contains("a, b and c") && read("docs/plan.json").contains("\"T3\""));

    // T1 and T2 ran at the same time; T3 only after both finished.
    let seen = seen.lock().unwrap();
    let (t1, t2, t3) = (&seen["T1"], &seen["T2"], &seen["T3"]);
    let gap = if t1[0] > t2[0] { t1[0] - t2[0] } else { t2[0] - t1[0] };
    assert!(gap < Duration::from_millis(100), "T1 and T2 started together (gap {gap:?})");
    assert!(t3[0] > *t1.last().unwrap() && t3[0] > *t2.last().unwrap(), "T3 waited for T1 and T2");

    // The JSONL log has every syscall result (replay data) and no streamed text.
    let text = String::from_utf8(log_buf.0.lock().unwrap().clone()).unwrap();
    let lines: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let done: Vec<&serde_json::Value> = lines.iter().filter(|l| l["type"] == "SyscallDone").collect();
    let issued = lines.iter().filter(|l| l["type"] == "SyscallIssued").count();
    assert_eq!(done.len(), issued, "every issued syscall has its result logged");
    assert!(done.iter().any(|l| l["result"]["Completion"]["message"]["content"].as_str().is_some_and(|c| c.contains("STATUS: done"))));
    assert!(!lines.iter().any(|l| l["type"] == "TextDelta"));
    let _ = std::io::stdout().flush();
    k.shutdown();
}

#[test]
fn invalid_tool_arguments_are_answered_without_a_syscall() {
    let mut a = LlmAgent::new(ModelBinding::default(), "s".into(), "go".into(), vec![], "x".into());
    let view = AcbView { pid: 1, ticket: 1, group: Group::Development, steps: 0, notes: vec![] };
    assert!(matches!(a.step(&view, vec![]), StepResult::Syscalls(v) if matches!(v[0], Syscall::Complete(_))));
    let mut msg = Message::assistant("");
    msg.tool_calls = vec![
        ToolCall {
            id: "a".into(),
            kind: "function".into(),
            function: zilch_ipc::llm::FunctionCall { name: "read_file".into(), arguments: "{\"path\":\"x\"}".into() },
        },
        ToolCall {
            id: "b".into(),
            kind: "function".into(),
            function: zilch_ipc::llm::FunctionCall { name: "read_file".into(), arguments: "[1]".into() },
        },
        ToolCall {
            id: "c".into(),
            kind: "function".into(),
            function: zilch_ipc::llm::FunctionCall { name: "submit_plan".into(), arguments: "{\"tasks\":3}".into() },
        },
    ];
    let r = a.step(&view, vec![SyscallResult::Completion { message: msg, usage: Usage::default(), cut: false }]);
    let StepResult::Syscalls(calls) = r else { panic!() };
    assert_eq!(calls.len(), 1, "only the valid call becomes a syscall");
    // Results come back for the issued call; the others were answered locally, in order.
    let view2 = AcbView { notes: vec!["[supervisor] stop repeating".into()], ..view };
    let StepResult::Syscalls(next) = a.step(&view2, vec![SyscallResult::Output("file x".into())]) else { panic!() };
    let Syscall::Complete(req) = &next[0] else { panic!() };
    let tail: Vec<String> = req.messages.iter().rev().take(4).rev().map(|m| m.text().to_string()).collect();
    assert_eq!(tail[0], "file x");
    assert!(tail[1].contains("must be a JSON object") && tail[2].contains("invalid plan"));
    assert_eq!(tail[3], "[supervisor] stop repeating", "the nudge comes last, before the next reply");
}

#[test]
fn a_restart_drops_the_conversation_and_starts_from_the_task_plus_summary() {
    let mut a = LlmAgent::new(ModelBinding::default(), "s".into(), "Write a.txt".into(), vec![], "x".into());
    let view = AcbView { pid: 1, ticket: 1, group: Group::Development, steps: 0, notes: vec![] };
    a.step(&view, vec![]);
    assert!(a.restart("[supervisor] you looped. What it did:\n- bash(x) → Error"));
    let StepResult::Syscalls(calls) = a.step(&view, vec![]) else { panic!() };
    let Syscall::Complete(req) = &calls[0] else { panic!() };
    assert_eq!(req.messages.len(), 2, "system + the new first message only");
    let first = req.messages[1].text();
    assert!(first.starts_with("Write a.txt") && first.contains("you looped"), "{first}");
}
