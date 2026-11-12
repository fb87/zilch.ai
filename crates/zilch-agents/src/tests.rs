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
        let usage = LlmEvent::Usage(Usage { prompt_tokens: 100, completion_tokens: 10, ..Default::default() });
        if system.contains("MAIN agent") {
            // A follow-up order: it must still have the first order in its context.
            let last_user = req.messages.last().filter(|m| m.role == Role::User).map(|m| m.text().to_string()).unwrap_or_default();
            if last_user.starts_with("Order #2") {
                let remembers = req.messages.iter().any(|m| m.text().contains("Summary: a, b and c were written."));
                let reply = if remembers { "Earlier I had a, b and c written." } else { "I do not know." };
                return vec![LlmEvent::TextDelta(reply.into()), usage];
            }
            // The main agent only dispatches, then replies with the report.
            let tools: Vec<&str> = req.tools.iter().map(|t| t.function.name.as_str()).collect();
            assert_eq!(tools, ["dispatch"]);
            return match last_tool_result(req) {
                Some(r) if r.starts_with("Report from the plan subagent:") => {
                    assert!(r.contains("planner: a, b and c are in place"), "{r}");
                    vec![LlmEvent::TextDelta("Summary: a, b and c were written.".into()), usage]
                }
                _ => vec![call(0, "dispatch", json!({"agent":"plan","brief":"Order: write a, b and c"})), usage],
            };
        }
        if system.contains("PLANNER") {
            assert_eq!(first_user, "Order: write a, b and c");
            let tools: Vec<&str> = req.tools.iter().map(|t| t.function.name.as_str()).collect();
            assert!(tools.contains(&"submit_plan") && !tools.contains(&"write_file"), "{tools:?}");
            return match last_tool_result(req) {
                Some(r) if r.starts_with("All tasks finished") => {
                    assert!(r.contains("[T3] Write c — Done"), "{r}");
                    vec![LlmEvent::TextDelta("planner: a, b and c are in place".into()), usage]
                }
                _ => vec![
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
    // A follow-up order goes to the same main agent, which remembers the first one.
    k.send(Cmd::Order("what did you do before?".into()));
    let follow_up = loop {
        let env = events.recv_timeout(deadline.saturating_duration_since(Instant::now())).expect("second order served");
        if let KernelEvent::TicketServed { ticket: 2, summary, .. } = &env.event {
            break summary.clone();
        }
        if let KernelEvent::Spawned { group: Group::Architecture, pid, .. } = &env.event {
            panic!("a second main agent ({pid}) was spawned");
        }
    };
    assert_eq!(follow_up, "Earlier I had a, b and c written.");
    let read = |p: &str| std::fs::read_to_string(ws.join(p)).unwrap_or_default();
    assert_eq!(
        (read("a.txt"), read("b.txt"), read("c.txt")),
        ("A".into(), "B".into(), "AB".into()),
        "b.txt is T2's, T1's write was refused"
    );
    assert!(!ws.join("docs").exists(), "the main agent leaves no files in the project");

    // T1 and T2 ran at the same time; T3 only after both finished.
    let seen = seen.lock().unwrap();
    let (t1, t2, t3) = (&seen["T1"], &seen["T2"], &seen["T3"]);
    let gap = if t1[0] > t2[0] { t1[0] - t2[0] } else { t2[0] - t1[0] };
    assert!(gap < Duration::from_millis(100), "T1 and T2 started together (gap {gap:?})");
    assert!(t3[0] > *t1.last().unwrap() && t3[0] > *t2.last().unwrap(), "T3 waited for T1 and T2");

    // The JSONL log has every syscall result (replay data) and no streamed text.
    let text = String::from_utf8(log_buf.0.lock().unwrap().clone()).unwrap();
    let lines: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    let done: Vec<&serde_json::Value> = lines.iter().filter(|l| l["type"] == "SyscallDone" && l["summary"] != "serve").collect();
    // (The main agent's last call, `serve`, stays open: it is waiting for the next order.)
    let issued = lines.iter().filter(|l| l["type"] == "SyscallIssued" && l["summary"] != "serve").count();
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

#[test]
fn a_note_arriving_with_the_final_reply_is_read_instead_of_exiting() {
    let mut a = LlmAgent::new(ModelBinding::default(), "s".into(), "task".into(), vec![], "x".into());
    let view = AcbView { pid: 1, ticket: 1, group: Group::Development, steps: 0, notes: vec![] };
    a.step(&view, vec![]);
    let done = || SyscallResult::Completion { message: Message::assistant("All done.\nSTATUS: done"), usage: Usage::default(), cut: false };
    let with_note = AcbView { notes: vec!["[guest] also add a test".into()], ..view.clone() };
    let StepResult::Syscalls(calls) = a.step(&with_note, vec![done()]) else { panic!("it must not exit with an unread note") };
    let Syscall::Complete(req) = &calls[0] else { panic!() };
    assert_eq!(req.messages.last().unwrap().text(), "[guest] also add a test");
    assert!(matches!(a.step(&view, vec![done()]), StepResult::Exit(_)), "without new input the final reply ends it");
}

#[test]
fn the_resident_main_agent_serves_waits_and_survives_a_failed_call() {
    let mut a = LlmAgent::new(ModelBinding::default(), "s".into(), "Order #1: hi".into(), vec![], "x".into()).resident();
    let view = AcbView { pid: 1, ticket: 1, group: Group::Architecture, steps: 0, notes: vec![] };
    a.step(&view, vec![]);
    let reply = |t: &str| SyscallResult::Completion { message: Message::assistant(t), usage: Usage::default(), cut: false };
    let StepResult::Syscalls(c) = a.step(&view, vec![reply("Hello!")]) else { panic!("a resident agent does not exit") };
    assert!(matches!(&c[0], Syscall::Serve { summary } if summary == "Hello!"));
    // The next order arrives as a message and becomes the next user turn, after the history.
    let order = SyscallResult::Message(zilch_ipc::Msg { from: 0, kind: zilch_ipc::MsgKind::Guest, body: "Order #2: and now?".into() });
    let StepResult::Syscalls(c) = a.step(&view, vec![order]) else { panic!() };
    let Syscall::Complete(req) = &c[0] else { panic!() };
    let texts: Vec<&str> = req.messages.iter().map(|m| m.text()).collect();
    assert_eq!(texts[1..], ["Order #1: hi", "Hello!", "Order #2: and now?"]);
    // A failed call is retried once, then reported to the guest, and the agent keeps waiting.
    let failed = || SyscallResult::Failed("provider returned 503".into());
    assert!(matches!(a.step(&view, vec![failed()]), StepResult::Syscalls(c) if matches!(c[0], Syscall::Complete(_))));
    assert!(
        matches!(a.step(&view, vec![failed()]), StepResult::Syscalls(c) if matches!(&c[0], Syscall::Serve { summary } if summary.contains("503")))
    );
}

#[test]
fn a_history_over_budget_is_summarized_through_a_syscall_before_the_next_request() {
    let mut a =
        LlmAgent::new(ModelBinding::default(), "s".into(), "Order #1: big job".into(), vec![], "x".into()).resident().with_budget(4_000);
    let view = AcbView { pid: 1, ticket: 1, group: Group::Architecture, steps: 0, notes: vec![] };
    a.step(&view, vec![]);
    // Grow the conversation well past the budget with tool turns.
    for i in 0..12 {
        let mut m = Message::assistant("");
        m.tool_calls = vec![ToolCall {
            id: format!("c{i}"),
            kind: "function".into(),
            function: zilch_ipc::llm::FunctionCall { name: "read_file".into(), arguments: "{}".into() },
        }];
        let r = a.step(&view, vec![SyscallResult::Completion { message: m, usage: Usage::default(), cut: false }]);
        assert!(matches!(r, StepResult::Syscalls(_)));
        let next = a.step(&view, vec![SyscallResult::Output("y".repeat(1_400))]);
        let StepResult::Syscalls(calls) = next else { panic!() };
        let Syscall::Complete(req) = &calls[0] else { panic!() };
        if req.messages[0].text().starts_with("You compress") {
            // The summary request: answer it, and the real request follows with a short history.
            let summary =
                SyscallResult::Completion { message: Message::assistant("- read many files"), usage: Usage::default(), cut: false };
            let StepResult::Syscalls(after) = a.step(&view, vec![summary]) else { panic!() };
            let Syscall::Complete(req) = &after[0] else { panic!() };
            assert!(
                req.messages[1].text().starts_with("<conversation-summary>") && req.messages[1].text().contains("read many files"),
                "{:?}",
                req.messages[1]
            );
            assert!(compact::estimate_tokens("", &req.messages, &[]) < 3_000, "back under the budget");
            return;
        }
    }
    panic!("the history never got compacted");
}

#[test]
fn a_session_is_saved_when_an_order_is_served_and_continued_by_a_new_kernel() {
    let dir = tempfile::tempdir().unwrap();
    let ws = dir.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let file = dir.path().join("sessions").join("1.json");
    let llm: Arc<dyn Backend> = Arc::new(ScriptedLlm::new(|req| {
        let last = req.messages.last().map(|m| m.text().to_string()).unwrap_or_default();
        let reply = if last.starts_with("Order #2") {
            let remembers = req.messages.iter().any(|m| m.text().contains("remember the word banana"));
            if remembers {
                "the word was banana"
            } else {
                "I do not know"
            }
        } else {
            "noted"
        };
        vec![LlmEvent::TextDelta(reply.into())]
    }));
    let binding = ModelBinding { provider: "fake".into(), model: "scripted".into(), ..Default::default() };
    let run = |first_ticket: u64, resume: Option<Vec<Message>>, order: &str| -> (u64, String) {
        let drivers =
            DriverSet::new(&ws, ToolPolicy::default(), [("fake".to_string(), llm.clone())].into()).with_session_file(file.clone());
        let book = Book::new(binding.clone(), binding.clone(), drivers.specs(ARCHITECT_TOOLS), drivers.specs(DEVELOPER_TOOLS), &ws);
        *book.resume.lock().unwrap() = resume;
        let bus = Arc::new(EventBus::new());
        let events = bus.subscribe(10_000);
        let cfg = KernelConfig { tick: Duration::from_millis(10), auto_approve: true, first_ticket, ..Default::default() };
        let k = zilch_kernel::start(cfg, Arc::new(drivers), Arc::new(book), bus);
        k.send(Cmd::Order(order.into()));
        let served = loop {
            let env = events.recv_timeout(Duration::from_secs(10)).expect("served");
            if let KernelEvent::TicketServed { ticket, summary, .. } = &env.event {
                break (*ticket, summary.clone());
            }
        };
        // The session is written off the kernel thread; give it a moment, then stop.
        let deadline = Instant::now() + Duration::from_secs(5);
        while !file.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        k.shutdown();
        served
    };
    assert_eq!(run(1, None, "remember the word banana"), (1, "noted".into()));
    let saved = zilch_drivers::session::load(&file).and_then(|v| snapshot_messages(&v)).expect("session saved");
    assert!(saved.iter().any(|m| m.text() == "Order #1: remember the word banana"), "{saved:?}");
    // `-c`: a new kernel, the main agent starts from the saved conversation, numbering goes on.
    assert_eq!(run(2, Some(saved), "which word?"), (2, "the word was banana".into()));
}

#[test]
fn prompt_overrides_apply_and_a_task_can_run_on_a_custom_profile() {
    let b = |m: &str| ModelBinding { provider: "p".into(), model: m.into(), ..Default::default() };
    let mut book = Book::new(b("main"), b("dev"), vec![], vec![], std::path::Path::new("/tmp"));
    book.prompts.insert(Group::Development, ("Prefer small diffs.".into(), false));
    book.prompts.insert(Group::Architecture, ("You are terse.".into(), true));
    book.profiles.push(SubagentProfile {
        name: "tester".into(),
        description: "runs tests".into(),
        binding: b("fast"),
        tools: vec![],
        prompt: Some(("Only tests.".into(), false)),
    });
    let main = book.system_prompt(Group::Architecture);
    assert!(main.starts_with("You are terse.") && !main.contains("MAIN agent"), "replace drops the built-in prompt");
    assert!(book.subagent_prompt("plan").unwrap().contains("- tester: runs tests"), "the planner learns the profiles");
    let dev = book.system_prompt(Group::Development);
    assert!(dev.contains("DEVELOPER") && dev.contains("Prefer small diffs."), "append keeps the built-in prompt");
    // A task naming the profile asks its model, with its prompt.
    let mut task = PlanTask {
        id: "T1".into(),
        title: "t".into(),
        task: "test it".into(),
        depends_on: vec![],
        owned_paths: vec![],
        acceptance: vec![],
        priority: None,
        worker: Some("tester".into()),
    };
    let view = AcbView { pid: 2, ticket: 1, group: Group::Development, steps: 0, notes: vec![] };
    let StepResult::Syscalls(c) = book.developer(1, &task, "").step(&view, vec![]) else { panic!() };
    let Syscall::Complete(req) = &c[0] else { panic!() };
    assert_eq!(req.binding.model, "fast");
    assert!(req.messages[0].text().contains("Only tests."));
    task.worker = None;
    let StepResult::Syscalls(c) = book.developer(1, &task, "").step(&view, vec![]) else { panic!() };
    let Syscall::Complete(req) = &c[0] else { panic!() };
    assert_eq!(req.binding.model, "dev", "no worker: the default subagent");
}

#[test]
fn reasoning_from_an_earlier_order_is_not_resent() {
    let mut a = LlmAgent::new(ModelBinding::default(), "s".into(), "Order #1: hi".into(), vec![], "x".into()).resident();
    let view = AcbView { pid: 1, ticket: 1, group: Group::Architecture, steps: 0, notes: vec![] };
    a.step(&view, vec![]);
    let mut thought = Message::assistant("Hello!");
    thought.reasoning_content = Some("the guest greets me".into());
    a.step(&view, vec![SyscallResult::Completion { message: thought, usage: Usage::default(), cut: false }]);
    let order = SyscallResult::Message(zilch_ipc::Msg { from: 0, kind: zilch_ipc::MsgKind::Guest, body: "Order #2: and now?".into() });
    let StepResult::Syscalls(c) = a.step(&view, vec![order]) else { panic!() };
    let Syscall::Complete(req) = &c[0] else { panic!() };
    assert!(req.messages.iter().all(|m| m.reasoning_content.is_none()), "{:?}", req.messages);
    assert!(req.messages.iter().any(|m| m.text() == "Hello!"), "the reply itself stays");
}

#[test]
fn the_agents_that_read_code_get_the_repository_overview() {
    let b = ModelBinding::default();
    let book = Book::new(b.clone(), b, vec![], vec![], std::path::Path::new("/tmp"));
    assert!(!book.subagent_prompt("plan").unwrap().contains("Repository map"), "no overview, no section");
    *book.repo_overview.write().unwrap() = "src/ — the parser (3 files, ~400 lines)\n".into();
    for p in [book.subagent_prompt("plan").unwrap(), book.subagent_prompt("answer").unwrap(), book.system_prompt(Group::Spec)] {
        assert!(p.contains("## Repository map") && p.contains("src/ — the parser"), "{p}");
    }
    assert!(!book.system_prompt(Group::Architecture).contains("Repository map"), "the main agent only routes");
}

#[test]
fn between_orders_a_long_history_is_digested_through_its_cached_prefix_first() {
    let mut a =
        LlmAgent::new(ModelBinding::default(), "s".into(), "Order #1: read".into(), vec![], "x".into()).resident().with_budget(16_000);
    let view = AcbView { pid: 1, ticket: 1, group: Group::Architecture, steps: 0, notes: vec![] };
    a.step(&view, vec![]);
    let reply = |t: &str| SyscallResult::Completion { message: Message::assistant(t), usage: Usage::default(), cut: false };
    // Order 1 ends with a long reply: the history is past 40% of the 16k budget.
    let big = format!("Here is all of it: {}", "z".repeat(30_000));
    let StepResult::Syscalls(c) = a.step(&view, vec![reply(&big)]) else { panic!() };
    assert!(matches!(&c[0], Syscall::Serve { .. }));
    // Order 2 arrives: first the digest request, with the agent's own prefix and the order held back.
    let order = SyscallResult::Message(zilch_ipc::Msg { from: 0, kind: zilch_ipc::MsgKind::Guest, body: "Order #2: and now?".into() });
    let StepResult::Syscalls(c) = a.step(&view, vec![order]) else { panic!() };
    let Syscall::Complete(req) = &c[0] else { panic!() };
    let texts: Vec<&str> = req.messages.iter().map(|m| m.text()).collect();
    assert_eq!(texts[0], "s");
    assert_eq!(texts[1], "Order #1: read");
    assert!(texts.last().unwrap().starts_with("[compaction]") && !texts.iter().any(|t| t.contains("Order #2")), "{texts:?}");
    // The digest arrives: the order goes out on top of it, the long reply is gone.
    let StepResult::Syscalls(c) = a.step(&view, vec![reply("- order 1: read everything")]) else { panic!() };
    let Syscall::Complete(req) = &c[0] else { panic!() };
    let texts: Vec<&str> = req.messages.iter().map(|m| m.text()).collect();
    assert!(texts[1].starts_with("<conversation-summary>") && texts[1].contains("order 1: read everything"), "{texts:?}");
    assert_eq!(*texts.last().unwrap(), "Order #2: and now?");
    assert!(compact::estimate_tokens("", &req.messages, &[]) < 10_000, "the long reply is not resent");
}
