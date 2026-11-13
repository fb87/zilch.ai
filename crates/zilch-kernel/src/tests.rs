//! Kernel behaviour with scripted agents and fake drivers (no network, no model).

use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;
use zilch_ipc::Envelope;

/// Fake equipment. Tools: `work` (sleeps), `hang` (until cancelled), `write` (path-checked),
/// `exec` (a command), anything else answers "ok".
#[derive(Default)]
struct FakeDrivers {
    spans: Mutex<Vec<(Pid, Instant, Instant)>>,
    hangs: AtomicUsize,
}

impl Drivers for FakeDrivers {
    fn classify(&self, name: &str, args: &Value) -> Access {
        match name {
            "write" => Access::Write(args["path"].as_str().unwrap_or("").into()),
            "exec" => Access::Exec,
            _ => Access::Read,
        }
    }
    fn run(&self, ctx: &DriverCtx, call: Syscall) -> SyscallResult {
        let name = match call {
            Syscall::Tool { name, .. } => name,
            // Every model call "costs" 30 tokens.
            _ => {
                return SyscallResult::Completion {
                    message: zilch_ipc::Message::assistant("thinking"),
                    usage: zilch_ipc::Usage { prompt_tokens: 25, completion_tokens: 5, ..Default::default() },
                    cut: false,
                }
            }
        };
        match name.as_str() {
            "work" => {
                let start = Instant::now();
                std::thread::sleep(Duration::from_millis(150));
                self.spans.lock().unwrap().push((ctx.pid, start, Instant::now()));
                SyscallResult::Output("worked".into())
            }
            "hang" => {
                self.hangs.fetch_add(1, Ordering::SeqCst);
                while !ctx.cancel.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                SyscallResult::Failed("cancelled".into())
            }
            "same" => SyscallResult::Output("same result".into()),
            // Slow but healthy: runs well past the hang threshold, sending heartbeats.
            "slow" => {
                let end = Instant::now() + Duration::from_millis(3500);
                while Instant::now() < end {
                    (ctx.alive)();
                    std::thread::sleep(Duration::from_millis(20));
                }
                SyscallResult::Output("slow done".into())
            }
            _ => SyscallResult::Output("ok".into()),
        }
    }
}

fn tool(name: &str, args: Value) -> Syscall {
    Syscall::Tool { call_id: "c".into(), name: name.into(), args }
}

/// An agent driven by a closure over (step number, wake results).
struct Scripted<F: FnMut(u32, Vec<SyscallResult>, &AcbView) -> StepResult + Send>(u32, F);

impl<F: FnMut(u32, Vec<SyscallResult>, &AcbView) -> StepResult + Send> Agent for Scripted<F> {
    fn step(&mut self, view: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
        self.0 += 1;
        (self.1)(self.0, wake, view)
    }
}

type DevScript = Arc<dyn Fn(&PlanTask) -> Box<dyn Agent> + Send + Sync>;

/// Architect: submits `tasks`, then exits with the digest (or the rejection text).
struct Book {
    tasks: Vec<PlanTask>,
    dev: DevScript,
}

impl Recipes for Book {
    fn architect(&self, _ticket: TicketId, _order: &str) -> Box<dyn Agent> {
        let tasks = self.tasks.clone();
        Box::new(Scripted(0, move |n, wake, _| match n {
            1 => StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: tasks.clone() }]),
            _ => StepResult::Exit(ExitStatus::ok(wake.first().map(SyscallResult::text).unwrap_or_default())),
        }))
    }
    fn developer(&self, _ticket: TicketId, task: &PlanTask, _context: &str) -> Box<dyn Agent> {
        (self.dev)(task)
    }
}

fn task(id: &str, deps: &[&str], paths: &[&str]) -> PlanTask {
    PlanTask {
        id: id.into(),
        title: id.into(),
        task: format!("do {id}"),
        depends_on: deps.iter().map(|s| s.to_string()).collect(),
        owned_paths: paths.iter().map(|s| s.to_string()).collect(),
        acceptance: vec![],
        priority: None,
        worker: None,
    }
}

/// A developer that calls `work` once and reports done.
fn worker() -> DevScript {
    Arc::new(|_| {
        Box::new(Scripted(0, |n, _, _| match n {
            1 => StepResult::Syscalls(vec![tool("work", json!({}))]),
            _ => StepResult::Exit(ExitStatus::ok("did it\nSTATUS: done")),
        }))
    })
}

use serde_json::json;

fn cfg() -> KernelConfig {
    KernelConfig { workers: 4, io_threads: 8, exec_threads: 2, tick: Duration::from_millis(10), auto_approve: true, ..Default::default() }
}

struct Run {
    k: KernelHandle,
    events: Receiver<Arc<Envelope>>,
    seen: Vec<KernelEvent>,
}

impl Run {
    fn new(cfg: KernelConfig, drivers: Arc<FakeDrivers>, book: Book) -> Self {
        let bus = Arc::new(EventBus::new());
        let events = bus.subscribe(10_000);
        let k = start(cfg, drivers, Arc::new(book), bus);
        Self { k, events, seen: vec![] }
    }
    /// Collect events until `until` matches one (or panic after 10 s).
    fn wait(&mut self, until: impl Fn(&KernelEvent) -> bool) -> KernelEvent {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let left = deadline.saturating_duration_since(Instant::now());
            let env = self.events.recv_timeout(left).unwrap_or_else(|_| panic!("timed out; saw {:#?}", self.seen));
            self.seen.push(env.event.clone());
            if until(&env.event) {
                return env.event.clone();
            }
        }
    }
    fn served(&mut self) -> (bool, String) {
        match self.wait(|e| matches!(e, KernelEvent::TicketServed { .. })) {
            KernelEvent::TicketServed { ok, summary, .. } => (ok, summary),
            _ => unreachable!(),
        }
    }
}

#[test]
fn independent_tasks_run_in_parallel_and_dependents_wait() {
    let drivers = Arc::new(FakeDrivers::default());
    let tasks = vec![task("T1", &[], &["a.rs"]), task("T2", &[], &["b.rs"]), task("T3", &["T1", "T2"], &["c.rs"])];
    let mut run = Run::new(cfg(), drivers.clone(), Book { tasks, dev: worker() });
    run.k.send(Cmd::Order("build it".into()));
    let (ok, summary) = run.served();
    assert!(ok, "{summary}");
    assert!(summary.contains("[T1]") && summary.contains("[T3] T3 — Done"), "{summary}");

    let spans = drivers.spans.lock().unwrap().clone();
    assert_eq!(spans.len(), 3);
    // pids: 1 = architect, then T1, T2 spawned together, T3 later.
    let span = |pid: Pid| spans.iter().find(|s| s.0 == pid).copied().unwrap();
    let (t1, t2, t3) = (span(2), span(3), span(4));
    assert!(t1.1 < t2.2 && t2.1 < t1.2, "T1 and T2 overlapped in time");
    assert!(t3.1 >= t1.2 && t3.1 >= t2.2, "T3 started after both");
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Question { .. })), "the plan went to the guest");
}

#[test]
fn capabilities_confine_writes_to_owned_paths_and_commands_to_developers() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, wake, _| match n {
            1 => StepResult::Syscalls(vec![
                tool("write", json!({"path":"src/mine.rs"})),
                tool("write", json!({"path":"src/other.rs"})),
                tool("exec", json!({})),
            ]),
            _ => {
                let r: Vec<String> = wake.iter().map(SyscallResult::text).collect();
                assert_eq!(r[0], "ok");
                assert!(r[1].contains("permission denied") && r[1].contains("src/mine.rs"), "{}", r[1]);
                assert_eq!(r[2], "ok", "developers may run commands");
                StepResult::Exit(ExitStatus::ok("STATUS: done"))
            }
        }))
    });
    // The main agent tries a command and writes before planning: it may do neither.
    struct Arch(DevScript);
    impl Recipes for Arch {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Scripted(0, |n, wake, _| match n {
                1 => StepResult::Syscalls(vec![
                    tool("exec", json!({})),
                    tool("write", json!({"path":"src/x.rs"})),
                    tool("write", json!({"path":"docs/design.md"})),
                ]),
                2 => {
                    let r: Vec<String> = wake.iter().map(SyscallResult::text).collect();
                    assert!(r[0].contains("may not run commands"), "{}", r[0]);
                    assert!(r[1].contains("permission denied"), "{}", r[1]);
                    assert!(r[2].contains("permission denied") && r[2].contains("(none)"), "{}", r[2]);
                    StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: vec![task("T1", &[], &["src/mine.rs"])] }])
                }
                _ => StepResult::Exit(ExitStatus::ok(wake[0].text())),
            }))
        }
        fn developer(&self, _: TicketId, t: &PlanTask, _: &str) -> Box<dyn Agent> {
            (self.0)(t)
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), drivers, Arc::new(Arch(dev)), bus);
    k.send(Cmd::Order("x".into()));
    let mut run = Run { k, events, seen: vec![] };
    let (ok, summary) = run.served();
    assert!(ok && summary.contains("Done"), "{summary}");
}

#[test]
fn a_hung_call_is_interrupted_then_the_process_killed_and_dependents_blocked() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|t| {
        let hang = t.id == "T1";
        Box::new(Scripted(0, move |n, _, _| match (hang, n) {
            (true, _) => StepResult::Syscalls(vec![tool("hang", json!({"n": n}))]),
            // A quick call (not the 150 ms "work"): only T1 should ever look hung.
            (false, 1) => StepResult::Syscalls(vec![tool("check", json!({}))]),
            _ => StepResult::Exit(ExitStatus::ok("STATUS: done")),
        }))
    });
    let tasks = vec![task("T1", &[], &["a"]), task("T2", &["T1"], &["b"]), task("T3", &[], &["c"])];
    let mut run = Run::new(KernelConfig { hang_ticks: 30, ..cfg() }, drivers.clone(), Book { tasks, dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { pid: 2, signal: Signal::Interrupt, .. })), "interrupted first");
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { pid: 2, signal: Signal::Kill, .. })), "then killed");
    assert!(drivers.hangs.load(Ordering::SeqCst) >= 2, "it was given a second chance after the interrupt");
    assert!(
        summary.contains("[T1] T1 — Failed") && summary.contains("[T2] T2 — Blocked") && summary.contains("[T3] T3 — Done"),
        "{summary}"
    );
}

#[test]
fn a_looping_process_is_nudged_then_stopped() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |_, _, view: &AcbView| {
            if view.notes.iter().any(|n| n.starts_with("[supervisor]")) {
                // It sees the nudge… and loops anyway.
            }
            StepResult::Syscalls(vec![tool("same", json!({"q": 1}))])
        }))
    });
    let mut run = Run::new(cfg(), drivers, Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { signal: Signal::Warn, reason, .. } if reason.contains("nudged"))));
    assert!(summary.contains("stopped by the supervisor") && summary.contains("Failed"), "{summary}");
    let steps = run.seen.iter().filter(|e| matches!(e, KernelEvent::SyscallIssued { pid: 2, .. })).count();
    assert_eq!(steps, 4, "3 identical calls, a nudge, the 4th stops it (not max_steps)");
}

#[test]
fn recv_parks_without_a_worker_and_send_wakes_it() {
    let drivers = Arc::new(FakeDrivers::default());
    // T1 (pid 2) waits for a message; T2 (pid 3) works a bit, then sends it one.
    let dev: DevScript = Arc::new(|t| {
        let receiver = t.id == "T1";
        Box::new(Scripted(0, move |n, wake, _| match (receiver, n) {
            (true, 1) => StepResult::Syscalls(vec![Syscall::Recv]),
            (true, _) => StepResult::Exit(ExitStatus::ok(format!("got: {}\nSTATUS: done", wake[0].text()))),
            (false, 1) => StepResult::Syscalls(vec![tool("work", json!({}))]),
            (false, 2) => StepResult::Syscalls(vec![Syscall::Send { to: 2, kind: MsgKind::Task, body: "hello T1".into() }]),
            (false, _) => StepResult::Exit(ExitStatus::ok("STATUS: done")),
        }))
    });
    let tasks = vec![task("T1", &[], &["a"]), task("T2", &[], &["b"])];
    // One worker only: if Recv held it, T2 could never run and the order would hang.
    let mut run = Run::new(KernelConfig { workers: 1, ..cfg() }, drivers, Book { tasks, dev });
    run.k.send(Cmd::Order("x".into()));
    let (ok, summary) = run.served();
    assert!(ok && summary.contains("got: [message from 3 (Task)]\nhello T1"), "{summary}");
}

#[test]
fn budgets_count_model_calls_with_a_step_cap_as_a_safety_net() {
    // The budget counts model calls (like zilch's max_steps): a model call then a tool call per
    // round trip, so 5 calls = 10 steps; it stops on the 6th call, not after 5 steps.
    let dev: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, _, _| match n % 2 {
            1 => StepResult::Syscalls(vec![llm_call()]),
            _ => StepResult::Syscalls(vec![tool("x", json!({"n": n}))]),
        }))
    });
    let k = KernelConfig { max_steps: 5, loop_limit: 0, ..cfg() };
    let mut run = Run::new(k.clone(), Arc::new(FakeDrivers::default()), Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    // It ignores the wrap-up note, so it is stopped after the grace (5 more calls).
    assert!(summary.contains("model-call budget exhausted (10 calls for this task, wrap-up included"), "{summary}");
    let calls =
        run.seen.iter().filter(|e| matches!(e, KernelEvent::SyscallIssued { pid: 2, summary, .. } if summary.starts_with("llm"))).count();
    assert_eq!(calls, 10);
    assert!(run
        .seen
        .iter()
        .any(|e| matches!(e, KernelEvent::Signal { pid: 2, signal: Signal::Warn, reason } if reason.contains("asked to wrap up"))));

    // Without model calls, raw steps are still capped (4× the budget).
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |n, _, _| StepResult::Syscalls(vec![tool("x", json!({"n": n}))]))));
    let mut run = Run::new(k, Arc::new(FakeDrivers::default()), Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(summary.contains("step budget exhausted (20 steps for this task)"), "{summary}");
}

#[test]
fn the_guest_can_reject_a_plan_or_kill_a_process() {
    let drivers = Arc::new(FakeDrivers::default());
    let mut run = Run::new(
        KernelConfig { auto_approve: false, ..cfg() },
        drivers.clone(),
        Book { tasks: vec![task("T1", &[], &["a"])], dev: worker() },
    );
    run.k.send(Cmd::Order("x".into()));
    let KernelEvent::Question { qid, text, .. } = run.wait(|e| matches!(e, KernelEvent::Question { .. })) else { unreachable!() };
    assert!(text.contains("wave 1") && text.contains("[T1]"), "{text}");
    run.k.send(Cmd::Answer { qid, text: "n".into() });
    let (_, summary) = run.served();
    assert!(summary.contains("rejected"), "{summary}");
    assert!(drivers.spans.lock().unwrap().is_empty(), "nothing ran");

    // Kill a hanging developer by hand: its task fails, the order still finishes.
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |_, _, _| StepResult::Syscalls(vec![tool("hang", json!({}))]))));
    let mut run = Run::new(cfg(), drivers.clone(), Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    run.wait(|e| matches!(e, KernelEvent::SyscallIssued { pid: 2, .. }));
    let ps = run.k.ps();
    assert!(ps.iter().any(|p| p.pid == 2 && p.state == "blocked(io)" && p.doing.starts_with("hang")), "{ps:?}");
    run.k.send(Cmd::Kill(2));
    let (_, summary) = run.served();
    assert!(summary.contains("killed by the user") && summary.contains("Failed"), "{summary}");
}

#[test]
fn higher_priority_runs_first_and_waiting_work_ages_upward() {
    let bus = Arc::new(EventBus::new());
    let mut k = Kernel::new(
        KernelConfig { age_ticks: 10, ..cfg() },
        Arc::new(FakeDrivers::default()),
        Arc::new(Book { tasks: vec![], dev: worker() }),
        bus,
    );
    let caps = || Caps { write: vec![], write_any: false, exec: false };
    let idle = || -> Box<dyn Agent> { Box::new(Scripted(0, |_, _, _| StepResult::Yield)) };
    let low = k.spawn(idle(), None, 1, Group::Development, "low".into(), caps(), CancelToken::new(), 2);
    let high = k.spawn(idle(), None, 1, Group::Architecture, "high".into(), caps(), CancelToken::new(), 0);
    let mid = k.spawn(idle(), None, 1, Group::Development, "mid".into(), caps(), CancelToken::new(), 1);
    assert_eq!(k.pick_next(), Some(high));
    // Put it back and let `low` wait long enough to age into level 1, behind `mid`.
    k.enqueue(high);
    k.tick = 10;
    k.on_tick();
    assert_eq!(k.acbs[&low].priority, 1, "aged one level");
    assert_eq!((k.pick_next(), k.pick_next(), k.pick_next()), (Some(high), Some(mid), Some(low)));
}

/// A developer that loops (or hangs) until it is restarted `cure_after` times, then reports done.
/// It records every restart summary it gets.
struct Stubborn {
    restarts: usize,
    cure_after: usize,
    hang: bool,
    summaries: Arc<Mutex<Vec<String>>>,
}

impl Agent for Stubborn {
    fn step(&mut self, _view: &AcbView, _wake: Vec<SyscallResult>) -> StepResult {
        if self.restarts >= self.cure_after {
            return match _wake.is_empty() {
                // A quick call: "work" takes 150 ms without heartbeats, past the 10-tick hang threshold
                // of the hang test, so a healthy developer would be taken for a hung one.
                true => StepResult::Syscalls(vec![tool("check", json!({}))]),
                false => StepResult::Exit(ExitStatus::ok("fixed it\nSTATUS: done")),
            };
        }
        let name = if self.hang { "hang" } else { "same" };
        StepResult::Syscalls(vec![tool(name, json!({"q": 1}))])
    }
    fn restart(&mut self, summary: &str) -> bool {
        self.restarts += 1;
        self.summaries.lock().unwrap().push(summary.to_string());
        true
    }
}

fn stubborn(cure_after: usize, hang: bool, summaries: Arc<Mutex<Vec<String>>>) -> DevScript {
    Arc::new(move |_| Box::new(Stubborn { restarts: 0, cure_after, hang, summaries: summaries.clone() }))
}

#[test]
fn a_loop_that_survives_the_nudge_is_restarted_with_a_summary_of_what_was_tried() {
    let summaries: Arc<Mutex<Vec<String>>> = Arc::default();
    let mut run = Run::new(
        cfg(),
        Arc::new(FakeDrivers::default()),
        Book { tasks: vec![task("T1", &[], &["a"])], dev: stubborn(1, false, summaries.clone()) },
    );
    run.k.send(Cmd::Order("x".into()));
    let (ok, summary) = run.served();
    assert!(ok && summary.contains("fixed it"), "{summary}");
    assert!(run
        .seen
        .iter()
        .any(|e| matches!(e, KernelEvent::Signal { pid: 2, signal: Signal::Restart, reason } if reason.contains("1 of 2"))));
    let s = summaries.lock().unwrap();
    assert_eq!(s.len(), 1);
    assert!(
        s[0].contains("same arguments") && s[0].contains("same(") && s[0].contains("→ same result"),
        "the summary says what was tried: {}",
        s[0]
    );
}

#[test]
fn restarts_are_limited_then_the_process_is_stopped() {
    let summaries: Arc<Mutex<Vec<String>>> = Arc::default();
    let k = KernelConfig { max_restarts: 1, ..cfg() };
    let mut run = Run::new(
        k,
        Arc::new(FakeDrivers::default()),
        Book { tasks: vec![task("T1", &[], &["a"])], dev: stubborn(usize::MAX, false, summaries.clone()) },
    );
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert_eq!(summaries.lock().unwrap().len(), 1, "one restart allowed");
    assert!(summary.contains("stopped by the supervisor") && summary.contains("Failed"), "{summary}");
}

#[test]
fn a_hung_process_that_can_restart_gets_a_fresh_start_instead_of_a_kill() {
    let summaries: Arc<Mutex<Vec<String>>> = Arc::default();
    let k = KernelConfig { hang_ticks: 10, ..cfg() };
    let mut run = Run::new(
        k,
        Arc::new(FakeDrivers::default()),
        Book { tasks: vec![task("T1", &[], &["a"])], dev: stubborn(1, true, summaries.clone()) },
    );
    run.k.send(Cmd::Order("x".into()));
    let (ok, summary) = run.served();
    assert!(ok && summary.contains("fixed it"), "{summary}");
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { signal: Signal::Interrupt, .. })));
    // Only the hung call is interrupted: once cured, the developer is healthy (a slow call of the cured
    // developer that crossed the 10-tick threshold was interrupted too, and on a slow machine its
    // late result led to more restarts until the limit killed it).
    let interrupts = run.seen.iter().filter(|e| matches!(e, KernelEvent::Signal { signal: Signal::Interrupt, .. })).count();
    assert_eq!(interrupts, 1, "only the hung call is interrupted");
    assert!(run
        .seen
        .iter()
        .any(|e| matches!(e, KernelEvent::Signal { signal: Signal::Restart, reason, .. } if reason.contains("no progress"))));
    assert!(!run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { signal: Signal::Kill, .. })));
}

#[test]
fn processes_waiting_on_each_other_are_woken_instead_of_waiting_forever() {
    // Both developers wait for a message; nobody will ever send one.
    let dev: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, wake, _| match n {
            1 => StepResult::Syscalls(vec![Syscall::Recv]),
            _ => StepResult::Exit(ExitStatus::ok(format!("gave up: {}\nSTATUS: done", wake[0].text()))),
        }))
    });
    let mut run =
        Run::new(cfg(), Arc::new(FakeDrivers::default()), Book { tasks: vec![task("T1", &[], &["a"]), task("T2", &[], &["b"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert_eq!(summary.matches("gave up: Error: deadlock").count(), 2, "{summary}");
    let order: Vec<Pid> = run
        .seen
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Signal { pid, reason, .. } if reason.starts_with("deadlock") => Some(*pid),
            _ => None,
        })
        .collect();
    assert_eq!(order, vec![3, 2], "the youngest waiter is woken first");
}

fn llm_call() -> Syscall {
    Syscall::Complete(zilch_ipc::ChatRequest { binding: Default::default(), messages: vec![], tools: vec![], session: None })
}

#[test]
fn urgent_tasks_start_first_when_workers_are_scarce() {
    let drivers = Arc::new(FakeDrivers::default());
    let mut low = task("LOW", &[], &["a"]);
    low.priority = Some(3);
    let mut high = task("HIGH", &[], &["b"]);
    high.priority = Some(0);
    // LOW is listed (and spawned) first; with one worker HIGH must still run first.
    let mut run = Run::new(KernelConfig { workers: 1, ..cfg() }, drivers.clone(), Book { tasks: vec![low, high], dev: worker() });
    run.k.send(Cmd::Order("x".into()));
    let KernelEvent::Question { text, .. } = run.wait(|e| matches!(e, KernelEvent::Question { .. })) else { unreachable!() };
    assert!(text.contains("[LOW] LOW  (a · priority 3)") && text.contains("[HIGH] HIGH  (b · priority 0)"), "{text}");
    run.served();
    let spans = drivers.spans.lock().unwrap().clone();
    let first = spans.iter().min_by_key(|s| s.1).unwrap().0;
    assert_eq!(first, 3, "pid 3 (HIGH) ran before pid 2 (LOW)");
}

#[test]
fn a_group_over_its_token_quota_is_stopped_and_stats_show_usage() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |_, _, _| StepResult::Syscalls(vec![llm_call()]))));
    // Loop detection off: the fake model always gives the same reply.
    let k = KernelConfig { group_tokens: [(Group::Development, 100)].into(), loop_limit: 0, ..cfg() };
    let mut run = Run::new(k, drivers, Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(summary.contains("the developer group used up its token quota (120 of 100 tokens)"), "{summary}");
    let stats = run.k.stats();
    assert_eq!(stats.groups, vec![(Group::Development, 120, Some(100))]);
    assert_eq!((stats.busy_workers, stats.queued.iter().sum::<usize>()), (0, 0));
}

#[test]
fn the_guest_can_send_input_to_a_process_or_a_group() {
    use zilch_ipc::Target;
    let drivers = Arc::new(FakeDrivers::default());
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let log = seen.clone();
    // T1 (pid 2) works for a while and records the notes it is given; T2 (pid 3) waits for a message.
    let dev: DevScript = Arc::new(move |t| {
        let log = log.clone();
        let waiter = t.id == "T2";
        Box::new(Scripted(0, move |n, wake, view: &AcbView| {
            log.lock().unwrap().extend(view.notes.iter().cloned());
            match (waiter, n) {
                (true, 1) => StepResult::Syscalls(vec![Syscall::Recv]),
                (true, _) => StepResult::Exit(ExitStatus::ok(format!("got {}\nSTATUS: done", wake[0].text()))),
                (false, 1..=3) => StepResult::Syscalls(vec![tool("work", json!({"n": n}))]),
                (false, _) => StepResult::Exit(ExitStatus::ok("STATUS: done")),
            }
        }))
    });
    let mut run = Run::new(cfg(), drivers, Book { tasks: vec![task("T1", &[], &["a"]), task("T2", &[], &["b"])], dev });
    run.k.send(Cmd::Order("x".into()));
    run.wait(|e| matches!(e, KernelEvent::SyscallIssued { pid: 3, .. }));
    run.wait(|e| matches!(e, KernelEvent::SyscallIssued { pid: 2, .. }));

    assert!(run.k.tell(Target::Pid(2), "use thiserror").unwrap().contains("before its next model call"));
    assert!(run.k.tell(Target::Pid(3), "here you go").unwrap().contains("delivered now"));
    assert!(run.k.tell(Target::Pid(99), "x").unwrap_err().contains("no process 99"));
    assert!(run.k.tell(Target::Pid(2), "   ").unwrap_err().contains("nothing to send"));
    assert!(run.k.tell(Target::Group(Group::Review), "x").unwrap_err().contains("no live reviewer"));
    let (_, summary) = run.served();
    assert!(summary.contains("got [message from 0 (Guest)]\nhere you go"), "{summary}");
    assert!(seen.lock().unwrap().iter().any(|n| n == "[guest] use thiserror"), "{:?}", seen.lock().unwrap());
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Told { pid: 2, text, .. } if text == "use thiserror")), "logged for replay");
    assert!(run.k.tell(Target::Pid(2), "late").is_err(), "a finished process cannot be told anything");
}

#[test]
fn a_group_message_reaches_every_live_member_of_the_newest_order() {
    use zilch_ipc::Target;
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |_, _, _| StepResult::Syscalls(vec![tool("hang", json!({}))]))));
    let mut run = Run::new(cfg(), drivers, Book { tasks: vec![task("T1", &[], &["a"]), task("T2", &[], &["b"])], dev });
    run.k.send(Cmd::Order("x".into()));
    run.wait(|e| matches!(e, KernelEvent::SyscallIssued { pid: 3, .. }));
    let how = run.k.tell(Target::Group(Group::Development), "stop and report").unwrap();
    assert!(how.contains("2:") && how.contains("3:"), "{how}");
    let arch = run.k.tell(Target::Group(Group::Architecture), "also add docs").unwrap();
    assert!(arch.contains("1: queued: it reads it when its tasks finish"), "{arch}");
    run.k.send(Cmd::KillTicket(1));
    run.served();
}

/// A resident main agent: "plan …" orders go through submit_plan, anything else is served at
/// once. It records every input and note it sees, in order.
struct Main {
    phase: u8,
    seen: Arc<Mutex<Vec<String>>>,
    tasks: Vec<PlanTask>,
}

impl Agent for Main {
    fn step(&mut self, view: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
        if idle_wake(&wake) {
            return StepResult::Syscalls(vec![Syscall::Recv]); // nothing to compact: keep waiting
        }
        let mut seen = self.seen.lock().unwrap();
        seen.extend(view.notes.iter().cloned());
        for w in &wake {
            seen.push(w.text());
        }
        let last = wake.last().map(SyscallResult::text).unwrap_or_default();
        drop(seen);
        match self.phase {
            // A new order (the first one comes as the program's input, later ones as messages).
            0 | 2 if last.contains("plan") || (self.phase == 0 && wake.is_empty()) => {
                self.phase = 1;
                StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: self.tasks.clone() }])
            }
            1 => {
                self.phase = 2;
                StepResult::Syscalls(vec![Syscall::Serve { summary: format!("served: {}", last.lines().next().unwrap_or("")) }])
            }
            _ => {
                self.phase = 2;
                StepResult::Syscalls(vec![Syscall::Serve { summary: format!("answered: {last}") }])
            }
        }
    }
}

struct MainBook {
    seen: Arc<Mutex<Vec<String>>>,
    tasks: Vec<PlanTask>,
    dev: DevScript,
    mains: Arc<AtomicUsize>,
}

impl Recipes for MainBook {
    fn architect(&self, _: TicketId, order: &str) -> Box<dyn Agent> {
        self.mains.fetch_add(1, Ordering::SeqCst);
        self.seen.lock().unwrap().push(order.to_string());
        Box::new(Main { phase: 0, seen: self.seen.clone(), tasks: self.tasks.clone() })
    }
    fn spec(&self, _: TicketId, order: &str) -> Option<Box<dyn Agent>> {
        self.seen.lock().unwrap().push(format!("spec: {order}"));
        // A spec agent never plans: phase 2 answers every order directly.
        Some(Box::new(Main { phase: 2, seen: self.seen.clone(), tasks: vec![] }))
    }
    fn developer(&self, _: TicketId, t: &PlanTask, _: &str) -> Box<dyn Agent> {
        (self.dev)(t)
    }
}

fn main_run(dev: DevScript) -> (Run, Arc<Mutex<Vec<String>>>, Arc<AtomicUsize>) {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let mains = Arc::new(AtomicUsize::new(0));
    let book = MainBook { seen: seen.clone(), tasks: vec![task("T1", &[], &["a"])], dev, mains: mains.clone() };
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), Arc::new(FakeDrivers::default()), Arc::new(book), bus);
    (Run { k, events, seen: vec![] }, seen, mains)
}

fn served_ticket(run: &mut Run) -> (TicketId, bool, String) {
    match run.wait(|e| matches!(e, KernelEvent::TicketServed { .. })) {
        KernelEvent::TicketServed { ticket, ok, summary } => (ticket, ok, summary),
        _ => unreachable!(),
    }
}

#[test]
fn one_main_agent_serves_every_order_in_one_context_and_queues_new_ones() {
    let (mut run, seen, mains) = main_run(worker());
    run.k.send(Cmd::Order("plan the first thing".into()));
    // A second order while the first is still running: it waits for the main agent.
    run.k.send(Cmd::Order("what did you do?".into()));
    let (t1, ok1, s1) = served_ticket(&mut run);
    assert_eq!((t1, ok1), (1, true));
    assert!(s1.starts_with("served: All tasks finished"), "{s1}");
    let (t2, _, s2) = served_ticket(&mut run);
    assert_eq!(t2, 2);
    assert_eq!(s2, "answered: [message from 0 (Guest)]\nOrder #2: what did you do?");
    assert_eq!(mains.load(Ordering::SeqCst), 1, "one main agent for the whole session");
    let ps = run.k.ps();
    assert!(ps.iter().any(|p| p.pid == 1 && p.state == "blocked(recv)"), "it waits for the next order: {ps:?}");
    assert!(!ps.iter().any(|p| p.pid == 2), "finished subagents are gone");
    let seen = seen.lock().unwrap();
    assert_eq!(seen[0], "Order #1: plan the first thing");
}

#[test]
fn cancelling_an_order_stops_its_subagents_but_keeps_the_main_agent() {
    use zilch_ipc::Target;
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |_, _, _| StepResult::Syscalls(vec![tool("hang", json!({}))]))));
    let (mut run, seen, mains) = main_run(dev);
    run.k.send(Cmd::Order("plan something long".into()));
    run.wait(|e| matches!(e, KernelEvent::SyscallIssued { pid: 2, .. }));
    // A note to the subagent is also heard by the main agent.
    run.k.tell(Target::Pid(2), "hurry up").unwrap();
    run.k.send(Cmd::KillTicket(1));
    let (t, ok, summary) = served_ticket(&mut run);
    assert_eq!((t, ok), (1, false));
    assert!(summary.contains("cancelled by the guest"), "{summary}");
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::Exited { pid: 2, ok: false, .. })), "the subagent was stopped");
    // The main agent is still there, and an @main note while it is idle becomes the next order.
    let deadline = Instant::now() + Duration::from_secs(5);
    while !run.k.ps().iter().any(|p| p.pid == 1 && p.state == "blocked(recv)") {
        assert!(Instant::now() < deadline, "main agent did not go idle");
        std::thread::sleep(Duration::from_millis(20));
    }
    let how = run.k.tell(Target::Group(Group::Architecture), "status?").unwrap();
    assert!(how.contains("order #2"), "{how}");
    let (t2, _, s2) = served_ticket(&mut run);
    assert_eq!(t2, 2);
    assert!(s2.contains("Order #2: status?"), "{s2}");
    assert_eq!(mains.load(Ordering::SeqCst), 1);
    let seen = seen.lock().unwrap();
    assert!(seen.iter().any(|n| n == "[guest → subagent 2] hurry up"), "{seen:?}");
    assert!(seen.iter().any(|n| n.contains("I cancelled order #1")), "{seen:?}");
}

#[test]
fn a_killed_main_agent_is_replaced_by_a_new_one_on_the_next_order() {
    let (mut run, _seen, mains) = main_run(worker());
    run.k.send(Cmd::Order("hello".into()));
    let (_, _, s) = served_ticket(&mut run);
    assert!(s.starts_with("served:") || s.starts_with("answered:"), "{s}");
    run.k.send(Cmd::Kill(1));
    run.wait(|e| matches!(e, KernelEvent::Exited { pid: 1, .. }));
    run.k.send(Cmd::Order("plan again".into()));
    let (t, ok, _) = served_ticket(&mut run);
    assert_eq!((t, ok), (2, true));
    assert_eq!(mains.load(Ordering::SeqCst), 2, "a fresh main agent for the next order");
}

#[test]
fn a_slow_call_that_sends_heartbeats_is_not_taken_for_a_hang() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, wake, _| match n {
            1 => StepResult::Syscalls(vec![tool("slow", json!({}))]),
            _ => StepResult::Exit(ExitStatus::ok(format!("{}\nSTATUS: done", wake[0].text()))),
        }))
    });
    // Hang threshold 2.5 s; the call takes 3.5 s but keeps saying it is alive (heartbeats reach
    // the kernel at most once a second, so the threshold leaves room for a slow machine).
    let mut run = Run::new(KernelConfig { hang_ticks: 250, ..cfg() }, drivers, Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (ok, summary) = run.served();
    assert!(ok && summary.contains("slow done"), "{summary}");
    assert!(
        !run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { signal: Signal::Interrupt | Signal::Kill, .. })),
        "no interrupt for a live call"
    );
}

#[test]
fn a_plan_is_approved_only_by_an_explicit_yes() {
    let drivers = Arc::new(FakeDrivers::default());
    let mut run = Run::new(
        KernelConfig { auto_approve: false, ..cfg() },
        drivers.clone(),
        Book { tasks: vec![task("T1", &[], &["a"])], dev: worker() },
    );
    run.k.send(Cmd::Order("x".into()));
    let KernelEvent::Question { qid, .. } = run.wait(|e| matches!(e, KernelEvent::Question { .. })) else { unreachable!() };
    // A stray Enter does nothing: the plan keeps waiting and nothing runs.
    run.k.send(Cmd::Answer { qid, text: "   ".into() });
    std::thread::sleep(Duration::from_millis(150));
    let ps = run.k.ps();
    assert!(ps.iter().any(|p| p.pid == 1 && p.state == "blocked(question)"), "{ps:?}");
    assert!(drivers.spans.lock().unwrap().is_empty());
    // "Y" (any case) approves.
    run.k.send(Cmd::Answer { qid, text: "Y".into() });
    let (ok, summary) = run.served();
    assert!(ok && summary.contains("[T1] T1 — Done"), "{summary}");
}

#[test]
fn stats_show_calls_in_flight_even_when_no_step_worker_is_busy() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |_, _, _| StepResult::Syscalls(vec![tool("hang", json!({}))]))));
    let mut run = Run::new(cfg(), drivers, Book { tasks: vec![task("T1", &[], &["a"]), task("T2", &[], &["b"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let deadline = Instant::now() + Duration::from_secs(5);
    let st = loop {
        let st = run.k.stats();
        if st.tool_calls == 2 || Instant::now() > deadline {
            break st;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!((st.tool_calls, st.model_calls, st.commands), (2, 0, 0), "{st:?}");
    assert_eq!((st.busy, st.waiting), (2, 1), "two subagents busy, the main agent waits on them: {st:?}");
    assert_eq!(st.busy_workers, 0, "step workers are free while calls run on the driver pools");
    run.k.send(Cmd::KillTicket(1));
    run.served();
}

#[test]
fn the_spec_lane_has_its_own_resident_agent_running_beside_the_code_lane() {
    use zilch_ipc::Target;
    let (mut run, seen, mains) = main_run(worker());
    run.k.send(Cmd::OrderTo(Group::Spec, "draft a spec".into()));
    run.k.send(Cmd::Order("plan some code".into()));
    let (a, b) = (served_ticket(&mut run), served_ticket(&mut run));
    let mut served = [a.0, b.0];
    served.sort();
    assert_eq!(served, [1, 2], "both lanes served");
    let groups: Vec<(Pid, Group)> = run
        .seen
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Spawned { pid, group, .. } if matches!(group, Group::Spec | Group::Architecture) => Some((*pid, *group)),
            _ => None,
        })
        .collect();
    assert_eq!(groups.len(), 2, "one resident per lane: {groups:?}");
    assert_ne!(groups[0].0, groups[1].0);
    assert!(run.seen.iter().any(|e| matches!(e, KernelEvent::TicketOpened { ticket: 1, lane: Group::Spec, .. })));
    assert_eq!(mains.load(Ordering::SeqCst), 1, "the code lane's main agent is not the spec agent");
    // @spec while the spec agent is idle starts the spec lane's next order.
    let spec_pid = groups.iter().find(|(_, g)| *g == Group::Spec).unwrap().0;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !run.k.ps().iter().any(|p| p.pid == spec_pid && p.state == "blocked(recv)") {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(20));
    }
    let how = run.k.tell(Target::Group(Group::Spec), "add a non-goal").unwrap();
    assert!(how.contains("order #3"), "{how}");
    let (t, _, s) = served_ticket(&mut run);
    assert_eq!(t, 3);
    assert!(s.contains("Order #3: add a non-goal"), "{s}");
    assert!(seen.lock().unwrap().iter().any(|x| x == "spec: Order #1: draft a spec"));
}

#[test]
fn with_approval_ask_commands_wait_for_the_guest() {
    let drivers = Arc::new(FakeDrivers::default());
    let results: Arc<Mutex<Vec<String>>> = Arc::default();
    let log = results.clone();
    let dev: DevScript = Arc::new(move |_| {
        let log = log.clone();
        Box::new(Scripted(0, move |n, wake, _| {
            if let Some(r) = wake.first() {
                log.lock().unwrap().push(r.text());
            }
            if n <= 3 {
                StepResult::Syscalls(vec![tool("exec", json!({"command": format!("run {n}")}))])
            } else {
                StepResult::Exit(ExitStatus::ok("STATUS: done"))
            }
        }))
    });
    let mut run = Run::new(KernelConfig { ask_commands: true, ..cfg() }, drivers, Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let ask = |run: &mut Run| match run.wait(|e| matches!(e, KernelEvent::Question { command: true, .. })) {
        KernelEvent::Question { qid, text, .. } => (qid, text),
        _ => unreachable!(),
    };
    let (q1, text) = ask(&mut run);
    assert!(text.contains("wants to run: exec"), "{text}");
    run.k.send(Cmd::Answer { qid: q1, text: "".into() }); // a stray Enter: still waiting
    run.k.send(Cmd::Answer { qid: q1, text: "n".into() });
    let (q2, _) = ask(&mut run);
    run.k.send(Cmd::Answer { qid: q2, text: "a".into() });
    let (ok, _) = run.served();
    assert!(ok);
    let questions = run.seen.iter().filter(|e| matches!(e, KernelEvent::Question { command: true, .. })).count();
    assert_eq!(questions, 2, "after `a` the third command ran without asking");
    let r = results.lock().unwrap();
    assert!(r[0].contains("did not allow"), "{r:?}");
    assert_eq!((r[1].as_str(), r[2].as_str()), ("ok", "ok"));
}

#[test]
fn explorers_answer_questions_in_parallel_and_report_together() {
    let drivers = Arc::new(FakeDrivers::default());
    struct Explorers;
    impl Recipes for Explorers {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Scripted(0, |n, wake, _| match n {
                1 => StepResult::Syscalls(vec![Syscall::Explore {
                    questions: vec!["where is X?".into(), "how does Y work?".into(), "  ".into()],
                }]),
                _ => StepResult::Exit(ExitStatus::ok(wake[0].text())),
            }))
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            unreachable!("no plan here")
        }
        fn explorer(&self, _: TicketId, q: &str) -> Option<Box<dyn Agent>> {
            let q = q.to_string();
            Some(Box::new(Scripted(0, move |n, _, _| match n {
                1 => StepResult::Syscalls(vec![tool("work", json!({}))]),
                _ => StepResult::Exit(ExitStatus::ok(format!("answer to {q}"))),
            })))
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), drivers.clone(), Arc::new(Explorers), bus);
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("investigate".into()));
    let (ok, summary) = run.served();
    assert!(ok, "{summary}");
    assert!(summary.starts_with("All explorers reported (level 1 of at most 3; 14 of 16 explorers left"), "{summary}");
    assert!(
        summary.contains("### where is X?\nanswer to where is X?") && summary.contains("### how does Y work?\nanswer to how does Y work?"),
        "{summary}"
    );
    let spans = drivers.spans.lock().unwrap().clone();
    assert_eq!(spans.len(), 2, "one explorer per non-empty question");
    assert!(spans[0].1 < spans[1].2 && spans[1].1 < spans[0].2, "the explorers ran at the same time");
}

#[test]
fn exploration_goes_in_bounded_levels_per_order() {
    // An asker that explores 7 questions per level until it is refused, then serves what it saw.
    struct Asker(Vec<String>);
    impl Agent for Asker {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            let last = wake.first().map(SyscallResult::text).unwrap_or_default();
            if last.contains("no more exploration") {
                self.0.push(last);
                return StepResult::Syscalls(vec![Syscall::Serve { summary: self.0.join("\n---\n") }]);
            }
            if !last.is_empty() {
                self.0.push(last.lines().next().unwrap_or("").to_string());
            }
            StepResult::Syscalls(vec![Syscall::Explore { questions: (1..=7).map(|i| format!("q{i}")).collect() }])
        }
    }
    struct B(Arc<std::sync::atomic::AtomicU32>);
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Asker(vec![]))
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            unreachable!("no plan here")
        }
        fn explorer(&self, _: TicketId, q: &str) -> Option<Box<dyn Agent>> {
            self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            Some(Box::new(Scripted(0, {
                let q = q.to_string();
                move |_, _, _| StepResult::Exit(ExitStatus::ok(format!("answer to {q}")))
            })))
        }
    }
    let spawned = Arc::new(std::sync::atomic::AtomicU32::new(0));
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), Arc::new(FakeDrivers::default()), Arc::new(B(spawned.clone())), bus);
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("investigate".into()));
    let (ok, summary) = run.served();
    assert!(ok, "{summary}");
    // 7 + 7, then the 2 explorers left; the next level is refused.
    assert_eq!(spawned.load(std::sync::atomic::Ordering::SeqCst), 16);
    for want in ["level 1 of at most 3; 9 of 16", "level 2 of at most 3; 2 of 16", "level 3 of at most 3; 0 of 16"] {
        assert!(summary.contains(want), "{want}: {summary}");
    }
    assert!(summary.contains("no more exploration for this order (3 levels, 16 explorers used)"), "{summary}");
}

#[test]
fn a_resident_agents_budget_is_per_order_and_a_finished_plan_can_be_followed_by_another() {
    // A main agent that plans twice in one order (the second plan fixes what the first left), then
    // serves; it takes more steps over three orders than one order's budget allows.
    struct Twice;
    impl Agent for Twice {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            // With submit_plan and a tool call in one step, the plan's result comes first.
            let last = wake.first().map(SyscallResult::text).unwrap_or_default();
            if wake.is_empty() || last.contains("Order #") {
                return StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: vec![task("T1", &[], &["a"])] }, tool("x", json!({}))]);
            }
            if last.starts_with("All tasks finished") && !last.contains("second") {
                return StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: vec![task("second", &[], &["b"])] }]);
            }
            StepResult::Syscalls(vec![Syscall::Serve { summary: format!("served after: {}", last.lines().next().unwrap_or("")) }])
        }
    }
    struct B;
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Twice)
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            worker()(&task("w", &[], &[]))
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(KernelConfig { max_steps: 3, loop_limit: 0, ..cfg() }, Arc::new(FakeDrivers::default()), Arc::new(B), bus);
    let mut run = Run { k, events, seen: vec![] };
    for n in 1..=3 {
        run.k.send(Cmd::Order(format!("order {n}")));
        let (t, ok, s) = served_ticket(&mut run);
        assert_eq!(t, n);
        assert!(ok, "order {n}: {s}");
    }
    // Two plans per order, both approved and run; submit_plan shared a step with a tool call.
    assert_eq!(run.seen.iter().filter(|e| matches!(e, KernelEvent::Question { .. })).count(), 6);
    assert!(!run.seen.iter().any(|e| matches!(e, KernelEvent::Exited { pid: 1, .. })), "the main agent was never stopped");
}

#[test]
fn an_order_runs_at_most_two_follow_up_plans() {
    // A main agent that would plan again after every finished plan, forever.
    struct Polisher(u32);
    impl Agent for Polisher {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            let last = wake.first().map(SyscallResult::text).unwrap_or_default();
            if last.contains("no more plans") {
                return StepResult::Syscalls(vec![Syscall::Serve { summary: last }]);
            }
            self.0 += 1;
            StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: vec![task(&format!("T{}", self.0), &[], &["a"])] }])
        }
    }
    struct B;
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Polisher(0))
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            worker()(&task("w", &[], &[]))
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(KernelConfig { loop_limit: 0, ..cfg() }, Arc::new(FakeDrivers::default()), Arc::new(B), bus);
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("polish".into()));
    let (_, ok, summary) = served_ticket(&mut run);
    assert!(ok && summary.contains("2 follow-up plans already ran"), "{summary}");
    // The first plan and two follow-ups were approved and run; the fourth was refused.
    assert_eq!(run.seen.iter().filter(|e| matches!(e, KernelEvent::Question { .. })).count(), 3);
    // The next order starts with a fresh allowance.
    run.k.send(Cmd::Order("again".into()));
    let (_, ok, _) = served_ticket(&mut run);
    assert!(ok);
    assert_eq!(run.seen.iter().filter(|e| matches!(e, KernelEvent::Question { .. })).count(), 6);
}

#[test]
fn an_agent_out_of_budget_is_asked_to_wrap_up_and_keeps_its_work() {
    // It works with model calls until the supervisor's wrap-up note arrives, then reports.
    let dev: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, _, view: &AcbView| {
            if view.notes.iter().any(|x| x.contains("used its budget")) {
                return StepResult::Exit(ExitStatus::ok("did parts 1-3; part 4 is left\nSTATUS: failed"));
            }
            match n % 2 {
                1 => StepResult::Syscalls(vec![llm_call()]),
                _ => StepResult::Syscalls(vec![tool("x", json!({"n": n}))]),
            }
        }))
    });
    let mut run = Run::new(
        KernelConfig { max_steps: 5, loop_limit: 0, ..cfg() },
        Arc::new(FakeDrivers::default()),
        Book { tasks: vec![task("T1", &[], &["a"])], dev },
    );
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(summary.contains("did parts 1-3; part 4 is left"), "the report is kept for the main agent: {summary}");
    assert!(!summary.contains("budget exhausted"), "it was not killed: {summary}");
    assert!(!run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { signal: Signal::Kill, .. })));
}

#[test]
fn a_task_still_making_progress_gets_more_budget_one_going_in_circles_does_not() {
    // Progressing: every tool call reads something new (its result changes), for 12 model calls,
    // then it reports. Budget 5: extended at 5 (+2) and again at 7, 9, 11 … up to 3× = 15.
    let progressing: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, _, _| match (n, n % 2) {
            (25.., _) => StepResult::Exit(ExitStatus::ok("all done\nSTATUS: done")),
            (_, 1) => StepResult::Syscalls(vec![llm_call()]),
            _ => StepResult::Syscalls(vec![tool("x", json!({"n": n}))]),
        }))
    });
    struct Fresh;
    impl Drivers for Fresh {
        fn classify(&self, _: &str, _: &Value) -> Access {
            Access::Read
        }
        fn run(&self, _: &DriverCtx, call: Syscall) -> SyscallResult {
            match call {
                // Each read finds something different: new information, i.e. progress.
                Syscall::Tool { args, .. } => SyscallResult::Output(format!("content {}", args["n"])),
                _ => SyscallResult::Completion { message: zilch_ipc::Message::assistant("hmm"), usage: Default::default(), cut: false },
            }
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(
        KernelConfig { max_steps: 5, loop_limit: 0, ..cfg() },
        Arc::new(Fresh),
        Arc::new(Book { tasks: vec![task("T1", &[], &["a"])], dev: progressing }),
        bus,
    );
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(summary.contains("all done") && summary.contains("Done"), "a productive task is not cut off: {summary}");
    let extended =
        run.seen.iter().filter(|e| matches!(e, KernelEvent::Signal { reason, .. } if reason.contains("budget extended"))).count();
    assert!(extended >= 3, "extended several times: {extended}");

    // In circles: the same read with the same result every time — no progress, so no extension.
    let circles: DevScript = Arc::new(|_| {
        Box::new(Scripted(0, |n, _, _| match n % 2 {
            1 => StepResult::Syscalls(vec![llm_call()]),
            _ => StepResult::Syscalls(vec![tool("same", json!({}))]),
        }))
    });
    let mut run = Run::new(
        KernelConfig { max_steps: 5, loop_limit: 0, ..cfg() },
        Arc::new(FakeDrivers::default()),
        Book { tasks: vec![task("T1", &[], &["a"])], dev: circles },
    );
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(!run.seen.iter().any(|e| matches!(e, KernelEvent::Signal { reason, .. } if reason.contains("budget extended"))), "{summary}");
    assert!(summary.contains("model-call budget exhausted (10 calls"), "wrap-up note, then stopped after the grace: {summary}");
}

#[test]
fn the_main_agent_dispatches_a_planner_whose_plan_runs_and_whose_report_comes_back() {
    // Main: dispatch the order to a planner, then serve with what it reported.
    struct Main;
    impl Agent for Main {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            let last = wake.first().map(SyscallResult::text).unwrap_or_default();
            if let Some(report) = last.strip_prefix("Report from the plan subagent:\n") {
                return StepResult::Syscalls(vec![Syscall::Serve { summary: format!("main: {report}") }]);
            }
            if last.contains("did not finish") || last.starts_with("no subagent") {
                return StepResult::Syscalls(vec![Syscall::Serve { summary: last }]);
            }
            StepResult::Syscalls(vec![Syscall::Dispatch { agent: "plan".into(), brief: "build a and b".into() }])
        }
    }
    // Planner: explores, plans, and reports when the plan has run.
    struct Planner(u8);
    impl Agent for Planner {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            self.0 += 1;
            let last = wake.first().map(SyscallResult::text).unwrap_or_default();
            match self.0 {
                1 => StepResult::Syscalls(vec![Syscall::Explore { questions: vec!["where?".into()] }]),
                2 => {
                    assert!(last.starts_with("All explorers reported"), "{last}");
                    StepResult::Syscalls(vec![Syscall::SubmitPlan { tasks: vec![task("T1", &[], &["a"]), task("T2", &[], &["b"])] }])
                }
                _ => {
                    assert!(last.starts_with("All tasks finished"), "{last}");
                    StepResult::Exit(ExitStatus::ok("a and b are built"))
                }
            }
        }
    }
    struct B;
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Main)
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            worker()(&task("w", &[], &[]))
        }
        fn explorer(&self, _: TicketId, _: &str) -> Option<Box<dyn Agent>> {
            Some(Box::new(Scripted(0, |_, _, _| StepResult::Exit(ExitStatus::ok("over there")))))
        }
        fn subagent(&self, _: TicketId, agent: &str, brief: &str) -> Option<Box<dyn Agent>> {
            assert_eq!(brief, "build a and b");
            (agent == "plan").then(|| Box::new(Planner(0)) as Box<dyn Agent>)
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), Arc::new(FakeDrivers::default()), Arc::new(B), bus);
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("add a and b".into()));
    let (ok, summary) = run.served();
    assert!(ok && summary == "main: a and b are built", "{summary}");
    // The developers were the planner's children, and the planner the main agent's.
    let parent = |p: Pid| {
        run.seen.iter().find_map(|e| match e {
            KernelEvent::Spawned { pid, parent, .. } if *pid == p => Some(*parent),
            _ => None,
        })
    };
    let devs: Vec<Pid> = run
        .seen
        .iter()
        .filter_map(|e| match e {
            KernelEvent::Spawned { pid, title, .. } if title.starts_with("[T") => Some(*pid),
            _ => None,
        })
        .collect();
    assert_eq!(devs.len(), 2);
    let planner = parent(devs[0]).flatten().expect("a parent");
    assert_eq!(parent(planner), Some(Some(1)));
}

#[test]
fn only_the_main_agent_dispatches_and_only_known_subagents() {
    struct Main;
    impl Agent for Main {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            match wake.first().map(SyscallResult::text) {
                None => StepResult::Syscalls(vec![Syscall::Dispatch { agent: "chef".into(), brief: "cook".into() }]),
                Some(t) if t.contains("Order #") => {
                    StepResult::Syscalls(vec![Syscall::Dispatch { agent: "chef".into(), brief: "cook".into() }])
                }
                Some(t) => StepResult::Syscalls(vec![Syscall::Serve { summary: t }]),
            }
        }
    }
    struct B;
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Main)
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            worker()(&task("w", &[], &[]))
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), Arc::new(FakeDrivers::default()), Arc::new(B), bus);
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(summary.contains("no subagent 'chef'"), "{summary}");
}

#[test]
fn a_repeated_dispatch_in_one_step_runs_once() {
    // The model asks twice in one step: the first dispatch runs, the second is refused.
    struct Main;
    impl Agent for Main {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            match wake.first().map(SyscallResult::text) {
                Some(t) if t.starts_with("Report from") => {
                    assert!(wake[1].text().contains("the first one in this step runs"), "{}", wake[1].text());
                    StepResult::Syscalls(vec![Syscall::Serve { summary: t }])
                }
                _ => {
                    let d = || Syscall::Dispatch { agent: "answer".into(), brief: "why?".into() };
                    StepResult::Syscalls(vec![d(), d()])
                }
            }
        }
    }
    struct B;
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Main)
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            worker()(&task("w", &[], &[]))
        }
        fn subagent(&self, _: TicketId, _: &str, _: &str) -> Option<Box<dyn Agent>> {
            Some(Box::new(Scripted(0, |_, _, _| StepResult::Exit(ExitStatus::ok("because")))))
        }
    }
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(cfg(), Arc::new(FakeDrivers::default()), Arc::new(B), bus);
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("why?".into()));
    let (ok, summary) = run.served();
    assert!(ok && summary == "Report from the answer subagent:\nbecause", "{summary}");
    let spawned = run.seen.iter().filter(|e| matches!(e, KernelEvent::Spawned { title, .. } if title.starts_with("answer"))).count();
    assert_eq!(spawned, 1);
}

/// An idle wake from the kernel (a served resident agent may compact; scripted ones just wait).
fn idle_wake(wake: &[SyscallResult]) -> bool {
    !wake.is_empty() && wake.iter().all(|r| matches!(r, SyscallResult::Message(m) if m.kind == zilch_ipc::MsgKind::Idle))
}

#[test]
fn a_served_main_agent_is_woken_when_idle_and_an_order_waits_for_its_compaction() {
    // Records the idle wakes it gets; on "cold" it compacts (a slow model call) before waiting again.
    struct Main(Arc<Mutex<Vec<String>>>);
    impl Agent for Main {
        fn step(&mut self, _: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
            if idle_wake(&wake) {
                let body = wake[0].text();
                self.0.lock().unwrap().push(body.clone());
                if body.contains("cold") {
                    return StepResult::Syscalls(vec![tool("slow", json!({}))]); // "compacting"
                }
                return StepResult::Syscalls(vec![Syscall::Recv]);
            }
            match wake.first().map(SyscallResult::text) {
                Some(t) if t == "done after 400ms" => StepResult::Syscalls(vec![Syscall::Recv]), // compaction finished
                Some(t) => StepResult::Syscalls(vec![Syscall::Serve { summary: format!("answered: {t}") }]),
                None => StepResult::Syscalls(vec![Syscall::Serve { summary: "answered: first".into() }]),
            }
        }
    }
    struct B(Arc<Mutex<Vec<String>>>);
    impl Recipes for B {
        fn architect(&self, _: TicketId, _: &str) -> Box<dyn Agent> {
            Box::new(Main(self.0.clone()))
        }
        fn developer(&self, _: TicketId, _: &PlanTask, _: &str) -> Box<dyn Agent> {
            worker()(&task("w", &[], &[]))
        }
    }
    let wakes: Arc<Mutex<Vec<String>>> = Arc::default();
    let bus = Arc::new(EventBus::new());
    let events = bus.subscribe(10_000);
    let k = start(
        KernelConfig { idle_cold: Duration::from_millis(300), ..cfg() },
        Arc::new(FakeDrivers::default()),
        Arc::new(B(wakes.clone())),
        bus,
    );
    let mut run = Run { k, events, seen: vec![] };
    run.k.send(Cmd::Order("hi".into()));
    assert_eq!(served_ticket(&mut run).0, 1);
    // "now" right away, "cold" after idle_cold — once each.
    let deadline = Instant::now() + Duration::from_secs(5);
    while wakes.lock().unwrap().len() < 2 {
        assert!(Instant::now() < deadline, "{:?}", wakes.lock().unwrap());
        std::thread::sleep(Duration::from_millis(20));
    }
    // An order sent while it compacts becomes the next order and is delivered once it waits again.
    let how = run.k.tell(zilch_ipc::Target::Group(Group::Architecture), "next please").unwrap();
    assert!(how.contains("order #2"), "{how}");
    let (t, ok, s) = served_ticket(&mut run);
    assert!(t == 2 && ok && s.contains("next please"), "{s}");
    std::thread::sleep(Duration::from_millis(400));
    let w = wakes.lock().unwrap().clone();
    // Each served order gets one "now" and, idle past idle_cold, one "cold".
    assert_eq!(w.iter().filter(|b| b.ends_with("now")).count(), 2, "{w:?}");
    assert_eq!(w.iter().filter(|b| b.ends_with("cold")).count(), 2, "{w:?}");
}
