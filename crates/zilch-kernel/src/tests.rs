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
        let Syscall::Tool { name, .. } = call else { return SyscallResult::Failed("no llm in this test".into()) };
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
    // The architect also tries a command and a write outside docs/ before planning.
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
                    assert_eq!(r[2], "ok");
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
            (false, 1) => StepResult::Syscalls(vec![tool("work", json!({}))]),
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
fn budgets_stop_a_process() {
    let drivers = Arc::new(FakeDrivers::default());
    let dev: DevScript = Arc::new(|_| Box::new(Scripted(0, |n, _, _| StepResult::Syscalls(vec![tool("x", json!({"n": n}))]))));
    let mut run = Run::new(KernelConfig { max_steps: 5, ..cfg() }, drivers, Book { tasks: vec![task("T1", &[], &["a"])], dev });
    run.k.send(Cmd::Order("x".into()));
    let (_, summary) = run.served();
    assert!(summary.contains("step budget exhausted (5 steps)") && summary.contains("Failed"), "{summary}");
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
