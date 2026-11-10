//! The zilch kernel: the kitchen.
//!
//! One kernel thread owns every process (agent) and decides everything; step workers (burners)
//! run one agent step at a time and hand it back; driver pools do the physical work (LLM calls,
//! files, commands). A timer tick, driven by `recv_timeout`, runs the supervisor (hung agents,
//! budgets) and ages waiting work. Nothing blocks the kernel thread: it only moves processes
//! between queues and reacts to completions.
//!
//! ```text
//!  CLI ──Cmd──▶ kernel thread ──job──▶ step workers ──StepResult──▶ kernel
//!                   │  ▲                                              │
//!                   │  └──────── SyscallDone ◀── driver pools ◀──syscall
//!                   └── events ──▶ EventBus ──▶ CLI, JSONL log
//! ```

pub mod cancel;
pub mod plan;
pub mod supervisor;

pub use cancel::CancelToken;

use plan::TaskStatus;
use serde_json::Value;
use std::collections::{BTreeMap, VecDeque};
use std::sync::mpsc::{channel, Receiver, RecvTimeoutError, Sender};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};
use supervisor::{nudge_message, CallRecord, Supervisor, Verdict};
use zilch_ipc::{
    EventBus, ExitStatus, Group, KernelEvent, Mailbox, Msg, MsgKind, Pid, PlanTask, Signal, Syscall, SyscallResult, TicketId, CLI_PID,
};

// ───────────────────────────── contracts ─────────────────────────────

/// What a process sees of itself when it steps.
#[derive(Debug, Clone)]
pub struct AcbView {
    pub pid: Pid,
    pub ticket: TicketId,
    pub group: Group,
    pub steps: u32,
    /// Messages from the supervisor (a nudge) to put in front of the model on this step.
    pub notes: Vec<String>,
}

pub enum StepResult {
    /// Ask the kernel to do things; they run in parallel and the next step gets all results.
    Syscalls(Vec<Syscall>),
    /// Give the worker back; run again later with no new input.
    Yield,
    Exit(ExitStatus),
}

/// A process's program: a resumable state machine. `step` must not do I/O; everything outside
/// the process goes through syscalls.
pub trait Agent: Send {
    fn step(&mut self, view: &AcbView, wake: Vec<SyscallResult>) -> StepResult;
}

/// What a tool call would touch, for the capability check.
#[derive(Debug, Clone, PartialEq)]
pub enum Access {
    Read,
    /// Writes this workspace-relative path.
    Write(String),
    /// Runs a command.
    Exec,
    /// The call is invalid (unknown tool, bad path); the reason goes back to the model.
    Invalid(String),
}

/// Which driver pool runs a syscall.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolClass {
    /// Network-bound (LLM calls) and quick file operations.
    Io,
    /// Commands: their own pool, so a long build never takes a step worker or an LLM slot.
    Exec,
}

/// Per-call context handed to a driver.
pub struct DriverCtx<'a> {
    pub pid: Pid,
    pub ticket: TicketId,
    pub cancel: &'a CancelToken,
    pub bus: &'a EventBus,
}

/// The equipment: everything with side effects. Called from driver pool threads.
pub trait Drivers: Send + Sync + 'static {
    fn classify(&self, name: &str, args: &Value) -> Access;
    fn class(&self, call: &Syscall) -> PoolClass {
        match call {
            Syscall::Tool { name, args, .. } if self.classify(name, args) == Access::Exec => PoolClass::Exec,
            _ => PoolClass::Io,
        }
    }
    /// Run a `Complete` or `Tool` syscall to completion (blocking; honour `ctx.cancel`).
    fn run(&self, ctx: &DriverCtx, call: Syscall) -> SyscallResult;
    /// Persist an approved plan (e.g. `docs/plan.json`).
    fn save_plan(&self, _ticket: TicketId, _tasks: &[PlanTask]) {}
    /// A process is gone: drop whatever the drivers kept for it.
    fn release(&self, _pid: Pid) {}
}

/// The recipe book: builds the programs the kernel runs.
pub trait Recipes: Send + Sync + 'static {
    fn architect(&self, ticket: TicketId, order: &str) -> Box<dyn Agent>;
    /// `context`: what the Developer should know about the whole plan.
    fn developer(&self, ticket: TicketId, task: &PlanTask, context: &str) -> Box<dyn Agent>;
}

#[derive(Debug, Clone)]
pub struct KernelConfig {
    /// Step workers (burners). Steps are cheap; this bounds CPU work, not LLM calls.
    pub workers: usize,
    /// Threads for LLM calls and file operations (LLM concurrency is also capped per provider).
    pub io_threads: usize,
    pub exec_threads: usize,
    pub tick: Duration,
    /// Ticks without any progress before a running or I/O-blocked process is interrupted, then killed.
    pub hang_ticks: u64,
    /// Supervisor loop limit (see `supervisor`); 0 = off.
    pub loop_limit: usize,
    /// Per-process step budget.
    pub max_steps: u32,
    /// Per-process token budget (None = unlimited).
    pub max_tokens: Option<u64>,
    /// Approve plans without asking (tests, --yes).
    pub auto_approve: bool,
    /// Ticks a process may wait in a lower queue before it moves up one level.
    pub age_ticks: u64,
}

impl Default for KernelConfig {
    fn default() -> Self {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
        Self {
            workers: cores,
            io_threads: 32,
            exec_threads: 4,
            tick: Duration::from_millis(100),
            hang_ticks: 3000, // 5 minutes at 100 ms
            loop_limit: supervisor::DEFAULT_LOOP_LIMIT,
            max_steps: 80,
            max_tokens: None,
            auto_approve: false,
            age_ticks: 50,
        }
    }
}

/// Commands from the CLI.
pub enum Cmd {
    Order(String),
    /// Answer a `Question` (plan approval: "" / "y" approve, "n" / "r" reject, else feedback).
    Answer {
        qid: u64,
        text: String,
    },
    Kill(Pid),
    KillTicket(TicketId),
    Ps(Sender<Vec<ProcInfo>>),
    Shutdown,
}

/// One row of the process table.
#[derive(Debug, Clone)]
pub struct ProcInfo {
    pub pid: Pid,
    pub parent: Option<Pid>,
    pub ticket: TicketId,
    pub group: Group,
    pub title: String,
    pub state: String,
    pub doing: String,
    pub steps: u32,
    pub tokens: u64,
    pub age_ticks: u64,
}

// ───────────────────────────── process table ─────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Block {
    /// Syscalls in flight.
    Io,
    /// Waiting for a message.
    Recv,
    /// Waiting for the guest (plan approval).
    Question,
    /// Waiting for its children (the plan's tasks).
    Child,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Ready,
    Running,
    Blocked(Block),
    Zombie,
    Killed,
}

impl State {
    fn label(self) -> String {
        match self {
            State::Blocked(b) => format!("blocked({})", format!("{b:?}").to_lowercase()),
            s => format!("{s:?}").to_lowercase(),
        }
    }
    fn alive(self) -> bool {
        !matches!(self, State::Zombie | State::Killed)
    }
}

/// Write permissions of a process.
#[derive(Debug, Clone)]
struct Caps {
    /// Globs this process may write; empty with `write_any` false = nothing.
    write: Vec<String>,
    write_any: bool,
    exec: bool,
}

impl Caps {
    fn allows(&self, access: &Access) -> Result<(), String> {
        match access {
            Access::Read => Ok(()),
            Access::Invalid(e) => Err(e.clone()),
            Access::Exec if self.exec => Ok(()),
            Access::Exec => Err("permission denied: this agent may not run commands".into()),
            Access::Write(path) => {
                let p = path.trim_start_matches("./");
                let ok = self.write_any
                    || self.write.iter().any(|g| g == p || globset::Glob::new(g).map(|g| g.compile_matcher().is_match(p)).unwrap_or(false));
                if ok {
                    Ok(())
                } else {
                    Err(format!(
                        "permission denied: '{path}' is outside this agent's writable paths ({})",
                        if self.write.is_empty() { "none".into() } else { self.write.join(", ") }
                    ))
                }
            }
        }
    }
}

/// One call of the batch in flight, kept for the supervisor and the process table.
struct InFlight {
    summary: String,
    class: PoolClass,
    /// For tool calls: (name, raw arguments) for loop fingerprints.
    tool: Option<(String, String)>,
    completion: bool,
}

/// Agent Control Block: everything the kernel knows about one process.
struct Acb {
    pid: Pid,
    parent: Option<Pid>,
    ticket: TicketId,
    group: Group,
    title: String,
    state: State,
    priority: u8,
    /// The program; `None` while a worker runs its step.
    agent: Option<Box<dyn Agent>>,
    caps: Caps,
    mailbox: Mailbox,
    cancel: CancelToken,
    /// Results to deliver on the next step.
    wake: Vec<SyscallResult>,
    notes: Vec<String>,
    // The batch of syscalls in flight.
    batch: u64,
    batch_cancel: CancelToken,
    inflight: Vec<InFlight>,
    pending: Vec<Option<SyscallResult>>,
    outstanding: usize,
    /// Slot index waiting on the mailbox / the guest / the children.
    parked_slot: Option<usize>,
    // Accounting.
    steps: u32,
    tokens: u64,
    born: u64,
    ready_since: u64,
    last_progress: u64,
    hang_strikes: u32,
    /// The batch in flight was cancelled as hung: finishing it is not progress.
    batch_interrupted: bool,
    supervisor: Supervisor,
    last_text: String,
}

struct TaskRun {
    status: TaskStatus,
    pid: Option<Pid>,
    report: String,
}

struct PlanRun {
    tasks: Vec<PlanTask>,
    runs: Vec<TaskRun>,
    approved: bool,
}

struct Ticket {
    root: Pid,
    cancel: CancelToken,
    plan: Option<PlanRun>,
    served: bool,
}

enum KEvent {
    WorkerDone { pid: Pid, agent: Option<Box<dyn Agent>>, result: Result<StepResult, String> },
    SyscallDone { pid: Pid, batch: u64, idx: usize, result: SyscallResult },
    Cmd(Cmd),
}

type Job = Box<dyn FnOnce() + Send>;

/// A fixed set of threads taking jobs from one queue.
struct Pool {
    tx: Option<Sender<Job>>,
    threads: Vec<JoinHandle<()>>,
}

impl Pool {
    fn new(name: &str, n: usize) -> Self {
        let (tx, rx) = channel::<Job>();
        let rx = Arc::new(Mutex::new(rx));
        let threads = (0..n.max(1))
            .map(|i| {
                let rx = rx.clone();
                std::thread::Builder::new()
                    .name(format!("{name}-{i}"))
                    .spawn(move || loop {
                        let job = { rx.lock().unwrap().recv() };
                        match job {
                            Ok(j) => j(),
                            Err(_) => break,
                        }
                    })
                    .expect("spawn pool thread")
            })
            .collect();
        Self { tx: Some(tx), threads }
    }
    fn run(&self, job: Job) {
        if let Some(tx) = &self.tx {
            let _ = tx.send(job);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.tx.take();
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

const QUEUES: usize = 4;
const MAILBOX_CAP: usize = 32;

pub struct Kernel {
    cfg: KernelConfig,
    bus: Arc<EventBus>,
    drivers: Arc<dyn Drivers>,
    recipes: Arc<dyn Recipes>,
    tx: Sender<KEvent>,
    rx: Receiver<KEvent>,
    tick: u64,
    started: Instant,
    next_pid: Pid,
    next_ticket: TicketId,
    next_qid: u64,
    acbs: BTreeMap<Pid, Acb>,
    queues: [VecDeque<Pid>; QUEUES],
    idle_workers: usize,
    tickets: BTreeMap<TicketId, Ticket>,
    /// Open questions: qid → ticket.
    questions: BTreeMap<u64, TicketId>,
    workers: Pool,
    io: Pool,
    exec: Pool,
}

/// The CLI's handle on a running kernel.
pub struct KernelHandle {
    tx: Sender<KEvent>,
    pub bus: Arc<EventBus>,
    thread: Option<JoinHandle<()>>,
}

impl KernelHandle {
    pub fn send(&self, cmd: Cmd) {
        let _ = self.tx.send(KEvent::Cmd(cmd));
    }
    pub fn ps(&self) -> Vec<ProcInfo> {
        let (tx, rx) = channel();
        self.send(Cmd::Ps(tx));
        rx.recv_timeout(Duration::from_secs(2)).unwrap_or_default()
    }
    /// Stop the kernel and wait for it (in-flight work is cancelled).
    pub fn shutdown(mut self) {
        self.send(Cmd::Shutdown);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

impl Drop for KernelHandle {
    fn drop(&mut self) {
        let _ = self.tx.send(KEvent::Cmd(Cmd::Shutdown));
    }
}

/// Start the kernel on its own thread.
pub fn start(cfg: KernelConfig, drivers: Arc<dyn Drivers>, recipes: Arc<dyn Recipes>, bus: Arc<EventBus>) -> KernelHandle {
    let mut k = Kernel::new(cfg, drivers, recipes, bus.clone());
    let tx = k.tx.clone();
    let thread = std::thread::Builder::new().name("zilch-kernel".into()).spawn(move || k.run()).expect("spawn kernel");
    KernelHandle { tx, bus, thread: Some(thread) }
}

impl Kernel {
    fn new(cfg: KernelConfig, drivers: Arc<dyn Drivers>, recipes: Arc<dyn Recipes>, bus: Arc<EventBus>) -> Self {
        let (tx, rx) = channel();
        Self {
            workers: Pool::new("zilch-step", cfg.workers),
            io: Pool::new("zilch-io", cfg.io_threads),
            exec: Pool::new("zilch-exec", cfg.exec_threads),
            idle_workers: cfg.workers.max(1),
            cfg,
            bus,
            drivers,
            recipes,
            tx,
            rx,
            tick: 0,
            started: Instant::now(),
            next_pid: 1,
            next_ticket: 1,
            next_qid: 1,
            acbs: BTreeMap::new(),
            queues: Default::default(),
            tickets: BTreeMap::new(),
            questions: BTreeMap::new(),
        }
    }

    fn run(&mut self) {
        loop {
            match self.rx.recv_timeout(self.cfg.tick) {
                Ok(KEvent::Cmd(Cmd::Shutdown)) => break,
                Ok(KEvent::Cmd(c)) => self.command(c),
                Ok(KEvent::WorkerDone { pid, agent, result }) => self.step_done(pid, agent, result),
                Ok(KEvent::SyscallDone { pid, batch, idx, result }) => self.syscall_done(pid, batch, idx, result),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => break,
            }
            let now = (self.started.elapsed().as_millis() / self.cfg.tick.as_millis().max(1)) as u64;
            if now > self.tick {
                self.tick = now;
                self.bus.set_tick(now);
                self.on_tick();
            }
            self.dispatch();
        }
        // Shutting down: stop everything in flight.
        for t in self.tickets.values() {
            t.cancel.cancel();
        }
    }

    // ── commands ──

    fn command(&mut self, c: Cmd) {
        match c {
            Cmd::Order(text) => self.open_ticket(text),
            Cmd::Answer { qid, text } => self.answer(qid, &text),
            Cmd::Kill(pid) => self.kill(pid, "killed by the user"),
            Cmd::KillTicket(t) => {
                if let Some(root) = self.tickets.get(&t).map(|t| t.root) {
                    self.kill(root, "order cancelled by the user");
                }
            }
            Cmd::Ps(reply) => {
                let rows = self
                    .acbs
                    .values()
                    .map(|a| ProcInfo {
                        pid: a.pid,
                        parent: a.parent,
                        ticket: a.ticket,
                        group: a.group,
                        title: a.title.clone(),
                        state: a.state.label(),
                        doing: a.inflight.iter().map(|f| f.summary.clone()).collect::<Vec<_>>().join(", "),
                        steps: a.steps,
                        tokens: a.tokens,
                        age_ticks: self.tick.saturating_sub(a.born),
                    })
                    .collect();
                let _ = reply.send(rows);
            }
            Cmd::Shutdown => {}
        }
    }

    fn open_ticket(&mut self, text: String) {
        let ticket = self.next_ticket;
        self.next_ticket += 1;
        let cancel = CancelToken::new();
        self.bus.publish(KernelEvent::TicketOpened { ticket, text: text.clone() });
        let agent = self.recipes.architect(ticket, &text);
        let caps = Caps { write: vec!["docs/**".into()], write_any: false, exec: false };
        let title: String = text.lines().next().unwrap_or("").chars().take(60).collect();
        let root = self.spawn(agent, None, ticket, Group::Architecture, title, caps, cancel.child(), 0);
        self.tickets.insert(ticket, Ticket { root, cancel, plan: None, served: false });
    }

    #[allow(clippy::too_many_arguments)]
    fn spawn(
        &mut self,
        agent: Box<dyn Agent>,
        parent: Option<Pid>,
        ticket: TicketId,
        group: Group,
        title: String,
        caps: Caps,
        cancel: CancelToken,
        priority: u8,
    ) -> Pid {
        let pid = self.next_pid;
        self.next_pid += 1;
        self.acbs.insert(
            pid,
            Acb {
                pid,
                parent,
                ticket,
                group,
                title: title.clone(),
                state: State::Ready,
                priority: priority.min(QUEUES as u8 - 1),
                agent: Some(agent),
                caps,
                mailbox: Mailbox::new(MAILBOX_CAP),
                batch_cancel: cancel.child(),
                cancel,
                wake: vec![],
                notes: vec![],
                batch: 0,
                inflight: vec![],
                pending: vec![],
                outstanding: 0,
                parked_slot: None,
                steps: 0,
                tokens: 0,
                born: self.tick,
                ready_since: self.tick,
                last_progress: self.tick,
                hang_strikes: 0,
                batch_interrupted: false,
                supervisor: Supervisor::new(self.cfg.loop_limit),
                last_text: String::new(),
            },
        );
        self.bus.publish(KernelEvent::Spawned { pid, parent, ticket, group, title });
        self.enqueue(pid);
        pid
    }

    // ── scheduling ──

    fn enqueue(&mut self, pid: Pid) {
        let tick = self.tick;
        if let Some(a) = self.acbs.get_mut(&pid) {
            a.state = State::Ready;
            a.ready_since = tick;
            self.queues[a.priority as usize].push_back(pid);
        }
    }

    fn set_state(&mut self, pid: Pid, state: State) {
        if let Some(a) = self.acbs.get_mut(&pid) {
            if a.state != state {
                a.state = state;
                self.bus.publish(KernelEvent::State { pid, state: state.label() });
            }
        }
    }

    /// Highest-priority Ready process; FIFO (round-robin) within a level.
    fn pick_next(&mut self) -> Option<Pid> {
        for q in self.queues.iter_mut() {
            while let Some(pid) = q.pop_front() {
                if self.acbs.get(&pid).is_some_and(|a| a.state == State::Ready && a.agent.is_some()) {
                    return Some(pid);
                }
            }
        }
        None
    }

    fn dispatch(&mut self) {
        while self.idle_workers > 0 {
            let Some(pid) = self.pick_next() else { break };
            if !self.within_budget(pid) {
                continue;
            }
            let a = self.acbs.get_mut(&pid).expect("picked");
            let mut agent = a.agent.take().expect("ready agents are parked");
            let wake = std::mem::take(&mut a.wake);
            let view = AcbView { pid, ticket: a.ticket, group: a.group, steps: a.steps, notes: std::mem::take(&mut a.notes) };
            self.idle_workers -= 1;
            self.set_state(pid, State::Running);
            let tx = self.tx.clone();
            self.workers.run(Box::new(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| agent.step(&view, wake)));
                let (agent, result) = match result {
                    Ok(r) => (Some(agent), Ok(r)),
                    Err(p) => {
                        let msg = p
                            .downcast_ref::<&str>()
                            .map(|s| s.to_string())
                            .or_else(|| p.downcast_ref::<String>().cloned())
                            .unwrap_or_default();
                        (None, Err(format!("crashed: {msg}")))
                    }
                };
                let _ = tx.send(KEvent::WorkerDone { pid, agent, result });
            }));
        }
    }

    fn within_budget(&mut self, pid: Pid) -> bool {
        let Some(a) = self.acbs.get(&pid) else { return false };
        let reason = if a.steps >= self.cfg.max_steps {
            Some(format!("step budget exhausted ({} steps)", self.cfg.max_steps))
        } else {
            self.cfg.max_tokens.filter(|m| a.tokens >= *m).map(|m| format!("token budget exhausted ({} of {m} tokens)", a.tokens))
        };
        match reason {
            Some(r) => {
                self.finish(pid, ExitStatus::failed(format!("[stopped: {r}]\nSTATUS: failed")), State::Killed);
                false
            }
            None => true,
        }
    }

    fn step_done(&mut self, pid: Pid, agent: Option<Box<dyn Agent>>, result: Result<StepResult, String>) {
        self.idle_workers += 1;
        let tick = self.tick;
        let Some(a) = self.acbs.get_mut(&pid) else { return };
        if !a.state.alive() {
            return; // killed while its step ran
        }
        a.steps += 1;
        a.last_progress = tick;
        a.agent = agent;
        match result {
            Err(e) => self.finish(pid, ExitStatus::failed(format!("[{e}]\nSTATUS: failed")), State::Killed),
            Ok(StepResult::Yield) => self.enqueue(pid),
            Ok(StepResult::Exit(st)) => self.finish(pid, st, State::Zombie),
            Ok(StepResult::Syscalls(calls)) if calls.is_empty() => self.enqueue(pid),
            Ok(StepResult::Syscalls(calls)) => self.issue(pid, calls),
        }
    }

    // ── syscalls ──

    fn issue(&mut self, pid: Pid, calls: Vec<Syscall>) {
        let n = calls.len();
        let (batch, cancel, ticket) = {
            let a = self.acbs.get_mut(&pid).expect("alive");
            a.batch += 1;
            a.batch_cancel = a.cancel.child();
            a.batch_interrupted = false;
            a.pending = vec![None; n];
            a.outstanding = n;
            a.parked_slot = None;
            a.inflight = calls
                .iter()
                .map(|c| InFlight {
                    summary: c.summary(),
                    class: PoolClass::Io,
                    tool: match c {
                        Syscall::Tool { name, args, .. } => Some((name.clone(), args.to_string())),
                        _ => None,
                    },
                    completion: matches!(c, Syscall::Complete(_)),
                })
                .collect();
            (a.batch, a.batch_cancel.clone(), a.ticket)
        };
        self.set_state(pid, State::Blocked(Block::Io));
        let exclusive = n > 1 && calls.iter().any(|c| matches!(c, Syscall::Recv | Syscall::SubmitPlan { .. }));
        for (idx, call) in calls.into_iter().enumerate() {
            self.bus.publish(KernelEvent::SyscallIssued { pid, idx, summary: call.summary() });
            match call {
                Syscall::Recv | Syscall::SubmitPlan { .. } if exclusive => {
                    self.fill(pid, batch, idx, SyscallResult::Failed("recv and submit_plan must be the only call in their step".into()));
                }
                Syscall::Send { to, kind, body } => {
                    let r = self.send_msg(pid, to, kind, body);
                    self.fill(pid, batch, idx, r);
                }
                Syscall::Recv => {
                    let msg = self.acbs.get_mut(&pid).and_then(|a| a.mailbox.recv());
                    match msg {
                        Some(m) => self.fill(pid, batch, idx, SyscallResult::Message(m)),
                        None => {
                            if let Some(a) = self.acbs.get_mut(&pid) {
                                a.parked_slot = Some(idx);
                            }
                            self.set_state(pid, State::Blocked(Block::Recv));
                        }
                    }
                }
                Syscall::SubmitPlan { tasks } => self.submit_plan(pid, batch, idx, tasks),
                call @ (Syscall::Complete(_) | Syscall::Tool { .. }) => {
                    if let Syscall::Tool { name, args, .. } = &call {
                        let access = self.drivers.classify(name, args);
                        let allowed = self.acbs.get(&pid).map(|a| a.caps.allows(&access)).unwrap_or(Ok(()));
                        if let Err(e) = allowed {
                            self.fill(pid, batch, idx, SyscallResult::Failed(e));
                            continue;
                        }
                    }
                    let class = self.drivers.class(&call);
                    if let Some(f) = self.acbs.get_mut(&pid).and_then(|a| a.inflight.get_mut(idx)) {
                        f.class = class;
                    }
                    let (drivers, bus, tx, cancel) = (self.drivers.clone(), self.bus.clone(), self.tx.clone(), cancel.clone());
                    let job: Job = Box::new(move || {
                        let result = if cancel.is_cancelled() {
                            SyscallResult::Failed("cancelled".into())
                        } else {
                            let ctx = DriverCtx { pid, ticket, cancel: &cancel, bus: &bus };
                            drivers.run(&ctx, call)
                        };
                        let _ = tx.send(KEvent::SyscallDone { pid, batch, idx, result });
                    });
                    match class {
                        PoolClass::Io => self.io.run(job),
                        PoolClass::Exec => self.exec.run(job),
                    }
                }
            }
        }
    }

    fn send_msg(&mut self, from: Pid, to: Pid, kind: MsgKind, body: String) -> SyscallResult {
        let same_ticket = self.acbs.get(&from).map(|a| a.ticket) == self.acbs.get(&to).map(|a| a.ticket);
        if to == CLI_PID {
            self.bus.publish(KernelEvent::Notice { pid: Some(from), text: body });
            return SyscallResult::Output("sent to the guest".into());
        }
        if !same_ticket || !self.acbs.get(&to).is_some_and(|a| a.state.alive()) {
            return SyscallResult::Failed(format!("no live process {to} on this order"));
        }
        let msg = Msg { from, kind, body };
        let target = self.acbs.get_mut(&to).expect("checked");
        if let Err(m) = target.mailbox.try_send(msg) {
            drop(m);
            return SyscallResult::Failed(format!("mailbox of {to} is full; try again later"));
        }
        // Wake a receiver waiting on its mailbox.
        if target.state == State::Blocked(Block::Recv) {
            if let (Some(slot), Some(m)) = (target.parked_slot.take(), target.mailbox.recv()) {
                let batch = target.batch;
                self.fill(to, batch, slot, SyscallResult::Message(m));
            }
        }
        SyscallResult::Output("sent".into())
    }

    fn syscall_done(&mut self, pid: Pid, batch: u64, idx: usize, result: SyscallResult) {
        let Some(a) = self.acbs.get_mut(&pid) else { return };
        if !a.state.alive() || a.batch != batch {
            return; // stale: killed, or a batch the supervisor gave up on
        }
        if let SyscallResult::Completion { usage, .. } = &result {
            a.tokens += usage.prompt_tokens + usage.completion_tokens;
            let model = a.inflight.get(idx).map(|f| f.summary.trim_start_matches("llm ").to_string()).unwrap_or_default();
            self.bus.publish(KernelEvent::Usage {
                pid,
                model,
                prompt_tokens: usage.prompt_tokens,
                completion_tokens: usage.completion_tokens,
            });
        }
        self.fill(pid, batch, idx, result);
    }

    /// Record one result of the batch; when it is the last, run the supervisor and wake the process.
    fn fill(&mut self, pid: Pid, batch: u64, idx: usize, result: SyscallResult) {
        let tick = self.tick;
        let Some(a) = self.acbs.get_mut(&pid) else { return };
        if a.batch != batch || a.pending.get(idx).is_none_or(|p| p.is_some()) {
            return;
        }
        a.last_progress = tick;
        let summary = a.inflight.get(idx).map(|f| f.summary.clone()).unwrap_or_default();
        self.bus.publish(KernelEvent::SyscallDone { pid, idx, ok: result.ok(), summary, result: result.clone() });
        let a = self.acbs.get_mut(&pid).expect("present");
        a.pending[idx] = Some(result);
        a.outstanding -= 1;
        if a.outstanding == 0 {
            self.finish_batch(pid);
        }
    }

    fn finish_batch(&mut self, pid: Pid) {
        let a = self.acbs.get_mut(&pid).expect("present");
        let results: Vec<SyscallResult> = a.pending.drain(..).map(|r| r.expect("all filled")).collect();
        let inflight = std::mem::take(&mut a.inflight);
        if !a.batch_interrupted {
            a.hang_strikes = 0;
        }
        // Supervisor: loops are judged on whole steps (the reply, or the tool calls with results).
        let mut verdict = Verdict::Ok;
        let completion = inflight.iter().zip(&results).find(|(f, _)| f.completion).map(|(_, r)| r);
        if let Some(SyscallResult::Completion { message, cut, .. }) = completion {
            a.last_text = message.text().to_string();
            if *cut {
                verdict = a.supervisor.strike("kept repeating the same text in one reply".into());
            } else if message.tool_calls.is_empty() {
                verdict = a.supervisor.observe(&a.last_text, &[]);
            }
        } else {
            let texts: Vec<String> = results.iter().map(SyscallResult::text).collect();
            let records: Vec<CallRecord> = inflight
                .iter()
                .zip(&texts)
                .filter_map(|(f, r)| f.tool.as_ref().map(|(name, args)| CallRecord { name, arguments: args, result: r }))
                .collect();
            if !records.is_empty() {
                let text = a.last_text.clone();
                verdict = a.supervisor.observe(&text, &records);
            }
        }
        a.wake = results;
        match verdict {
            Verdict::Ok => {}
            Verdict::Nudge(reason) => {
                a.notes.push(nudge_message(&reason));
                self.bus.publish(KernelEvent::Signal { pid, signal: Signal::Warn, reason: format!("looping: {reason} — nudged") });
            }
            Verdict::Stop(reason) => {
                let report = format!("{} it {reason}, again after a warning]\nSTATUS: failed", supervisor::STOPPED_BY_SUPERVISOR);
                self.finish(pid, ExitStatus::failed(report), State::Killed);
                return;
            }
        }
        self.enqueue(pid);
    }

    // ── plans ──

    fn submit_plan(&mut self, pid: Pid, batch: u64, idx: usize, tasks: Vec<PlanTask>) {
        let (ticket, is_root) = match self.acbs.get(&pid) {
            Some(a) => (a.ticket, self.tickets.get(&a.ticket).is_some_and(|t| t.root == pid)),
            None => return,
        };
        if !is_root {
            return self.fill(pid, batch, idx, SyscallResult::Failed("only the order's architect may submit a plan".into()));
        }
        if self.tickets.get(&ticket).and_then(|t| t.plan.as_ref()).is_some_and(|p| p.approved) {
            return self.fill(pid, batch, idx, SyscallResult::Failed("this order's plan was already approved and run".into()));
        }
        if let Err(e) = plan::validate(&tasks) {
            return self.fill(pid, batch, idx, SyscallResult::Failed(format!("invalid plan: {e}. Fix it and submit again.")));
        }
        let text = render_plan(&tasks);
        let runs = tasks.iter().map(|_| TaskRun { status: TaskStatus::Pending, pid: None, report: String::new() }).collect();
        if let Some(t) = self.tickets.get_mut(&ticket) {
            t.plan = Some(PlanRun { tasks, runs, approved: false });
        }
        if let Some(a) = self.acbs.get_mut(&pid) {
            a.parked_slot = Some(idx);
        }
        self.set_state(pid, State::Blocked(Block::Question));
        let qid = self.next_qid;
        self.next_qid += 1;
        self.questions.insert(qid, ticket);
        self.bus.publish(KernelEvent::Question { qid, ticket, pid, text });
        if self.cfg.auto_approve {
            self.answer(qid, "");
        }
    }

    fn answer(&mut self, qid: u64, text: &str) {
        let Some(ticket) = self.questions.remove(&qid) else { return };
        let Some(root) = self.tickets.get(&ticket).map(|t| t.root) else { return };
        let reply = match text.trim().to_lowercase().as_str() {
            "" | "a" | "y" | "yes" | "approve" => None,
            "r" | "n" | "no" | "reject" => {
                Some("The guest rejected this plan. Do not implement anything; stop and say so briefly.".to_string())
            }
            _ => Some(format!(
                "The guest wants changes to the plan before approving it: {}\nRevise it and call submit_plan again.",
                text.trim()
            )),
        };
        match reply {
            Some(r) => self.wake_parked(root, SyscallResult::Output(r)),
            None => {
                let tasks = {
                    let Some(p) = self.tickets.get_mut(&ticket).and_then(|t| t.plan.as_mut()) else { return };
                    p.approved = true;
                    p.tasks.clone()
                };
                let drivers = self.drivers.clone();
                self.io.run(Box::new(move || drivers.save_plan(ticket, &tasks)));
                self.set_state(root, State::Blocked(Block::Child));
                self.bus
                    .publish(KernelEvent::Notice { pid: Some(root), text: format!("plan approved: {} task(s)", self.plan_len(ticket)) });
                self.fire(ticket);
            }
        }
    }

    fn plan_len(&self, ticket: TicketId) -> usize {
        self.tickets.get(&ticket).and_then(|t| t.plan.as_ref()).map_or(0, |p| p.tasks.len())
    }

    /// Deliver the result of the parked syscall (mailbox / question / children) and wake the process.
    fn wake_parked(&mut self, pid: Pid, result: SyscallResult) {
        let Some(a) = self.acbs.get_mut(&pid) else { return };
        if let Some(slot) = a.parked_slot.take() {
            let batch = a.batch;
            self.fill(pid, batch, slot, result);
        }
    }

    /// Start every task that can run; when none is left running, hand the digest to the architect.
    fn fire(&mut self, ticket: TicketId) {
        let Some(t) = self.tickets.get_mut(&ticket) else { return };
        let (root, cancel) = (t.root, t.cancel.clone());
        let Some(p) = t.plan.as_mut().filter(|p| p.approved) else { return };
        let mut status: Vec<TaskStatus> = p.runs.iter().map(|r| r.status).collect();
        let blocked = plan::propagate_failures(&p.tasks, &mut status);
        for i in &blocked {
            p.runs[*i].status = TaskStatus::Blocked;
            p.runs[*i].report = "Blocked: a dependency failed.".into();
        }
        let start = plan::startable(&p.tasks, &status);
        let context = plan_context(&p.tasks);
        let jobs: Vec<(usize, PlanTask)> = start.iter().map(|i| (*i, p.tasks[*i].clone())).collect();
        for (i, task) in jobs {
            let agent = self.recipes.developer(ticket, &task, &context);
            let caps = Caps { write: task.owned_paths.clone(), write_any: task.owned_paths.is_empty(), exec: true };
            let title = format!("[{}] {}", task.id, task.title);
            let pid = self.spawn(agent, Some(root), ticket, Group::Development, title, caps, cancel.child(), 1);
            let run = &mut self.tickets.get_mut(&ticket).unwrap().plan.as_mut().unwrap().runs[i];
            run.status = TaskStatus::Running;
            run.pid = Some(pid);
        }
        let p = self.tickets.get(&ticket).and_then(|t| t.plan.as_ref()).expect("plan");
        if p.runs.iter().all(|r| r.status.is_terminal()) {
            let digest = digest(p);
            self.wake_parked(root, SyscallResult::Output(digest));
        }
    }

    // ── endings ──

    /// A process ended (exit, budget, supervisor, kill). Children of an ended architect go too.
    fn finish(&mut self, pid: Pid, st: ExitStatus, state: State) {
        let Some(a) = self.acbs.get_mut(&pid) else { return };
        if !a.state.alive() {
            return;
        }
        a.cancel.cancel();
        a.inflight.clear();
        let (ticket, group) = (a.ticket, a.group);
        self.set_state(pid, state);
        self.bus.publish(KernelEvent::Exited { pid, ok: st.ok, report: st.report.clone() });
        self.drivers.release(pid);
        let Some(t) = self.tickets.get_mut(&ticket) else { return };
        if t.root == pid {
            // The order is finished (served, failed or cancelled): stop whatever still runs for it.
            t.served = true;
            t.cancel.cancel();
            let children: Vec<Pid> =
                self.acbs.values().filter(|c| c.ticket == ticket && c.pid != pid && c.state.alive()).map(|c| c.pid).collect();
            for c in children {
                self.finish(c, ExitStatus::failed("[stopped: the order ended]\nSTATUS: failed"), State::Killed);
            }
            self.questions.retain(|_, tk| *tk != ticket);
            self.bus.publish(KernelEvent::TicketServed { ticket, ok: st.ok, summary: st.report });
            self.acbs.retain(|_, a| a.ticket != ticket);
        } else if group == Group::Development {
            if let Some(p) = t.plan.as_mut() {
                if let Some(run) = p.runs.iter_mut().find(|r| r.pid == Some(pid)) {
                    run.status = if st.ok && report_ok(&st.report) { TaskStatus::Done } else { TaskStatus::Failed };
                    run.report = st.report;
                }
            }
            if !t.served {
                self.fire(ticket);
            }
        }
    }

    fn kill(&mut self, pid: Pid, reason: &str) {
        if self.acbs.get(&pid).is_some_and(|a| a.state.alive()) {
            self.bus.publish(KernelEvent::Signal { pid, signal: Signal::Kill, reason: reason.into() });
            self.finish(pid, ExitStatus::failed(format!("[{reason}]\nSTATUS: failed")), State::Killed);
        }
    }

    // ── timer tick ──

    fn on_tick(&mut self) {
        let tick = self.tick;
        // Hung processes: no progress for hang_ticks while running or waiting on I/O. Commands
        // have their own timeout and are left to the exec driver.
        let mut interrupt = Vec::new();
        let mut kill = Vec::new();
        for a in self.acbs.values() {
            let watched = matches!(a.state, State::Running | State::Blocked(Block::Io));
            let exec_busy = a.inflight.iter().any(|f| f.class == PoolClass::Exec);
            if watched && !exec_busy && tick.saturating_sub(a.last_progress) > self.cfg.hang_ticks {
                if a.hang_strikes == 0 && a.state == State::Blocked(Block::Io) {
                    interrupt.push(a.pid);
                } else {
                    kill.push(a.pid);
                }
            }
        }
        for pid in interrupt {
            let secs = self.cfg.hang_ticks * self.cfg.tick.as_millis() as u64 / 1000;
            if let Some(a) = self.acbs.get_mut(&pid) {
                a.batch_cancel.cancel();
                a.batch_interrupted = true;
                a.hang_strikes += 1;
                a.last_progress = tick;
            }
            self.bus.publish(KernelEvent::Signal {
                pid,
                signal: Signal::Interrupt,
                reason: format!("hung: no progress for {secs}s — in-flight calls cancelled"),
            });
        }
        for pid in kill {
            self.kill(pid, "hung: no progress after an interrupt");
        }
        // Aging: a process waiting too long in a lower queue moves up one level.
        if self.cfg.age_ticks > 0 && tick.is_multiple_of(self.cfg.age_ticks) {
            for level in 1..QUEUES {
                let waiting: Vec<Pid> = self.queues[level].iter().copied().collect();
                for pid in waiting {
                    if let Some(a) = self.acbs.get_mut(&pid) {
                        if a.state == State::Ready
                            && tick.saturating_sub(a.ready_since) >= self.cfg.age_ticks
                            && a.priority as usize == level
                        {
                            a.priority -= 1;
                            a.ready_since = tick;
                            self.queues[level].retain(|p| *p != pid);
                            self.queues[level - 1].push_back(pid);
                        }
                    }
                }
            }
        }
    }
}

/// Did a Developer report success? (Reports end with `STATUS: done|failed`.)
pub fn report_ok(report: &str) -> bool {
    !report.lines().any(|l| {
        let l = l.trim().to_ascii_lowercase();
        l.starts_with("status:") && l.contains("failed")
    })
}

/// The plan as shown to the guest: tasks grouped into waves that run in parallel.
pub fn render_plan(tasks: &[PlanTask]) -> String {
    let waves = plan::waves(tasks).unwrap_or_default();
    let mut s = String::new();
    for (i, wave) in waves.iter().enumerate() {
        s.push_str(&format!("wave {} — runs in parallel\n", i + 1));
        for id in wave {
            if let Some(t) = tasks.iter().find(|t| &t.id == id) {
                let paths = if t.owned_paths.is_empty() { "any file".into() } else { t.owned_paths.join(", ") };
                s.push_str(&format!("  ☐ [{}] {}  ({paths})\n", t.id, t.title));
                for a in &t.acceptance {
                    s.push_str(&format!("      ✓ {a}\n"));
                }
            }
        }
    }
    s.trim_end().to_string()
}

fn plan_context(tasks: &[PlanTask]) -> String {
    tasks.iter().map(|t| format!("[{}] {} (owns: {})", t.id, t.title, t.owned_paths.join(", "))).collect::<Vec<_>>().join("\n")
}

fn digest(p: &PlanRun) -> String {
    let mut out = String::from("All tasks finished.\n");
    for (t, r) in p.tasks.iter().zip(&p.runs) {
        let report: String = r.report.trim().chars().take(4000).collect();
        out.push_str(&format!("\n### [{}] {} — {:?}\n{report}\n", t.id, t.title, r.status));
    }
    out
}

#[cfg(test)]
mod tests;
