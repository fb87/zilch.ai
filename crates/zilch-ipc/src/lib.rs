//! The vocabulary shared by the zilch kernel, its drivers, the agents and the CLI.
//!
//! Restaurant view: a guest's order becomes a **ticket**; cooks (agents) are **processes** with a
//! pid, they talk through **mailboxes**, ask the kitchen for things with **syscalls**, and
//! everything the kitchen does is announced on the **event bus** (the ticket rail everyone can
//! read). No async runtime anywhere: plain `std` channels and mutexes.

pub mod llm;

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TrySendError};
use std::sync::{Arc, Mutex};

pub use llm::{ChatRequest, Message, ModelBinding, ToolCall, ToolSpec, Usage};

pub type Pid = u64;
pub type TicketId = u64;

/// The CLI (the waiter) is not a process, but it can be addressed like one.
pub const CLI_PID: Pid = 0;

/// Stations of the kitchen; each has its own default capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum Group {
    /// The main agent (head chef): one per session, keeps the whole conversation; analyses each
    /// order, writes requirements/design/plan (docs only) and hands tasks to subagents.
    Architecture,
    /// Subagents (line cooks): one per task, fresh isolated context, write only owned paths.
    Development,
    /// Expediter at the pass (milestone 4).
    Review,
    /// The spec lane's requirements engineer: one per session, maintains spec/design/plan.
    Spec,
}

impl Group {
    pub fn name(self) -> &'static str {
        match self {
            Group::Architecture => "main",
            Group::Development => "developer",
            Group::Review => "reviewer",
            Group::Spec => "spec",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MsgKind {
    Task,
    Result,
    ReviewFeedback,
    Question,
    Answer,
    Cancel,
    /// Extra input from the guest (`@pid …` / `@group …` at the prompt).
    Guest,
}

/// A note passed between cooks. Large payloads live in files; messages carry text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Msg {
    pub from: Pid,
    pub kind: MsgKind,
    pub body: String,
}

/// Bounded per-process queue. Sending never blocks: a full mailbox hands the message back, so
/// two agents can never deadlock on each other's queues.
#[derive(Debug)]
pub struct Mailbox {
    q: VecDeque<Msg>,
    cap: usize,
}

impl Mailbox {
    pub fn new(cap: usize) -> Self {
        Self { q: VecDeque::new(), cap: cap.max(1) }
    }
    pub fn try_send(&mut self, m: Msg) -> Result<(), Msg> {
        if self.q.len() >= self.cap {
            return Err(m);
        }
        self.q.push_back(m);
        Ok(())
    }
    pub fn recv(&mut self) -> Option<Msg> {
        self.q.pop_front()
    }
    pub fn len(&self) -> usize {
        self.q.len()
    }
    pub fn is_empty(&self) -> bool {
        self.q.is_empty()
    }
}

/// Who a guest message is for: one process, or every live process of a group in the newest order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Pid(Pid),
    Group(Group),
}

impl Target {
    /// `"3"` → pid 3; `"architect"`, `"developer"`, `"reviewer"` (or plurals) → that group.
    pub fn parse(s: &str) -> Option<Self> {
        if let Ok(pid) = s.parse::<Pid>() {
            return Some(Target::Pid(pid));
        }
        match s.to_ascii_lowercase().trim_end_matches('s') {
            "main" | "architect" => Some(Target::Group(Group::Architecture)),
            "developer" | "dev" => Some(Target::Group(Group::Development)),
            "reviewer" => Some(Target::Group(Group::Review)),
            "spec" => Some(Target::Group(Group::Spec)),
            _ => None,
        }
    }
}

/// One task of the Architect's plan (`docs/plan.json`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PlanTask {
    pub id: String,
    pub title: String,
    /// Self-contained instructions for the Developer.
    pub task: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Globs the Developer may write. Empty = may touch anything (runs alone).
    #[serde(default)]
    pub owned_paths: Vec<String>,
    /// How to tell the task is done ("cargo test export::json passes").
    #[serde(default)]
    pub acceptance: Vec<String>,
    /// Scheduling priority: 0 most urgent … 3 background (default 1).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub priority: Option<u8>,
    /// A custom subagent profile to run the task on (default: the developer).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub worker: Option<String>,
}

/// What an agent asks the kernel to do. A step may issue several; they run in parallel and the
/// agent is woken with all results, in order, once the last one completes.
#[derive(Debug, Clone)]
pub enum Syscall {
    /// One LLM call (it is I/O like any other: the step worker never waits on the network).
    Complete(ChatRequest),
    /// A tool call by name (read_file, write_file, edit_file, glob, grep, git, bash, …).
    Tool { call_id: String, name: String, args: serde_json::Value },
    /// Put a message in another process's mailbox (never blocks).
    Send { to: Pid, kind: MsgKind, body: String },
    /// Take the next message; blocks the agent (not a worker) while the mailbox is empty.
    Recv,
    /// Architect only: hand the plan to the kernel. Blocks until the guest approved it and every
    /// task finished (or the guest rejected it); the result is the digest of all task reports.
    SubmitPlan { tasks: Vec<PlanTask> },
    /// Main agent only: this order is done — give `summary` to the guest, then sleep until the
    /// next order (the result is that order, as a message).
    Serve { summary: String },
    /// A lane's main agent: investigate these questions with read-only explorers, one per question,
    /// in parallel; the result holds every explorer's report.
    Explore { questions: Vec<String> },
    /// The code lane's main agent hands the order to one subagent (`agent`: "answer" or "plan") with a
    /// self-contained brief; the result is that subagent's report.
    Dispatch { agent: String, brief: String },
}

impl Syscall {
    /// Short label for the process table and the event stream.
    pub fn summary(&self) -> String {
        match self {
            Syscall::Complete(r) => format!("llm {}", r.binding.key()),
            Syscall::Tool { name, args, .. } => {
                let detail = ["path", "command", "pattern", "doc"]
                    .iter()
                    .find_map(|k| args.get(*k).and_then(|v| v.as_str()))
                    .map(|s| s.lines().next().unwrap_or("").chars().take(80).collect::<String>())
                    .unwrap_or_default();
                format!("{name}({detail})")
            }
            Syscall::Send { to, kind, .. } => format!("send {kind:?} → {to}"),
            Syscall::Recv => "recv".into(),
            Syscall::SubmitPlan { tasks } => format!("submit_plan ({} tasks)", tasks.len()),
            Syscall::Serve { .. } => "serve".into(),
            Syscall::Explore { questions } => format!("explore ({} question(s))", questions.len()),
            Syscall::Dispatch { agent, .. } => format!("dispatch ({agent})"),
        }
    }
}

/// What a syscall came back with.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SyscallResult {
    /// The LLM's reply. `cut`: the stream was stopped because it kept repeating itself.
    Completion { message: Message, usage: Usage, cut: bool },
    /// Tool output (tool-level errors are text starting with "Error:", for the model to read).
    Output(String),
    /// A message taken from the mailbox.
    Message(Msg),
    /// The syscall could not run: denied by a capability, cancelled, or a driver failure.
    Failed(String),
}

impl SyscallResult {
    pub fn ok(&self) -> bool {
        match self {
            SyscallResult::Failed(_) => false,
            SyscallResult::Output(s) => !s.starts_with("Error"),
            _ => true,
        }
    }
    /// The result as text for a model (failures become "Error: …").
    pub fn text(&self) -> String {
        match self {
            SyscallResult::Completion { message, .. } => message.text().to_string(),
            SyscallResult::Output(s) => s.clone(),
            SyscallResult::Message(m) => format!("[message from {} ({:?})]\n{}", m.from, m.kind, m.body),
            SyscallResult::Failed(e) => format!("Error: {e}"),
        }
    }
}

/// How a process ended. `report` is its final word (a Developer's report, an answer, a reason).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExitStatus {
    pub ok: bool,
    pub report: String,
}

impl ExitStatus {
    pub fn ok(report: impl Into<String>) -> Self {
        Self { ok: true, report: report.into() }
    }
    pub fn failed(report: impl Into<String>) -> Self {
        Self { ok: false, report: report.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Signal {
    /// A supervisor warning (nudge) or an interrupted call.
    Warn,
    /// In-flight calls cancelled, the process keeps running.
    Interrupt,
    /// Conversation discarded; the process starts its task again with a summary.
    Restart,
    /// The process is stopped.
    Kill,
}

/// Everything the kernel does, as seen by observers (CLI, JSONL log).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum KernelEvent {
    TicketOpened {
        ticket: TicketId,
        text: String,
        /// Architecture = the code lane, Spec = the spec lane.
        lane: Group,
    },
    Spawned {
        pid: Pid,
        parent: Option<Pid>,
        ticket: TicketId,
        group: Group,
        title: String,
    },
    State {
        pid: Pid,
        state: String,
    },
    SyscallIssued {
        pid: Pid,
        idx: usize,
        summary: String,
    },
    /// `result` carries the full result for the log (replay); observers usually need `summary`.
    SyscallDone {
        pid: Pid,
        idx: usize,
        ok: bool,
        summary: String,
        result: SyscallResult,
    },
    /// Streamed reply text (not written to the log: the Completion result has it all).
    TextDelta {
        pid: Pid,
        text: String,
    },
    Usage {
        pid: Pid,
        model: String,
        prompt_tokens: u64,
        completion_tokens: u64,
        /// Of `prompt_tokens`, served from the provider's prompt cache.
        #[serde(default, skip_serializing_if = "is_zero_u64")]
        cached_tokens: u64,
    },
    /// Estimated tokens per part of a model request about to be sent (system, tools, user,
    /// assistant, reasoning, `tool:<name>` outputs), to see where input tokens go.
    RequestParts {
        pid: Pid,
        parts: Vec<(String, u64)>,
    },
    /// The kernel needs the guest: a plan to approve, or (approval = "ask") a command to allow.
    /// Answer with the CLI's `Answer` command.
    Question {
        qid: u64,
        ticket: TicketId,
        pid: Pid,
        text: String,
        /// A command waiting to run (y allow, n deny, a allow all commands this session).
        #[serde(default)]
        command: bool,
    },
    Notice {
        pid: Option<Pid>,
        text: String,
    },
    /// The guest sent extra input to a process (`@pid …`); `how` says when it will be read.
    Told {
        pid: Pid,
        text: String,
        how: String,
    },
    Signal {
        pid: Pid,
        signal: Signal,
        reason: String,
    },
    Exited {
        pid: Pid,
        ok: bool,
        report: String,
    },
    TicketServed {
        ticket: TicketId,
        ok: bool,
        summary: String,
    },
    /// This subscriber was too slow and missed `missed` events.
    Lagged {
        missed: u64,
    },
}

/// An event with its place in history: `seq` is gap-free across the bus, `tick` is kernel time.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Envelope {
    pub seq: u64,
    pub tick: u64,
    #[serde(flatten)]
    pub event: KernelEvent,
}

struct Subscriber {
    tx: SyncSender<Arc<Envelope>>,
    missed: u64,
}

struct BusInner {
    seq: u64,
    subs: Vec<Subscriber>,
    log: Option<Box<dyn Write + Send>>,
}

/// Publish/subscribe of kernel events. Publishing never blocks: each subscriber has a bounded
/// queue, and a slow one gets a `Lagged` marker instead of stalling the kitchen. The JSONL log is
/// written synchronously, so it never misses an event (it is the audit trail for replay).
pub struct EventBus {
    inner: Mutex<BusInner>,
    tick: AtomicU64,
}

impl Default for EventBus {
    fn default() -> Self {
        Self::new()
    }
}

impl EventBus {
    pub fn new() -> Self {
        Self { inner: Mutex::new(BusInner { seq: 0, subs: Vec::new(), log: None }), tick: AtomicU64::new(0) }
    }

    /// Write every event (except streamed text) as one JSON line to `w`.
    pub fn set_log(&self, w: Box<dyn Write + Send>) {
        self.inner.lock().unwrap().log = Some(w);
    }

    pub fn subscribe(&self, capacity: usize) -> Receiver<Arc<Envelope>> {
        let (tx, rx) = sync_channel(capacity.max(1));
        self.inner.lock().unwrap().subs.push(Subscriber { tx, missed: 0 });
        rx
    }

    pub fn set_tick(&self, t: u64) {
        self.tick.store(t, Ordering::Relaxed);
    }

    pub fn tick(&self) -> u64 {
        self.tick.load(Ordering::Relaxed)
    }

    pub fn publish(&self, event: KernelEvent) {
        let mut inner = self.inner.lock().unwrap();
        inner.seq += 1;
        let env = Arc::new(Envelope { seq: inner.seq, tick: self.tick(), event });
        if !matches!(env.event, KernelEvent::TextDelta { .. }) {
            if let Some(log) = inner.log.as_mut() {
                if let Ok(line) = serde_json::to_string(&*env) {
                    let _ = writeln!(log, "{line}");
                    let _ = log.flush();
                }
            }
        }
        inner.subs.retain_mut(|s| {
            if s.missed > 0 {
                let lag = Arc::new(Envelope { seq: env.seq, tick: env.tick, event: KernelEvent::Lagged { missed: s.missed } });
                match s.tx.try_send(lag) {
                    Ok(()) => s.missed = 0,
                    Err(TrySendError::Full(_)) => {
                        s.missed += 1;
                        return true;
                    }
                    Err(TrySendError::Disconnected(_)) => return false,
                }
            }
            match s.tx.try_send(env.clone()) {
                Ok(()) => true,
                Err(TrySendError::Full(_)) => {
                    s.missed += 1;
                    true
                }
                Err(TrySendError::Disconnected(_)) => false,
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_mailbox_hands_the_message_back_instead_of_blocking() {
        let mut m = Mailbox::new(2);
        let msg = |b: &str| Msg { from: 1, kind: MsgKind::Task, body: b.into() };
        assert!(m.try_send(msg("a")).is_ok());
        assert!(m.try_send(msg("b")).is_ok());
        assert_eq!(m.try_send(msg("c")).unwrap_err().body, "c");
        assert_eq!(m.recv().unwrap().body, "a", "in order");
        assert!(m.try_send(msg("c")).is_ok());
    }

    #[test]
    fn a_slow_subscriber_gets_a_lag_marker_and_never_blocks_the_publisher() {
        let bus = EventBus::new();
        let slow = bus.subscribe(2);
        let fast = bus.subscribe(100);
        for i in 0..10 {
            bus.publish(KernelEvent::Notice { pid: None, text: format!("n{i}") });
        }
        assert_eq!(fast.try_iter().count(), 10);
        let got: Vec<_> = slow.try_iter().map(|e| e.event.clone()).collect();
        assert_eq!(got.len(), 2, "only what fit");
        bus.publish(KernelEvent::Notice { pid: None, text: "after".into() });
        let next: Vec<_> = slow.try_iter().map(|e| e.event.clone()).collect();
        assert_eq!(next[0], KernelEvent::Lagged { missed: 8 });
        assert!(matches!(&next[1], KernelEvent::Notice { text, .. } if text == "after"));
    }

    #[test]
    fn the_log_gets_every_event_but_streamed_text_with_seq_and_tick() {
        #[derive(Clone, Default)]
        struct Buf(Arc<Mutex<Vec<u8>>>);
        impl Write for Buf {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(b);
                Ok(b.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let bus = EventBus::new();
        let buf = Buf::default();
        bus.set_log(Box::new(buf.clone()));
        bus.set_tick(7);
        bus.publish(KernelEvent::TextDelta { pid: 1, text: "x".into() });
        bus.publish(KernelEvent::Exited { pid: 1, ok: true, report: "done".into() });
        let text = String::from_utf8(buf.0.lock().unwrap().clone()).unwrap();
        assert_eq!(text.lines().count(), 1);
        let line: serde_json::Value = serde_json::from_str(text.trim()).unwrap();
        assert_eq!((line["seq"].as_u64(), line["tick"].as_u64(), line["type"].as_str()), (Some(2), Some(7), Some("Exited")));
    }
}

#[cfg(test)]
mod target_tests {
    use super::*;

    #[test]
    fn mentions_parse_to_pids_and_groups() {
        assert_eq!(Target::parse("3"), Some(Target::Pid(3)));
        assert_eq!(Target::parse("architect"), Some(Target::Group(Group::Architecture)));
        assert_eq!(Target::parse("Developers"), Some(Target::Group(Group::Development)));
        assert_eq!(Target::parse("dev"), Some(Target::Group(Group::Development)));
        assert_eq!(Target::parse("T2"), None);
    }
}

fn is_zero_u64(n: &u64) -> bool {
    *n == 0
}
