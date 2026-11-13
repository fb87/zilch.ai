//! `zilch`: the waiter. Takes the guest's order, hands it to the kitchen (the kernel), shows
//! what the kitchen does as it happens, relays the plan for approval, and serves the result.
//! It never waits on the agents: the kernel runs on its own threads and reports on the event bus.

mod completion;
mod costs;
mod daemon;
mod debug;
mod front;
mod ipc;
mod markdown;
mod session;
mod status;
mod wrap;

use clap::Parser;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zilch_agents::Book;
use zilch_drivers::config::{Config, Overrides};

use zilch_ipc::{Group, KernelEvent, Pid, TicketId};
use zilch_kernel::KernelHandle;

#[derive(Parser)]
#[command(
    name = "zilch",
    version,
    about = "zilch: a coding agent built like a small OS — a main agent plans your order, subagents cook it in parallel, the result is served."
)]
struct Cli {
    /// Place one order and exit when it is served.
    #[arg(short = 'p', long = "prompt", global = true)]
    prompt: Option<String>,
    /// Approve plans without asking.
    #[arg(long, global = true)]
    yes: bool,
    /// Nothing asks: plans approved, commands run (same as --approval auto).
    #[arg(long, global = true)]
    yolo: bool,
    /// Outside sources for this session: fetch_source, web_fetch (and web_search if configured).
    #[arg(long, global = true)]
    web: bool,
    /// ask (plans and commands) | plan (plans; default) | auto (nothing).
    #[arg(long, value_name = "MODE", global = true)]
    approval: Option<String>,
    /// Model for every group.
    #[arg(long, global = true)]
    model: Option<String>,
    #[arg(long, global = true)]
    architect_model: Option<String>,
    #[arg(long, global = true)]
    developer_model: Option<String>,
    /// Step workers (default: CPU cores).
    #[arg(long, global = true)]
    workers: Option<usize>,
    /// Write the event log here (default: <data dir>/zilch/logs/os-<time>.jsonl).
    #[arg(long, global = true)]
    log: Option<PathBuf>,
    #[arg(long, global = true)]
    no_log: bool,
    /// Continue the most recent session in this directory (the main agent remembers it).
    #[arg(short = 'c', long = "continue", global = true)]
    r#continue: bool,
    /// Do not save this session.
    #[arg(long, global = true)]
    no_session: bool,
    /// Continue a specific session of this directory (ids from --sessions).
    #[arg(long, value_name = "ID", global = true)]
    resume: Option<String>,
    /// List the saved sessions of this directory and exit.
    #[arg(long, global = true)]
    sessions: bool,
    /// Show the resolved setup and ping every model, then exit.
    #[arg(long, global = true)]
    doctor: bool,
    /// Start verbose: every model call and syscall result, and the models' streamed text and
    /// reasoning (/verbose, /quiet switch while running).
    #[arg(short, long, global = true)]
    verbose: bool,
    /// Start in debug mode: verbose, plus timings and sizes of every model and tool call (time to
    /// first token, tokens, cost, output size, zilch's own overhead) and a breakdown per order
    /// (/debug switches while running).
    #[arg(long, global = true)]
    debug: bool,
    /// With -p: the order goes to this daemon session (created for this directory if missing).
    #[arg(long, value_name = "ID", global = true)]
    session: Option<String>,
    #[command(subcommand)]
    cmd: Option<Sub>,
}

#[derive(clap::Subcommand)]
enum Sub {
    /// The zilch daemon: run (foreground, default) | start (background) | stop | status.
    Daemon { action: Option<String> },
    /// Create a session in the daemon (prints its id).
    New {
        /// The workspace (default: this directory).
        #[arg(long)]
        workspace: Option<PathBuf>,
        /// The session's id (default: <directory name>-<n>).
        #[arg(long)]
        name: Option<String>,
    },
    /// A REPL on a daemon session (default: this directory's, created if missing). /detach leaves it running.
    Attach { id: Option<String> },
    /// The daemon's sessions.
    Ls,
    /// End a daemon session (its conversation stays resumable).
    Stop { id: String },
}

const BOLD: &str = "\x1b[1m";
const DIM: &str = "\x1b[2m";
const RESET: &str = "\x1b[0m";
const GREEN: &str = "\x1b[32m";
const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const CYAN: &str = "\x1b[36m";

/// Turns kernel events into lines for the guest, and remembers what the prompt needs.
struct View {
    color: bool,
    /// `/verbose` (or -v): model calls, syscall results and the models' streamed text and
    /// reasoning too; `/quiet` switches back.
    verbose: std::sync::atomic::AtomicBool,
    /// Streamed text not yet shown, per process: (is reasoning, the unfinished line).
    streams: Mutex<HashMap<Pid, (bool, String)>>,
    /// Per process: the markdown state of its streamed reply (inside a code block, a table …).
    md: Mutex<HashMap<Pid, markdown::Markdown>>,
    /// The workspace as models may spell it (canonical, and as given); paths below it are shown
    /// relative to it.
    workspace: Vec<String>,
    procs: Mutex<HashMap<Pid, (Group, String)>>,
    /// Plans waiting for the guest, oldest first: (qid, ticket).
    questions: Mutex<std::collections::VecDeque<(u64, TicketId)>>,
    /// Questions that are commands waiting to run (approval = "ask"): qid → "3 wants to run: …".
    commands: Mutex<HashMap<u64, String>>,
    open: Mutex<HashSet<TicketId>>,
    /// Live processes for the status row.
    live: Mutex<BTreeMap<Pid, Live>>,
    /// A short message shown in the status row for a few seconds.
    note: Mutex<Option<(String, Instant)>>,
    /// Run when an order is served (brings the repository overview up to date).
    on_served: std::sync::OnceLock<Box<dyn Fn() + Send + Sync>>,
    started: Instant,
    /// Usage, prices, the high-cost warning.
    costs: Arc<costs::Costs>,
    /// The last served answer, for /redraw.
    last_summary: Mutex<Option<String>>,
    /// This session's orders for /recap: (number, text, answer once served).
    orders: Mutex<Vec<(TicketId, String, Option<String>)>>,
    /// The lane typed orders go to (Architecture = code, Spec = spec), and each order's lane.
    lane: Mutex<Group>,
    lanes_of: Mutex<HashMap<TicketId, Group>>,
    /// The order each subagent works for (the resident agents work for their lane's open order).
    tickets_of: Mutex<HashMap<Pid, TicketId>>,
    /// Per order: when it was placed, and its model requests, tokens and cost so far.
    order_stats: Mutex<HashMap<TicketId, OrderStats>>,
    /// `--debug` / `/debug`: timings and sizes of every call, and a breakdown per order.
    debug: std::sync::atomic::AtomicBool,
    probe: debug::Probe,
    /// Between orders: since when, and what the main agent's last wake-up did.
    idle: Mutex<Idle>,
    /// Quiet mode: each process.s tool calls of the current step, shown as one line when done.
    steps: Mutex<HashMap<Pid, Step>>,
}

/// The status row between orders: `idle 5m12s · woke at 5m00s: nothing to condense`.
#[derive(Default)]
struct Idle {
    /// None while an order is open.
    since: Option<Instant>,
    /// A wake-up in progress: when (into the idle time), and whether it called the model.
    waking: Option<(Duration, bool)>,
    /// The last finished wake-up.
    last: Option<(Duration, bool)>,
}

/// `75s` → `1m15s`, `9s` → `9s`.
fn short_duration(d: Duration) -> String {
    let s = d.as_secs();
    if s >= 60 {
        format!("{}m{:02}s", s / 60, s % 60)
    } else {
        format!("{s}s")
    }
}

#[derive(Default)]
struct OrderStats {
    started: Option<Instant>,
    requests: u64,
    usage: zilch_ipc::Usage,
    usd: f64,
    /// Some request's price was unknown: the total is a lower bound.
    unpriced: bool,
}

impl OrderStats {
    /// `2m29s · 16 requests · 421k in (63% cached) · 10k out · $0.0123`
    fn line(&self) -> String {
        let secs = self.started.map_or(0, |s| s.elapsed().as_secs());
        let time = if secs >= 60 { format!("{}m{:02}s", secs / 60, secs % 60) } else { format!("{secs}s") };
        if self.requests == 0 {
            return time;
        }
        let u = &self.usage;
        let cost = match (self.unpriced, self.usd > 0.0) {
            (true, false) => "price unknown".to_string(),
            (true, true) => format!("≥ ${:.4}", self.usd),
            (false, _) => format!("${:.4}", self.usd),
        };
        format!(
            "{time} · {} request{} · {} in ({}) · {} out · {cost}",
            self.requests,
            if self.requests == 1 { "" } else { "s" },
            costs::fmt_tokens(u.prompt_tokens),
            costs::cached_share(u.cached_tokens, u.prompt_tokens),
            costs::fmt_tokens(u.completion_tokens),
        )
    }
}

/// What the status row knows about one process.
struct Live {
    group: Group,
    ticket: TicketId,
    state: String,
    /// What it is doing and since when.
    doing: Option<(String, Instant)>,
}

impl View {
    fn c(&self, code: &'static str) -> &'static str {
        if self.color {
            code
        } else {
            ""
        }
    }

    /// Paths inside the workspace relative to it (`/ws/src/a.rs` → `src/a.rs`, `/ws` → `.`);
    /// paths outside it stay absolute.
    fn rel(&self, text: &str) -> String {
        rel_to(&self.workspace, text)
    }

    /// The order a process works for: a subagent's own, or the open order of a resident agent's
    /// lane (none between orders: idle housekeeping counts for the session only).
    fn ticket_for(&self, pid: Pid) -> Option<TicketId> {
        self.tickets_of.lock().unwrap().get(&pid).copied().or_else(|| {
            let lane = self.procs.lock().unwrap().get(&pid).map(|(g, _)| *g)?;
            let lanes = self.lanes_of.lock().unwrap();
            self.open.lock().unwrap().iter().filter(|t| lanes.get(t) == Some(&lane)).max().copied()
        })
    }

    /// A process as the debug breakdown names it: `main`, `spec`, `2 answer`, `5 [T1]`.
    fn label(&self, pid: Pid) -> String {
        match self.procs.lock().unwrap().get(&pid) {
            Some((Group::Architecture, _)) => "main".into(),
            Some((Group::Spec, _)) => "spec".into(),
            Some((_, title)) => format!("{pid} {}", title.split([':', ' ']).next().unwrap_or("")),
            None => pid.to_string(),
        }
    }

    fn debug(&self) -> bool {
        self.debug.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `/debug` on or off (on implies verbose; `/quiet` and `/verbose` switch it off).
    fn set_debug(&self, on: bool) {
        self.debug.store(on, std::sync::atomic::Ordering::Relaxed);
    }

    fn verbose(&self) -> bool {
        self.verbose.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// `/verbose` or `/quiet`: whether the log shows everything or only what the guest needs.
    fn set_verbose(&self, on: bool) {
        self.verbose.store(on, std::sync::atomic::Ordering::Relaxed);
        self.streams.lock().unwrap().clear();
        self.md.lock().unwrap().clear();
        self.steps.lock().unwrap().clear();
    }

    /// One streamed line of a process: `┊` (dim, as written) for reasoning, `│` for reply text,
    /// which is markdown: rendered line by line on a terminal (a table comes out once it ends).
    fn stream_line(&self, pid: Pid, reasoning: bool, line: &str) -> Option<String> {
        let line = line.trim_end();
        let (d, r) = (self.c(DIM), self.c(RESET));
        if reasoning || !self.color {
            if line.trim().is_empty() {
                return None;
            }
            return Some(if reasoning { format!("    {d}{pid} ┊ {line}{r}") } else { format!("    {d}{pid} │{r} {line}") });
        }
        let width = terminal_width().saturating_sub(8 + pid.to_string().len()).max(20);
        let rows = self.md.lock().unwrap().entry(pid).or_insert_with(|| markdown::Markdown::new(true)).line(line, width)?;
        self.md_rows(pid, &rows)
    }

    /// Rendered markdown rows under a process's `│`.
    fn md_rows(&self, pid: Pid, rows: &str) -> Option<String> {
        let (d, r) = (self.c(DIM), self.c(RESET));
        let rows: Vec<String> = rows.split('\n').filter(|l| !l.trim().is_empty()).map(|l| format!("    {d}{pid} │{r} {l}")).collect();
        (!rows.is_empty()).then(|| rows.join("\n"))
    }

    /// Streamed text of one process: whole lines come out at once, the rest waits for more (or
    /// for the call's end), so parallel processes don't interleave mid-line.
    fn stream(&self, pid: Pid, reasoning: bool, text: &str) -> Option<String> {
        let mut streams = self.streams.lock().unwrap();
        let (kind, buf) = streams.entry(pid).or_insert_with(|| (reasoning, String::new()));
        let mut out = Vec::new();
        if *kind != reasoning {
            // Reasoning ended and the reply began (or the other way round): finish the line.
            out.extend(self.stream_line(pid, *kind, buf));
            buf.clear();
            *kind = reasoning;
        }
        buf.push_str(text);
        while let Some(i) = buf.find('\n') {
            let line: String = buf.drain(..=i).collect();
            out.extend(self.stream_line(pid, reasoning, &line));
        }
        (!out.is_empty()).then(|| out.join("\n"))
    }

    /// What is left of a process's stream when its model call ends.
    fn flush_stream(&self, pid: Pid) -> Option<String> {
        let (kind, buf) = self.streams.lock().unwrap().remove(&pid)?;
        let last = self.stream_line(pid, kind, &buf);
        // A table still being collected ends with the reply.
        let width = terminal_width().saturating_sub(8 + pid.to_string().len()).max(20);
        let table = self.md.lock().unwrap().remove(&pid).and_then(|mut m| m.finish(width)).and_then(|t| self.md_rows(pid, &t));
        match (last, table) {
            (Some(a), Some(b)) => Some(format!("{a}\n{b}")),
            (a, b) => a.or(b),
        }
    }

    /// The oldest plan waiting for an answer.
    fn question(&self) -> Option<(u64, TicketId)> {
        self.questions.lock().unwrap().front().copied()
    }

    fn latest_open(&self) -> Option<TicketId> {
        self.open.lock().unwrap().iter().max().copied()
    }

    /// Idle time and the main agent's wake-ups between orders: a resident agent's `recv` ending
    /// with no order open is a wake-up; a model call before its next `recv` means it condensed its
    /// conversation.
    fn track_idle(&self, ev: &KernelEvent) {
        let resident = |pid: &Pid| matches!(self.procs.lock().unwrap().get(pid).map(|(g, _)| *g), Some(Group::Architecture | Group::Spec));
        let mut idle = self.idle.lock().unwrap();
        match ev {
            KernelEvent::TicketOpened { .. } => *idle = Idle::default(),
            KernelEvent::TicketServed { ticket, .. } => {
                if self.open.lock().unwrap().iter().all(|t| t == ticket) {
                    *idle = Idle { since: Some(Instant::now()), ..Default::default() };
                }
            }
            KernelEvent::SyscallDone { pid, summary, .. } if summary == "recv" && resident(pid) => {
                if let Some(since) = idle.since {
                    idle.waking = Some((since.elapsed(), false));
                }
            }
            KernelEvent::SyscallIssued { pid, summary, .. } if summary.starts_with("llm ") && resident(pid) => {
                if let Some((_, called)) = idle.waking.as_mut() {
                    *called = true;
                }
            }
            KernelEvent::SyscallIssued { pid, summary, .. } if summary == "recv" && resident(pid) => {
                if let Some(w) = idle.waking.take() {
                    idle.last = Some(w);
                }
            }
            _ => {}
        }
    }

    /// `idle 5m12s · woke at 5m00s: nothing to condense` (before the first order: since start).
    fn idle_text(&self) -> String {
        let idle = self.idle.lock().unwrap();
        let since = idle.since.map_or_else(|| self.started.elapsed(), |s| s.elapsed());
        let mut s = format!("idle {}", short_duration(since));
        match (idle.waking, idle.last) {
            (Some(_), _) => s.push_str(" · main agent awake: checking its conversation"),
            (None, Some((at, condensed))) => {
                s.push_str(&format!(" · woke at {}: {}", short_duration(at), if condensed { "condensed" } else { "nothing to condense" }))
            }
            _ => {}
        }
        s
    }

    /// Keep the status row's picture of live processes up to date.
    fn track(&self, ev: &KernelEvent) {
        if let KernelEvent::RequestParts { parts, .. } = ev {
            self.costs.record_parts(parts);
        }
        self.track_idle(ev);
        match ev {
            KernelEvent::TicketOpened { ticket, .. } => {
                self.order_stats.lock().unwrap().insert(*ticket, OrderStats { started: Some(Instant::now()), ..Default::default() });
            }
            KernelEvent::Spawned { pid, ticket, group, .. } if !matches!(group, Group::Architecture | Group::Spec) => {
                self.tickets_of.lock().unwrap().insert(*pid, *ticket);
            }
            _ => {}
        }
        // Measure for --debug (always: /debug switched on mid-order still gets the whole order).
        match ev {
            KernelEvent::TicketOpened { ticket, .. } => self.probe.order_opened(*ticket),
            KernelEvent::SyscallIssued { pid, idx, summary } => {
                self.probe.issued(*pid, *idx, summary, self.ticket_for(*pid), self.debug());
            }
            KernelEvent::TextDelta { pid, .. } => self.probe.token(*pid),
            KernelEvent::SyscallDone { pid, idx, summary, result, .. } => {
                let who = self.label(*pid);
                let at = debug::At { who: &who, ticket: self.ticket_for(*pid), keep: self.debug() };
                self.probe.done(*pid, *idx, summary, result.text().chars().count(), at)
            }
            _ => {}
        }
        if let KernelEvent::Usage { pid, model, prompt_tokens, completion_tokens, cached_tokens } = ev {
            let group = self.procs.lock().unwrap().get(pid).map(|(g, _)| *g);
            let who = match group {
                Some(Group::Architecture) => "main",
                Some(Group::Spec) => "spec",
                _ => "subagents",
            };
            let u =
                zilch_ipc::Usage { prompt_tokens: *prompt_tokens, completion_tokens: *completion_tokens, cached_tokens: *cached_tokens };
            self.costs.record(who, model, u);
            self.probe.usage(*pid, u, self.costs.price(model, &u));
            if let Some(t) = self.ticket_for(*pid) {
                let price = self.costs.price(model, &u);
                if let Some(s) = self.order_stats.lock().unwrap().get_mut(&t) {
                    s.requests += 1;
                    s.usage += u;
                    match price {
                        Some(p) => s.usd += p,
                        None => s.unpriced = true,
                    }
                }
            }
        }
        let mut live = self.live.lock().unwrap();
        match ev {
            KernelEvent::Spawned { pid, ticket, group, .. } => {
                live.insert(*pid, Live { group: *group, ticket: *ticket, state: "ready".into(), doing: None });
            }
            KernelEvent::State { pid, state } => {
                if let Some(l) = live.get_mut(pid) {
                    l.state = state.clone();
                    if state == "ready" || state == "running" {
                        l.doing = None;
                    }
                }
            }
            KernelEvent::SyscallIssued { pid, summary, .. } => {
                if let Some(l) = live.get_mut(pid) {
                    let what = if summary.starts_with("llm ") { "thinking".to_string() } else { self.rel(summary) };
                    // Several calls in one step: show the first, count the rest.
                    l.doing = match l.doing.take() {
                        Some((d, t)) if !d.ends_with(" more") && d != what => Some((format!("{d} +1 more"), t)),
                        Some((d, t)) => Some((bump(&d), t)),
                        None => Some((what, Instant::now())),
                    };
                }
            }
            KernelEvent::Exited { pid, .. } => {
                live.remove(pid);
            }
            // The resident main agent outlives every order.
            KernelEvent::TicketServed { ticket, .. } => {
                live.retain(|_, l| l.ticket != *ticket || matches!(l.group, Group::Architecture | Group::Spec));
                if let Some(f) = self.on_served.get() {
                    f();
                }
            }
            _ => {}
        }
    }

    /// Show `msg` in the status row for a few seconds (printed instead when there is no status row).
    fn note(&self, msg: &str) {
        *self.note.lock().unwrap() = Some((msg.to_string(), Instant::now()));
    }

    /// The status row: a pinned plan question, or what each busy process is doing.
    fn status_line(&self) -> (String, bool) {
        let (b, d, r, y, cy) = (self.c(BOLD), self.c(DIM), self.c(RESET), self.c(YELLOW), self.c(CYAN));
        let note =
            self.note.lock().unwrap().as_ref().filter(|(_, at)| at.elapsed() < Duration::from_secs(3)).map(|(m, _)| format!("{y}{m}{r}  "));
        let mut note = note.unwrap_or_default();
        // The high-cost model warning pulses like a heartbeat (highlighted 1 s, plain 1 s).
        if let Some(seg) = self.costs.status_segment() {
            let on = (self.started.elapsed().as_millis() / 1000).is_multiple_of(2);
            let style = if on { format!("{b}{}", self.c("\x1b[7m")) } else { y.to_string() };
            note = format!("{style} {seg} {r}  {note}");
        }
        let spent = self.costs.meter.spent();
        let spent = if spent > 0.0 { format!("  {d}· ${spent:.4}{r}") } else { String::new() };
        if let Some((qid, ticket)) = self.question() {
            let rev = self.c("\x1b[7m");
            if let Some(cmd) = self.commands.lock().unwrap().get(&qid) {
                return (format!("{note}{b}{rev}? {cmd} — y allows · n denies · a allows all commands {r}"), false);
            }
            return (
                format!("{note}{b}{rev}? plan #{ticket} waits for you — y approves · n rejects · anything else is feedback {r}"),
                false,
            );
        }
        let live = self.live.lock().unwrap();
        let busy: Vec<(&Pid, &Live)> = live.iter().filter(|(_, l)| l.state == "running" || l.state == "blocked(io)").collect();
        // The main agent waiting for the next order is idle, not "waiting" on anything.
        let waiting =
            live.values().filter(|l| !(matches!(l.group, Group::Architecture | Group::Spec) && l.state == "blocked(recv)")).count()
                - busy.len();
        if busy.is_empty() {
            let rest = if waiting > 0 { format!("{waiting} waiting") } else { self.idle_text() };
            return (format!("{note}{d}{rest}{r}{spent}"), false);
        }
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame = FRAMES[(self.started.elapsed().as_millis() / 100) as usize % FRAMES.len()];
        let cells: Vec<String> = busy
            .iter()
            .map(|(pid, l)| {
                let (what, secs) = l.doing.as_ref().map(|(w, t)| (w.as_str(), t.elapsed().as_secs())).unwrap_or(("thinking", 0));
                let who = match l.group {
                    Group::Architecture => "main".to_string(),
                    Group::Spec => "spec".to_string(),
                    _ => format!("{pid}"),
                };
                format!("{who} {d}{what} · {secs}s{r}")
            })
            .collect();
        let more = if waiting > 0 { format!("  {d}│ {waiting} waiting{r}") } else { String::new() };
        (format!("{note}{cy}{frame}{r} {} working: {}{more}{spent}", busy.len(), cells.join(&format!("{d}  │  {r}"))), true)
    }

    fn render(&self, ev: &KernelEvent) -> Option<String> {
        let out = if let KernelEvent::TextDelta { pid, text, reasoning } = ev {
            if !self.verbose() {
                return None;
            }
            self.stream(*pid, *reasoning, text)?
        } else {
            // A model call ended: the rest of its stream comes before the ✓ line.
            let rest = match ev {
                KernelEvent::SyscallDone { pid, summary, .. } if summary.starts_with("llm ") => self.flush_stream(*pid),
                _ => None,
            };
            match (rest, self.render_line(ev)) {
                (Some(a), Some(b)) => format!("{a}\n{b}"),
                (a, b) => a.or(b)?,
            }
        };
        if !self.color {
            return Some(out); // not a terminal: leave long lines to whatever reads them
        }
        // Wrap ourselves, so a long line continues under its text rather than at column 0
        // (code fences and tables are left as they are).
        let width = terminal_width();
        let mut w = wrap::Wrapper::default();
        Some(out.split('\n').map(|l| w.line(l, width)).collect::<Vec<_>>().join("\n"))
    }

    fn render_line(&self, ev: &KernelEvent) -> Option<String> {
        let (b, d, r, g, red, y, cy) = (self.c(BOLD), self.c(DIM), self.c(RESET), self.c(GREEN), self.c(RED), self.c(YELLOW), self.c(CYAN));
        let first = |s: &str, n: usize| {
            s.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("STATUS:")).unwrap_or("").chars().take(n).collect::<String>()
        };
        Some(match ev {
            KernelEvent::TicketOpened { ticket, text, lane } => {
                self.open.lock().unwrap().insert(*ticket);
                self.orders.lock().unwrap().push((*ticket, text.clone(), None));
                self.lanes_of.lock().unwrap().insert(*ticket, *lane);
                let mark = if *lane == Group::Spec { "spec " } else { "" };
                format!("{cy}● {mark}order #{ticket}{r} {}", first(text, 100))
            }
            KernelEvent::Spawned { pid, group, title, .. } => {
                self.procs.lock().unwrap().insert(*pid, (*group, title.clone()));
                match group {
                    Group::Architecture => format!("  {d}+ {pid} main agent{r}"),
                    Group::Spec => format!("  {d}+ {pid} spec agent{r}"),
                    // Dispatched subagents and explorers say what they are in their title.
                    _ if ["answer:", "plan:", "explore:"].iter().any(|p| title.starts_with(p)) => format!("  {cy}+{r} {pid} {b}{title}{r}"),
                    _ => format!("  {cy}+{r} {pid} {} {b}{title}{r}", group.name()),
                }
            }
            KernelEvent::SyscallIssued { pid, idx, summary } if self.debug() && summary != "recv" => {
                // How long zilch took since this process's previous results came back.
                let gap = self.probe.gap(*pid, *idx).map(|g| format!(" (+{} after its results)", debug::ms(g))).unwrap_or_default();
                format!("    {d}{pid} ▸ {}{gap}{r}", self.rel(summary))
            }
            KernelEvent::SyscallIssued { pid, summary, .. } => {
                // A resident agent waiting for its next order (also after an idle wake) is never
                // news; model calls are only in /verbose.
                if summary == "recv" || (summary.starts_with("llm ") && !self.verbose()) {
                    return None;
                }
                // Quiet: a step's tool calls become one line once they have all finished.
                if !self.verbose() && is_tool(summary) {
                    let mut steps = self.steps.lock().unwrap();
                    let step = steps.entry(*pid).or_default();
                    step.inflight += 1;
                    return None;
                }
                format!("    {d}{pid} ▸ {}{r}", self.rel(summary))
            }
            KernelEvent::SyscallDone { summary, .. } if summary == "recv" || summary == "serve" => return None,
            KernelEvent::SyscallDone { pid, ok, summary, result, .. } if !self.verbose() && is_tool(summary) => {
                let failed = (!ok).then(|| format!("    {red}{pid} ✗ {}{r}{d}: {}{r}", self.rel(summary), first(&result.text(), 200)));
                let done = {
                    let mut steps = self.steps.lock().unwrap();
                    let step = steps.entry(*pid).or_default();
                    step.inflight = step.inflight.saturating_sub(1);
                    if *ok {
                        step.add(summary); // a failed call has its ✗ line and is not counted
                    }
                    if step.inflight == 0 {
                        steps.remove(pid).and_then(|s| s.text())
                    } else {
                        None
                    }
                }
                .map(|t| format!("    {d}{pid} ▸ {}{r}", self.rel(&t)));
                match (failed, done) {
                    (Some(a), Some(b)) => format!("{a}\n{b}"),
                    (a, b) => a.or(b)?,
                }
            }
            KernelEvent::SyscallDone { pid, idx, ok: false, summary, result } => {
                let dbg = self.probe.note(*pid, *idx).map(|n| format!(" · {n}")).unwrap_or_default();
                format!("    {red}{pid} ✗ {}{r}{d}{dbg}: {}{r}", self.rel(summary), first(&result.text(), 200))
            }
            KernelEvent::SyscallDone { pid, idx, summary, result, .. } if self.verbose() => {
                let text = result.text();
                let dbg = self.probe.note(*pid, *idx).map(|n| format!(" · {n}")).unwrap_or_default();
                let head = match result_note(summary, &text) {
                    Some(note) => format!("    {d}{pid} ✓ {} → {note}{dbg}{r}", self.rel(summary)),
                    None => format!("    {d}{pid} ✓ {}{dbg}{r}", self.rel(summary)),
                };
                // A tool's output under its line (model replies stream; reports have their own line).
                let tool = !(summary.starts_with("llm ") || ["dispatch", "explore", "submit_plan"].iter().any(|p| summary.starts_with(p)));
                match tool.then(|| tool_output(&text, d, r)).flatten() {
                    Some(out) => format!("{head}\n{out}"),
                    None => head,
                }
            }
            KernelEvent::Question { qid, ticket, text, command: true, .. } => {
                self.questions.lock().unwrap().push_back((*qid, *ticket));
                self.commands.lock().unwrap().insert(*qid, text.clone());
                format!("{y}?{r} {b}{text}{r}\n{y}answer:{r} y = allow · n = deny · a = allow all commands this session {d}(at the allow prompt){r}")
            }
            KernelEvent::Question { qid, ticket, text, .. } => {
                self.questions.lock().unwrap().push_back((*qid, *ticket));
                format!("{y}?{r} {b}plan for order #{ticket}{r}\n{text}\n{y}answer:{r} y = approve · n = reject · anything else = feedback to the main agent {d}(at the plan prompt){r}")
            }
            KernelEvent::Signal { pid, signal, reason } => format!("{y}⚠ {pid} {signal:?}{r}: {reason}"),
            KernelEvent::Exited { pid, ok, report } => {
                let (group, title) = self.procs.lock().unwrap().get(pid).cloned().unwrap_or((Group::Development, String::new()));
                if matches!(group, Group::Architecture | Group::Spec) {
                    return None; // the served ticket shows its report
                }
                let mark = if *ok && zilch_kernel::report_ok(report) { format!("{g}✓{r}") } else { format!("{red}✗{r}") };
                // The report is markdown: its first line with inline markup rendered (code, bold).
                let line = first(report, 160);
                let line = if self.color { markdown::inline(&line) } else { line };
                format!("  {mark} {pid} {title}{d}:{r} {line}")
            }
            KernelEvent::TicketServed { ticket, ok, summary } => {
                self.open.lock().unwrap().remove(ticket);
                *self.last_summary.lock().unwrap() = Some(summary.clone());
                if let Some(o) = self.orders.lock().unwrap().iter_mut().find(|o| o.0 == *ticket) {
                    o.2 = Some(if *ok { summary.clone() } else { format!("(not served) {summary}") });
                }
                self.questions.lock().unwrap().retain(|(_, t)| t != ticket);
                let width = terminal_width().saturating_sub(2).max(20);
                let body = markdown::Markdown::new(self.color).with_indent(2).text(summary.trim(), width);
                let mark = if self.lanes_of.lock().unwrap().get(ticket) == Some(&Group::Spec) { "spec " } else { "" };
                let stats = self.order_stats.lock().unwrap().remove(ticket);
                let wall = stats.as_ref().and_then(|s| s.started).map(|s| s.elapsed()).unwrap_or_default();
                let stats = stats.map(|s| format!(" · {}", s.line())).unwrap_or_default();
                let end = if *ok {
                    format!("{g}── {mark}order #{ticket} served{stats} ──{r}")
                } else {
                    format!("{red}── {mark}order #{ticket} not served{stats} ──{r}")
                };
                // --debug: where the order's time and tokens went.
                let report = self.probe.order_report(*ticket, wall);
                match report.filter(|_| self.debug()) {
                    Some(rep) => format!("{body}\n{end}\n{}", rep.lines().map(|l| format!("{d}{l}{r}")).collect::<Vec<_>>().join("\n")),
                    None => format!("{body}\n{end}"),
                }
            }
            KernelEvent::Notice { text, .. } => format!("  {d}{text}{r}"),
            KernelEvent::Lagged { missed } => format!("  {d}(display missed {missed} events; the log has them){r}"),
            KernelEvent::Told { .. }
            | KernelEvent::State { .. }
            | KernelEvent::TextDelta { .. }
            | KernelEvent::Usage { .. }
            | KernelEvent::RequestParts { .. }
            | KernelEvent::SyscallDone { .. } => return None,
        })
    }
}

/// A width set by the client (the daemon has no terminal of its own; 0 = ask the terminal).
static WIDTH: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn set_width(w: usize) {
    if w > 0 {
        WIDTH.store(w, std::sync::atomic::Ordering::SeqCst);
    }
}

fn terminal_width() -> usize {
    let w = WIDTH.load(std::sync::atomic::Ordering::SeqCst);
    if w > 0 {
        return w;
    }
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        #[cfg(target_os = "linux")]
        let request = libc::TIOCGWINSZ as libc::Ioctl;
        #[cfg(not(target_os = "linux"))]
        let request = libc::TIOCGWINSZ;
        if libc::ioctl(1, request, &mut ws) == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).unwrap_or(100)
}

fn open_log(log: Option<PathBuf>, no_log: bool) -> Option<(PathBuf, std::fs::File)> {
    if no_log {
        return None;
    }
    let path = log.or_else(|| {
        let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        dirs::data_dir().map(|d| d.join("zilch").join("logs").join(format!("os-{secs}.jsonl")))
    })?;
    if let Some(dir) = path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let file = std::fs::OpenOptions::new().create(true).append(true).open(&path).ok()?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600));
    }
    Some((path, file))
}

fn ps(k: &KernelHandle) -> String {
    let st = k.stats();
    let groups: Vec<String> = st
        .groups
        .iter()
        .map(|(g, used, quota)| match quota {
            Some(q) => format!("{} {}k/{}k", g.name(), used / 1000, q / 1000),
            None => format!("{} {}k", g.name(), used / 1000),
        })
        .collect();
    let tokens = if groups.is_empty() { "0".to_string() } else { groups.join(", ") };
    // What is happening (calls in flight, busy processes) first; the kernel internals after.
    let queued: usize = st.queued.iter().sum();
    let ready = if queued > 0 { format!(" · ready for a step {queued} {:?}", st.queued) } else { String::new() };
    let header = format!(
        "in flight: {} model call(s), {} command(s), {} tool call(s) · processes {} busy, {} waiting · step workers {}/{}{ready} · tokens {tokens}",
        st.model_calls, st.commands, st.tool_calls, st.busy, st.waiting, st.busy_workers, st.workers
    );
    let rows = k.ps();
    if rows.is_empty() {
        return format!("{header}\nNo processes (place an order).");
    }
    let mut s = format!("{header}\n");
    s += &format!("{:>4} {:>6} {:<10} {:<16} {:>5} {:>8}  {}\n", "PID", "TICKET", "GROUP", "STATE", "STEPS", "TOKENS", "DOING / TITLE");
    for p in rows {
        let doing = if p.doing.is_empty() { p.title.clone() } else { format!("{} — {}", p.doing, p.title) };
        s.push_str(&format!(
            "{:>4} {:>6} {:<10} {:<16} {:>5} {:>8}  {}\n",
            p.pid,
            p.ticket,
            p.group.name(),
            p.state,
            p.steps,
            p.tokens,
            doing
        ));
    }
    s.trim_end().to_string()
}

const HELP: &str = "\
Type an order to place it. While a plan waits for you: y approves, n rejects, anything else is feedback (an empty line does nothing).
@<pid> text  more input for one process (read before its next model call)
@main text   more input for the main agent (when it is idle, this is simply the next order)
@developer text   the same for every live subagent of the newest order
/spec        switch to the spec lane (requirements → spec.md, design.md, plan.md, released versions)
/code        switch back to the code lane (the main agent and its subagents)
/status      which lane you type into, what runs, and the spec documents' state
/ps          the process table (pid, group, state, what it is doing)
/kill <pid>  stop one process (its task fails; dependents are blocked)
/cancel      cancel the newest open order
/usage       tokens and USD per agent and model, session total
/models      the model of the main agent and the subagents
/model <main|developer> <model> [provider]   switch for this session (`default` switches back)
/ack         silence the high-cost model warning until its next step
/prompt [main|answerer|planner|developer|spec]   the system prompt
/sessions    this directory's saved sessions (--resume <id> continues one)
/recap       this session's orders and the last answer
/clear       start a fresh session (new main agent, empty context; the old one stays resumable)
/about       version, repository, workspace, session file, config files
/redraw      clear the screen and re-render the last answer at the current width
/verbose     also show model calls, results and the models' streamed text and reasoning
/debug       verbose plus timings, tokens, cost and sizes per call, and a breakdown per order
/quiet       back to the short log (the default): one line per step, e.g. read 2 files, edited a.rs +3 -1
/help        this help
/exit        quit (Ctrl-D)
/detach      (daemon sessions, zilch attach) leave; the session keeps running · /stop ends it
Tab completes commands, their arguments (groups, models, pids) and @ mentions.
Ctrl-C cancels the newest open order.";

/// A tool call (as opposed to a model call or the kernel's own syscalls, which have lines of
/// their own).
fn is_tool(summary: &str) -> bool {
    !["llm ", "recv", "serve", "dispatch", "explore", "submit_plan", "send "].iter().any(|p| summary.starts_with(p))
}

/// The quiet log's account of one step's tool calls (a process's calls issued together):
/// `read ci.yml, edited ledger/parse.py +3 -1, ran 2 shell commands`.
#[derive(Default, Debug)]
struct Step {
    /// Calls issued and not yet finished.
    inflight: u32,
    reads: BTreeSet<String>,
    searches: u32,
    commands: u32,
    git: u32,
    edits: BTreeSet<String>,
    added: u64,
    removed: u64,
    writes: Vec<(String, u64)>,
    fetches: u32,
    web_searches: u32,
    other: Vec<String>,
}

impl Step {
    /// Count one call from its summary (`read_file(ci.yml, all)`, `edit_file(a.rs, +3 -1)` …).
    fn add(&mut self, summary: &str) {
        let (tool, detail) = summary.split_once('(').map(|(t, d)| (t, d.strip_suffix(')').unwrap_or(d))).unwrap_or((summary, ""));
        let path = detail.split([',', ':']).next().unwrap_or(detail).trim().to_string();
        let num =
            |prefix: char| detail.split_whitespace().find_map(|w| w.strip_prefix(prefix)).and_then(|n| n.parse::<u64>().ok()).unwrap_or(0);
        match tool {
            "read_file" => {
                self.reads.insert(path);
            }
            "grep" | "glob" | "find_definition" | "repo_map" => self.searches += 1,
            "bash" => self.commands += 1,
            "git" => self.git += 1,
            "edit_file" | "edit_doc" => {
                self.edits.insert(path);
                self.added += num('+');
                self.removed += num('-');
            }
            "write_file" | "write_doc" => {
                let lines = detail.rsplit(", ").next().and_then(|l| l.strip_suffix(" lines")).and_then(|n| n.parse().ok()).unwrap_or(0);
                self.writes.push((path, lines));
            }
            "web_fetch" | "fetch_source" => self.fetches += 1,
            "web_search" => self.web_searches += 1,
            _ => self.other.push(summary.to_string()),
        }
    }

    fn text(&self) -> Option<String> {
        let n = |k: u32, one: &str, many: &str| format!("{k} {}", if k == 1 { one } else { many });
        let mut parts = Vec::new();
        match self.edits.len() {
            0 => {}
            1 => parts.push(format!("edited {} +{} -{}", self.edits.first().unwrap(), self.added, self.removed)),
            k => parts.push(format!("edited {k} files +{} -{}", self.added, self.removed)),
        }
        match self.writes.as_slice() {
            [] => {}
            [(p, lines)] => parts.push(format!("wrote {p} ({lines} lines)")),
            w => parts.push(format!("wrote {} files ({} lines)", w.len(), w.iter().map(|(_, l)| l).sum::<u64>())),
        }
        match self.reads.len() {
            0 => {}
            1 => parts.push(format!("read {}", self.reads.first().unwrap())),
            k => parts.push(format!("read {k} files")),
        }
        if self.searches > 0 {
            parts.push(format!("searched {}", if self.searches == 1 { "once".into() } else { format!("{} times", self.searches) }));
        }
        if self.commands > 0 {
            parts.push(format!("ran {}", n(self.commands, "shell command", "shell commands")));
        }
        if self.git > 0 {
            parts.push(format!("ran {}", n(self.git, "git command", "git commands")));
        }
        if self.fetches > 0 {
            parts.push(format!("fetched {}", n(self.fetches, "page", "pages")));
        }
        if self.web_searches > 0 {
            parts.push(format!(
                "searched the web {}",
                if self.web_searches == 1 { "once".into() } else { format!("{} times", self.web_searches) }
            ));
        }
        parts.extend(self.other.iter().cloned());
        (!parts.is_empty()).then(|| parts.join(", "))
    }
}

/// Lines of a tool's output shown under its ✓ line in /verbose.
const OUTPUT_LINES: usize = 12;

/// A tool's output for /verbose: the first lines, dim and indented under its ✓ line, with a note of
/// what was left out (None for empty output).
fn tool_output(text: &str, d: &str, r: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
    if lines.is_empty() {
        return None;
    }
    let mut out: Vec<String> = lines
        .iter()
        .take(OUTPUT_LINES)
        .map(|l| {
            // read_file's numbered lines (`    12\tcode`) read better as `12  code`.
            let l = match l.split_once('\t') {
                Some((n, rest)) if n.trim().bytes().all(|b| b.is_ascii_digit()) && !n.trim().is_empty() => format!("{}  {rest}", n.trim()),
                _ => l.replace('\t', "  "),
            };
            let l: String = l.chars().take(300).collect();
            format!("        {d}│ {l}{r}")
        })
        .collect();
    if lines.len() > OUTPUT_LINES {
        out.push(format!("        {d}│ … {} more lines{r}", lines.len() - OUTPUT_LINES));
    }
    Some(out.join("\n"))
}

/// What a finished read or search brought back, for its ✓ line: `84 lines`, `lines 140-419 of
/// 2223`, `outline + first 40 of 2223 lines`, `unchanged, not resent`, `12 matches`, `3 files`.
fn result_note(summary: &str, text: &str) -> Option<String> {
    let tool = summary.split('(').next().unwrap_or("");
    let footer = |prefix: &str| {
        text.lines()
            .rev()
            .find_map(|l| l.strip_prefix(prefix))
            .and_then(|r| r.split_whitespace().next())
            .and_then(|n| n.parse::<usize>().ok())
    };
    match tool {
        "read_file" => {
            if text.contains("unchanged since you read them") {
                return Some("unchanged, not resent".into());
            }
            if let Some((head, _)) = text.split_once(" lines. Definitions") {
                let total = head.rsplit(' ').next().unwrap_or("?");
                return Some(format!("outline + first lines of {total}"));
            }
            if text == "(empty file)" {
                return Some("empty file".into());
            }
            // "[lines 140-419 of 2223; the rest: …]" or "[lines 140-419 of 2223]" after a range.
            if let Some(range) = text.lines().rev().find_map(|l| l.strip_prefix("[lines ")) {
                let range = range.split([';', ']']).next().unwrap_or(range);
                return Some(format!("lines {range}"));
            }
            let n = text
                .lines()
                .filter(|l| l.trim_start().split('\t').next().is_some_and(|d| !d.is_empty() && d.bytes().all(|b| b.is_ascii_digit())))
                .count();
            Some(format!("{n} lines"))
        }
        "grep" | "glob" => {
            let (none, one, many) = if tool == "grep" { ("No matches", "match", "matches") } else { ("No files found", "file", "files") };
            if text.starts_with(none) {
                return Some(format!("no {many}"));
            }
            // Shown lines, plus "[N more … not shown]".
            let shown = text.lines().filter(|l| !l.starts_with('[') && !l.trim().is_empty()).count();
            let n = shown + footer("[").unwrap_or(0);
            Some(format!("{n} {}", if n == 1 { one } else { many }))
        }
        _ => None,
    }
}

/// `text` with paths inside one of the `workspaces` (spellings of the same directory) made
/// relative to it: `/ws/src/a.rs` → `src/a.rs`, a bare `/ws` → `.`. A path that merely starts
/// with the same letters (`/ws2/x`, `/mnt/ws/x`) is left alone, as is everything outside.
fn rel_to(workspaces: &[String], text: &str) -> String {
    let part_of_path = |c: char| c.is_alphanumeric() || matches!(c, '/' | '.' | '-' | '_');
    let mut out = text.to_string();
    for ws in workspaces.iter().filter(|w| w.len() > 1) {
        let ws = ws.trim_end_matches('/');
        let mut result = String::with_capacity(out.len());
        let mut rest = out.as_str();
        while let Some(i) = rest.find(ws) {
            let before = rest[..i].chars().next_back();
            let after = rest[i + ws.len()..].chars().next();
            result.push_str(&rest[..i]);
            if before.is_some_and(part_of_path) {
                result.push_str(ws);
            } else if after == Some('/') {
                rest = &rest[i + ws.len() + 1..];
                continue;
            } else if after.is_some_and(part_of_path) {
                result.push_str(ws);
            } else {
                result.push('.');
            }
            rest = &rest[i + ws.len()..];
        }
        result.push_str(rest);
        out = result;
    }
    out
}

fn main() {
    // Sandbox helper mode: before any thread exists (unshare(CLONE_NEWUSER) refuses
    // multithreaded processes).
    let mut args = std::env::args();
    if args.nth(1).as_deref() == Some("__sandbox") {
        #[cfg(target_os = "linux")]
        zilch_drivers::tools::sandbox_linux::helper_main(args);
        #[cfg(not(target_os = "linux"))]
        std::process::exit(126);
    }
    if let Ok(exe) = std::env::current_exe() {
        zilch_drivers::tools::sandbox::set_helper(exe, vec!["__sandbox".into()]);
    }
    match run(Cli::parse()) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
    }
}

fn run(cli: Cli) -> anyhow::Result<i32> {
    let cwd = std::env::current_dir()?;
    let opts = |workspace: PathBuf, color: bool, editor_link: bool| session::SessionOpts {
        workspace,
        model: cli.model.clone(),
        architect_model: cli.architect_model.clone(),
        developer_model: cli.developer_model.clone(),
        web: cli.web,
        approval: if cli.yes || cli.yolo { Some("auto".into()) } else { cli.approval.clone() },
        workers: cli.workers,
        log: cli.log.clone(),
        no_log: cli.no_log,
        r#continue: cli.r#continue,
        resume: cli.resume.clone(),
        no_session: cli.no_session,
        verbose: cli.verbose || cli.debug,
        debug: cli.debug,
        color,
        editor_link,
    };
    let tty = std::io::stdout().is_terminal();
    match &cli.cmd {
        Some(Sub::Daemon { action }) => {
            return match action.as_deref().unwrap_or("run") {
                "run" => {
                    if daemon::run()? {
                        Ok(0)
                    } else {
                        eprintln!("a zilch daemon already runs ({})", ipc::daemon_endpoint());
                        Ok(1)
                    }
                }
                "start" => daemon::start().map(|_| 0),
                "stop" => {
                    daemon::request(&daemon::Up::Shutdown)?;
                    println!("zilch daemon stopped");
                    Ok(0)
                }
                "status" => {
                    if !daemon::running() {
                        println!("no zilch daemon is running");
                        return Ok(1);
                    }
                    let sessions = match daemon::request(&daemon::Up::List)? {
                        daemon::Down::Ok { sessions, .. } => sessions,
                        _ => vec![],
                    };
                    println!("zilch daemon running ({}), {} session(s)", ipc::daemon_endpoint(), sessions.len());
                    Ok(0)
                }
                other => anyhow::bail!("zilch daemon {other}: use run | start | stop | status"),
            };
        }
        Some(Sub::New { workspace, name }) => {
            let ws = workspace.clone().unwrap_or_else(|| cwd.clone());
            let ws = if ws.is_absolute() { ws } else { cwd.join(ws) };
            match daemon::request(&daemon::Up::New { opts: opts(ws, true, true), name: name.clone() })? {
                daemon::Down::Ok { id, .. } => {
                    let id = id.unwrap_or_default();
                    println!("{id}\n(zilch attach {id} to talk to it; zilch -p \"…\" --session {id} for one order)");
                }
                other => anyhow::bail!("unexpected answer: {other:?}"),
            }
            return Ok(0);
        }
        Some(Sub::Ls) => {
            if let daemon::Down::Ok { sessions, .. } = daemon::request(&daemon::Up::List)? {
                println!("{}", daemon::list_text(&sessions));
            }
            return Ok(0);
        }
        Some(Sub::Stop { id }) => {
            daemon::request(&daemon::Up::Stop { id: id.clone() })?;
            println!("session {id} stopped (its conversation stays resumable)");
            return Ok(0);
        }
        Some(Sub::Attach { id }) => {
            // Attach to the session, or this directory's newest one, or a new one for it.
            let up = daemon::Up::Attach {
                id: id.clone(),
                workspace: Some(cwd.clone()),
                create: id.is_none().then(|| opts(cwd.clone(), true, true)),
                replay: 200,
                width: terminal_width(),
            };
            let (link, sid, banner) = daemon::Remote::attach(&up)?;
            let leave = front::repl(link.clone(), &banner, 200, true)?;
            link.detach();
            match leave {
                front::Leave::Stop => {
                    daemon::request(&daemon::Up::Stop { id: sid.clone() })?;
                    println!("session {sid} stopped");
                }
                _ => println!("detached; session {sid} keeps running (zilch attach {sid})"),
            }
            return Ok(0);
        }
        None => {}
    }
    if cli.doctor {
        let ov = Overrides {
            model: cli.model.clone(),
            architect_model: cli.architect_model.clone(),
            developer_model: cli.developer_model.clone(),
            web: cli.web.then_some(true),
        };
        let cfg = Config::load(&cwd, &ov)?;
        let meter = zilch_drivers::llm::catalog::CostMeter::new(cfg.max_session_cost);
        *meter.catalog.write().unwrap() = cfg.discover_catalog();
        return Ok(doctor(&cfg, &meter, &cwd));
    }
    if cli.sessions {
        let dir = zilch_drivers::session::default_root().map(|r| zilch_drivers::session::dir_for(&r, &cwd));
        match dir {
            Some(d) if !cli.no_session => println!("{}", sessions_list(&d, None)),
            _ => println!("Sessions are off."),
        }
        return Ok(0);
    }
    // One order into a daemon session (created for this directory if missing).
    if let (Some(order), Some(id)) = (&cli.prompt, &cli.session) {
        let up = daemon::Up::Attach {
            id: Some(id.clone()),
            workspace: None,
            create: Some(opts(cwd.clone(), true, true)),
            replay: 0,
            width: terminal_width(),
        };
        let (link, _, banner) = daemon::Remote::attach(&up)?;
        let code = front::one_order(link.as_ref(), &banner, order)?;
        link.detach();
        return Ok(code);
    }
    if cli.session.is_some() {
        anyhow::bail!("--session goes with -p (for a REPL on a daemon session: zilch attach <id>)");
    }
    // In this process.
    let session = session::Session::open(&opts(cwd.clone(), tty, cli.prompt.is_none()))?;
    let banner = session.banner.clone();
    let link = Arc::new(session::Local::new(session));
    if let Some(order) = &cli.prompt {
        let code = front::one_order(link.as_ref(), &banner, order)?;
        link.stop("served");
        return Ok(code);
    }
    front::repl(link.clone(), &banner, 0, false)?;
    let d = if tty { DIM } else { "" };
    let r = if tty { RESET } else { "" };
    for line in link.stop("the guest left") {
        println!("{d}{line}{r}");
    }
    Ok(0)
}

/// "x +1 more" → "x +2 more".
fn bump(d: &str) -> String {
    match d.rsplit_once(" +").and_then(|(head, tail)| tail.strip_suffix(" more").and_then(|n| n.parse::<u32>().ok()).map(|n| (head, n))) {
        Some((head, n)) => format!("{head} +{} more", n + 1),
        None => d.to_string(),
    }
}

/// A model with its price, for /models and /model.
fn priced(meter: &zilch_drivers::llm::catalog::CostMeter, b: &zilch_ipc::ModelBinding) -> String {
    match meter.info(&b.key()) {
        Some(zilch_drivers::llm::catalog::ModelInfo { input_per_mtok: Some(i), output_per_mtok: Some(o), .. }) => {
            format!("{} (${i}/${o} per 1M tokens in/out)", b.key())
        }
        _ => b.key(),
    }
}

/// /models: what each group uses now (and what it was configured with, if switched).
fn models_list(
    book: &Book,
    meter: &zilch_drivers::llm::catalog::CostMeter,
    configured: &(zilch_ipc::ModelBinding, zilch_ipc::ModelBinding),
) -> String {
    [("main", Group::Architecture, &configured.0), ("developer", Group::Development, &configured.1)]
        .iter()
        .map(|(name, g, conf)| {
            let now = book.binding(*g);
            let note = if &now != *conf { format!("  (switched; configured: {})", conf.key()) } else { String::new() };
            format!("{name:<10} {}{note}", priced(meter, &now))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// /model <main|developer> <model> [provider] — or `default` to switch back.
fn model_cmd(
    arg: &str,
    book: &Book,
    meter: &zilch_drivers::llm::catalog::CostMeter,
    providers: &[String],
    configured: &(zilch_ipc::ModelBinding, zilch_ipc::ModelBinding),
) -> String {
    let parts: Vec<&str> = arg.split_whitespace().collect();
    let (who, model, provider) = match parts.as_slice() {
        [] => return models_list(book, meter, configured),
        [w, m] => (*w, *m, None),
        [w, m, p] => (*w, *m, Some(*p)),
        _ => return "usage: /model <main|developer> <model> [provider]   ·   /model <main|developer> default".into(),
    };
    let (group, conf) = match zilch_ipc::Target::parse(who) {
        Some(zilch_ipc::Target::Group(Group::Architecture)) => (Group::Architecture, &configured.0),
        Some(zilch_ipc::Target::Group(Group::Development)) => (Group::Development, &configured.1),
        _ => return format!("unknown group '{who}' (main, developer)"),
    };
    if model == "default" {
        book.set_binding(group, conf.clone());
        return format!("{who} → {} (configured model)", conf.key());
    }
    let mut b = book.binding(group);
    if let Some(p) = provider {
        if !providers.iter().any(|x| x == p) {
            return format!("unknown or unusable provider '{p}' (usable: {})", providers.join(", "));
        }
        b.provider = p.to_string();
    }
    b.model = model.to_string();
    // A model the provider does not list would fail with a 400 in the middle of a task.
    let prefix = format!("{}/", b.provider);
    let listed: Vec<String> = meter.catalog.read().unwrap().keys().filter(|k| k.starts_with(&prefix)).cloned().collect();
    if !listed.is_empty() && !listed.contains(&b.key()) {
        return format!("'{model}' is not in provider '{}''s model list", b.provider);
    }
    let shown = priced(meter, &b);
    book.set_binding(group, b);
    format!("{who} → {shown}, from its next model call")
}

/// /about: version, project, where things are.
fn about(cwd: &std::path::Path, sources: &[PathBuf], session: Option<&PathBuf>) -> String {
    let mut lines = vec![
        format!("zilch {} — {}", env!("CARGO_PKG_VERSION"), env!("CARGO_PKG_DESCRIPTION")),
        format!("  repository  {}", env!("CARGO_PKG_REPOSITORY")),
        format!("  author      {}", env!("CARGO_PKG_AUTHORS").replace(':', ", ")),
        format!("  license     {}", env!("CARGO_PKG_LICENSE")),
        format!("  platform    {}/{}", std::env::consts::OS, std::env::consts::ARCH),
        format!("  workspace   {}", cwd.display()),
        format!("  session     {}", session.map_or("not saved (--no-session)".into(), |p| p.display().to_string())),
    ];
    if sources.is_empty() {
        lines.push("  config      built-in defaults + environment (no config file found)".into());
    }
    for (i, s) in sources.iter().enumerate() {
        lines.push(format!("{}{}", if i == 0 { "  config      " } else { "              " }, s.display()));
    }
    lines.join("\n")
}

/// How long ago a millisecond timestamp was, roughly.
fn ago(ms: u64) -> String {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let mins = now.saturating_sub(ms) / 60_000;
    match mins {
        0 => "just now".into(),
        m if m < 60 => format!("{m} min ago"),
        m if m < 48 * 60 => format!("{} h ago", m / 60),
        m => format!("{} days ago", m / (24 * 60)),
    }
}

/// --sessions and /sessions: this directory's saved sessions, newest first.
fn sessions_list(dir: &std::path::Path, current: Option<&PathBuf>) -> String {
    let list = zilch_drivers::session::list(dir);
    if list.is_empty() {
        return "No saved sessions in this directory.".into();
    }
    let mut s = String::new();
    for e in list {
        let mark = if Some(&e.path) == current { "▸" } else { " " };
        let last: String = e.orders.last().map(|t| t.split_once(": ").map(|x| x.1).unwrap_or(t)).unwrap_or("").chars().take(60).collect();
        s.push_str(&format!("{mark} {}  {:<12} {:>3} order(s)  {last}\n", e.id, ago(e.updated_ms), e.orders.len()));
    }
    s.push_str("continue one with: zilch --resume <id>   (or -c for the newest)");
    s
}

/// /recap: this session's orders and the last answer.
fn recap(view: &View, session: Option<&PathBuf>) -> String {
    let orders = view.orders.lock().unwrap();
    let id = session.and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "(not saved)".into());
    if orders.is_empty() {
        return format!("Session {id}: no orders yet.");
    }
    let mut s = format!("Session {id}: {} order(s)\n", orders.len());
    for (n, text, answer) in orders.iter() {
        let status = match answer {
            None => "open".to_string(),
            Some(a) if a.starts_with("(earlier session)") => "earlier session".to_string(),
            Some(a) => a.lines().map(str::trim).find(|l| !l.is_empty()).unwrap_or("").chars().take(70).collect(),
        };
        let text: String = text.chars().take(60).collect();
        s.push_str(&format!("  #{n} {text}  → {status}\n"));
    }
    if let Some(last) = view.last_summary.lock().unwrap().as_ref() {
        let width = terminal_width().saturating_sub(2).max(20);
        s.push_str(&format!("\nLast answer:\n{}", markdown::Markdown::new(view.color).with_indent(2).text(last.trim(), width)));
    }
    s.trim_end().to_string()
}

/// The spec documents' state, one paragraph.
fn spec_status(dir: &std::path::Path) -> String {
    zilch_drivers::tools::spec_tools::SpecStore::new(dir.to_path_buf()).status().describe(dir)
}

/// --doctor: the resolved setup, then one tiny request to every distinct model.
fn doctor(cfg: &Config, meter: &zilch_drivers::llm::catalog::CostMeter, cwd: &std::path::Path) -> i32 {
    println!("zilch {} doctor — workspace {}", env!("CARGO_PKG_VERSION"), cwd.display());
    if cfg.sources.is_empty() {
        println!("  config      built-in defaults + environment (no config file found)");
    }
    for (i, s) in cfg.sources.iter().enumerate() {
        println!("{}{}", if i == 0 { "  config      " } else { "              " }, s.display());
    }
    for w in &cfg.warnings {
        println!("  warning     {w}");
    }
    println!("  providers   {}", cfg.usable_providers().join(", "));
    println!("  main/spec   {}", priced(meter, &cfg.architect));
    println!("  subagents   {}", priced(meter, &cfg.developer));
    let (_, sandbox) = zilch_drivers::tools::sandbox::status(cfg.policy.bash_sandbox, cfg.policy.bash.network);
    println!("  commands    {sandbox}");
    let caps = [cfg.max_session_tokens.map(|t| format!("{t} tokens")), cfg.max_session_cost.map(|c| format!("${c}"))]
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    println!("  caps        {}", if caps.is_empty() { "none".to_string() } else { caps.join(", ") });
    println!("  spec dir    {}", cfg.spec_dir.display());
    println!(
        "  sessions    {}",
        zilch_drivers::session::default_root()
            .map_or("unavailable".into(), |r| zilch_drivers::session::dir_for(&r, cwd).display().to_string())
    );
    let backends = cfg.backends();
    let mut seen = std::collections::BTreeSet::new();
    let mut failed = false;
    for b in [&cfg.architect, &cfg.developer] {
        if !seen.insert(b.key()) {
            continue;
        }
        let listed = {
            let prefix = format!("{}/", b.provider);
            let cat = meter.catalog.read().unwrap();
            let any = cat.keys().any(|k| k.starts_with(&prefix));
            match (any, cat.contains_key(&b.key())) {
                (false, _) => "",
                (true, true) => "",
                (true, false) => " (not in the provider's model list!)",
            }
        };
        let Some(backend) = backends.get(&b.provider) else {
            println!("  ✗ {} — provider not usable (missing key?)", b.key());
            failed = true;
            continue;
        };
        let req = zilch_ipc::ChatRequest {
            binding: zilch_ipc::ModelBinding { max_tokens: Some(256), ..b.clone() },
            messages: vec![zilch_ipc::Message::user("Reply with the single word OK.")],
            tools: vec![],
            session: Some(format!("zilch-doctor-{}", std::process::id())),
        };
        let start = Instant::now();
        match zilch_drivers::llm::complete(backend.as_ref(), &req, &zilch_kernel::CancelToken::new(), &|| {}, |_, _| {}) {
            Ok((msg, _, _)) => {
                let reply: String = msg.text().trim().chars().take(30).collect();
                let reply = if reply.is_empty() { "(empty reply — the call works)".to_string() } else { reply };
                println!("  ✓ {} answered in {:.1}s: {reply}{listed}", b.key(), start.elapsed().as_secs_f64());
            }
            Err(e) => {
                println!("  ✗ {} — {e:#}{listed}", b.key());
                failed = true;
            }
        }
    }
    if failed {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod rel_tests {
    #[test]
    fn paths_inside_the_workspace_are_shown_relative_others_in_full() {
        let ws = vec!["/pst/conf".to_string()];
        let r = |s: &str| super::rel_to(&ws, s);
        assert_eq!(r("read_file(/pst/conf/flake.nix:1-40)"), "read_file(flake.nix:1-40)");
        assert_eq!(r("bash(cd /pst/conf && sed -n '1,9p' flake.nix)"), "bash(cd . && sed -n '1,9p' flake.nix)");
        assert_eq!(r("write_file(/tmp/spots.txt, 3 lines)"), "write_file(/tmp/spots.txt, 3 lines)");
        assert_eq!(r("glob(/pst/config/*.nix)"), "glob(/pst/config/*.nix)", "a sibling with a longer name");
        assert_eq!(r("read_file(/mnt/pst/conf/a)"), "read_file(/mnt/pst/conf/a)", "the same letters inside another path");
        assert_eq!(r("grep(x in src)"), "grep(x in src)");
    }
}

#[cfg(test)]
mod result_note_tests {
    use super::result_note;

    #[test]
    fn a_finished_read_or_search_says_what_came_back() {
        let whole = "     1\tname: ci\n     2\ton: push\n     3\tjobs:\n";
        assert_eq!(result_note("read_file(ci.yml, all)", whole).as_deref(), Some("3 lines"));
        let part = "   140\ta\n   141\tb\n[lines 140-141 of 2223; the rest: offset=142. Several ranges: …]\n";
        assert_eq!(result_note("read_file(f.nix:140-141)", part).as_deref(), Some("lines 140-141 of 2223"));
        let outline = "big.rs: 1601 lines. Definitions (line: signature):\n   100: pub fn f()\n";
        assert_eq!(result_note("read_file(big.rs, all)", outline).as_deref(), Some("outline + first lines of 1601"));
        let same = "[a.rs lines 1-40: unchanged since you read them earlier in this conversation. …]";
        assert_eq!(result_note("read_file(a.rs:1-40)", same).as_deref(), Some("unchanged, not resent"));
        assert_eq!(result_note("grep(fn main)", "src/a.rs:1: fn main() {}\nsrc/b.rs:9: fn main() {}").as_deref(), Some("2 matches"));
        assert_eq!(result_note("grep(x)", "a:1: x\n[12 more matches not shown]").as_deref(), Some("13 matches"));
        assert_eq!(result_note("grep(zzz)", "No matches").as_deref(), Some("no matches"));
        assert_eq!(result_note("glob(**/*.toml)", "Cargo.toml").as_deref(), Some("1 file"));
        assert_eq!(result_note("bash(ls)", "x"), None);
    }
}

#[cfg(test)]
mod tool_output_tests {
    #[test]
    fn verbose_shows_the_first_lines_of_a_tools_output() {
        let read: String = (1..=20).map(|i| format!("{i:>6}\tline {i}\n")).collect();
        let out = super::tool_output(&read, "", "").unwrap();
        let rows: Vec<&str> = out.lines().collect();
        assert_eq!(rows.len(), 13, "{out}");
        assert_eq!(rows[0], "        │ 1  line 1");
        assert_eq!(rows[12], "        │ … 8 more lines");
        assert_eq!(super::tool_output("ok", "", "").unwrap(), "        │ ok");
        assert!(super::tool_output("  \n", "", "").is_none());
    }

    #[test]
    fn idle_durations_read_short() {
        assert_eq!(super::short_duration(std::time::Duration::from_secs(9)), "9s");
        assert_eq!(super::short_duration(std::time::Duration::from_secs(312)), "5m12s");
    }
}

#[cfg(test)]
mod step_tests {
    use super::Step;

    #[test]
    fn a_steps_tool_calls_read_as_one_short_line() {
        let mut s = Step::default();
        for c in ["read_file(ci.yml, all)", "edit_file(ledger/parse.py, +3 -1)", "bash(cargo test)", "bash(ls)", "grep(fn main in src)"] {
            s.add(c);
        }
        assert_eq!(s.text().unwrap(), "edited ledger/parse.py +3 -1, read ci.yml, searched once, ran 2 shell commands");
        let mut s = Step::default();
        for c in [
            "read_file(a.rs:1-40)",
            "read_file(a.rs:41-80)",
            "read_file(b.rs, all)",
            "edit_file(a.rs, +2 -0)",
            "edit_file(b.rs, +1 -4 in 2 edits)",
        ] {
            s.add(c);
        }
        assert_eq!(s.text().unwrap(), "edited 2 files +3 -4, read 2 files", "two ranges of a.rs are one file");
        let mut s = Step::default();
        s.add("write_file(report.md, 63 lines)");
        s.add("web_search(rust serde)");
        assert_eq!(s.text().unwrap(), "wrote report.md (63 lines), searched the web once");
        assert!(Step::default().text().is_none());
    }
}
