//! `zilch`: the waiter. Takes the guest's order, hands it to the kitchen (the kernel), shows
//! what the kitchen does as it happens, relays the plan for approval, and serves the result.
//! It never waits on the agents: the kernel runs on its own threads and reports on the event bus.

mod completion;
mod costs;
mod daemon;
mod front;
mod ipc;
#[allow(dead_code)] // ported whole from zilch-cli; streaming parts come later
mod markdown;
mod session;
mod status;
#[allow(dead_code)]
mod wrap;

use clap::Parser;
use std::collections::{BTreeMap, HashMap, HashSet};
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
    /// Show every model call and syscall result.
    #[arg(short, long, global = true)]
    verbose: bool,
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
    verbose: bool,
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

    /// The oldest plan waiting for an answer.
    fn question(&self) -> Option<(u64, TicketId)> {
        self.questions.lock().unwrap().front().copied()
    }

    fn latest_open(&self) -> Option<TicketId> {
        self.open.lock().unwrap().iter().max().copied()
    }

    /// Keep the status row's picture of live processes up to date.
    fn track(&self, ev: &KernelEvent) {
        if let KernelEvent::RequestParts { parts, .. } = ev {
            self.costs.record_parts(parts);
        }
        if let KernelEvent::Usage { pid, model, prompt_tokens, completion_tokens, cached_tokens } = ev {
            let who = match self.procs.lock().unwrap().get(pid).map(|(g, _)| *g) {
                Some(Group::Architecture) => "main",
                Some(Group::Spec) => "spec",
                _ => "subagents",
            };
            self.costs.record(
                who,
                model,
                zilch_ipc::Usage { prompt_tokens: *prompt_tokens, completion_tokens: *completion_tokens, cached_tokens: *cached_tokens },
            );
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
                    let what = if summary.starts_with("llm ") { "thinking".to_string() } else { summary.clone() };
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
            let rest = if waiting > 0 { format!("{waiting} waiting") } else { "idle".into() };
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
                    _ => format!("  {cy}+{r} {pid} {} {b}{title}{r}", group.name()),
                }
            }
            KernelEvent::SyscallIssued { pid, summary, .. } => {
                // Model calls, and the resident agent waiting again after an idle wake, are not news.
                if (summary.starts_with("llm ") || summary == "recv") && !self.verbose {
                    return None;
                }
                format!("    {d}{pid} ▸ {summary}{r}")
            }
            KernelEvent::SyscallDone { pid, ok: false, summary, result, .. } => {
                format!("    {red}{pid} ✗ {summary}{r}{d}: {}{r}", first(&result.text(), 200))
            }
            KernelEvent::SyscallDone { pid, summary, .. } if self.verbose => format!("    {d}{pid} ✓ {summary}{r}"),
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
                format!("  {mark} {pid} {title}{d}: {}{r}", first(report, 160))
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
                let end = if *ok {
                    format!("{g}── {mark}order #{ticket} served ──{r}")
                } else {
                    format!("{red}── {mark}order #{ticket} not served ──{r}")
                };
                format!("{body}\n{end}")
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
/help        this help
/exit        quit (Ctrl-D)
/detach      (daemon sessions, zilch attach) leave; the session keeps running · /stop ends it
Tab completes commands, their arguments (groups, models, pids) and @ mentions.
Ctrl-C cancels the newest open order.";

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
        verbose: cli.verbose,
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
        match zilch_drivers::llm::complete(backend.as_ref(), &req, &zilch_kernel::CancelToken::new(), &|| {}, |_| {}) {
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
