//! `zilch-os`: the waiter. Takes the guest's order, hands it to the kitchen (the kernel), shows
//! what the kitchen does as it happens, relays the plan for approval, and serves the result.
//! It never waits on the agents: the kernel runs on its own threads and reports on the event bus.

#[allow(dead_code)] // ported whole from zilch-cli; streaming parts come later
mod markdown;
mod status;
#[allow(dead_code)]
mod wrap;

use clap::Parser;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, IsTerminal, Write};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zilch_agents::Book;
use zilch_drivers::config::{Config, Overrides};
use zilch_drivers::{DriverSet, ARCHITECT_TOOLS, DEVELOPER_TOOLS};
use zilch_ipc::{EventBus, Group, KernelEvent, Pid, TicketId};
use zilch_kernel::{Cmd, KernelConfig, KernelHandle};

#[derive(Parser)]
#[command(
    name = "zilch-os",
    version,
    about = "zilch as an agent OS: the architect plans your order, developers cook it in parallel, the result is served."
)]
struct Cli {
    /// Place one order and exit when it is served.
    #[arg(short = 'p', long = "prompt")]
    prompt: Option<String>,
    /// Approve plans without asking.
    #[arg(long)]
    yes: bool,
    /// Model for every group.
    #[arg(long)]
    model: Option<String>,
    #[arg(long)]
    architect_model: Option<String>,
    #[arg(long)]
    developer_model: Option<String>,
    /// Step workers (default: CPU cores).
    #[arg(long)]
    workers: Option<usize>,
    /// Write the event log here (default: <data dir>/zilch/logs/os-<time>.jsonl).
    #[arg(long)]
    log: Option<PathBuf>,
    #[arg(long)]
    no_log: bool,
    /// Show every model call and syscall result.
    #[arg(short, long)]
    verbose: bool,
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
    open: Mutex<HashSet<TicketId>>,
    /// Live processes for the status row.
    live: Mutex<BTreeMap<Pid, Live>>,
    /// A short message shown in the status row for a few seconds.
    note: Mutex<Option<(String, Instant)>>,
    started: Instant,
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
            KernelEvent::TicketServed { ticket, .. } => live.retain(|_, l| l.ticket != *ticket || l.group == Group::Architecture),
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
        let note = note.unwrap_or_default();
        if let Some((_, ticket)) = self.question() {
            let rev = self.c("\x1b[7m");
            return (
                format!("{note}{b}{rev}? plan #{ticket} waits for you — Enter approves · n rejects · anything else is feedback {r}"),
                false,
            );
        }
        let live = self.live.lock().unwrap();
        let busy: Vec<(&Pid, &Live)> = live.iter().filter(|(_, l)| l.state == "running" || l.state == "blocked(io)").collect();
        // The main agent waiting for the next order is idle, not "waiting" on anything.
        let waiting = live.values().filter(|l| !(l.group == Group::Architecture && l.state == "blocked(recv)")).count() - busy.len();
        if busy.is_empty() {
            let rest = if waiting > 0 { format!("{waiting} waiting") } else { "idle".into() };
            return (format!("{note}{d}{rest}{r}"), false);
        }
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let frame = FRAMES[(self.started.elapsed().as_millis() / 100) as usize % FRAMES.len()];
        let cells: Vec<String> = busy
            .iter()
            .map(|(pid, l)| {
                let (what, secs) = l.doing.as_ref().map(|(w, t)| (w.as_str(), t.elapsed().as_secs())).unwrap_or(("thinking", 0));
                let who = if l.group == Group::Architecture { "main".to_string() } else { format!("{pid}") };
                format!("{who} {d}{what} · {secs}s{r}")
            })
            .collect();
        let more = if waiting > 0 { format!("  {d}│ {waiting} waiting{r}") } else { String::new() };
        (format!("{note}{cy}{frame}{r} {} working: {}{more}", busy.len(), cells.join(&format!("{d}  │  {r}"))), true)
    }

    fn render(&self, ev: &KernelEvent) -> Option<String> {
        let (b, d, r, g, red, y, cy) = (self.c(BOLD), self.c(DIM), self.c(RESET), self.c(GREEN), self.c(RED), self.c(YELLOW), self.c(CYAN));
        let first = |s: &str, n: usize| {
            s.lines().map(str::trim).find(|l| !l.is_empty() && !l.starts_with("STATUS:")).unwrap_or("").chars().take(n).collect::<String>()
        };
        Some(match ev {
            KernelEvent::TicketOpened { ticket, text } => {
                self.open.lock().unwrap().insert(*ticket);
                format!("{cy}● order #{ticket}{r} {}", first(text, 100))
            }
            KernelEvent::Spawned { pid, group, title, .. } => {
                self.procs.lock().unwrap().insert(*pid, (*group, title.clone()));
                match group {
                    Group::Architecture => format!("  {d}+ {pid} main agent{r}"),
                    _ => format!("  {cy}+{r} {pid} {} {b}{title}{r}", group.name()),
                }
            }
            KernelEvent::SyscallIssued { pid, summary, .. } => {
                if summary.starts_with("llm ") && !self.verbose {
                    return None;
                }
                format!("    {d}{pid} ▸ {summary}{r}")
            }
            KernelEvent::SyscallDone { pid, ok: false, summary, result, .. } => {
                format!("    {red}{pid} ✗ {summary}{r}{d}: {}{r}", first(&result.text(), 200))
            }
            KernelEvent::SyscallDone { pid, summary, .. } if self.verbose => format!("    {d}{pid} ✓ {summary}{r}"),
            KernelEvent::Question { qid, ticket, text, .. } => {
                self.questions.lock().unwrap().push_back((*qid, *ticket));
                format!("{y}?{r} {b}plan for order #{ticket}{r}\n{text}\n{y}answer:{r} Enter = approve · n = reject · anything else = feedback to the main agent {d}(at the plan prompt; press Enter on an empty line if it is not showing){r}")
            }
            KernelEvent::Signal { pid, signal, reason } => format!("{y}⚠ {pid} {signal:?}{r}: {reason}"),
            KernelEvent::Exited { pid, ok, report } => {
                let (group, title) = self.procs.lock().unwrap().get(pid).cloned().unwrap_or((Group::Development, String::new()));
                if group == Group::Architecture {
                    return None; // the served ticket shows its report
                }
                let mark = if *ok && zilch_kernel::report_ok(report) { format!("{g}✓{r}") } else { format!("{red}✗{r}") };
                format!("  {mark} {pid} {title}{d}: {}{r}", first(report, 160))
            }
            KernelEvent::TicketServed { ticket, ok, summary } => {
                self.open.lock().unwrap().remove(ticket);
                self.questions.lock().unwrap().retain(|(_, t)| t != ticket);
                let width = terminal_width().saturating_sub(2).max(20);
                let body = markdown::Markdown::new(self.color).with_indent(2).text(summary.trim(), width);
                let end = if *ok {
                    format!("{g}── order #{ticket} served ──{r}")
                } else {
                    format!("{red}── order #{ticket} not served ──{r}")
                };
                format!("{body}\n{end}")
            }
            KernelEvent::Notice { text, .. } => format!("  {d}{text}{r}"),
            KernelEvent::Lagged { missed } => format!("  {d}(display missed {missed} events; the log has them){r}"),
            KernelEvent::Told { .. }
            | KernelEvent::State { .. }
            | KernelEvent::TextDelta { .. }
            | KernelEvent::Usage { .. }
            | KernelEvent::SyscallDone { .. } => return None,
        })
    }
}

fn terminal_width() -> usize {
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

fn open_log(cli: &Cli) -> Option<(PathBuf, std::fs::File)> {
    if cli.no_log {
        return None;
    }
    let path = cli.log.clone().or_else(|| {
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
    let header = format!("workers {}/{} busy · ready queues {:?} · tokens {tokens}", st.busy_workers, st.workers, st.queued);
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
Type an order to place it. While a plan waits for you: Enter approves, n rejects, anything else is feedback.
@<pid> text  more input for one process (read before its next model call)
@main text   more input for the main agent (when it is idle, this is simply the next order)
@developer text   the same for every live subagent of the newest order
/ps          the process table (pid, group, state, what it is doing)
/kill <pid>  stop one process (its task fails; dependents are blocked)
/cancel      cancel the newest open order
/help        this help
/exit        quit (Ctrl-D)
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
    let ov =
        Overrides { model: cli.model.clone(), architect_model: cli.architect_model.clone(), developer_model: cli.developer_model.clone() };
    let cfg = Config::load(&cwd, &ov)?;
    let drivers = DriverSet::new(&cwd, cfg.policy.clone(), cfg.backends());
    let mut book =
        Book::new(cfg.architect.clone(), cfg.developer.clone(), drivers.specs(ARCHITECT_TOOLS), drivers.specs(DEVELOPER_TOOLS), &cwd);
    if let Some(t) = cfg.main_context_tokens {
        book.main_context_tokens = t;
    }
    let bus = Arc::new(EventBus::new());
    let log = open_log(&cli);
    if let Some((_, file)) = &log {
        bus.set_log(Box::new(file.try_clone()?));
    }
    let events = bus.subscribe(4096);
    let defaults = KernelConfig::default();
    let kcfg = KernelConfig {
        workers: cli.workers.unwrap_or(defaults.workers),
        io_threads: cfg.io_threads,
        loop_limit: cfg.loop_limit.unwrap_or(defaults.loop_limit),
        max_steps: cfg.max_steps,
        max_tokens: cfg.max_task_tokens,
        group_tokens: cfg.group_tokens.clone(),
        auto_approve: cli.yes,
        ..defaults
    };
    let color = std::io::stdout().is_terminal();
    let view = Arc::new(View {
        color,
        verbose: cli.verbose,
        procs: Mutex::default(),
        questions: Mutex::default(),
        open: Mutex::default(),
        live: Mutex::default(),
        note: Mutex::default(),
        started: Instant::now(),
    });
    let (d, r) = (view.c(DIM), view.c(RESET));
    let (_, sandbox) = zilch_drivers::tools::sandbox::status(cfg.policy.bash_sandbox, cfg.policy.bash.network);
    let banner = format!(
        "{}zilch-os {}{} {d}· main {} · subagents {} · {} workers{r}\n{d}commands: {sandbox}{r}{}",
        view.c(BOLD),
        env!("CARGO_PKG_VERSION"),
        r,
        cfg.architect.key(),
        cfg.developer.key(),
        kcfg.workers,
        log.as_ref().map(|(p, _)| format!("\n{d}event log: {}{r}", p.display())).unwrap_or_default()
    );
    let kernel = zilch_kernel::start(kcfg, Arc::new(drivers), Arc::new(book), bus);

    if let Some(order) = &cli.prompt {
        eprintln!("{banner}");
        kernel.send(Cmd::Order(order.clone()));
        let stdin = std::io::stdin();
        for env in events.iter() {
            if let Some(s) = view.render(&env.event) {
                println!("{s}");
                let _ = std::io::stdout().flush();
            }
            match &env.event {
                KernelEvent::Question { qid, .. } if !cli.yes => {
                    eprint!("plan? › ");
                    let mut line = String::new();
                    stdin.lock().read_line(&mut line)?;
                    kernel.send(Cmd::Answer { qid: *qid, text: line.trim().to_string() });
                }
                KernelEvent::TicketServed { ok, .. } => {
                    kernel.shutdown();
                    return Ok(if *ok { 0 } else { 1 });
                }
                _ => {}
            }
        }
        return Ok(1);
    }

    println!("{banner}\n{d}Type an order, or /help.{r}");
    let mut rl: rustyline::Editor<status::StatusHelper, rustyline::history::DefaultHistory> = rustyline::Editor::new()?;
    let mut printer = rl.create_external_printer()?;
    // The live status row above the prompt (terminals only), redrawn while something runs.
    let bar = color.then(|| Arc::new(status::StatusBar::default()));
    if let Some(bar) = &bar {
        rl.set_helper(Some(status::StatusHelper(bar.clone())));
        status::spawn_refresher(bar.clone(), Box::new(rl.create_external_printer()?));
        let (v, bar) = (view.clone(), bar.clone());
        std::thread::Builder::new().name("zilch-os-status".into()).spawn(move || loop {
            let (text, busy) = v.status_line();
            bar.set(status::clip(&text, terminal_width()), busy);
            std::thread::sleep(Duration::from_millis(100));
        })?;
    }
    let v = view.clone();
    std::thread::Builder::new().name("zilch-os-view".into()).spawn(move || {
        for env in events.iter() {
            v.track(&env.event);
            if let Some(s) = v.render(&env.event) {
                if rustyline::ExternalPrinter::print(&mut printer, format!("{s}\n")).is_err() {
                    break;
                }
            }
        }
    })?;
    loop {
        // A line answers the plan only if the plan prompt was showing when it was typed.
        let asked = view.question();
        // With a status row the prompt is two rows; the first is filled with the status (status.rs).
        let row = if bar.is_some() { "\n" } else { "" };
        let prompt = match asked {
            Some((_, t)) => format!("{row}plan #{t}? › "),
            None => format!("{row}zilch-os › "),
        };
        if let Some(b) = &bar {
            b.reading.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let read = rl.readline(&prompt);
        if let Some(b) = &bar {
            b.reading.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let line = match read {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                match view.latest_open() {
                    Some(t) => {
                        kernel.send(Cmd::KillTicket(t));
                        println!("⏹ cancelling order #{t}…");
                    }
                    None => println!("(Ctrl-D to quit)"),
                }
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let input = line.trim();
        // The order may have ended (cancelled, killed) after this prompt was drawn.
        let asked = asked.filter(|(_, t)| view.open.lock().unwrap().contains(t));
        if let (Some((qid, _)), false) = (asked, input.starts_with('/') || input.starts_with('@')) {
            view.questions.lock().unwrap().retain(|(q, _)| *q != qid);
            kernel.send(Cmd::Answer { qid, text: input.to_string() });
            continue;
        }
        if input.is_empty() {
            // Nothing to order: say so instead of silently doing nothing (the plan prompt, if one
            // arrived meanwhile, shows from the next line on).
            let msg = if view.question().is_some() { "empty input ignored — a plan is waiting" } else { "empty input ignored" };
            match &bar {
                Some(_) => view.note(msg),
                None => println!("({msg})"),
            }
            continue;
        }
        let _ = rl.add_history_entry(input);
        if let Some(rest) = input.strip_prefix('@') {
            let (who, text) = rest.split_once(char::is_whitespace).map(|(w, t)| (w, t.trim())).unwrap_or((rest, ""));
            let msg = match zilch_ipc::Target::parse(who) {
                None => format!("@{who}: not a pid or group (architect, developer)"),
                Some(_) if text.is_empty() => format!("@{who}: nothing to send"),
                Some(target) => match kernel.tell(target, text) {
                    Ok(how) => {
                        println!("  {}✉ you → @{who}: {text}{}", view.c(DIM), view.c(RESET));
                        format!("✉ {how}")
                    }
                    Err(e) => format!("@{who}: {e}"),
                },
            };
            match &bar {
                Some(_) => view.note(&msg),
                None => println!("({msg})"),
            }
            continue;
        }
        let (cmd, arg) = input.split_once(' ').map(|(c, a)| (c, a.trim())).unwrap_or((input, ""));
        match cmd {
            "/exit" | "/quit" => break,
            "/help" => println!("{HELP}"),
            "/ps" => println!("{}", ps(&kernel)),
            "/kill" => match arg.parse::<Pid>() {
                Ok(pid) => kernel.send(Cmd::Kill(pid)),
                Err(_) => println!("usage: /kill <pid>  (see /ps)"),
            },
            "/cancel" => match view.latest_open() {
                Some(t) => kernel.send(Cmd::KillTicket(t)),
                None => println!("No open order."),
            },
            c if c.starts_with('/') => println!("Unknown command {c}. /help lists them."),
            _ => kernel.send(Cmd::Order(input.to_string())),
        }
    }
    kernel.shutdown();
    Ok(0)
}

/// "x +1 more" → "x +2 more".
fn bump(d: &str) -> String {
    match d.rsplit_once(" +").and_then(|(head, tail)| tail.strip_suffix(" more").and_then(|n| n.parse::<u32>().ok()).map(|n| (head, n))) {
        Some((head, n)) => format!("{head} +{} more", n + 1),
        None => d.to_string(),
    }
}
