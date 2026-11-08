//! Renders agent events: streamed master text, tool lines, and a live TODO/worker panel
//! redrawn with plain ANSI escapes (no TUI crate).

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::{mpsc, Mutex};
use zilch_core::llm::Usage;
use zilch_core::team::todo::{Todo, TodoStatus};
use zilch_core::{AgentEvent, EventKind};

pub const DIM: &str = "\x1b[2m";
pub const BOLD: &str = "\x1b[1m";
pub const RESET: &str = "\x1b[0m";
pub const GREEN: &str = "\x1b[32m";
pub const RED: &str = "\x1b[31m";
pub const YELLOW: &str = "\x1b[33m";
pub const CYAN: &str = "\x1b[36m";

/// Terminal state shared by the renderer and the interactive prompts (approver, plan review).
pub struct Screen {
    out: Box<dyn Write + Send>,
    fd: i32,
    color: bool,
    pub verbose: bool,
    /// Whether to stream the master's text (off in -p mode, where the final answer is printed at the end).
    pub stream_master: bool,
    panel_lines: usize,
    at_line_start: bool,
    todos: Vec<Todo>,
    /// agent -> what it is doing right now (for the live status lines).
    busy: BTreeMap<String, Busy>,
    started: Instant,
    /// instance -> task, for delegations not tied to a TODO.
    delegations: BTreeMap<String, String>,
    worker_text: HashMap<String, String>,
    pub usage: BTreeMap<(String, String), Usage>,
    pub cost: BTreeMap<(String, String), f64>,
}

pub type SharedScreen = Arc<Mutex<Screen>>;

/// What one agent is doing right now.
struct Busy {
    label: String,
    since: Instant,
    /// Latest output line of a running command.
    progress: Option<String>,
    /// Heartbeats already printed (plain output mode).
    noted: u64,
}

fn fmt_secs(s: u64) -> String {
    if s < 60 {
        format!("{s}s")
    } else {
        format!("{}m{:02}s", s / 60, s % 60)
    }
}

impl Screen {
    /// `fd` is the descriptor `out` writes to (1 or 2), used to ask the terminal its width.
    pub fn new(out: Box<dyn Write + Send>, fd: i32, color: bool, verbose: bool, stream_master: bool) -> SharedScreen {
        Arc::new(Mutex::new(Self {
            out,
            fd,
            color,
            verbose,
            stream_master,
            panel_lines: 0,
            at_line_start: true,
            todos: vec![],
            busy: BTreeMap::new(),
            started: Instant::now(),
            delegations: BTreeMap::new(),
            worker_text: HashMap::new(),
            usage: BTreeMap::new(),
            cost: BTreeMap::new(),
        }))
    }

    pub fn c(&self, code: &'static str) -> &'static str {
        if self.color {
            code
        } else {
            ""
        }
    }

    /// Erase the panel so normal output (or a prompt) can be printed.
    pub fn clear_panel(&mut self) {
        if self.panel_lines > 0 && self.color {
            // Erase line by line (cursor up + erase line). Not "erase below" (ESC[J): when the
            // panel sits on the top row, that clears the *entire* screen, and terminals with
            // scroll-on-clear (tmux >= 3.3 default) then archive every frame into scrollback.
            let _ = write!(self.out, "\r{}", "\x1b[1A\x1b[2K".repeat(self.panel_lines));
        }
        self.panel_lines = 0;
    }

    fn ensure_newline(&mut self) {
        if !self.at_line_start {
            let _ = writeln!(self.out);
            self.at_line_start = true;
        }
    }

    pub fn line(&mut self, s: &str) {
        self.clear_panel();
        self.ensure_newline();
        let _ = writeln!(self.out, "{s}");
        self.draw_panel();
    }

    pub fn write_raw(&mut self, s: &str) {
        let _ = write!(self.out, "{s}");
        let _ = self.out.flush();
    }

    fn draw_panel(&mut self) {
        if !self.at_line_start || !self.color {
            let _ = self.out.flush();
            return;
        }
        let running: Vec<&Todo> = self.todos.iter().filter(|t| t.status == TodoStatus::InProgress).collect();
        let mut lines = Vec::new();
        if !running.is_empty() || !self.delegations.is_empty() {
            let working = running.len() + self.delegations.len();
            let header = if self.todos.is_empty() {
                format!("team: {working} working")
            } else {
                let done = self.todos.iter().filter(|t| t.status == TodoStatus::Done).count();
                format!("team: {working} working · {done}/{} TODOs done", self.todos.len())
            };
            lines.push(format!("{}── {header} ──{}", self.c(DIM), self.c(RESET)));
            for t in &running {
                let inst = t.instance.clone().unwrap_or_else(|| "queued".into());
                let act = self.activity_line(&inst);
                lines.push(format!("  {} [{}] {} {}· {inst} · {act}{}", self.spinner(YELLOW), t.id, t.title, self.c(DIM), self.c(RESET)));
            }
            for (inst, task) in &self.delegations {
                let act = self.activity_line(inst);
                lines.push(format!("  {} {inst} {}· {task} · {act}{}", self.spinner(CYAN), self.c(DIM), self.c(RESET)));
            }
        }
        // The master is always shown while it works, so the screen never looks frozen.
        if self.busy.contains_key("master") {
            let act = self.activity_line("master");
            lines.push(format!("{} {}master{} {}· {act}{}", self.spinner(CYAN), self.c(BOLD), self.c(RESET), self.c(DIM), self.c(RESET)));
        }
        let width = terminal_width(self.fd);
        for l in &lines {
            let _ = writeln!(self.out, "{}", clip(l, width));
        }
        self.panel_lines = lines.len();
        let _ = self.out.flush();
    }

    fn set_busy(&mut self, agent: &str, label: String) {
        self.busy.insert(agent.to_string(), Busy { label, since: Instant::now(), progress: None, noted: 0 });
    }

    /// "bash(cargo test) · 37s · Compiling serde v1.0" for an agent (or "starting…").
    fn activity_line(&self, agent: &str) -> String {
        match self.busy.get(agent) {
            Some(b) => {
                let mut s = format!("{} · {}", b.label, fmt_secs(b.since.elapsed().as_secs()));
                if let Some(p) = &b.progress {
                    s.push_str(&format!(" · {p}"));
                }
                s
            }
            None => "starting…".into(),
        }
    }

    fn spinner(&self, color: &'static str) -> String {
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let i = (self.started.elapsed().as_millis() / 100) as usize % FRAMES.len();
        format!("{}{}{}", self.c(color), FRAMES[i], self.c(RESET))
    }

    /// Called ~8×/s: animate the panel on a TTY; on plain output, print a heartbeat line every
    /// 30s for long-running work so logs never look frozen.
    pub fn tick(&mut self) {
        if self.busy.is_empty() {
            return;
        }
        if self.color {
            if self.at_line_start {
                self.clear_panel();
                self.draw_panel();
            }
            return;
        }
        let mut notes = Vec::new();
        for (agent, b) in self.busy.iter_mut() {
            let secs = b.since.elapsed().as_secs();
            if secs >= 30 && secs / 30 > b.noted {
                b.noted = secs / 30;
                notes.push(format!("… {agent} still running {} ({})", b.label, fmt_secs(secs)));
            }
        }
        for n in notes {
            self.line(&n);
        }
    }

    pub fn handle(&mut self, ev: AgentEvent) {
        let master = ev.agent == "master";
        match ev.kind {
            EventKind::TextDelta(t) if master => {
                if !self.stream_master {
                    return;
                }
                self.clear_panel();
                let _ = write!(self.out, "{t}");
                self.at_line_start = t.ends_with('\n');
                let _ = self.out.flush();
            }
            EventKind::TextDelta(t) => {
                if self.verbose {
                    let buf = self.worker_text.entry(ev.agent.clone()).or_default();
                    buf.push_str(&t);
                    let mut complete = Vec::new();
                    while let Some(pos) = buf.find('\n') {
                        complete.push(buf.drain(..=pos).collect::<String>());
                    }
                    for line in complete {
                        let msg = format!("{}  │ {}: {}{}", self.c(DIM), ev.agent, line.trim_end(), self.c(RESET));
                        self.line(&msg);
                    }
                }
            }
            EventKind::TextDone => {
                if master && self.stream_master {
                    self.clear_panel();
                    self.ensure_newline();
                    self.draw_panel();
                } else if let Some(rest) = self.worker_text.remove(&ev.agent) {
                    if self.verbose && !rest.trim().is_empty() {
                        let msg = format!("{}  │ {}: {}{}", self.c(DIM), ev.agent, rest.trim_end(), self.c(RESET));
                        self.line(&msg);
                    }
                }
            }
            EventKind::ToolStart { name, summary, .. } => {
                if master {
                    if !["todo_write", "propose_plan", "run_todos", "delegate"].contains(&name.as_str()) || self.verbose {
                        let msg = format!("{}●{} {}{}{}({})", self.c(CYAN), self.c(RESET), self.c(BOLD), name, self.c(RESET), summary);
                        self.line(&msg);
                    } else if name == "run_todos" {
                        let msg = format!("{}●{} running TODOs in parallel…", self.c(CYAN), self.c(RESET));
                        self.line(&msg);
                    }
                } else {
                    self.set_busy(&ev.agent, format!("{name}({summary})"));
                    if self.verbose {
                        let msg = format!("{}  │ {} ● {}({}){}", self.c(DIM), ev.agent, name, summary, self.c(RESET));
                        self.line(&msg);
                    } else {
                        self.clear_panel();
                        self.draw_panel();
                    }
                }
            }
            EventKind::ToolEnd { ok, summary, name, .. } => {
                if master && (self.verbose || !["todo_write", "propose_plan", "run_todos", "delegate"].contains(&name.as_str())) {
                    let mark = if ok {
                        format!("{}⎿{}", self.c(DIM), self.c(RESET))
                    } else {
                        format!("{}⎿ ✗{}", self.c(RED), self.c(RESET))
                    };
                    let msg = format!("  {mark} {}{}{}", self.c(DIM), summary, self.c(RESET));
                    self.line(&msg);
                } else if !master && !ok && self.verbose {
                    let msg = format!("{}  │ {} ✗ {}{}", self.c(RED), ev.agent, summary, self.c(RESET));
                    self.line(&msg);
                }
            }
            EventKind::TodosChanged(new) => {
                let old: HashMap<String, TodoStatus> = self.todos.iter().map(|t| (t.id.clone(), t.status.clone())).collect();
                let first = self.todos.is_empty() && !new.is_empty();
                self.todos = new.clone();
                if first {
                    let mut s = format!("{}● TODOs{}", self.c(CYAN), self.c(RESET));
                    for t in &new {
                        s.push_str(&format!(
                            "\n  {} [{}] {} {}({}){}",
                            t.status.icon(),
                            t.id,
                            t.title,
                            self.c(DIM),
                            t.worker,
                            self.c(RESET)
                        ));
                    }
                    self.line(&s);
                }
                for t in &new {
                    let before = old.get(&t.id);
                    if before == Some(&t.status) {
                        continue;
                    }
                    let inst = t.instance.clone().unwrap_or_default();
                    let summary = t.report.as_deref().and_then(summary_line).unwrap_or_default();
                    let msg = match t.status {
                        TodoStatus::Done => format!(
                            "  {}✓{} [{}] {} {}({inst}) {summary}{}",
                            self.c(GREEN),
                            self.c(RESET),
                            t.id,
                            t.title,
                            self.c(DIM),
                            self.c(RESET)
                        ),
                        TodoStatus::Failed => format!(
                            "  {}✗{} [{}] {} {}({inst}) {summary}{}",
                            self.c(RED),
                            self.c(RESET),
                            t.id,
                            t.title,
                            self.c(DIM),
                            self.c(RESET)
                        ),
                        TodoStatus::Blocked => {
                            format!("  {}⊘{} [{}] {} blocked by a failed dependency", self.c(YELLOW), self.c(RESET), t.id, t.title)
                        }
                        TodoStatus::Cancelled => format!("  – [{}] {} cancelled", t.id, t.title),
                        _ => {
                            if !first && before.is_none() {
                                format!("  ☐ [{}] {} {}(added){}", t.id, t.title, self.c(DIM), self.c(RESET))
                            } else {
                                String::new()
                            }
                        }
                    };
                    if !msg.is_empty() {
                        self.line(&msg);
                    }
                }
                self.clear_panel();
                self.draw_panel();
            }
            EventKind::DelegateStart { worker, task } => {
                self.delegations.insert(worker.clone(), task.clone());
                let msg = format!("{}●{} delegate → {}{}{}: {}", self.c(CYAN), self.c(RESET), self.c(BOLD), worker, self.c(RESET), task);
                self.line(&msg);
            }
            EventKind::DelegateEnd { worker, ok } => {
                self.delegations.remove(&worker);
                self.busy.remove(&worker);
                let msg = if ok {
                    format!("  {}✓{} {} reported back", self.c(GREEN), self.c(RESET), worker)
                } else {
                    format!("  {}✗{} {} reported a failure", self.c(RED), self.c(RESET), worker)
                };
                self.line(&msg);
            }
            EventKind::InstanceSpawned { id, model } => {
                if self.verbose {
                    let msg = format!("{}  + spawned {id} ({model}){}", self.c(DIM), self.c(RESET));
                    self.line(&msg);
                }
            }
            EventKind::Usage { model, usage, cost } => {
                let key = (ev.agent, model);
                if let Some(c) = cost {
                    *self.cost.entry(key.clone()).or_default() += c;
                }
                *self.usage.entry(key).or_default() += usage;
            }
            EventKind::Notice(n) => {
                let msg = format!("{}{n}{}", self.c(DIM), self.c(RESET));
                self.line(&msg);
            }
            EventKind::TodoStatus { .. } => {}
            EventKind::Thinking => self.set_busy(&ev.agent, "thinking".into()),
            EventKind::ToolProgress { line, .. } => {
                if let Some(b) = self.busy.get_mut(&ev.agent) {
                    b.progress = Some(line);
                }
            }
            EventKind::Idle => {
                self.busy.remove(&ev.agent);
                self.clear_panel();
                self.draw_panel();
            }
        }
    }

    pub fn usage_report(&self) -> String {
        let mut s = String::new();
        let (mut pi, mut co) = (0, 0);
        let mut total_cost: Option<f64> = None;
        for ((agent, model), u) in &self.usage {
            let cost = self.cost.get(&(agent.clone(), model.clone())).copied();
            let c = cost.map_or("      ?".to_string(), |c| format!("${c:>8.4}"));
            s.push_str(&format!("  {agent:<14} {model:<36} in {:>8}  out {:>7}  {c}\n", u.prompt_tokens, u.completion_tokens));
            pi += u.prompt_tokens;
            co += u.completion_tokens;
            if let Some(c) = cost {
                *total_cost.get_or_insert(0.0) += c;
            }
        }
        let tc = total_cost.map_or("(prices unknown)".to_string(), |c| format!("${c:>8.4}"));
        s.push_str(&format!("  {:<51} in {pi:>8}  out {co:>7}  {tc}", "total"));
        s
    }

    pub fn total_tokens(&self) -> (u64, u64) {
        self.usage.values().fold((0, 0), |(a, b), u| (a + u.prompt_tokens, b + u.completion_tokens))
    }

    /// Total USD across agents whose model price is known (None if no price is known at all).
    pub fn total_cost(&self) -> Option<f64> {
        if self.cost.is_empty() {
            None
        } else {
            Some(self.cost.values().sum())
        }
    }
}

fn summary_line(report: &str) -> Option<String> {
    report
        .lines()
        .find_map(|l| l.trim().strip_prefix("SUMMARY:").map(|s| s.trim().to_string()))
        .or_else(|| report.lines().next().map(|s| s.chars().take(100).collect()))
}

/// Width of the terminal behind file descriptor `fd` (asked from the kernel on every redraw, so
/// resizing works), then $COLUMNS, then 80. `COLUMNS` alone is unreliable: shells rarely export it.
fn terminal_width(fd: i32) -> usize {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        if libc::ioctl(fd, libc::TIOCGWINSZ as libc::Ioctl, &mut ws) == 0 && ws.ws_col > 0 {
            return ws.ws_col as usize;
        }
    }
    let _ = fd;
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).unwrap_or(80)
}

/// Clip a line so it occupies at most `width - 1` terminal columns (ANSI escapes take none, wide
/// characters like emoji/CJK take two). A panel line must never wrap: the redraw moves the cursor
/// up one row per line, so a wrapped line would leave stale rows behind on every frame.
fn clip(s: &str, width: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    let max = width.saturating_sub(1).max(10);
    let mut out = String::new();
    let mut cols = 0;
    let mut in_esc = false;
    for ch in s.chars() {
        if in_esc {
            out.push(ch);
            if ch.is_ascii_alphabetic() {
                in_esc = false;
            }
            continue;
        }
        if ch == '\x1b' {
            in_esc = true;
            out.push(ch);
            continue;
        }
        let w = ch.width().unwrap_or(0);
        if cols + w > max - 1 {
            out.push('…');
            out.push_str(RESET);
            break;
        }
        out.push(ch);
        cols += w;
    }
    out
}

pub fn spawn_renderer(
    screen: SharedScreen,
    mut rx: mpsc::UnboundedReceiver<AgentEvent>,
    mut log: Option<crate::log::EventLog>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        // Events drive the screen; a timer keeps spinners and elapsed times moving between events.
        let mut ticker = tokio::time::interval(std::time::Duration::from_millis(125));
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                ev = rx.recv() => {
                    let Some(ev) = ev else { break };
                    if let Some(l) = log.as_mut() {
                        l.handle(&ev);
                    }
                    screen.lock().await.handle(ev);
                }
                _ = ticker.tick() => {
                    // Never block on the lock: an approval prompt may be holding it.
                    if let Ok(mut s) = screen.try_lock() {
                        s.tick();
                    }
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// A Write sink we can read back.
    #[derive(Clone, Default)]
    struct Buf(Arc<StdMutex<Vec<u8>>>);
    impl Write for Buf {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    impl Buf {
        fn text(&self) -> String {
            String::from_utf8_lossy(&self.0.lock().unwrap()).replace("\x1b[", "\u{241b}[")
        }
    }
    fn ev(agent: &str, kind: EventKind) -> AgentEvent {
        AgentEvent { agent: agent.into(), kind }
    }

    #[tokio::test]
    async fn long_command_shows_spinner_elapsed_and_latest_output() {
        let buf = Buf::default();
        let screen = Screen::new(Box::new(buf.clone()), -1, true, false, true);
        let mut s = screen.lock().await;
        s.handle(ev("coder#1", EventKind::DelegateStart { worker: "coder#1".into(), task: "build".into() }));
        s.handle(ev("coder#1", EventKind::ToolStart { call_id: "c".into(), name: "bash".into(), summary: "cargo build".into() }));
        s.handle(ev("coder#1", EventKind::ToolProgress { call_id: String::new(), line: "Compiling serde v1.0.200".into() }));
        s.busy.get_mut("coder#1").unwrap().since = Instant::now() - std::time::Duration::from_secs(75);
        s.tick();
        let out = buf.text();
        assert!(out.contains("bash(cargo build) · 1m15s · Compiling serde v1.0.200"), "{out}");
        assert!(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"].iter().any(|f| out.contains(f)), "spinner drawn");
        // The master gets its own status line while it works.
        s.handle(ev("master", EventKind::Thinking));
        s.tick();
        assert!(buf.text().contains("master") && buf.text().contains("thinking · 0s"));
        // Idle clears it: no more redraws.
        s.handle(ev("master", EventKind::Idle));
        s.handle(ev("coder#1", EventKind::Idle));
        s.handle(ev("coder#1", EventKind::DelegateEnd { worker: "coder#1".into(), ok: true }));
        let before = buf.text().len();
        s.tick();
        assert_eq!(buf.text().len(), before, "nothing busy → tick draws nothing");
    }

    #[test]
    fn clipped_panel_lines_never_wrap() {
        use unicode_width::UnicodeWidthStr;
        let strip = |s: &str| {
            let mut out = String::new();
            let mut esc = false;
            for c in s.chars() {
                if esc {
                    esc = !c.is_ascii_alphabetic();
                    continue;
                }
                if c == '\x1b' {
                    esc = true;
                    continue;
                }
                out.push(c);
            }
            out
        };
        let line =
            format!("{CYAN}⠹{RESET} coder#2 · bash(cargo test) · 1m07s · 編譯中 🦀🦀🦀 Compiling serde v1.0.200 and a lot more output");
        for width in [20, 40, 60, 80] {
            let visible = strip(&clip(&line, width));
            assert!(visible.width() < width, "width {width}: {} cols: {visible}", visible.width());
        }
        // Short lines are untouched.
        assert_eq!(strip(&clip("⠹ master · thinking · 4s", 80)), "⠹ master · thinking · 4s");
    }

    #[tokio::test]
    async fn plain_output_prints_heartbeats_every_30s() {
        let buf = Buf::default();
        let screen = Screen::new(Box::new(buf.clone()), -1, false, false, false);
        let mut s = screen.lock().await;
        s.handle(ev("coder#2", EventKind::ToolStart { call_id: "c".into(), name: "bash".into(), summary: "npm test".into() }));
        s.tick();
        assert!(!buf.text().contains("still running"), "no heartbeat before 30s");
        s.busy.get_mut("coder#2").unwrap().since = Instant::now() - std::time::Duration::from_secs(31);
        s.tick();
        s.tick(); // same 30s window: printed once
        assert_eq!(buf.text().matches("still running bash(npm test) (31s)").count(), 1, "{}", buf.text());
        assert!(!buf.text().contains('\u{241b}'), "no ANSI escapes in plain mode");
    }
}
