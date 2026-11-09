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
pub const REVERSE: &str = "\x1b[7m";

/// Terminal state shared by the renderer and the interactive prompts (approver, plan review).
pub struct Screen {
    out: Box<dyn Write + Send>,
    fd: i32,
    pub(crate) color: bool,
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
    /// Animate the status panel with cursor movement (off in the REPL: the input line is live).
    pub live_panel: bool,
    /// Lane-aware output (REPL): only the active lane streams; the other one is summarized.
    pub lanes: bool,
    /// Agent id of the active lane: "master" (/code) or "spec" (/spec).
    pub active: String,
    /// Answers produced by a background lane, shown when the user switches to it.
    background_answer: HashMap<String, String>,
    /// Raw markdown of each lane root's current reply while it streams, and of its last
    /// complete reply (kept for /redraw).
    answer_buf: HashMap<String, String>,
    last_answer: HashMap<String, String>,
    /// Markdown renderer for the lane root's streamed answer (VT100/VT220 attributes only).
    md: crate::markdown::Markdown,
    /// Indent of rendered answers (config `markdown_indent`).
    md_indent: usize,
    /// Streamed master text not yet ending in a newline (printed, wrapped, line by line).
    master_buf: String,
    /// REPL status bar above the input line (None: plain output, heartbeat lines instead).
    pub status_bar: Option<Arc<crate::status::StatusBar>>,
    /// A question waiting for the user's answer (approval, plan review): pinned in the status
    /// row until answered, so output from other agents cannot scroll it out of sight.
    question: Option<String>,
    /// Set while handling an event of the background lane: put before each printed line.
    bg_prefix: String,
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
            live_panel: true,
            lanes: false,
            active: "master".into(),
            background_answer: HashMap::new(),
            answer_buf: HashMap::new(),
            last_answer: HashMap::new(),
            md: crate::markdown::Markdown::new(color),
            md_indent: 0,
            master_buf: String::new(),
            status_bar: None,
            question: None,
            bg_prefix: String::new(),
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

    /// Indent rendered markdown answers by `n` columns (0 = off).
    pub fn set_markdown_indent(&mut self, n: usize) {
        self.md_indent = n;
        self.md = crate::markdown::Markdown::new(self.color).with_indent(n);
    }

    /// Columns to wrap at: the terminal's current width (asked each time, so resizing applies to
    /// the next line), or 0 = don't wrap when the output is not a terminal.
    fn wrap_width(&self) -> usize {
        tty_width(self.fd).unwrap_or(0)
    }

    /// Print what is left of the streamed master text as a (wrapped) line of its own.
    fn flush_master(&mut self) {
        if self.master_buf.is_empty() {
            return;
        }
        let rest = std::mem::take(&mut self.master_buf);
        let w = self.wrap_width();
        if let Some(s) = self.md.line(&rest, w) {
            let _ = writeln!(self.out, "{s}");
        }
        self.at_line_start = true;
    }

    /// Print a table the markdown renderer is still collecting (before other output, so the
    /// answer keeps its order).
    fn flush_md_table(&mut self) {
        let w = self.wrap_width();
        if let Some(t) = self.md.take_table(w) {
            let _ = writeln!(self.out, "{t}");
        }
    }

    pub fn line(&mut self, s: &str) {
        self.clear_panel();
        self.flush_master();
        self.flush_md_table();
        self.ensure_newline();
        let w = self.wrap_width();
        let s = if self.bg_prefix.is_empty() {
            s.to_string()
        } else {
            s.lines().map(|l| format!("{}{l}", self.bg_prefix)).collect::<Vec<_>>().join("\n")
        };
        let _ = writeln!(self.out, "{}", crate::wrap::Wrapper::default().text(&s, w));
        self.draw_panel();
    }

    pub fn write_raw(&mut self, s: &str) {
        let _ = write!(self.out, "{s}");
        let _ = self.out.flush();
    }

    fn draw_panel(&mut self) {
        if !self.at_line_start || !self.color || !self.live_panel {
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
        if self.status_bar.is_some() {
            // The REPL status bar shows spinners and timers live; no heartbeat lines needed.
            self.update_status();
            return;
        }
        if self.color && self.live_panel {
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
                let what = if b.label == "thinking" { "still thinking".to_string() } else { format!("still running {}", b.label) };
                notes.push(format!("… {agent} {what} ({})", fmt_secs(secs)));
            }
        }
        for n in notes {
            self.line(&n);
        }
    }

    /// The lane an agent belongs to: the spec master is its own lane; master and workers are code.
    fn lane_of(agent: &str) -> &'static str {
        if agent == "spec" {
            "spec"
        } else {
            "master"
        }
    }

    fn lane_name(root: &str) -> &'static str {
        if root == "spec" {
            "spec"
        } else {
            "code"
        }
    }

    /// Lane routing (REPL). Returns true if the event should be handled normally. Events of the
    /// background lane update its status silently, its answer is buffered for later, and a short
    /// notice appears when it finishes; important events (TODO changes, notices, costs) always
    /// pass through.
    fn route_background(&mut self, ev: &AgentEvent) -> bool {
        let lane = Self::lane_of(&ev.agent);
        if lane == self.active {
            return true;
        }
        let root = ev.agent == lane;
        match &ev.kind {
            EventKind::TextDelta(t) if root => {
                self.background_answer.entry(ev.agent.clone()).or_default().push_str(t);
                false
            }
            EventKind::Thinking => {
                self.set_busy(&ev.agent, "thinking".into());
                false
            }
            EventKind::ToolStart { name, summary, .. } => {
                self.set_busy(&ev.agent, format!("{name}({summary})"));
                false
            }
            EventKind::ToolProgress { line, .. } => {
                if let Some(b) = self.busy.get_mut(&ev.agent) {
                    b.progress = Some(line.clone());
                }
                false
            }
            EventKind::Idle if root => {
                self.busy.remove(&ev.agent);
                let name = Self::lane_name(lane);
                let has_answer = self.background_answer.get(&ev.agent).is_some_and(|a| !a.trim().is_empty());
                let msg = if has_answer {
                    format!("{}✓ {name} lane finished — /{name} to see its answer{}", self.c(GREEN), self.c(RESET))
                } else {
                    format!("{}✓ {name} lane finished{}", self.c(GREEN), self.c(RESET))
                };
                self.line(&msg);
                false
            }
            EventKind::Idle => {
                self.busy.remove(&ev.agent);
                false
            }
            EventKind::TodosChanged(_) | EventKind::Notice(_) | EventKind::Usage { .. } | EventKind::TodoStatus { .. } => true,
            _ => false,
        }
    }

    /// Make `root` ("master" or "spec") the foreground lane and show anything it said meanwhile.
    pub fn switch_lane(&mut self, root: &str) {
        self.active = root.to_string();
        // A rule marks where the other lane's output ends.
        let label = self.lane_label(root);
        let width = match self.wrap_width() {
            0 => 40,
            w => w.min(100),
        };
        let tail = "─".repeat(width.saturating_sub(6 + unicode_width::UnicodeWidthStr::width(label.as_str())).max(4));
        let rule = format!("{}──── {label} {tail}{}", self.c(DIM), self.c(RESET));
        self.line(&rule);
        if let Some(answer) = self.background_answer.remove(root).filter(|a| !a.trim().is_empty()) {
            let msg = format!("{}── {} lane answered while you were away ──{}", self.c(DIM), Self::lane_name(root), self.c(RESET));
            self.line(&msg);
            // Rendered as markdown and printed as is (code lines must not be re-wrapped).
            let body = crate::markdown::Markdown::new(self.color).with_indent(self.md_indent).text(answer.trim_end(), self.wrap_width());
            self.clear_panel();
            let _ = writeln!(self.out, "{body}");
            self.draw_panel();
            self.last_answer.insert(root.to_string(), answer);
        }
    }

    /// /redraw: clear the screen and render the active lane's last answer again at the current
    /// terminal width (after a resize, or when other output garbled the view). Uses only VT100
    /// controls: cursor home + erase display; what was on screen stays in the scrollback.
    pub fn redraw(&mut self) {
        let root = self.active.clone();
        self.clear_panel();
        if self.color {
            let _ = write!(self.out, "\x1b[H\x1b[2J");
        }
        self.at_line_start = true;
        match self.last_answer.get(&root).filter(|a| !a.trim().is_empty()).cloned() {
            Some(answer) => {
                let w = self.wrap_width();
                let cols = if w == 0 { String::new() } else { format!(" at {w} columns") };
                let msg = format!("{}── last {} answer, redrawn{cols} ──{}", self.c(DIM), Self::lane_name(&root), self.c(RESET));
                self.line(&msg);
                let body = crate::markdown::Markdown::new(self.color).with_indent(self.md_indent).text(answer.trim_end(), w);
                let _ = writeln!(self.out, "{body}");
            }
            None => {
                let msg = format!("{}(no answer in the {} lane yet){}", self.c(DIM), Self::lane_name(&root), self.c(RESET));
                self.line(&msg);
            }
        }
        self.draw_panel();
        self.update_status();
        let _ = self.out.flush();
    }

    pub fn handle(&mut self, ev: AgentEvent) {
        // Lines the background lane still prints (TODO results, notices, "finished") carry its
        // lane mark so they are not mistaken for the foreground lane's output.
        self.bg_prefix = if self.lanes && Self::lane_of(&ev.agent) != self.active {
            let root = Self::lane_of(&ev.agent);
            if self.color {
                format!("{} ", lane_icon(root))
            } else {
                format!("[{}] ", Self::lane_name(root))
            }
        } else {
            String::new()
        };
        self.handle_event(ev);
        self.bg_prefix.clear();
        self.update_status();
    }

    /// What one lane is doing, for the status bar; None when it is idle.
    fn lane_status(&self, root: &str) -> Option<String> {
        let agents: Vec<&String> = self.busy.keys().filter(|a| Self::lane_of(a) == root).collect();
        let running: Vec<&Todo> = self.todos.iter().filter(|t| t.status == TodoStatus::InProgress).collect();
        let team = root == "master" && (!running.is_empty() || !self.delegations.is_empty());
        if agents.is_empty() && !team {
            return None;
        }
        let mut parts = Vec::new();
        if team {
            let working = running.len() + self.delegations.len();
            parts.push(format!("{working} working"));
            if !self.todos.is_empty() {
                let done = self.todos.iter().filter(|t| t.status == TodoStatus::Done).count();
                parts.push(format!("{done}/{} TODOs", self.todos.len()));
            }
        }
        // The longest-running worker activity, else the lane root's own.
        let worker = agents.iter().filter(|a| a.as_str() != root).max_by_key(|a| self.busy[a.as_str()].since.elapsed());
        match worker {
            Some(w) => parts.push(format!("{w} {}", self.activity_line(w))),
            None if self.busy.contains_key(root) => parts.push(self.activity_line(root)),
            None => {}
        }
        Some(parts.join(" · "))
    }

    /// Refresh the REPL status bar text (cheap; called on every event and tick).
    pub fn update_status(&mut self) {
        let Some(bar) = self.status_bar.clone() else { return };
        if let Some(q) = &self.question {
            let text = format!("{}{}? {q} {}", self.c(BOLD), self.c(REVERSE), self.c(RESET));
            bar.set(clip(&text, terminal_width(self.fd)), false);
            return;
        }
        let mut busy = false;
        let mut cells = Vec::new();
        for root in ["master", "spec"] {
            // On a terminal the lane's icon goes before its name.
            let name = self.lane_label(root);
            let active = self.active == root;
            let mark = if active { "▸ " } else { "" };
            match self.lane_status(root) {
                Some(s) => {
                    busy = true;
                    let color = if root == "spec" { YELLOW } else { CYAN };
                    cells.push(format!("{} {mark}{name}: {}{s}{}", self.spinner(color), self.c(DIM), self.c(RESET)));
                }
                None => cells.push(format!("{}{mark}{name}: idle{}", self.c(DIM), self.c(RESET))),
            }
        }
        let sep = format!("{}  │  {}", self.c(DIM), self.c(RESET));
        let text = clip(&cells.join(&sep), terminal_width(self.fd));
        bar.set(text, busy);
    }

    /// Pin (or with None, unpin) a one-line question in the status row.
    pub fn set_question(&mut self, q: Option<String>) {
        self.question = q;
        self.update_status();
    }

    /// The lane an agent belongs to, as shown to the user: "⧉ code" on a terminal, "code" otherwise.
    pub fn lane_label(&self, agent: &str) -> String {
        let root = Self::lane_of(agent);
        if self.color {
            format!("{} {}", lane_icon(root), Self::lane_name(root))
        } else {
            Self::lane_name(root).to_string()
        }
    }

    fn handle_event(&mut self, ev: AgentEvent) {
        if self.lanes && !self.route_background(&ev) {
            return;
        }
        // The lane root (master or spec) gets the full treatment; workers are summarized.
        let master = ev.agent == "master" || ev.agent == "spec";
        match ev.kind {
            EventKind::TextDelta(t) if master => {
                if !self.stream_master {
                    return;
                }
                self.answer_buf.entry(ev.agent.clone()).or_default().push_str(&t);
                self.clear_panel();
                let w = self.wrap_width();
                if w == 0 {
                    // Not a terminal: pass the stream through untouched.
                    let _ = write!(self.out, "{t}");
                    self.at_line_start = t.ends_with('\n');
                } else {
                    // Wrap whole lines (the REPL printer shows complete lines only anyway).
                    self.master_buf.push_str(&t);
                    while let Some(pos) = self.master_buf.find('\n') {
                        let line: String = self.master_buf.drain(..=pos).collect();
                        if let Some(s) = self.md.line(line.trim_end_matches(['\n', '\r']), w) {
                            let _ = writeln!(self.out, "{s}");
                        }
                    }
                }
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
                    // Keep the reply for /redraw (a turn's final text replaces earlier ones).
                    if let Some(a) = self.answer_buf.remove(&ev.agent).filter(|a| !a.trim().is_empty()) {
                        self.last_answer.insert(ev.agent.clone(), a);
                    }
                    self.clear_panel();
                    self.flush_master();
                    self.ensure_newline();
                    // A table at the very end; an unclosed fence ends with the answer.
                    let w = self.wrap_width();
                    if let Some(t) = self.md.finish(w) {
                        let _ = writeln!(self.out, "{t}");
                    }
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

/// The lane's symbol on a terminal: joined frames for the code lane, a hexagon (structure,
/// architecture) for the spec lane. Both one column wide, text presentation.
pub fn lane_icon(root: &str) -> &'static str {
    if root == "spec" {
        "\u{232C}" // ⌬
    } else {
        "\u{29C9}" // ⧉
    }
}

/// Width of the terminal behind file descriptor `fd`, asked from the kernel on every call so a
/// resized window takes effect on the next line. `None` when `fd` is not a terminal.
fn tty_width(fd: i32) -> Option<usize> {
    #[cfg(unix)]
    unsafe {
        let mut ws: libc::winsize = std::mem::zeroed();
        // The request type differs per platform: libc::Ioctl (int on musl, unsigned long on glibc)
        // exists only for Linux; macOS/BSD take an unsigned long, which TIOCGWINSZ already is.
        #[cfg(target_os = "linux")]
        let request = libc::TIOCGWINSZ as libc::Ioctl;
        #[cfg(not(target_os = "linux"))]
        let request = libc::TIOCGWINSZ;
        if libc::ioctl(fd, request, &mut ws) == 0 && ws.ws_col > 0 {
            return Some(ws.ws_col as usize);
        }
    }
    let _ = fd;
    None
}

/// `text` (markdown) rendered and wrapped for the terminal behind `fd`; unchanged when `fd` is a
/// pipe or file. With NO_COLOR it is wrapped but not styled.
pub fn wrap_for_tty(fd: i32, text: &str, indent: usize) -> String {
    match tty_width(fd) {
        Some(w) => crate::markdown::Markdown::new(std::env::var_os("NO_COLOR").is_none()).with_indent(indent).text(text, w),
        None => text.to_string(),
    }
}

/// Terminal width, then $COLUMNS, then 80. `COLUMNS` alone is unreliable: shells rarely export it.
fn terminal_width(fd: i32) -> usize {
    tty_width(fd).or_else(|| std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok())).unwrap_or(80)
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
        // A line break (a multi-line bash command in a tool summary) would make the line two
        // rows and leave a stale copy behind on every redraw; other controls are dropped.
        let ch = match ch {
            '\n' | '\r' | '\t' if out.ends_with(' ') => continue,
            '\n' | '\r' | '\t' => ' ',
            c if c.is_control() => continue,
            c => c,
        };
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

/// `Write` adapter for the REPL: complete lines are handed to rustyline's external printer, which
/// prints them above the input line and redraws what the user is typing, so output from either
/// lane never garbles the prompt.
pub struct PrinterOut {
    printer: Box<dyn rustyline::ExternalPrinter + Send>,
    buf: Vec<u8>,
}

impl PrinterOut {
    pub fn new(printer: Box<dyn rustyline::ExternalPrinter + Send>) -> Self {
        Self { printer, buf: Vec::new() }
    }
}

impl Write for PrinterOut {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.extend_from_slice(data);
        while let Some(pos) = self.buf.iter().position(|b| *b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let text = String::from_utf8_lossy(&line[..line.len() - 1]).into_owned();
            // rustyline writes the message as-is in raw mode (no output post-processing), and only
            // appends a bare "\n" when the message lacks one: end with "\r\n" ourselves so lines
            // start at column 0 instead of drifting right ("staircase").
            self.printer.print(format!("{text}\r\n")).map_err(std::io::Error::other)?;
        }
        Ok(data.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        // Partial lines (streaming text) are kept until their newline arrives.
        Ok(())
    }
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
    fn multi_line_tool_summaries_stay_on_one_status_row() {
        let s = clip("⠸ code: coder#1 bash(set -u\nD=/home/x\r\n\tls)  │  spec: idle", 80);
        assert_eq!(s, "⠸ code: coder#1 bash(set -u D=/home/x ls)  │  spec: idle");
        assert!(!clip("a\x07b\x1b[2mc", 80).contains('\x07'), "controls other than escapes are dropped");
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

    #[tokio::test]
    async fn repl_status_bar_follows_both_lanes() {
        let screen = Screen::new(Box::new(Buf::default()), -1, false, false, true);
        let mut s = screen.lock().await;
        s.lanes = true;
        let bar = Arc::new(crate::status::StatusBar::default());
        s.status_bar = Some(bar.clone());
        s.update_status();
        assert_eq!(bar.text(), "▸ code: idle  │  spec: idle");

        s.handle(ev("spec", EventKind::Thinking));
        assert!(bar.text().contains("code: idle") && bar.text().contains("spec: thinking · 0s"), "{}", bar.text());
        s.handle(ev("coder#1", EventKind::DelegateStart { worker: "coder#1".into(), task: "build".into() }));
        s.handle(ev("coder#1", EventKind::ToolStart { call_id: "c".into(), name: "bash".into(), summary: "cargo test".into() }));
        s.handle(ev("coder#1", EventKind::ToolProgress { call_id: String::new(), line: "running 12 tests".into() }));
        let t = bar.text();
        assert!(t.contains("▸ code: 1 working · coder#1 bash(cargo test) · 0s · running 12 tests"), "{t}");
        assert!(t.contains(" s"), "spec cell still present (clipped at 80 columns): {t}");

        // Ticks keep it current while busy; finishing returns both lanes to idle.
        s.busy.get_mut("coder#1").unwrap().since = Instant::now() - std::time::Duration::from_secs(42);
        s.tick();
        assert!(bar.text().contains("42s"), "{}", bar.text());
        s.handle(ev("coder#1", EventKind::Idle));
        s.handle(ev("coder#1", EventKind::DelegateEnd { worker: "coder#1".into(), ok: true }));
        s.handle(ev("spec", EventKind::Idle));
        assert_eq!(bar.text(), "▸ code: idle  │  spec: idle");
    }

    #[tokio::test]
    async fn switching_lanes_draws_a_rule_and_background_lines_carry_their_lane() {
        let buf = Buf::default();
        let screen = Screen::new(Box::new(buf.clone()), -1, false, false, true);
        let mut s = screen.lock().await;
        s.lanes = true;
        s.switch_lane("spec");
        assert!(buf.text().contains("──── spec ────"), "{}", buf.text());
        // The code lane works in the background: its notice is marked, the spec lane's is not.
        s.handle(ev("master", EventKind::Notice("retrying in 2s".into())));
        s.handle(ev("spec", EventKind::Notice("spec saved".into())));
        let out = buf.text();
        assert!(out.contains("[code] retrying in 2s"), "{out}");
        assert!(out.contains("\nspec saved"), "{out}");
    }
    #[tokio::test]
    async fn redraw_clears_and_renders_the_last_answer_again() {
        let buf = Buf::default();
        let screen = Screen::new(Box::new(buf.clone()), -1, true, false, true);
        let mut s = screen.lock().await;
        s.redraw();
        assert!(buf.text().contains("no answer in the code lane yet"), "{}", buf.text());
        s.handle(ev("master", EventKind::TextDelta("Let me look first.\n".into())));
        s.handle(ev("master", EventKind::TextDone));
        s.handle(ev("master", EventKind::TextDelta("The **final** answer\n".into())));
        s.handle(ev("master", EventKind::TextDone));
        buf.0.lock().unwrap().clear();
        s.redraw();
        let out = buf.text();
        assert!(out.starts_with("\u{241b}[H\u{241b}[2J"), "VT100 home + erase display: {out}");
        assert!(out.contains("last code answer") && out.contains("final"), "{out}");
        assert!(!out.contains("Let me look"), "only the last reply: {out}");
    }

    #[tokio::test]
    async fn lanes_show_as_icons_on_a_terminal() {
        let screen = Screen::new(Box::new(Buf::default()), -1, true, false, true);
        let mut s = screen.lock().await;
        s.lanes = true;
        let bar = Arc::new(crate::status::StatusBar::default());
        s.status_bar = Some(bar.clone());
        s.update_status();
        let plain = crate::wrap::visible(&bar.text());
        assert_eq!(plain, "▸ ⧉ code: idle  │  ⌬ spec: idle");
        s.handle(ev("spec", EventKind::Thinking));
        let plain = crate::wrap::visible(&bar.text());
        assert!(plain.contains("⌬ spec: thinking"), "{plain}");
    }
}
