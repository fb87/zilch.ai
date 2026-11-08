//! Renders agent events: streamed master text, tool lines, and a live TODO/worker panel
//! redrawn with plain ANSI escapes (no TUI crate).

use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::Arc;
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
    color: bool,
    pub verbose: bool,
    /// Whether to stream the master's text (off in -p mode, where the final answer is printed at the end).
    pub stream_master: bool,
    panel_lines: usize,
    at_line_start: bool,
    todos: Vec<Todo>,
    /// instance -> current activity (last tool), for running workers.
    activity: HashMap<String, String>,
    /// instance -> task, for delegations not tied to a TODO.
    delegations: BTreeMap<String, String>,
    worker_text: HashMap<String, String>,
    pub usage: BTreeMap<(String, String), Usage>,
    pub cost: BTreeMap<(String, String), f64>,
}

pub type SharedScreen = Arc<Mutex<Screen>>;

impl Screen {
    pub fn new(out: Box<dyn Write + Send>, color: bool, verbose: bool, stream_master: bool) -> SharedScreen {
        Arc::new(Mutex::new(Self {
            out,
            color,
            verbose,
            stream_master,
            panel_lines: 0,
            at_line_start: true,
            todos: vec![],
            activity: HashMap::new(),
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
            let _ = write!(self.out, "\x1b[{}F\x1b[J", self.panel_lines);
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
        if running.is_empty() && self.delegations.is_empty() {
            let _ = self.out.flush();
            return;
        }
        let done = self.todos.iter().filter(|t| t.status == TodoStatus::Done).count();
        let mut lines = vec![format!(
            "{}── team: {} running · {}/{} TODOs done ──{}",
            self.c(DIM),
            running.len() + self.delegations.len(),
            done,
            self.todos.len(),
            self.c(RESET)
        )];
        for t in &running {
            let inst = t.instance.clone().unwrap_or_else(|| "queued".into());
            let act = self.activity.get(&inst).cloned().unwrap_or_else(|| "thinking…".into());
            lines.push(format!(
                "  {}◐{} [{}] {} {}· {} · {}{}",
                self.c(YELLOW),
                self.c(RESET),
                t.id,
                t.title,
                self.c(DIM),
                inst,
                act,
                self.c(RESET)
            ));
        }
        for (inst, task) in &self.delegations {
            let act = self.activity.get(inst).cloned().unwrap_or_else(|| "thinking…".into());
            lines.push(format!("  {}◐{} {} {}· {} · {}{}", self.c(CYAN), self.c(RESET), inst, self.c(DIM), task, act, self.c(RESET)));
        }
        let width = terminal_width();
        for l in &lines {
            let _ = writeln!(self.out, "{}", clip(l, width));
        }
        self.panel_lines = lines.len();
        let _ = self.out.flush();
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
                    self.activity.insert(ev.agent.clone(), format!("{name} {summary}"));
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
                self.activity.remove(&worker);
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

fn terminal_width() -> usize {
    std::env::var("COLUMNS").ok().and_then(|c| c.parse().ok()).unwrap_or(120)
}

/// Clip a line to `width` visible chars (ANSI escapes don't count).
fn clip(s: &str, width: usize) -> String {
    let mut out = String::new();
    let mut visible = 0;
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
        if visible + 1 >= width {
            out.push('…');
            out.push_str(RESET);
            break;
        }
        out.push(ch);
        visible += 1;
    }
    out
}

pub fn spawn_renderer(
    screen: SharedScreen,
    mut rx: mpsc::UnboundedReceiver<AgentEvent>,
    mut log: Option<crate::log::EventLog>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        while let Some(ev) = rx.recv().await {
            if let Some(l) = log.as_mut() {
                l.handle(&ev);
            }
            screen.lock().await.handle(ev);
        }
    })
}
