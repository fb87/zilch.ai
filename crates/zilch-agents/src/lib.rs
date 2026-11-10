//! zilch OS agents: the recipe book.
//!
//! One generic program, [`LlmAgent`], drives a model through the kernel: every step either asks
//! for one completion or hands the model's tool calls to the kernel as syscalls (they run in
//! parallel), and is woken with the results. It never does I/O itself. The Architect and the
//! Developer are the same program with different prompts, tools and inputs.

use serde_json::{json, Value};
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use zilch_ipc::{ChatRequest, ExitStatus, Group, Message, ModelBinding, PlanTask, Syscall, SyscallResult, TicketId, ToolCall, ToolSpec};
use zilch_kernel::{AcbView, Agent, Recipes, StepResult};

pub mod compact;

pub const MAIN_PROMPT: &str = include_str!("../prompts/main.md");
pub const DEVELOPER_PROMPT: &str = include_str!("../prompts/developer.md");

/// The Architect's extra tool: hand the plan to the kernel.
pub fn submit_plan_spec() -> ToolSpec {
    ToolSpec::new(
        "submit_plan",
        "Submit the implementation plan. The guest reviews it; once approved, every task runs on its own developer \
         (independent tasks in parallel). Returns when all tasks finished, with each developer's report — or the \
         guest's feedback / rejection.",
        json!({"type":"object","properties":{"tasks":{"type":"array","items":{"type":"object","properties":{
            "id":{"type":"string","description":"Short id, e.g. T1"},
            "title":{"type":"string"},
            "task":{"type":"string","description":"Complete, self-contained brief for the developer"},
            "depends_on":{"type":"array","items":{"type":"string"}},
            "owned_paths":{"type":"array","items":{"type":"string"},"description":"Files/globs this task may write (disjoint from parallel tasks)"},
            "acceptance":{"type":"array","items":{"type":"string"}},
            "priority":{"type":"integer","description":"0 most urgent … 3 background (default 1)"}
        },"required":["id","title","task","owned_paths"]}}},"required":["tasks"]}),
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum Phase {
    Start,
    AwaitCompletion,
    AwaitTools,
    /// Resident only: served the last order, waiting for the next one.
    AwaitInput,
    /// Waiting for the summary of `history[..cut]`.
    Compacting(usize),
}

/// A model-driven process.
pub struct LlmAgent {
    binding: ModelBinding,
    system: String,
    /// The task as first given (a restart starts from it again).
    task: String,
    /// The first message of the current attempt: the task, plus a restart summary after a restart.
    input: String,
    tools: Vec<ToolSpec>,
    session: String,
    messages: Vec<Message>,
    phase: Phase,
    /// The tool calls of the last reply, in order.
    calls: Vec<ToolCall>,
    /// For each call: `Some(text)` if answered locally (invalid arguments), else it was issued.
    local: Vec<Option<String>>,
    /// Resident (the main agent): a final answer serves the order and waits for the next one
    /// instead of exiting, and a failed model call is retried once instead of ending it.
    resident: bool,
    failures: u32,
    /// Context budget in tokens (see `compact`).
    budget: usize,
    /// The history was just compacted: send the next request without checking again.
    just_compacted: bool,
}

impl LlmAgent {
    pub fn new(binding: ModelBinding, system: String, input: String, tools: Vec<ToolSpec>, session: String) -> Self {
        Self {
            binding,
            system,
            task: input.clone(),
            input,
            tools,
            session,
            messages: vec![],
            phase: Phase::Start,
            calls: vec![],
            local: vec![],
            resident: false,
            failures: 0,
            budget: 128_000,
            just_compacted: false,
        }
    }

    /// Make this the resident main agent (see `resident`).
    pub fn resident(mut self) -> Self {
        self.resident = true;
        self
    }

    fn request(&self) -> Syscall {
        let mut messages = Vec::with_capacity(self.messages.len() + 1);
        messages.push(Message::system(&self.system));
        messages.extend(self.messages.iter().cloned());
        Syscall::Complete(ChatRequest {
            binding: self.binding.clone(),
            messages,
            tools: self.tools.clone(),
            session: Some(self.session.clone()),
        })
    }

    /// Ask the model, compacting the history first if it outgrew the budget.
    fn ask(&mut self) -> StepResult {
        if !std::mem::take(&mut self.just_compacted) {
            let plan =
                compact::prepare(&self.binding, Some(self.session.clone()), &self.system, &self.tools, &mut self.messages, self.budget);
            if let compact::Plan::Summarize { cut, request } = plan {
                self.phase = Phase::Compacting(cut);
                return StepResult::Syscalls(vec![Syscall::Complete(*request)]);
            }
        }
        self.phase = Phase::AwaitCompletion;
        StepResult::Syscalls(vec![self.request()])
    }

    /// Context budget in tokens (compaction starts at 75% of it).
    pub fn with_budget(mut self, tokens: usize) -> Self {
        self.budget = tokens.max(4_000);
        self
    }

    /// Turn the model's tool calls into syscalls (`submit_plan` becomes the kernel's own call).
    fn dispatch(&mut self, calls: Vec<ToolCall>) -> StepResult {
        let mut syscalls = Vec::new();
        self.local = calls
            .iter()
            .map(|c| {
                let args: Value = match serde_json::from_str(&c.function.arguments) {
                    Ok(v @ Value::Object(_)) => v,
                    _ => {
                        return Some(format!(
                            "Error: arguments for '{}' must be a JSON object; got: {}",
                            c.function.name, c.function.arguments
                        ))
                    }
                };
                if c.function.name == "submit_plan" {
                    match serde_json::from_value::<Vec<PlanTask>>(args.get("tasks").cloned().unwrap_or(Value::Null)) {
                        Ok(tasks) => syscalls.push(Syscall::SubmitPlan { tasks }),
                        Err(e) => return Some(format!("Error: invalid plan: {e}. Each task needs id, title, task and owned_paths.")),
                    }
                } else {
                    syscalls.push(Syscall::Tool { call_id: c.id.clone(), name: c.function.name.clone(), args });
                }
                None
            })
            .collect();
        self.calls = calls;
        self.phase = Phase::AwaitTools;
        StepResult::Syscalls(syscalls)
    }
}

impl Agent for LlmAgent {
    fn restart(&mut self, summary: &str) -> bool {
        self.messages.clear();
        self.calls.clear();
        self.local.clear();
        self.input = format!("{}\n\n{summary}", self.task);
        self.phase = Phase::Start;
        true
    }

    fn step(&mut self, view: &AcbView, wake: Vec<SyscallResult>) -> StepResult {
        let result = match self.phase {
            Phase::Start => {
                self.messages.push(Message::user(&self.input));
                None
            }
            Phase::AwaitCompletion => match wake.into_iter().next() {
                Some(SyscallResult::Completion { message, .. }) => {
                    let calls = message.tool_calls.clone();
                    let text = message.text().to_string();
                    self.messages.push(message);
                    self.failures = 0;
                    if calls.is_empty() && view.notes.is_empty() && self.resident {
                        self.phase = Phase::AwaitInput;
                        Some(StepResult::Syscalls(vec![Syscall::Serve { summary: text }]))
                    } else if calls.is_empty() && view.notes.is_empty() {
                        Some(StepResult::Exit(ExitStatus::ok(text)))
                    } else if calls.is_empty() {
                        None // new input arrived (guest note, nudge): read it and answer again
                    } else {
                        Some(self.dispatch(calls))
                    }
                }
                Some(other) if self.resident => {
                    // Keep the session alive: retry once, then report the failure and wait.
                    self.failures += 1;
                    if self.failures < 2 {
                        None
                    } else {
                        self.failures = 0;
                        self.phase = Phase::AwaitInput;
                        Some(StepResult::Syscalls(vec![Syscall::Serve { summary: format!("[the model call failed: {}]", other.text()) }]))
                    }
                }
                Some(other) => {
                    Some(StepResult::Exit(ExitStatus::failed(format!("[the model call failed: {}]\nSTATUS: failed", other.text()))))
                }
                None => Some(StepResult::Exit(ExitStatus::failed("[the model call returned nothing]\nSTATUS: failed"))),
            },
            Phase::Compacting(cut) => {
                // The summary of the older turns (on failure the elided history is kept as is).
                if let Some(SyscallResult::Completion { message, .. }) = wake.into_iter().next() {
                    if !message.text().trim().is_empty() && cut < self.messages.len() {
                        self.messages = compact::splice(&self.messages, cut, message.text());
                    }
                }
                self.just_compacted = true;
                None
            }
            Phase::AwaitInput => {
                // The next order (or a note) from the guest.
                for r in wake {
                    let text = match r {
                        SyscallResult::Message(m) => m.body,
                        other => other.text(),
                    };
                    self.messages.push(Message::user(text));
                }
                None
            }
            Phase::AwaitTools => {
                let mut issued = wake.into_iter();
                for (call, local) in self.calls.iter().zip(std::mem::take(&mut self.local)) {
                    let text = local.unwrap_or_else(|| issued.next().map(|r| r.text()).unwrap_or_else(|| "Error: no result".into()));
                    self.messages.push(Message::tool(&call.id, text));
                }
                None
            }
        };
        if let Some(r) = result {
            return r;
        }
        // A supervisor nudge goes in front of the model before its next reply.
        for note in &view.notes {
            self.messages.push(Message::user(note));
        }
        self.ask()
    }
}

/// The recipe book the kernel builds processes from.
pub struct Book {
    pub architect: ModelBinding,
    pub developer: ModelBinding,
    pub architect_tools: Vec<ToolSpec>,
    pub developer_tools: Vec<ToolSpec>,
    /// Environment section shared by every prompt (workspace, OS, date, AGENTS.md).
    pub environment: String,
    /// Context budget of the resident main agent in tokens (compaction starts at 75%).
    pub main_context_tokens: usize,
    sessions: AtomicU64,
}

impl Book {
    /// `architect_tools` should not include `submit_plan`; it is added here.
    pub fn new(
        architect: ModelBinding,
        developer: ModelBinding,
        mut architect_tools: Vec<ToolSpec>,
        developer_tools: Vec<ToolSpec>,
        cwd: &Path,
    ) -> Self {
        architect_tools.push(submit_plan_spec());
        Self {
            architect,
            developer,
            architect_tools,
            developer_tools,
            environment: environment(cwd),
            main_context_tokens: 64_000,
            sessions: AtomicU64::new(1),
        }
    }

    fn session(&self, ticket: TicketId, group: Group) -> String {
        let n = self.sessions.fetch_add(1, Ordering::Relaxed);
        let salt = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        format!("zilch-os-t{ticket}-{}-{n}-{salt:08x}", group.name())
    }
}

impl Recipes for Book {
    fn architect(&self, ticket: TicketId, order: &str) -> Box<dyn Agent> {
        let system = format!("{MAIN_PROMPT}\n{}", self.environment);
        Box::new(
            LlmAgent::new(
                self.architect.clone(),
                system,
                order.to_string(),
                self.architect_tools.clone(),
                self.session(ticket, Group::Architecture),
            )
            .resident()
            // The main agent hears everything all session: compact early so it stays fast.
            .with_budget(self.main_context_tokens),
        )
    }

    fn developer(&self, ticket: TicketId, task: &PlanTask, context: &str) -> Box<dyn Agent> {
        let system = format!("{DEVELOPER_PROMPT}\n{}", self.environment);
        Box::new(LlmAgent::new(
            self.developer.clone(),
            system,
            developer_brief(task, context),
            self.developer_tools.clone(),
            self.session(ticket, Group::Development),
        ))
    }
}

/// The Developer's first message: its task, what it may write, and the rest of the plan.
pub fn developer_brief(task: &PlanTask, context: &str) -> String {
    let owned = if task.owned_paths.is_empty() { "any file (you run alone)".to_string() } else { task.owned_paths.join(", ") };
    let mut s = format!("Task [{}] {}\n\n{}\n\nYou may write only: {owned}\n", task.id, task.title, task.task.trim());
    if !task.acceptance.is_empty() {
        s.push_str("\nDone when:\n");
        for a in &task.acceptance {
            s.push_str(&format!("- {a}\n"));
        }
    }
    s.push_str(&format!("\nThe whole plan (other developers work on the other tasks in parallel; do not touch their files):\n{context}\n"));
    s
}

/// Workspace, OS, date and AGENTS.md, read once when the book is made.
pub fn environment(cwd: &Path) -> String {
    let mut s = format!(
        "\n# Environment\n- Workspace: {}\n- OS: {}/{}\n- Date: {}\n",
        cwd.display(),
        std::env::consts::OS,
        std::env::consts::ARCH,
        today()
    );
    if let Some((path, text)) =
        cwd.ancestors().find_map(|d| std::fs::read_to_string(d.join("AGENTS.md")).ok().map(|t| (d.join("AGENTS.md"), t)))
    {
        s.push_str(&format!("\n# Project instructions ({})\n{}\n", path.display(), text.trim()));
    }
    s
}

/// Today's date (UTC) as YYYY-MM-DD without a date crate (Howard Hinnant's civil_from_days).
pub fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests;
