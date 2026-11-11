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

use std::sync::{Arc, RwLock};

/// A model binding several agents follow (one per group); `/model` changes it.
pub type SharedBinding = Arc<RwLock<ModelBinding>>;

pub const MAIN_PROMPT: &str = include_str!("../prompts/main.md");
pub const DEVELOPER_PROMPT: &str = include_str!("../prompts/developer.md");
pub const SPEC_PROMPT: &str = include_str!("../prompts/spec.md");
pub const EXPLORER_PROMPT: &str = include_str!("../prompts/explorer.md");
pub const ANSWERER_PROMPT: &str = include_str!("../prompts/answerer.md");
pub const PLANNER_PROMPT: &str = include_str!("../prompts/planner.md");

/// The main agent's only tool: hand the order to one subagent.
pub fn dispatch_spec() -> ToolSpec {
    ToolSpec::new(
        "dispatch",
        "Hand the guest's order to one subagent and wait for its report. agent \"answer\": questions, explanations, \
         checks, anything conversational. agent \"plan\": any change to the project (it plans, the guest approves, \
         developers carry it out). The brief is all the subagent sees: the request word for word plus the session \
         context it needs.",
        json!({"type":"object","properties":{
            "agent":{"type":"string","enum":["answer","plan"]},
            "brief":{"type":"string","description":"Self-contained: the guest's request verbatim and the context needed"}
        },"required":["agent","brief"]}),
    )
}

/// The main agent's fan-out tool: questions answered by read-only explorers in parallel.
pub fn explore_spec() -> ToolSpec {
    ToolSpec::new(
        "explore",
        "Investigate several questions at once: each goes to its own read-only explorer and they run in parallel. \
         Returns every explorer's report. Only for a broad investigation across code you have not seen (where is X \
         handled, how does Y work in a large project); a few files are cheaper to read yourself, and a question \
         answerable from the conversation needs no tools. Ask precise, independent questions (at most 8).",
        json!({"type":"object","properties":{"questions":{"type":"array","items":{"type":"string"}}},"required":["questions"]}),
    )
}

/// The Architect's extra tool: hand the plan to the kernel.
pub fn submit_plan_spec() -> ToolSpec {
    ToolSpec::new(
        "submit_plan",
        "Only for a change to the project (never to answer a question). Submit the implementation plan. The guest reviews it; once approved, every task runs on its own developer \
         (independent tasks in parallel). Returns when all tasks finished, with each developer's report — or the \
         guest's feedback / rejection.",
        json!({"type":"object","properties":{"tasks":{"type":"array","items":{"type":"object","properties":{
            "id":{"type":"string","description":"Short id, e.g. T1"},
            "title":{"type":"string"},
            "task":{"type":"string","description":"Complete, self-contained brief for the developer"},
            "depends_on":{"type":"array","items":{"type":"string"}},
            "owned_paths":{"type":"array","items":{"type":"string"},"description":"Files, directories (dir/ covers everything below) or globs this task may write; disjoint from parallel tasks"},
            "acceptance":{"type":"array","items":{"type":"string"}},
            "priority":{"type":"integer","description":"0 most urgent … 3 background (default 1)"},
            "worker":{"type":"string","description":"A subagent profile from the environment section (default: developer)"}
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
    /// Shared with the recipe book, so `/model` applies from the next call.
    binding: SharedBinding,
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
            binding: Arc::new(RwLock::new(binding)),
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

    /// Use (and follow) a binding shared with the recipe book.
    pub fn with_shared_binding(mut self, b: SharedBinding) -> Self {
        self.binding = b;
        self
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
            binding: self.binding.read().unwrap().clone(),
            messages,
            tools: self.tools.clone(),
            session: Some(self.session.clone()),
        })
    }

    /// Ask the model, compacting the history first if it outgrew the budget.
    fn ask(&mut self) -> StepResult {
        if !std::mem::take(&mut self.just_compacted) {
            let plan = compact::prepare(
                &self.binding.read().unwrap().clone(),
                Some(self.session.clone()),
                &self.system,
                &self.tools,
                &mut self.messages,
                self.budget,
            );
            if let compact::Plan::Summarize { cut, request } = plan {
                self.phase = Phase::Compacting(cut);
                return StepResult::Syscalls(vec![Syscall::Complete(*request)]);
            }
        }
        self.phase = Phase::AwaitCompletion;
        StepResult::Syscalls(vec![self.request()])
    }

    /// Start from an earlier session's conversation (`-c`): the input becomes the next turn.
    pub fn with_history(mut self, messages: Vec<Message>) -> Self {
        self.messages = messages;
        self
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
                if c.function.name == "explore" {
                    match serde_json::from_value::<Vec<String>>(args.get("questions").cloned().unwrap_or(Value::Null)) {
                        Ok(questions) => syscalls.push(Syscall::Explore { questions }),
                        Err(e) => return Some(format!("Error: explore needs `questions`, a list of strings: {e}")),
                    }
                } else if c.function.name == "dispatch" {
                    let field = |k: &str| args.get(k).and_then(Value::as_str).unwrap_or("").to_string();
                    syscalls.push(Syscall::Dispatch { agent: field("agent"), brief: field("brief") });
                } else if c.function.name == "submit_plan" {
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
    fn snapshot(&self) -> Option<Value> {
        Some(json!({ "messages": self.messages }))
    }

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
                Some(other) => {
                    // A failed call (provider error, or interrupted by the hang detector) gets one
                    // retry; a second failure in a row is reported: the main agent tells the guest and
                    // waits, a subagent fails its task.
                    self.failures += 1;
                    if self.failures < 2 {
                        None
                    } else if self.resident {
                        self.failures = 0;
                        self.phase = Phase::AwaitInput;
                        Some(StepResult::Syscalls(vec![Syscall::Serve { summary: format!("[the model call failed: {}]", other.text()) }]))
                    } else {
                        Some(StepResult::Exit(ExitStatus::failed(format!(
                            "[the model call failed twice: {}]\nSTATUS: failed",
                            other.text()
                        ))))
                    }
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

/// A custom subagent (from config): its own model, tools and prompt.
#[derive(Clone)]
pub struct SubagentProfile {
    pub name: String,
    pub description: String,
    pub binding: ModelBinding,
    pub tools: Vec<ToolSpec>,
    /// (text, replace) — see `Book::prompts`.
    pub prompt: Option<(String, bool)>,
}

/// The built-in prompt with an override applied: appended, or replacing it.
fn with_override(base: &str, o: Option<&(String, bool)>) -> String {
    match o {
        Some((text, true)) => text.trim().to_string(),
        Some((text, false)) => format!("{}\n\n{}", base.trim(), text.trim()),
        None => base.trim().to_string(),
    }
}

/// The recipe book the kernel builds processes from.
pub struct Book {
    pub architect: SharedBinding,
    pub developer: SharedBinding,
    /// The main agent's tools (just `dispatch`).
    pub architect_tools: Vec<ToolSpec>,
    /// Read-only tools of the subagents the main agent dispatches (answerer, planner).
    pub read_tools: Vec<ToolSpec>,
    pub developer_tools: Vec<ToolSpec>,
    /// Environment section shared by every prompt (workspace, OS, date, AGENTS.md).
    pub environment: String,
    /// Context budget of the resident main agent in tokens (compaction starts at 75%).
    pub main_context_tokens: usize,
    /// A saved conversation for the next main agent to continue (`-c`), used once.
    pub resume: std::sync::Mutex<Option<Vec<Message>>>,
    /// The spec lane: its tools (empty = no spec lane), where its documents live, its saved
    /// conversation. It follows the main agent's model.
    pub spec_tools: Vec<ToolSpec>,
    pub spec_dir: String,
    pub resume_spec: std::sync::Mutex<Option<Vec<Message>>>,
    /// Prompt overrides per group: (text, replace the built-in prompt instead of appending).
    pub prompts: std::collections::HashMap<Group, (String, bool)>,
    /// Custom subagent profiles a plan task can name in `worker`.
    pub profiles: Vec<SubagentProfile>,
    /// Read-only tools for explorers (empty = no explorers).
    pub explorer_tools: Vec<ToolSpec>,
    sessions: AtomicU64,
}

impl Book {
    /// `read_tools`: what the answerer and planner read with (explore and submit_plan are added
    /// here); the main agent itself only dispatches.
    pub fn new(
        architect: ModelBinding,
        developer: ModelBinding,
        read_tools: Vec<ToolSpec>,
        developer_tools: Vec<ToolSpec>,
        cwd: &Path,
    ) -> Self {
        Self {
            architect: Arc::new(RwLock::new(architect)),
            developer: Arc::new(RwLock::new(developer)),
            architect_tools: vec![dispatch_spec()],
            read_tools,
            developer_tools,
            environment: environment(cwd),
            main_context_tokens: 64_000,
            resume: std::sync::Mutex::new(None),
            spec_tools: vec![],
            spec_dir: String::new(),
            resume_spec: std::sync::Mutex::new(None),
            prompts: Default::default(),
            profiles: vec![],
            explorer_tools: vec![],
            sessions: AtomicU64::new(1),
        }
    }

    /// Give this kitchen a spec lane with these tools; its documents live in `dir`.
    pub fn with_spec(mut self, tools: Vec<ToolSpec>, dir: &Path) -> Self {
        self.spec_tools = tools;
        self.spec_dir = dir.display().to_string();
        self
    }

    /// The model a group uses now.
    pub fn binding(&self, group: Group) -> ModelBinding {
        match group {
            Group::Architecture => self.architect.read().unwrap().clone(),
            _ => self.developer.read().unwrap().clone(),
        }
    }

    /// `/model`: switch a group's model; its agents use it from their next model call.
    pub fn set_binding(&self, group: Group, b: ModelBinding) {
        let slot = if group == Group::Architecture { &self.architect } else { &self.developer };
        *slot.write().unwrap() = b;
    }

    /// `/prompt`: the system prompt a group's agents get.
    pub fn system_prompt(&self, group: Group) -> String {
        match group {
            Group::Architecture => {
                format!("{}\n{}", with_override(MAIN_PROMPT, self.prompts.get(&Group::Architecture)), self.environment)
            }
            Group::Spec => format!(
                "{}\n- Spec directory: {}\n{}",
                with_override(SPEC_PROMPT, self.prompts.get(&Group::Spec)),
                self.spec_dir,
                self.environment
            ),
            _ => format!("{}\n{}", with_override(DEVELOPER_PROMPT, self.prompts.get(&Group::Development)), self.environment),
        }
    }

    /// The prompt of a subagent the main agent dispatches to ("answer" or "plan").
    pub fn subagent_prompt(&self, agent: &str) -> Option<String> {
        match agent {
            "answer" => Some(format!("{ANSWERER_PROMPT}\n{}", self.environment)),
            "plan" => {
                let profiles = if self.profiles.is_empty() {
                    String::new()
                } else {
                    let list: String = self.profiles.iter().map(|p| format!("\n  - {}: {}", p.name, p.description)).collect();
                    format!("\nDeveloper profiles a task can name in `worker` (default: developer):{list}\n")
                };
                // Mention the spec lane's design and plan only when they exist, so the planner does
                // not go looking for them on every change.
                let spec = if !self.spec_dir.is_empty()
                    && ["design.md", "plan.md"].iter().any(|f| Path::new(&self.spec_dir).join(f).is_file())
                {
                    format!(
                        "\nThe spec lane has written {dir}/design.md and {dir}/plan.md. When the brief asks to build from the spec or the plan, read them and work from them (do not edit them); otherwise ignore them.\n",
                        dir = self.spec_dir
                    )
                } else {
                    String::new()
                };
                Some(format!("{PLANNER_PROMPT}{profiles}{spec}\n{}", self.environment))
            }
            _ => None,
        }
    }

    fn session(&self, ticket: TicketId, group: Group) -> String {
        let n = self.sessions.fetch_add(1, Ordering::Relaxed);
        let salt = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos()).unwrap_or(0);
        format!("zilch-t{ticket}-{}-{n}-{salt:08x}", group.name())
    }
}

impl Recipes for Book {
    fn architect(&self, ticket: TicketId, order: &str) -> Box<dyn Agent> {
        let system = self.system_prompt(Group::Architecture);
        Box::new(
            LlmAgent::new(
                self.binding(Group::Architecture),
                system,
                order.to_string(),
                self.architect_tools.clone(),
                self.session(ticket, Group::Architecture),
            )
            .with_shared_binding(self.architect.clone())
            .resident()
            // The main agent hears everything all session: compact early so it stays fast.
            .with_budget(self.main_context_tokens)
            .with_history(self.resume.lock().unwrap().take().unwrap_or_default()),
        )
    }

    fn explorer(&self, ticket: TicketId, question: &str) -> Option<Box<dyn Agent>> {
        if self.explorer_tools.is_empty() {
            return None;
        }
        let system = format!("{EXPLORER_PROMPT}\n{}", self.environment);
        Some(Box::new(
            LlmAgent::new(
                self.binding(Group::Development),
                system,
                question.to_string(),
                self.explorer_tools.clone(),
                self.session(ticket, Group::Development),
            )
            .with_shared_binding(self.developer.clone()),
        ))
    }

    fn subagent(&self, ticket: TicketId, agent: &str, brief: &str) -> Option<Box<dyn Agent>> {
        let system = self.subagent_prompt(agent)?;
        let mut tools = self.read_tools.clone();
        if !self.explorer_tools.is_empty() {
            tools.push(explore_spec());
        }
        if agent == "plan" {
            tools.push(submit_plan_spec());
        }
        // They do the thinking the guest is waiting for: the main agent's model.
        Some(Box::new(
            LlmAgent::new(self.binding(Group::Architecture), system, brief.to_string(), tools, self.session(ticket, Group::Architecture))
                .with_shared_binding(self.architect.clone()),
        ))
    }

    fn spec(&self, ticket: TicketId, order: &str) -> Option<Box<dyn Agent>> {
        if self.spec_tools.is_empty() {
            return None;
        }
        Some(Box::new(
            LlmAgent::new(
                self.binding(Group::Architecture),
                self.system_prompt(Group::Spec),
                order.to_string(),
                self.spec_tools.clone(),
                self.session(ticket, Group::Spec),
            )
            .with_shared_binding(self.architect.clone())
            .resident()
            .with_budget(self.main_context_tokens)
            .with_history(self.resume_spec.lock().unwrap().take().unwrap_or_default()),
        ))
    }

    fn developer(&self, ticket: TicketId, task: &PlanTask, context: &str) -> Box<dyn Agent> {
        // A task naming a custom profile runs on that profile's model, tools and prompt.
        if let Some(p) = task.worker.as_deref().and_then(|w| self.profiles.iter().find(|p| p.name == w)) {
            let system = format!("{}\n{}", with_override(DEVELOPER_PROMPT, p.prompt.as_ref()), self.environment);
            return Box::new(LlmAgent::new(
                p.binding.clone(),
                system,
                developer_brief(task, context),
                p.tools.clone(),
                self.session(ticket, Group::Development),
            ));
        }
        let system = self.system_prompt(Group::Development);
        Box::new(
            LlmAgent::new(
                self.binding(Group::Development),
                system,
                developer_brief(task, context),
                self.developer_tools.clone(),
                self.session(ticket, Group::Development),
            )
            .with_shared_binding(self.developer.clone()),
        )
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

/// The conversation saved by `LlmAgent::snapshot` (None if the snapshot is not one).
pub fn snapshot_messages(snapshot: &Value) -> Option<Vec<Message>> {
    serde_json::from_value(snapshot.get("messages")?.clone()).ok()
}
