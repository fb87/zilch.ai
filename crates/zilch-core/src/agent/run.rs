//! The agent loop shared by the master and all workers.

use super::approval::ApprovalRequest;
use super::events::EventKind;
use super::supervisor::{nudge_message, CallRecord, StreamGuard, Supervisor, Verdict, STOPPED_BY_SUPERVISOR};
use crate::llm::{complete_until, ChatRequest, LlmClient, Message, ModelBinding, ToolCall, Usage};
use crate::tools::{ToolCtx, ToolRegistry};
use anyhow::{bail, Result};
use futures::future::join_all;
use serde_json::Value;
use std::sync::Arc;

/// Optional per-turn hooks (the master uses these for TODO context and the end-of-turn reminder).
pub trait TurnHooks: Send + Sync {
    /// Ephemeral context appended to every request (not stored in history).
    fn context(&self) -> Option<String> {
        None
    }
    /// Called when the model wants to stop. Return a message to push back and keep going.
    fn on_finish(&self) -> Option<String> {
        None
    }
}

pub struct Agent {
    pub id: String,
    pub profile: String,
    pub llm: Arc<dyn LlmClient>,
    pub binding: ModelBinding,
    pub system: String,
    pub tools: ToolRegistry,
    pub history: Vec<Message>,
    pub max_steps: usize,
    /// Loop protection: repeats tolerated before the supervisor nudges, then stops (0 = off).
    pub loop_limit: usize,
    pub context_tokens: usize,
    pub ctx: Arc<ToolCtx>,
    pub hooks: Option<Arc<dyn TurnHooks>>,
    pub usage: Usage,
    /// Stable conversation id (prompt caching / routing on providers that support it).
    pub session: String,
}

/// Random conversation id like `zilch-coder#2-9f3a…`.
pub fn new_session_id(agent_id: &str) -> String {
    format!("zilch-{}-{:016x}", agent_id.replace('#', "-"), rand::random::<u64>())
}

impl Agent {
    pub fn reset(&mut self) {
        self.history.clear();
        self.ctx.read_hashes.lock().unwrap().clear();
        self.ctx.shown.lock().unwrap().clear();
        self.session = new_session_id(&self.id);
    }

    fn emit(&self, kind: EventKind) {
        self.ctx.events.emit(&self.id, kind);
    }

    /// Run one user/task turn to completion and return the final assistant text.
    ///
    /// Blank input is rejected, and if the turn fails before the model produced anything
    /// (e.g. the provider rejected the request), the history is rolled back so the agent stays
    /// usable instead of replaying the failing request forever.
    pub async fn run(&mut self, input: &str) -> Result<String> {
        if input.trim().is_empty() {
            bail!("empty input: nothing to do");
        }
        let start = self.history.len();
        let res = self.run_turn(input).await;
        self.emit(EventKind::Idle);
        if res.is_err() && !self.history[start..].iter().any(|m| m.role == crate::llm::Role::Assistant) {
            self.history.truncate(start);
            // Files read in the dropped part are no longer in the context.
            self.ctx.shown.lock().unwrap().clear();
        }
        res
    }

    async fn run_turn(&mut self, input: &str) -> Result<String> {
        let cancel = self.ctx.cancel.current();
        // A model's reasoning is needed back only within the tool loop of the turn that produced it;
        // from earlier turns it is just resent tokens. Dropped once per turn (one cache miss).
        for m in &mut self.history {
            m.reasoning_content = None;
        }
        self.history.push(Message::user(input));
        let mut reminded = false;
        let mut supervisor = Supervisor::new(self.loop_limit);
        let mut budget = super::budget::StepBudget::new(self.max_steps);

        loop {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            match budget.before_call() {
                super::budget::Decision::Go => {}
                super::budget::Decision::Extended { limit, writes, fresh } => self.emit(EventKind::Notice(format!(
                    "{}: still making progress ({writes} write(s), {fresh} new result(s)) — budget extended to {limit} model calls",
                    self.id
                ))),
                super::budget::Decision::WrapUp { grace } => {
                    self.emit(EventKind::Notice(format!(
                        "⚠ {}: used its budget of {} model calls — asked to wrap up ({grace} more)",
                        self.id, self.max_steps
                    )));
                    self.history.push(Message::user(super::budget::wrap_up_message(self.max_steps, grace)));
                }
                super::budget::Decision::Stop { limit } => {
                    let note =
                        format!("[stopped after reaching max_steps={} ({limit} model calls with extensions and wrap-up)]", self.max_steps);
                    self.history.push(Message::assistant(&note));
                    return Ok(note);
                }
            }
            let specs = self.tools.specs();
            if let Some(r) = super::compact::maybe_compact(
                self.llm.as_ref(),
                &self.binding,
                Some(self.session.clone()),
                &self.system,
                &specs,
                &mut self.history,
                self.context_tokens,
            )
            .await
            {
                // Older read_file outputs may have been elided or summarized away: read for real again.
                self.ctx.shown.lock().unwrap().clear();
                let how = if r.summarized { "summarized older turns" } else { "elided old tool output" };
                self.emit(EventKind::Notice(format!(
                    "{}: compacted history ~{:.1}k → ~{:.1}k tokens ({how})",
                    self.id,
                    r.before as f64 / 1000.0,
                    r.after as f64 / 1000.0
                )));
            }
            let mut messages = Vec::with_capacity(self.history.len() + 2);
            messages.push(Message::system(&self.system));
            messages.extend(self.history.iter().cloned());
            let extra = self.hooks.as_ref().and_then(|h| h.context());
            if let Some(extra) = &extra {
                messages.push(Message::user(format!("<system-context>\n{extra}\n</system-context>")));
            }
            self.emit(EventKind::RequestParts(super::compact::breakdown(&self.system, &self.history, extra.as_deref(), &specs)));
            let req = ChatRequest { binding: self.binding.clone(), messages, tools: specs, session: Some(self.session.clone()) };
            // Strict caps: refuse to SEND a request that would exceed the token or dollar budget,
            // instead of noticing after the fact.
            let estimate = super::compact::estimate_tokens("", &req.messages, &req.tools) as u64;
            if let Some(b) = &self.ctx.budget {
                if b.used() + estimate > b.limit {
                    bail!(
                        "session token budget reached ({} used + ~{estimate} for the next request > {}); raise max_session_tokens to continue",
                        b.used(),
                        b.limit
                    );
                }
            }
            if self.ctx.meter.would_exceed(&self.binding.key(), estimate) {
                bail!(
                    "session cost cap of ${:.4} would be exceeded (spent ${:.4}); raise max_session_cost to continue",
                    self.ctx.meter.limit_usd.unwrap_or(0.0),
                    self.ctx.meter.spent()
                );
            }

            let events = self.ctx.events.clone();
            let id = self.id.clone();
            let llm = self.llm.clone();
            self.emit(EventKind::Thinking);
            let mut guard = supervisor.enabled().then(StreamGuard::new);
            let on_text = |t: &str| {
                events.emit(&id, EventKind::TextDelta(t.to_string()));
                guard.as_mut().is_none_or(|g| !g.push(t))
            };
            let (mut msg, mut usage, cut) = tokio::select! {
                r = complete_until(llm.as_ref(), &req, on_text) => r?,
                _ = cancel.cancelled() => bail!("cancelled"),
            };
            if cut {
                // The provider never sent usage for a dropped stream: count an estimate so the
                // budget still sees it. Keep only a short head of the runaway text in history.
                let text = msg.content.take().unwrap_or_default();
                if usage.prompt_tokens == 0 {
                    usage.prompt_tokens = estimate;
                    usage.completion_tokens = (text.len() / 4) as u64;
                }
                let head: String = text.chars().take(400).collect();
                msg.content = Some(format!("{head}… [cut by the supervisor: this reply kept repeating itself]"));
                msg.tool_calls.clear();
            }
            self.usage += usage;
            if let Some(b) = &self.ctx.budget {
                b.add(usage.prompt_tokens + usage.completion_tokens);
            }
            let cost = self.ctx.meter.record(&self.binding.key(), &usage);
            self.emit(EventKind::Usage { model: self.binding.key(), usage, cost });
            if msg.content.is_some() {
                self.emit(EventKind::TextDone);
            }
            let calls = msg.tool_calls.clone();
            let text = msg.text().to_string();
            self.history.push(msg);

            if cut {
                let verdict = supervisor.strike("kept repeating the same text in one reply".into());
                if let Some(stop) = self.supervise(verdict) {
                    return Ok(stop);
                }
                continue;
            }

            if calls.is_empty() {
                let verdict = supervisor.observe(&text, &[]);
                if let Some(stop) = self.supervise(verdict) {
                    return Ok(stop);
                }
                if matches!(self.history.last(), Some(m) if m.role == crate::llm::Role::User) {
                    continue; // nudged: let it answer differently
                }
                if !reminded {
                    if let Some(nudge) = self.hooks.as_ref().and_then(|h| h.on_finish()) {
                        reminded = true;
                        self.history.push(Message::user(nudge));
                        continue;
                    }
                }
                return Ok(self.history.last().map(|m| m.text().to_string()).unwrap_or_default());
            }

            let results = self.execute_calls(&calls).await;
            let records: Vec<CallRecord> = calls
                .iter()
                .zip(&results)
                .map(|(c, r)| CallRecord { name: &c.function.name, arguments: &c.function.arguments, result: r })
                .collect();
            let verdict = supervisor.observe(&text, &records);
            for (call, result) in calls.iter().zip(&results) {
                budget.record(&call.function.name, result);
            }
            for (call, result) in calls.iter().zip(results) {
                self.history.push(Message::tool(&call.id, result));
            }
            if let Some(stop) = self.supervise(verdict) {
                return Ok(stop);
            }
        }
    }

    /// Safe-point compaction (call between turns): replace the history with a digest, when it uses
    /// at least `min_share` of the context budget. Returns what it did, or None (not needed or failed).
    pub async fn compact_idle(&mut self, min_share: f64) -> Option<super::compact::CompactReport> {
        let specs = self.tools.specs();
        let before = super::compact::estimate_tokens(&self.system, &self.history, &specs);
        if self.history.len() < 4 || (before as f64) < self.context_tokens as f64 * min_share {
            return None;
        }
        self.emit(EventKind::Notice(format!("{}: compacting finished work (~{:.1}k tokens)…", self.id, before as f64 / 1000.0)));
        let new = super::compact::digest(self.llm.as_ref(), &self.binding, Some(self.session.clone()), &self.system, &specs, &self.history)
            .await
            .ok()?;
        let after = super::compact::estimate_tokens(&self.system, &new, &specs);
        if after >= before {
            return None;
        }
        self.history = new;
        self.ctx.shown.lock().unwrap().clear();
        self.emit(EventKind::Notice(format!(
            "{}: compacted finished work ~{:.1}k → ~{:.1}k tokens",
            self.id,
            before as f64 / 1000.0,
            after as f64 / 1000.0
        )));
        Some(super::compact::CompactReport { before, after, summarized: true })
    }

    /// Act on the supervisor's verdict: a nudge goes into the history (the turn continues); a
    /// stop returns the note that ends the turn. Both are announced to the user.
    fn supervise(&mut self, verdict: Verdict) -> Option<String> {
        match verdict {
            Verdict::Ok => None,
            Verdict::Nudge(reason) => {
                self.emit(EventKind::Notice(format!("⚠ {} looping: {reason} — nudged", self.id)));
                self.history.push(Message::user(nudge_message(&reason)));
                None
            }
            Verdict::Stop(reason) => {
                self.emit(EventKind::Notice(format!("⚠ {} stopped by the supervisor: {reason}", self.id)));
                let note = format!("{STOPPED_BY_SUPERVISOR} it {reason}, again after a warning]\nSTATUS: failed");
                self.history.push(Message::assistant(&note));
                Some(note)
            }
        }
    }

    /// Approve sequentially (prompts are serialized anyway), then execute all approved calls concurrently.
    async fn execute_calls(&self, calls: &[ToolCall]) -> Vec<String> {
        enum Prepared {
            Ready(Value),
            Done(String),
        }
        let mut prepared = Vec::with_capacity(calls.len());
        for call in calls {
            let name = &call.function.name;
            let Some(tool) = self.tools.get(name) else {
                prepared.push(Prepared::Done(format!("Error: unknown tool '{name}'. Available: {}", self.tools.names().join(", "))));
                continue;
            };
            let args: Value = match serde_json::from_str(&call.function.arguments) {
                Ok(v @ Value::Object(_)) => v,
                Ok(_) | Err(_) => {
                    prepared.push(Prepared::Done(format!(
                        "Error: arguments for '{name}' must be a JSON object; got: {}",
                        call.function.arguments
                    )));
                    continue;
                }
            };
            if tool.mutating() {
                let in_scope = tool.target_path(&args).is_some_and(|p| self.ctx.in_scope(&p));
                let req = ApprovalRequest { agent: self.id.clone(), tool: name.clone(), summary: tool.summarize(&args) };
                if !self.ctx.gate.check(&req, in_scope).await {
                    prepared.push(Prepared::Done(
                        "Error: the user denied this action. Do not retry it; adjust your approach or report back.".into(),
                    ));
                    continue;
                }
            }
            prepared.push(Prepared::Ready(args));
        }

        let cancel = self.ctx.cancel.current();
        let futs = calls.iter().zip(prepared).map(|(call, p)| {
            let cancel = cancel.clone();
            async move {
                let args = match p {
                    Prepared::Done(s) => return s,
                    Prepared::Ready(a) => a,
                };
                let tool = self.tools.get(&call.function.name).expect("checked");
                let summary = tool.summarize(&args);
                self.emit(EventKind::ToolStart { call_id: call.id.clone(), name: call.function.name.clone(), summary: summary.clone() });
                let res = tokio::select! {
                    r = tool.call(args, &self.ctx) => r,
                    _ = cancel.cancelled() => Err(anyhow::anyhow!("cancelled")),
                };
                let (ok, out) = match res {
                    Ok(s) => (true, s),
                    Err(e) => (false, format!("Error: {e:#}")),
                };
                // Failures keep more detail: the reason is usually what the user needs to see.
                let first = out.lines().next().unwrap_or("").chars().take(if ok { 100 } else { 400 }).collect::<String>();
                self.emit(EventKind::ToolEnd { call_id: call.id.clone(), name: call.function.name.clone(), ok, summary: first });
                out
            }
        });
        join_all(futs).await
    }
}
