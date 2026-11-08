//! The agent loop shared by the master and all workers.

use super::approval::ApprovalRequest;
use super::events::EventKind;
use crate::llm::{complete, ChatRequest, LlmClient, Message, ModelBinding, ToolCall, Usage};
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
        if res.is_err() && !self.history[start..].iter().any(|m| m.role == crate::llm::Role::Assistant) {
            self.history.truncate(start);
        }
        res
    }

    async fn run_turn(&mut self, input: &str) -> Result<String> {
        let cancel = self.ctx.cancel.current();
        self.history.push(Message::user(input));
        let mut reminded = false;

        for _step in 0..self.max_steps {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            if let Some(b) = &self.ctx.budget {
                if b.exhausted() {
                    bail!("session token budget exhausted ({} of {} tokens used); raise max_session_tokens to continue", b.used(), b.limit);
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
            if let Some(extra) = self.hooks.as_ref().and_then(|h| h.context()) {
                messages.push(Message::user(format!("<system-context>\n{extra}\n</system-context>")));
            }
            let req = ChatRequest { binding: self.binding.clone(), messages, tools: specs, session: Some(self.session.clone()) };

            let events = self.ctx.events.clone();
            let id = self.id.clone();
            let llm = self.llm.clone();
            let (msg, usage) = tokio::select! {
                r = complete(llm.as_ref(), &req, |t| events.emit(&id, EventKind::TextDelta(t.to_string()))) => r?,
                _ = cancel.cancelled() => bail!("cancelled"),
            };
            self.usage += usage;
            if let Some(b) = &self.ctx.budget {
                b.add(usage.prompt_tokens + usage.completion_tokens);
            }
            self.emit(EventKind::Usage { model: self.binding.key(), usage });
            if msg.content.is_some() {
                self.emit(EventKind::TextDone);
            }
            let calls = msg.tool_calls.clone();
            self.history.push(msg);

            if calls.is_empty() {
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
            for (call, result) in calls.iter().zip(results) {
                self.history.push(Message::tool(&call.id, result));
            }
        }
        let note = format!("[stopped after reaching max_steps={}]", self.max_steps);
        self.history.push(Message::assistant(&note));
        Ok(note)
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
