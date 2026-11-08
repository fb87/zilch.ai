//! Events emitted by every agent; frontends (CLI, TUI, server) render them.

use crate::llm::Usage;
use crate::team::todo::{Todo, TodoStatus};
use tokio::sync::mpsc;

#[derive(Debug, Clone)]
pub struct AgentEvent {
    /// "master", "coder#2", ...
    pub agent: String,
    pub kind: EventKind,
}

#[derive(Debug, Clone)]
pub enum EventKind {
    TextDelta(String),
    /// An assistant turn finished streaming text (used to end lines cleanly).
    TextDone,
    ToolStart { call_id: String, name: String, summary: String },
    ToolEnd { call_id: String, name: String, ok: bool, summary: String },
    TodosChanged(Vec<Todo>),
    TodoStatus { id: String, status: TodoStatus, instance: Option<String> },
    DelegateStart { worker: String, task: String },
    DelegateEnd { worker: String, ok: bool },
    InstanceSpawned { id: String, model: String },
    Usage { model: String, usage: Usage },
    Notice(String),
}

/// Cheap, cloneable event sender. Sending never fails (events are dropped if nobody listens).
#[derive(Clone, Debug)]
pub struct EventSink(Option<mpsc::UnboundedSender<AgentEvent>>);

impl EventSink {
    pub fn new() -> (Self, mpsc::UnboundedReceiver<AgentEvent>) {
        let (tx, rx) = mpsc::unbounded_channel();
        (Self(Some(tx)), rx)
    }
    pub fn null() -> Self {
        Self(None)
    }
    pub fn emit(&self, agent: &str, kind: EventKind) {
        if let Some(tx) = &self.0 {
            let _ = tx.send(AgentEvent { agent: agent.to_string(), kind });
        }
    }
}
