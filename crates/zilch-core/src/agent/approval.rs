//! Approval of mutating tool calls.

use async_trait::async_trait;
use serde::Deserialize;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ApprovalMode {
    /// Ask for every mutating call.
    Ask,
    /// An approved TODO authorizes writes/edits inside its declared files; everything else asks.
    #[default]
    Plan,
    /// Never ask (`--yolo`).
    Auto,
}

#[derive(Debug, Clone)]
pub struct ApprovalRequest {
    pub agent: String,
    pub tool: String,
    pub summary: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny,
    /// Allow this and all future calls of the same tool in this session.
    AlwaysForTool,
}

/// Frontend hook that asks a human. Implementations must serialize prompts themselves.
#[async_trait]
pub trait Approver: Send + Sync {
    async fn approve(&self, req: &ApprovalRequest) -> Decision;
}

pub struct AutoApprove;

#[async_trait]
impl Approver for AutoApprove {
    async fn approve(&self, _: &ApprovalRequest) -> Decision {
        Decision::Allow
    }
}

/// Shared policy: mode + "always allow" memory + the human approver.
pub struct ApprovalGate {
    pub mode: ApprovalMode,
    approver: Arc<dyn Approver>,
    always: Mutex<HashSet<String>>,
}

impl ApprovalGate {
    pub fn new(mode: ApprovalMode, approver: Arc<dyn Approver>) -> Self {
        Self { mode, approver, always: Mutex::new(HashSet::new()) }
    }

    /// `in_scope`: the call only touches files inside the current TODO's approved scope.
    pub async fn check(&self, req: &ApprovalRequest, in_scope: bool) -> bool {
        match self.mode {
            ApprovalMode::Auto => return true,
            ApprovalMode::Plan if in_scope => return true,
            _ => {}
        }
        if self.always.lock().unwrap().contains(&req.tool) {
            return true;
        }
        match self.approver.approve(req).await {
            Decision::Allow => true,
            Decision::Deny => false,
            Decision::AlwaysForTool => {
                self.always.lock().unwrap().insert(req.tool.clone());
                true
            }
        }
    }
}
