//! Plan review (user approval of the TODO list) and the derived plan state.

use super::todo::{Todo, TodoList, TodoStatus};
use async_trait::async_trait;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlanDecision {
    Approve,
    /// Revise according to this feedback and propose again.
    Feedback(String),
    Reject(String),
}

/// Frontend hook that shows the TODO list (grouped into parallel waves) and asks for a decision.
#[async_trait]
pub trait PlanReviewer: Send + Sync {
    async fn review(&self, summary: &str, todos: &[Todo], waves: &[Vec<String>]) -> PlanDecision;
}

pub struct AutoReviewer;

#[async_trait]
impl PlanReviewer for AutoReviewer {
    async fn review(&self, _: &str, _: &[Todo], _: &[Vec<String>]) -> PlanDecision {
        PlanDecision::Approve
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanState {
    /// Open items exist that the user hasn't approved.
    Drafting,
    /// Approved items are waiting to run.
    Approved,
    Running,
    /// Nothing to do.
    Idle,
}

pub fn plan_state(list: &TodoList) -> PlanState {
    if list.items.iter().any(|t| t.status == TodoStatus::InProgress) {
        PlanState::Running
    } else if !list.unapproved_open().is_empty() {
        PlanState::Drafting
    } else if list.items.iter().any(|t| t.approved && t.status == TodoStatus::Pending) {
        PlanState::Approved
    } else {
        PlanState::Idle
    }
}
