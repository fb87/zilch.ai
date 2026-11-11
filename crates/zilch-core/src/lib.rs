//! zilch-core: a minimal, parallel, TODO-driven multi-agent coding engine.
//!
//! One master (read-only planner/coordinator) + a pool of worker instances (read/write),
//! talking to any OpenAI-compatible endpoint. No terminal I/O here; frontends consume events.

pub mod agent;
pub mod config;
pub mod llm;
pub mod profile;
pub mod prompt;
pub mod repomap;
pub mod team;
pub mod tools;

pub use agent::{AgentEvent, ApprovalMode, Approver, AutoApprove, EventKind, EventSink};
pub use config::{Config, Overrides};
pub use team::plan_state::{AutoReviewer, PlanDecision, PlanReviewer};
pub use team::Team;
