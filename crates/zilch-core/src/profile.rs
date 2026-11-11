//! Agent profiles: role, model binding, tools and system prompt. Workers are spawned from profiles.

use crate::llm::ModelBinding;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    Master,
    Worker,
    /// Requirements engineer of the /spec lane: reads the code, writes only the spec file.
    Spec,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PromptMode {
    /// Built-in default prompt + the profile's prompt.
    #[default]
    Append,
    /// Only the profile's prompt (the role contract is still added).
    Replace,
}

#[derive(Debug, Clone)]
pub struct Profile {
    pub name: String,
    pub description: String,
    pub role: AgentRole,
    pub binding: ModelBinding,
    /// Tool allowlist (empty = role default).
    pub tools: Vec<String>,
    pub max_steps: usize,
    /// Context window budget in tokens; history is compacted near this limit.
    pub context_tokens: usize,
    /// True if context_tokens was set in config (otherwise it may be discovered).
    pub context_explicit: bool,
    /// Explicitly configured prices (override discovered ones).
    pub pricing: crate::llm::catalog::ModelInfo,
    /// Built-in default prompt for this profile.
    pub default_prompt: String,
    /// User-provided prompt text (inline or loaded from file).
    pub custom_prompt: Option<String>,
    pub prompt_mode: PromptMode,
}

/// Compaction budget of the long-lived lane agents (master, spec master) unless configured: they keep
/// the whole session, so a full model window (up to 1M tokens) would make every call slow and costly.
pub const LANE_CONTEXT_TOKENS: usize = 64_000;

impl Profile {
    /// The compaction budget for a model with this context window: the configured `context_tokens`,
    /// else the window (capped for lane agents at [`LANE_CONTEXT_TOKENS`]); at least 16k.
    pub fn context_for(&self, window: Option<u64>) -> usize {
        if self.context_explicit {
            return self.context_tokens;
        }
        let w = window.map_or(128_000, |w| w as usize);
        let w = if matches!(self.role, AgentRole::Master | AgentRole::Spec) { w.min(LANE_CONTEXT_TOKENS) } else { w };
        w.max(16_000)
    }

    pub fn is_master(&self) -> bool {
        self.role == AgentRole::Master
    }
}
