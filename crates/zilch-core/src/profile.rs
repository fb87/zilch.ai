//! Agent profiles: role, model binding, tools and system prompt. Workers are spawned from profiles.

use crate::llm::ModelBinding;
use serde::Deserialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentRole {
    Master,
    Worker,
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
    /// Built-in default prompt for this profile.
    pub default_prompt: String,
    /// User-provided prompt text (inline or loaded from file).
    pub custom_prompt: Option<String>,
    pub prompt_mode: PromptMode,
}

impl Profile {
    pub fn is_master(&self) -> bool {
        self.role == AgentRole::Master
    }
}
