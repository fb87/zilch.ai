//! Hand-written serde types for the OpenAI Chat Completions API.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    pub role: Role,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tool_calls: Vec<ToolCall>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Reasoning text from thinking models (DeepSeek-style `reasoning_content`); echoed back on
    /// later requests because some providers require it for tool-call turns.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
}

impl Message {
    pub fn system(s: impl Into<String>) -> Self {
        Self { role: Role::System, content: Some(s.into()), tool_calls: vec![], tool_call_id: None, reasoning_content: None }
    }
    pub fn user(s: impl Into<String>) -> Self {
        Self { role: Role::User, content: Some(s.into()), tool_calls: vec![], tool_call_id: None, reasoning_content: None }
    }
    pub fn assistant(s: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: Some(s.into()), tool_calls: vec![], tool_call_id: None, reasoning_content: None }
    }
    pub fn tool(id: impl Into<String>, s: impl Into<String>) -> Self {
        Self { role: Role::Tool, content: Some(s.into()), tool_calls: vec![], tool_call_id: Some(id.into()), reasoning_content: None }
    }
    pub fn text(&self) -> &str {
        self.content.as_deref().unwrap_or("")
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type", default = "function_type")]
    pub kind: String,
    pub function: FunctionCall,
}

fn function_type() -> String {
    "function".into()
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FunctionCall {
    pub name: String,
    /// JSON-encoded arguments, exactly as the model produced them.
    pub arguments: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolSpec {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub function: FunctionSpec,
}

#[derive(Debug, Clone, Serialize)]
pub struct FunctionSpec {
    pub name: String,
    pub description: String,
    pub parameters: Value,
}

impl ToolSpec {
    pub fn new(name: impl Into<String>, description: impl Into<String>, parameters: Value) -> Self {
        Self {
            kind: "function",
            function: FunctionSpec { name: name.into(), description: description.into(), parameters },
        }
    }
}

/// Per-agent model configuration: which provider, which model, and sampling knobs.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ModelBinding {
    pub provider: String,
    pub model: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_effort: Option<String>,
    /// Passed through verbatim into the request body (provider-specific knobs).
    #[serde(default, skip_serializing_if = "Map::is_empty")]
    pub extra: Map<String, Value>,
}

impl ModelBinding {
    /// Stable key used to group worker instances by binding.
    pub fn key(&self) -> String {
        format!("{}/{}", self.provider, self.model)
    }
}

#[derive(Debug, Clone)]
pub struct ChatRequest {
    pub binding: ModelBinding,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolSpec>,
    /// Stable id of the conversation (one per agent instance); sent if the provider asks for it.
    pub session: Option<String>,
}

impl ChatRequest {
    pub fn to_body(&self) -> Value {
        let b = &self.binding;
        let mut body = serde_json::json!({
            "model": b.model,
            "messages": self.messages,
            "stream": true,
            "stream_options": { "include_usage": true },
        });
        let obj = body.as_object_mut().expect("object");
        if !self.tools.is_empty() {
            obj.insert("tools".into(), serde_json::to_value(&self.tools).expect("tools"));
        }
        if let Some(t) = b.temperature {
            obj.insert("temperature".into(), t.into());
        }
        if let Some(t) = b.top_p {
            obj.insert("top_p".into(), t.into());
        }
        if let Some(t) = b.max_tokens {
            obj.insert("max_tokens".into(), t.into());
        }
        if let Some(r) = &b.reasoning_effort {
            obj.insert("reasoning_effort".into(), r.clone().into());
        }
        for (k, v) in &b.extra {
            obj.insert(k.clone(), v.clone());
        }
        body
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Usage {
    #[serde(default)]
    pub prompt_tokens: u64,
    #[serde(default)]
    pub completion_tokens: u64,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, o: Self) {
        self.prompt_tokens += o.prompt_tokens;
        self.completion_tokens += o.completion_tokens;
    }
}

/// One decoded streaming event.
#[derive(Debug, Clone, PartialEq)]
pub enum LlmEvent {
    TextDelta(String),
    ReasoningDelta(String),
    ToolCallDelta { index: usize, id: Option<String>, name: Option<String>, args: String },
    Usage(Usage),
    Finish(String),
}
