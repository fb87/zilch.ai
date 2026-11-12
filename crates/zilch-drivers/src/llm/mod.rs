//! LLM driver: the gas supply. Ported wire formats (SSE, Chat Completions / Responses /
//! Anthropic) over a blocking HTTP client, plus a scripted fake for tests.

pub mod accumulate;
pub mod catalog;
pub mod client;
pub mod fake;
pub mod protocol;
pub mod sse;

pub use zilch_ipc::llm::*;

use anyhow::Result;
use zilch_kernel::supervisor::StreamGuard;
use zilch_kernel::CancelToken;

/// Something that streams a chat completion. `on_event` returns false to stop the stream.
pub trait Backend: Send + Sync {
    /// `alive` is called while the call is still making progress (queued, retrying, streaming).
    fn stream(&self, req: &ChatRequest, cancel: &CancelToken, alive: &dyn Fn(), on_event: &mut dyn FnMut(LlmEvent) -> bool) -> Result<()>;
}

/// Run a request to completion: accumulate the reply, forward text as it streams, and cut a
/// reply that keeps repeating itself (`true` as the third value; its usage is then estimated).
pub fn complete(
    backend: &dyn Backend,
    req: &ChatRequest,
    cancel: &CancelToken,
    alive: &dyn Fn(),
    mut on_text: impl FnMut(&str),
) -> Result<(Message, Usage, bool)> {
    let mut acc = accumulate::Accumulator::default();
    let mut guard = StreamGuard::new();
    let mut cut = false;
    backend.stream(req, cancel, alive, &mut |ev| {
        alive();
        if let LlmEvent::TextDelta(t) = &ev {
            on_text(t);
            if guard.push(t) {
                cut = true;
            }
        }
        acc.push(&ev);
        !cut && !cancel.is_cancelled()
    })?;
    if cancel.is_cancelled() {
        anyhow::bail!("cancelled");
    }
    let mut usage = acc.usage;
    let mut msg = acc.finish();
    if cut {
        let text = msg.content.take().unwrap_or_default();
        if usage.prompt_tokens == 0 {
            usage.prompt_tokens = (serde_json::to_string(&req.messages).map(|s| s.len()).unwrap_or(0) / 4) as u64;
            usage.completion_tokens = (text.len() / 4) as u64;
        }
        let head: String = text.chars().take(400).collect();
        msg.content = Some(format!("{head}… [cut by the supervisor: this reply kept repeating itself]"));
        msg.tool_calls.clear();
    }
    Ok((msg, usage, cut))
}

/// Estimated tokens per part of a request (≈4 characters per token): system, tools, user, assistant
/// (text and tool calls), reasoning, and each tool's outputs as `tool:<name>`.
pub fn request_parts(req: &zilch_ipc::ChatRequest) -> Vec<(String, u64)> {
    use zilch_ipc::llm::Role;
    let tok = |chars: usize| (chars / 4) as u64;
    let mut parts: std::collections::BTreeMap<String, u64> = Default::default();
    let mut add = |k: &str, n: u64| *parts.entry(k.to_string()).or_default() += n;
    add("tools", tok(serde_json::to_string(&req.tools).map(|s| s.len()).unwrap_or(0)));
    let mut names: std::collections::HashMap<&str, &str> = Default::default();
    for m in &req.messages {
        for c in &m.tool_calls {
            names.insert(c.id.as_str(), c.function.name.as_str());
        }
    }
    for m in &req.messages {
        let text = m.content.as_ref().map_or(0, String::len) + 16;
        match m.role {
            Role::System => add("system", tok(text)),
            Role::Tool => {
                let name = m.tool_call_id.as_deref().and_then(|id| names.get(id)).copied().unwrap_or("?");
                add(&format!("tool:{name}"), tok(text));
            }
            Role::Assistant => {
                let calls: usize = m.tool_calls.iter().map(|c| c.function.name.len() + c.function.arguments.len() + 16).sum();
                add("assistant", tok(text + calls));
            }
            Role::User => add("user", tok(text)),
        }
        if let Some(r) = &m.reasoning_content {
            add("reasoning", tok(r.len()));
        }
    }
    parts.into_iter().filter(|(_, n)| *n > 0).collect()
}

#[cfg(test)]
mod parts_tests {
    use super::*;
    use zilch_ipc::llm::{ChatRequest, FunctionCall, Message, ModelBinding, Role, ToolCall};

    #[test]
    fn each_tool_output_is_attributed_to_its_tool() {
        let call = Message {
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall {
                id: "c1".into(),
                kind: "function".into(),
                function: FunctionCall { name: "read_file".into(), arguments: "{}".into() },
            }],
            tool_call_id: None,
            reasoning_content: Some("r".repeat(400)),
        };
        let messages = vec![Message::system("s".repeat(784)), Message::user("u".repeat(384)), call, Message::tool("c1", "x".repeat(3984))];
        let req = ChatRequest { binding: ModelBinding::default(), messages, tools: vec![], session: None };
        let parts: std::collections::HashMap<String, u64> = request_parts(&req).into_iter().collect();
        assert_eq!((parts["system"], parts["user"], parts["tool:read_file"], parts["reasoning"]), (200, 100, 1000, 100));
    }
}
