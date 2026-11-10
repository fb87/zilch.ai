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
    fn stream(&self, req: &ChatRequest, cancel: &CancelToken, on_event: &mut dyn FnMut(LlmEvent) -> bool) -> Result<()>;
}

/// Run a request to completion: accumulate the reply, forward text as it streams, and cut a
/// reply that keeps repeating itself (`true` as the third value; its usage is then estimated).
pub fn complete(
    backend: &dyn Backend,
    req: &ChatRequest,
    cancel: &CancelToken,
    mut on_text: impl FnMut(&str),
) -> Result<(Message, Usage, bool)> {
    let mut acc = accumulate::Accumulator::default();
    let mut guard = StreamGuard::new();
    let mut cut = false;
    backend.stream(req, cancel, &mut |ev| {
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
