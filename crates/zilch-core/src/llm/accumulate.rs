//! Folds streaming deltas into a final assistant message.

use super::types::{FunctionCall, LlmEvent, Message, Role, ToolCall, Usage};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct Accumulator {
    text: String,
    reasoning: String,
    calls: BTreeMap<usize, (String, String, String)>, // index -> (id, name, args)
    pub usage: Usage,
    pub finish: Option<String>,
}

impl Accumulator {
    pub fn push(&mut self, ev: &LlmEvent) {
        match ev {
            LlmEvent::TextDelta(t) => self.text.push_str(t),
            LlmEvent::ReasoningDelta(t) => self.reasoning.push_str(t),
            LlmEvent::ToolCallDelta { index, id, name, args } => {
                let e = self.calls.entry(*index).or_default();
                if let Some(id) = id {
                    if e.0.is_empty() {
                        e.0 = id.clone();
                    }
                }
                if let Some(n) = name {
                    // Some providers resend the full name; others stream it in pieces.
                    if !e.1.ends_with(n.as_str()) {
                        e.1.push_str(n);
                    }
                }
                e.2.push_str(args);
            }
            LlmEvent::Usage(u) => self.usage = *u,
            LlmEvent::Finish(r) => self.finish = Some(r.clone()),
        }
    }

    pub fn finish(self) -> Message {
        let tool_calls = self
            .calls
            .into_iter()
            .map(|(i, (id, name, args))| ToolCall {
                id: if id.is_empty() { format!("call_{i}") } else { id },
                kind: "function".into(),
                function: FunctionCall {
                    name,
                    arguments: if args.trim().is_empty() { "{}".into() } else { args },
                },
            })
            .collect();
        Message {
            role: Role::Assistant,
            content: if self.text.is_empty() { None } else { Some(self.text) },
            tool_calls,
            tool_call_id: None,
            reasoning_content: if self.reasoning.is_empty() { None } else { Some(self.reasoning) },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::sse::{parse_chunk, SseDecoder};

    #[test]
    fn merges_split_args_and_parallel_calls() {
        let raw = concat!(
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"Let me look.\"}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"a\",\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"{\\\"path\\\":\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":1,\"id\":\"b\",\"function\":{\"name\":\"glob\",\"arguments\":\"{\\\"pattern\\\":\\\"*.rs\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"a.txt\\\"}\"}}]}}]}\n\n",
            "data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n",
            "data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n\n",
            "data: [DONE]\n\n",
        );
        let mut d = SseDecoder::default();
        let mut acc = Accumulator::default();
        // Feed in awkward 7-byte pieces to exercise buffering.
        for piece in raw.as_bytes().chunks(7) {
            for data in d.push(piece) {
                if let Some(evs) = parse_chunk(&data).unwrap() {
                    evs.iter().for_each(|e| acc.push(e));
                }
            }
        }
        assert_eq!(acc.usage.prompt_tokens, 10);
        let m = acc.finish();
        assert_eq!(m.text(), "Let me look.");
        assert_eq!(m.tool_calls.len(), 2);
        assert_eq!(m.tool_calls[0].function.name, "read_file");
        assert_eq!(m.tool_calls[0].function.arguments, "{\"path\":\"a.txt\"}");
        assert_eq!(m.tool_calls[1].id, "b");
        assert_eq!(m.tool_calls[1].function.arguments, "{\"pattern\":\"*.rs\"}");
    }
}
