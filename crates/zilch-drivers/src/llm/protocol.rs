//! Wire protocols: OpenAI Chat Completions, OpenAI Responses, Anthropic Messages.
//! Each maps zilch's provider-neutral `ChatRequest`/`Message`s to the wire format and decodes
//! its stream into the same `LlmEvent`s, so the agent loop never knows which one is in use.

use super::sse::parse_chunk as parse_chat_chunk;
use anyhow::{bail, Result};
use serde::Deserialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use zilch_ipc::llm::{ChatRequest, LlmEvent, Role, Usage};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Protocol {
    /// `POST {base}/chat/completions` (OpenAI and nearly every compatible server).
    #[default]
    Chat,
    /// `POST {base}/responses` (OpenAI Responses API).
    Responses,
    /// `POST {base}/messages` (Anthropic Messages API).
    Anthropic,
}

impl Protocol {
    pub fn path(self) -> &'static str {
        match self {
            Protocol::Chat => "/chat/completions",
            Protocol::Responses => "/responses",
            Protocol::Anthropic => "/messages",
        }
    }

    pub fn body(self, req: &ChatRequest) -> Value {
        match self {
            Protocol::Chat => req.to_body(),
            Protocol::Responses => responses_body(req),
            Protocol::Anthropic => anthropic_body(req),
        }
    }

    pub fn parser(self) -> StreamParser {
        StreamParser { protocol: self, input_tokens: 0, done: false, tool_index: HashMap::new() }
    }
}

fn system_text(req: &ChatRequest) -> String {
    req.messages.iter().filter(|m| m.role == Role::System).map(|m| m.text()).collect::<Vec<_>>().join("\n\n")
}

fn insert_common(obj: &mut Map<String, Value>, req: &ChatRequest, max_tokens_key: &str) {
    let b = &req.binding;
    if let Some(t) = b.temperature {
        obj.insert("temperature".into(), t.into());
    }
    if let Some(t) = b.top_p {
        obj.insert("top_p".into(), t.into());
    }
    if let Some(t) = b.max_tokens {
        obj.insert(max_tokens_key.into(), t.into());
    }
    for (k, v) in &b.extra {
        obj.insert(k.clone(), v.clone());
    }
}

// ---------------- OpenAI Responses ----------------

fn responses_body(req: &ChatRequest) -> Value {
    let mut input = Vec::new();
    for m in &req.messages {
        match m.role {
            Role::System => {}
            Role::User => input.push(json!({"type": "message", "role": "user", "content": m.text()})),
            Role::Assistant => {
                if !m.text().is_empty() {
                    input.push(json!({"type": "message", "role": "assistant", "content": m.text()}));
                }
                for c in &m.tool_calls {
                    input.push(
                        json!({"type": "function_call", "call_id": c.id, "name": c.function.name, "arguments": c.function.arguments}),
                    );
                }
            }
            Role::Tool => {
                input.push(json!({"type": "function_call_output", "call_id": m.tool_call_id, "output": m.text()}));
            }
        }
    }
    let mut body = json!({"model": req.binding.model, "input": input, "stream": true, "store": false});
    let obj = body.as_object_mut().expect("object");
    let sys = system_text(req);
    if !sys.is_empty() {
        obj.insert("instructions".into(), sys.into());
    }
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({"type": "function", "name": t.function.name, "description": t.function.description, "parameters": t.function.parameters}))
            .collect();
        obj.insert("tools".into(), tools.into());
    }
    if let Some(e) = &req.binding.reasoning_effort {
        obj.insert("reasoning".into(), json!({"effort": e}));
    }
    insert_common(obj, req, "max_output_tokens");
    body
}

// ---------------- Anthropic Messages ----------------

fn anthropic_body(req: &ChatRequest) -> Value {
    // Anthropic wants alternating user/assistant turns of content blocks; tool results are user
    // blocks. Consecutive same-role messages are merged.
    let mut msgs: Vec<(String, Vec<Value>)> = Vec::new();
    let mut push = |role: &str, block: Value| match msgs.last_mut() {
        Some((r, blocks)) if r == role => blocks.push(block),
        _ => msgs.push((role.to_string(), vec![block])),
    };
    for m in &req.messages {
        match m.role {
            Role::System => {}
            Role::User => {
                if !m.text().is_empty() {
                    push("user", json!({"type": "text", "text": m.text()}));
                }
            }
            Role::Assistant => {
                if !m.text().is_empty() {
                    push("assistant", json!({"type": "text", "text": m.text()}));
                }
                for c in &m.tool_calls {
                    let input: Value = serde_json::from_str(&c.function.arguments).unwrap_or_else(|_| json!({}));
                    let input = if input.is_object() { input } else { json!({}) };
                    push("assistant", json!({"type": "tool_use", "id": c.id, "name": c.function.name, "input": input}));
                }
            }
            Role::Tool => push(
                "user",
                json!({"type": "tool_result", "tool_use_id": m.tool_call_id, "content": if m.text().is_empty() { "(empty)" } else { m.text() }}),
            ),
        }
    }
    let messages: Vec<Value> = msgs.into_iter().map(|(role, content)| json!({"role": role, "content": content})).collect();
    let mut body = json!({
        "model": req.binding.model,
        "messages": messages,
        "stream": true,
        "max_tokens": req.binding.max_tokens.unwrap_or(8192),
    });
    let obj = body.as_object_mut().expect("object");
    let sys = system_text(req);
    if !sys.is_empty() {
        obj.insert("system".into(), sys.into());
    }
    if !req.tools.is_empty() {
        let tools: Vec<Value> = req
            .tools
            .iter()
            .map(|t| json!({"name": t.function.name, "description": t.function.description, "input_schema": t.function.parameters}))
            .collect();
        obj.insert("tools".into(), tools.into());
    }
    insert_common(obj, req, "max_tokens");
    body
}

// ---------------- streaming ----------------

/// Stateful decoder of one response stream (`data:` payloads in, `LlmEvent`s out).
pub struct StreamParser {
    protocol: Protocol,
    input_tokens: u64,
    done: bool,
    /// Responses: output_index → our tool-call index.
    tool_index: HashMap<u64, usize>,
}

impl StreamParser {
    /// `Ok(None)` = end of stream.
    pub fn parse(&mut self, data: &str) -> Result<Option<Vec<LlmEvent>>> {
        if self.done {
            return Ok(None);
        }
        match self.protocol {
            Protocol::Chat => parse_chat_chunk(data),
            Protocol::Responses => self.parse_responses(data),
            Protocol::Anthropic => self.parse_anthropic(data),
        }
    }

    fn parse_responses(&mut self, data: &str) -> Result<Option<Vec<LlmEvent>>> {
        let data = data.trim();
        if data == "[DONE]" {
            return Ok(None);
        }
        let v: Value = serde_json::from_str(data).map_err(|e| anyhow::anyhow!("bad stream chunk: {e}: {data}"))?;
        let s = |k: &str| v.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let mut out = Vec::new();
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "response.output_text.delta" => out.push(LlmEvent::TextDelta(s("delta"))),
            "response.reasoning_summary_text.delta" | "response.reasoning_text.delta" => out.push(LlmEvent::ReasoningDelta(s("delta"))),
            "response.output_item.added" => {
                let item = &v["item"];
                if item["type"] == "function_call" {
                    let oi = v["output_index"].as_u64().unwrap_or(0);
                    let next = self.tool_index.len();
                    let idx = *self.tool_index.entry(oi).or_insert(next);
                    out.push(LlmEvent::ToolCallDelta {
                        index: idx,
                        id: item["call_id"].as_str().map(str::to_string),
                        name: item["name"].as_str().map(str::to_string),
                        args: item["arguments"].as_str().unwrap_or("").to_string(),
                    });
                }
            }
            "response.function_call_arguments.delta" => {
                let oi = v["output_index"].as_u64().unwrap_or(0);
                if let Some(&idx) = self.tool_index.get(&oi) {
                    out.push(LlmEvent::ToolCallDelta { index: idx, id: None, name: None, args: s("delta") });
                }
            }
            "response.completed" | "response.incomplete" => {
                let u = &v["response"]["usage"];
                out.push(LlmEvent::Usage(Usage {
                    prompt_tokens: u["input_tokens"].as_u64().unwrap_or(0),
                    completion_tokens: u["output_tokens"].as_u64().unwrap_or(0),
                }));
                out.push(LlmEvent::Finish(v["response"]["status"].as_str().unwrap_or("completed").to_string()));
                self.done = true;
            }
            "response.failed" | "error" => {
                let err = v.get("response").and_then(|r| r.get("error")).or_else(|| v.get("error")).unwrap_or(&v);
                bail!("provider error: {err}");
            }
            _ => {}
        }
        Ok(Some(out))
    }

    fn parse_anthropic(&mut self, data: &str) -> Result<Option<Vec<LlmEvent>>> {
        let v: Value = serde_json::from_str(data.trim()).map_err(|e| anyhow::anyhow!("bad stream chunk: {e}: {data}"))?;
        let mut out = Vec::new();
        match v.get("type").and_then(Value::as_str).unwrap_or("") {
            "message_start" => {
                self.input_tokens = v["message"]["usage"]["input_tokens"].as_u64().unwrap_or(0)
                    + v["message"]["usage"]["cache_read_input_tokens"].as_u64().unwrap_or(0)
                    + v["message"]["usage"]["cache_creation_input_tokens"].as_u64().unwrap_or(0);
            }
            "content_block_start" => {
                let b = &v["content_block"];
                if b["type"] == "tool_use" {
                    out.push(LlmEvent::ToolCallDelta {
                        index: v["index"].as_u64().unwrap_or(0) as usize,
                        id: b["id"].as_str().map(str::to_string),
                        name: b["name"].as_str().map(str::to_string),
                        args: String::new(),
                    });
                }
            }
            "content_block_delta" => {
                let d = &v["delta"];
                let idx = v["index"].as_u64().unwrap_or(0) as usize;
                match d["type"].as_str().unwrap_or("") {
                    "text_delta" => out.push(LlmEvent::TextDelta(d["text"].as_str().unwrap_or("").into())),
                    "input_json_delta" => out.push(LlmEvent::ToolCallDelta {
                        index: idx,
                        id: None,
                        name: None,
                        args: d["partial_json"].as_str().unwrap_or("").into(),
                    }),
                    "thinking_delta" => out.push(LlmEvent::ReasoningDelta(d["thinking"].as_str().unwrap_or("").into())),
                    _ => {}
                }
            }
            "message_delta" => {
                out.push(LlmEvent::Usage(Usage {
                    prompt_tokens: self.input_tokens,
                    completion_tokens: v["usage"]["output_tokens"].as_u64().unwrap_or(0),
                }));
                if let Some(r) = v["delta"]["stop_reason"].as_str() {
                    out.push(LlmEvent::Finish(r.to_string()));
                }
            }
            "message_stop" => {
                self.done = true;
                return Ok(None);
            }
            "error" => bail!("provider error: {}", v.get("error").unwrap_or(&v)),
            _ => {} // ping, content_block_stop
        }
        Ok(Some(out))
    }
}

/// Reasoning text is only echoed back on Chat Completions (DeepSeek-style); other protocols carry
/// it in their own opaque formats, which zilch doesn't round-trip.
pub fn keeps_reasoning(p: Protocol) -> bool {
    p == Protocol::Chat
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::accumulate::Accumulator;
    use crate::llm::{FunctionCall, Message, ModelBinding, ToolCall, ToolSpec};

    fn req() -> ChatRequest {
        let call = Message {
            role: Role::Assistant,
            content: Some("checking".into()),
            tool_calls: vec![
                ToolCall {
                    id: "c1".into(),
                    kind: "function".into(),
                    function: FunctionCall { name: "glob".into(), arguments: "{\"pattern\":\"*\"}".into() },
                },
                ToolCall {
                    id: "c2".into(),
                    kind: "function".into(),
                    function: FunctionCall { name: "grep".into(), arguments: "not json".into() },
                },
            ],
            tool_call_id: None,
            reasoning_content: None,
        };
        ChatRequest {
            binding: ModelBinding { model: "m".into(), max_tokens: Some(100), ..Default::default() },
            messages: vec![
                Message::system("SYS"),
                Message::user("hi"),
                call,
                Message::tool("c1", "a.rs"),
                Message::tool("c2", ""),
                Message::user("next"),
            ],
            tools: vec![ToolSpec::new("glob", "find", json!({"type":"object"}))],
            session: None,
        }
    }

    #[test]
    fn anthropic_body_shape() {
        let b = Protocol::Anthropic.body(&req());
        assert_eq!(b["system"], "SYS");
        assert_eq!(b["max_tokens"], 100);
        let m = b["messages"].as_array().unwrap();
        let roles: Vec<&str> = m.iter().map(|x| x["role"].as_str().unwrap()).collect();
        assert_eq!(roles, vec!["user", "assistant", "user"], "alternating, tool results merged with next user text");
        assert_eq!(m[1]["content"][1]["type"], "tool_use");
        assert_eq!(m[1]["content"][1]["input"]["pattern"], "*");
        assert_eq!(m[1]["content"][2]["input"], json!({}), "invalid JSON args become {{}}");
        assert_eq!(m[2]["content"][0]["type"], "tool_result");
        assert_eq!(m[2]["content"][1]["content"], "(empty)");
        assert_eq!(m[2]["content"][2]["text"], "next");
        assert_eq!(b["tools"][0]["input_schema"]["type"], "object");
    }

    #[test]
    fn responses_body_shape() {
        let b = Protocol::Responses.body(&req());
        assert_eq!(b["instructions"], "SYS");
        assert_eq!(b["max_output_tokens"], 100);
        assert_eq!(b["store"], false);
        let types: Vec<&str> = b["input"].as_array().unwrap().iter().map(|x| x["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            vec!["message", "message", "function_call", "function_call", "function_call_output", "function_call_output", "message"]
        );
        assert_eq!(b["input"][4]["call_id"], "c1");
        assert_eq!(b["tools"][0]["type"], "function");
    }

    fn run(p: Protocol, chunks: &[&str]) -> (crate::llm::Message, Usage) {
        let mut parser = p.parser();
        let mut acc = Accumulator::default();
        for c in chunks {
            match parser.parse(c).unwrap() {
                Some(evs) => evs.iter().for_each(|e| acc.push(e)),
                None => break,
            }
        }
        let u = acc.usage;
        (acc.finish(), u)
    }

    #[test]
    fn responses_stream_decodes_text_and_tools() {
        // Recorded from a real provider (trimmed).
        let (m, u) = run(
            Protocol::Responses,
            &[
                r#"{"type":"response.created","response":{"status":"in_progress"}}"#,
                r#"{"type":"response.output_item.added","output_index":0,"item":{"type":"reasoning","summary":[]}}"#,
                r#"{"type":"response.output_text.delta","output_index":1,"delta":"Calling glob."}"#,
                r#"{"type":"response.output_item.added","output_index":2,"item":{"type":"function_call","name":"glob","call_id":"call_1","arguments":""}}"#,
                r#"{"type":"response.function_call_arguments.delta","output_index":2,"delta":"{\"pattern\""}"#,
                r#"{"type":"response.function_call_arguments.delta","output_index":2,"delta":":\"*.rs\"}"}"#,
                r#"{"type":"response.output_item.added","output_index":3,"item":{"type":"function_call","name":"grep","call_id":"call_2","arguments":"{}"}}"#,
                r#"{"type":"response.completed","response":{"status":"completed","usage":{"input_tokens":50,"output_tokens":7}}}"#,
            ],
        );
        assert_eq!(m.text(), "Calling glob.");
        assert_eq!(m.tool_calls.len(), 2);
        assert_eq!((m.tool_calls[0].id.as_str(), m.tool_calls[0].function.arguments.as_str()), ("call_1", "{\"pattern\":\"*.rs\"}"));
        assert_eq!(m.tool_calls[1].function.name, "grep");
        assert_eq!((u.prompt_tokens, u.completion_tokens), (50, 7));
    }

    #[test]
    fn anthropic_stream_decodes_text_tools_and_usage() {
        let (m, u) = run(
            Protocol::Anthropic,
            &[
                r#"{"type":"message_start","message":{"usage":{"input_tokens":30,"cache_read_input_tokens":10}}}"#,
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Let me look"}}"#,
                r#"{"type":"ping"}"#,
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file","input":{}}}"#,
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\"path\":"}}"#,
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"\"a.rs\"}"}}"#,
                r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":12}}"#,
                r#"{"type":"message_stop"}"#,
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"IGNORED after stop"}}"#,
            ],
        );
        assert_eq!(m.text(), "Let me look");
        assert_eq!((m.tool_calls[0].id.as_str(), m.tool_calls[0].function.arguments.as_str()), ("toolu_1", "{\"path\":\"a.rs\"}"));
        assert_eq!((u.prompt_tokens, u.completion_tokens), (40, 12));
    }

    #[test]
    fn errors_surface() {
        assert!(Protocol::Anthropic.parser().parse(r#"{"type":"error","error":{"type":"overloaded_error"}}"#).is_err());
        assert!(Protocol::Responses.parser().parse(r#"{"type":"response.failed","response":{"error":{"message":"boom"}}}"#).is_err());
    }
}
