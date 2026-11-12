//! Minimal hand-written Server-Sent-Events decoder for chat completion streams.

use serde_json::Value;
use zilch_ipc::llm::{LlmEvent, Usage};

/// Incrementally splits a byte stream into SSE `data:` payloads.
///
/// Buffers raw bytes and only decodes complete events, so a multi-byte UTF-8 character split
/// across network chunks is never corrupted (found by fuzzing).
#[derive(Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
}

impl SseDecoder {
    /// Feed raw bytes; returns complete `data:` payloads (one per event).
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut out = Vec::new();
        while let Some((end, sep)) = find_boundary(&self.buf) {
            let event: Vec<u8> = self.buf.drain(..end + sep).collect();
            if let Some(data) = event_data(&event[..end]) {
                out.push(data);
            }
        }
        out
    }

    /// Flush a trailing event that wasn't terminated by a blank line.
    pub fn finish(&mut self) -> Vec<String> {
        let rest = std::mem::take(&mut self.buf);
        event_data(&rest).into_iter().collect()
    }
}

/// Position of the first blank line ("\n\n", "\r\n\r\n", or mixed) and its length.
fn find_boundary(b: &[u8]) -> Option<(usize, usize)> {
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'\n' {
            // Accept "\n\n" and "\n\r\n" (the latter completes "\r\n\r\n").
            if b.get(i + 1) == Some(&b'\n') {
                return Some((i, 2));
            }
            if b.get(i + 1) == Some(&b'\r') && b.get(i + 2) == Some(&b'\n') {
                return Some((i, 3));
            }
        }
        i += 1;
    }
    None
}

/// Join the `data:` lines of one event (UTF-8 decoded once the event is complete).
fn event_data(event: &[u8]) -> Option<String> {
    let text = String::from_utf8_lossy(event);
    let data: Vec<&str> = text
        .split('\n')
        .map(|l| l.strip_suffix('\r').unwrap_or(l))
        .filter_map(|l| l.strip_prefix("data:"))
        .map(|d| d.strip_prefix(' ').unwrap_or(d))
        .collect();
    if data.is_empty() {
        None
    } else {
        Some(data.join("\n"))
    }
}

/// Decode one `data:` payload into events. Returns `None` for `[DONE]`.
pub fn parse_chunk(data: &str) -> anyhow::Result<Option<Vec<LlmEvent>>> {
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(data).map_err(|e| anyhow::anyhow!("bad stream chunk: {e}: {data}"))?;
    if let Some(err) = v.get("error") {
        anyhow::bail!("provider error: {err}");
    }
    let mut events = Vec::new();
    if let Some(choice) = v.get("choices").and_then(|c| c.get(0)) {
        if let Some(delta) = choice.get("delta") {
            if let Some(t) = delta.get("content").and_then(Value::as_str) {
                if !t.is_empty() {
                    events.push(LlmEvent::TextDelta(t.to_string()));
                }
            }
            // DeepSeek/OpenCode use `reasoning_content`; OpenRouter uses `reasoning`.
            for key in ["reasoning_content", "reasoning"] {
                if let Some(t) = delta.get(key).and_then(Value::as_str).filter(|t| !t.is_empty()) {
                    events.push(LlmEvent::ReasoningDelta(t.to_string()));
                    break;
                }
            }
            if let Some(calls) = delta.get("tool_calls").and_then(Value::as_array) {
                for (pos, c) in calls.iter().enumerate() {
                    let index = c.get("index").and_then(Value::as_u64).map(|i| i as usize).unwrap_or(pos);
                    let f = c.get("function");
                    events.push(LlmEvent::ToolCallDelta {
                        index,
                        id: c.get("id").and_then(Value::as_str).map(str::to_string),
                        name: f.and_then(|f| f.get("name")).and_then(Value::as_str).map(str::to_string),
                        args: f.and_then(|f| f.get("arguments")).and_then(Value::as_str).unwrap_or("").to_string(),
                    });
                }
            }
        }
        if let Some(r) = choice.get("finish_reason").and_then(Value::as_str) {
            events.push(LlmEvent::Finish(r.to_string()));
        }
    }
    if let Some(raw) = v.get("usage").filter(|u| !u.is_null()) {
        if let Ok(mut u) = serde_json::from_value::<Usage>(raw.clone()) {
            // OpenAI and most compatible providers: prompt_tokens_details.cached_tokens; DeepSeek:
            // prompt_cache_hit_tokens.
            u.cached_tokens = raw["prompt_tokens_details"]["cached_tokens"]
                .as_u64()
                .or_else(|| raw["prompt_cache_hit_tokens"].as_u64())
                .unwrap_or(0)
                .min(u.prompt_tokens);
            events.push(LlmEvent::Usage(u));
        }
    }
    Ok(Some(events))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_events_across_chunks() {
        let mut d = SseDecoder::default();
        assert!(d.push(b"data: {\"a\":").is_empty());
        let out = d.push(b"1}\n\ndata: [DONE]\n\n");
        assert_eq!(out, vec!["{\"a\":1}".to_string(), "[DONE]".to_string()]);
    }

    #[test]
    fn utf8_split_across_chunks_is_not_corrupted() {
        // Regression (fuzz): "¡" is 0xC2 0xA1; split between two network chunks.
        let raw = "data: {\"choices\":[{\"delta\":{\"content\":\"¡Xin chào 🦀\"}}]}\n\n".as_bytes();
        for cut in 1..raw.len() {
            let mut d = SseDecoder::default();
            let mut out = d.push(&raw[..cut]);
            out.extend(d.push(&raw[cut..]));
            let ev = parse_chunk(&out[0]).unwrap().unwrap();
            assert_eq!(ev[0], LlmEvent::TextDelta("¡Xin chào 🦀".into()), "cut at {cut}");
        }
    }

    #[test]
    fn handles_crlf_and_comments() {
        let mut d = SseDecoder::default();
        let out = d.push(b": keepalive\r\n\r\ndata: x\r\n\r\n");
        assert_eq!(out, vec!["x".to_string()]);
    }

    #[test]
    fn parses_text_and_tool_deltas() {
        let ev = parse_chunk(r#"{"choices":[{"delta":{"content":"hi","tool_calls":[{"index":0,"id":"c1","function":{"name":"read_file","arguments":"{\"pa"}}]}}]}"#)
            .unwrap()
            .unwrap();
        assert_eq!(ev[0], LlmEvent::TextDelta("hi".into()));
        assert!(
            matches!(&ev[1], LlmEvent::ToolCallDelta { index: 0, id: Some(id), name: Some(n), args } if id == "c1" && n == "read_file" && args == "{\"pa")
        );
        assert!(parse_chunk("[DONE]").unwrap().is_none());
    }
}

#[cfg(test)]
mod cache_tests {
    use super::*;

    fn usage(payload: &str) -> Usage {
        let evs = parse_chunk(payload).unwrap().unwrap();
        evs.into_iter().find_map(|e| if let LlmEvent::Usage(u) = e { Some(u) } else { None }).expect("usage")
    }

    #[test]
    fn cached_prompt_tokens_are_read_from_openai_and_deepseek_usage() {
        let openai = r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":768}}}"#;
        assert_eq!(usage(openai).cached_tokens, 768);
        let deepseek = r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":5,"prompt_cache_hit_tokens":900}}"#;
        assert_eq!(usage(deepseek).cached_tokens, 900);
        let none = r#"{"choices":[],"usage":{"prompt_tokens":1000,"completion_tokens":5}}"#;
        assert_eq!(usage(none).cached_tokens, 0);
    }
}
