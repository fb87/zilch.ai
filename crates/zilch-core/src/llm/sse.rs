//! Minimal hand-written Server-Sent-Events decoder for chat completion streams.

use super::types::{LlmEvent, Usage};
use serde_json::Value;

/// Incrementally splits a byte stream into SSE `data:` payloads.
#[derive(Default)]
pub struct SseDecoder {
    buf: String,
}

impl SseDecoder {
    /// Feed raw bytes; returns complete `data:` payloads (one per event).
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.push_str(&String::from_utf8_lossy(chunk));
        // Normalize CRLF so event boundaries are always "\n\n".
        if self.buf.contains('\r') {
            self.buf = self.buf.replace("\r\n", "\n");
        }
        let mut out = Vec::new();
        while let Some(pos) = self.buf.find("\n\n") {
            let event: String = self.buf.drain(..pos + 2).collect();
            let data: Vec<&str> = event
                .lines()
                .filter_map(|l| l.strip_prefix("data:"))
                .map(|d| d.strip_prefix(' ').unwrap_or(d))
                .collect();
            if !data.is_empty() {
                out.push(data.join("\n"));
            }
        }
        out
    }

    /// Flush a trailing event that wasn't terminated by a blank line.
    pub fn finish(&mut self) -> Vec<String> {
        if self.buf.trim().is_empty() {
            return vec![];
        }
        self.buf.push_str("\n\n");
        self.push(&[])
    }
}

/// Decode one `data:` payload into events. Returns `None` for `[DONE]`.
pub fn parse_chunk(data: &str) -> anyhow::Result<Option<Vec<LlmEvent>>> {
    let data = data.trim();
    if data == "[DONE]" {
        return Ok(None);
    }
    let v: Value = serde_json::from_str(data)
        .map_err(|e| anyhow::anyhow!("bad stream chunk: {e}: {data}"))?;
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
                        args: f
                            .and_then(|f| f.get("arguments"))
                            .and_then(Value::as_str)
                            .unwrap_or("")
                            .to_string(),
                    });
                }
            }
        }
        if let Some(r) = choice.get("finish_reason").and_then(Value::as_str) {
            events.push(LlmEvent::Finish(r.to_string()));
        }
    }
    if let Some(u) = v.get("usage").filter(|u| !u.is_null()) {
        if let Ok(u) = serde_json::from_value::<Usage>(u.clone()) {
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
        assert!(matches!(&ev[1], LlmEvent::ToolCallDelta { index: 0, id: Some(id), name: Some(n), args } if id == "c1" && n == "read_file" && args == "{\"pa"));
        assert!(parse_chunk("[DONE]").unwrap().is_none());
    }
}
