//! OpenAI-compatible streaming client on a blocking HTTP stack (no async runtime).
//!
//! Same behaviour as zilch's reqwest client: a per-provider concurrency limit held for the whole
//! stream, retries on 429/408/5xx and connection errors (honouring `Retry-After`), an idle timeout
//! (socket read timeout: a provider that stops sending fails instead of hanging), and the stable
//! per-conversation header some providers require. Cancellation is checked between chunks.

use super::protocol::Protocol;
use super::sse::SseDecoder;
use super::{Backend, ChatRequest, LlmEvent};
use anyhow::{bail, Result};
use std::io::Read;
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant};
use zilch_kernel::CancelToken;

pub const USER_AGENT: &str = concat!("zilch/", env!("CARGO_PKG_VERSION"));

/// Counting semaphore for in-flight requests to one provider.
struct Slots {
    free: Mutex<usize>,
    cv: Condvar,
}

impl Slots {
    fn acquire(&self, cancel: &CancelToken, alive: &dyn Fn()) -> Result<SlotGuard<'_>> {
        let mut free = self.free.lock().unwrap();
        while *free == 0 {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            alive(); // queued for a provider slot: slow, not stuck
            free = self.cv.wait_timeout(free, Duration::from_millis(100)).unwrap().0;
        }
        *free -= 1;
        Ok(SlotGuard(self))
    }
}

struct SlotGuard<'a>(&'a Slots);

impl Drop for SlotGuard<'_> {
    fn drop(&mut self) {
        *self.0.free.lock().unwrap() += 1;
        self.0.cv.notify_one();
    }
}

pub struct HttpClient {
    pub name: String,
    base_url: String,
    api_key: Option<String>,
    agent: ureq::Agent,
    slots: Slots,
    max_retries: u32,
    headers: Vec<(String, String)>,
    session_header: Option<String>,
    protocol: Protocol,
    idle_timeout: Duration,
}

impl HttpClient {
    pub fn new(name: impl Into<String>, base_url: impl Into<String>, api_key: Option<String>, max_concurrent: usize) -> Self {
        let mut c = Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            agent: ureq::AgentBuilder::new().build(),
            slots: Slots { free: Mutex::new(max_concurrent.max(1)), cv: Condvar::new() },
            max_retries: 4,
            headers: vec![],
            session_header: None,
            protocol: Protocol::Chat,
            idle_timeout: Duration::from_secs(180),
        };
        c.rebuild();
        c
    }

    fn rebuild(&mut self) {
        self.agent = ureq::AgentBuilder::new()
            .user_agent(USER_AGENT)
            .timeout_connect(Duration::from_secs(30))
            .timeout_read(self.idle_timeout)
            .timeout_write(Duration::from_secs(60))
            .build();
    }

    pub fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.headers = headers;
        self
    }
    pub fn with_idle_timeout(mut self, d: Duration) -> Self {
        self.idle_timeout = d;
        self.rebuild();
        self
    }
    pub fn with_protocol(mut self, p: Protocol) -> Self {
        self.protocol = p;
        self
    }
    pub fn with_session_header(mut self, name: Option<String>) -> Self {
        self.session_header = name;
        self
    }
    pub fn with_max_retries(mut self, n: u32) -> Self {
        self.max_retries = n;
        self
    }

    fn sleep_cancellable(d: Duration, cancel: &CancelToken, alive: &dyn Fn()) -> Result<()> {
        let end = Instant::now() + d;
        while Instant::now() < end {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            alive();
            std::thread::sleep(Duration::from_millis(50).min(end.saturating_duration_since(Instant::now())));
        }
        Ok(())
    }
}

fn jitter_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.subsec_nanos() as u64 % 250).unwrap_or(0)
}

impl Backend for HttpClient {
    fn stream(&self, req: &ChatRequest, cancel: &CancelToken, alive: &dyn Fn(), on_event: &mut dyn FnMut(LlmEvent) -> bool) -> Result<()> {
        // Hold a provider slot for the whole lifetime of the stream.
        let _slot = self.slots.acquire(cancel, alive)?;
        let url = format!("{}{}", self.base_url, self.protocol.path());
        let body = self.protocol.body(req).to_string();
        let mut attempt = 0;
        let resp = loop {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            let mut rb = self.agent.post(&url).set("Content-Type", "application/json");
            match (self.protocol, &self.api_key) {
                (Protocol::Anthropic, Some(k)) => rb = rb.set("x-api-key", k).set("anthropic-version", "2023-06-01"),
                (Protocol::Anthropic, None) => rb = rb.set("anthropic-version", "2023-06-01"),
                (_, Some(k)) => rb = rb.set("Authorization", &format!("Bearer {k}")),
                (_, None) => {}
            }
            for (k, v) in &self.headers {
                rb = rb.set(k, v);
            }
            if let (Some(h), Some(s)) = (&self.session_header, &req.session) {
                rb = rb.set(h, s);
            }
            let (retryable, retry_after, err) = match rb.send_string(&body) {
                Ok(r) => break r,
                Err(ureq::Error::Status(code, r)) => {
                    let after = r.header("retry-after").and_then(|v| v.trim().parse::<u64>().ok());
                    let text = r.into_string().unwrap_or_default();
                    let text: String = text.chars().take(2000).collect();
                    (code == 429 || code == 408 || code >= 500, after, format!("provider '{}' returned {code}: {text}", self.name))
                }
                Err(ureq::Error::Transport(t)) => (true, None, format!("request to provider '{}' failed: {t}", self.name)),
            };
            if retryable && attempt < self.max_retries {
                attempt += 1;
                let wait = match retry_after {
                    Some(secs) => Duration::from_secs(secs.min(60)),
                    None => Duration::from_millis(500u64 * 2u64.pow(attempt) + jitter_ms()),
                };
                Self::sleep_cancellable(wait, cancel, alive)?;
                continue;
            }
            bail!(err);
        };

        let mut reader = resp.into_reader();
        let mut dec = SseDecoder::default();
        let mut parser = self.protocol.parser();
        let mut chunk = [0u8; 8192];
        loop {
            if cancel.is_cancelled() {
                bail!("cancelled");
            }
            let n = match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if matches!(e.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) => {
                    bail!("stream stalled: no data for {}s", self.idle_timeout.as_secs())
                }
                Err(e) => bail!("stream error: {e}"),
            };
            for data in dec.push(&chunk[..n]) {
                match parser.parse(&data)? {
                    None => return Ok(()),
                    Some(evs) => {
                        for ev in evs {
                            if !on_event(ev) {
                                return Ok(()); // dropping the reader closes the connection
                            }
                        }
                    }
                }
            }
        }
        for data in dec.finish() {
            if let Ok(Some(evs)) = parser.parse(&data) {
                for ev in evs {
                    if !on_event(ev) {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{complete, ModelBinding};
    use std::io::{BufRead, BufReader, Write};
    use std::net::TcpListener;

    /// A one-shot HTTP server: answers each connection with the next canned response, and
    /// records the request heads it saw.
    fn serve(responses: Vec<String>) -> (String, std::sync::Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let seen = std::sync::Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        std::thread::spawn(move || {
            for (stream, resp) in listener.incoming().zip(responses) {
                let mut s = stream.unwrap();
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut head = String::new();
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    r.read_line(&mut line).unwrap();
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                    if line == "\r\n" {
                        break;
                    }
                    head.push_str(&line);
                }
                let mut body = vec![0; len];
                std::io::Read::read_exact(&mut r, &mut body).unwrap();
                log.lock().unwrap().push(head);
                s.write_all(resp.as_bytes()).unwrap();
            }
        });
        (url, seen)
    }

    fn sse(events: &[&str]) -> String {
        let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        format!("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\n\r\n{body}", body.len())
    }

    fn req() -> ChatRequest {
        ChatRequest {
            binding: ModelBinding { provider: "p".into(), model: "m".into(), ..Default::default() },
            messages: vec![crate::llm::Message::user("hi")],
            tools: vec![],
            session: Some("conv-1".into()),
        }
    }

    #[test]
    fn streams_text_and_retries_rate_limits_with_retry_after() {
        let busy = "HTTP/1.1 429 Too Many Requests\r\nRetry-After: 0\r\nContent-Length: 4\r\n\r\nslow".to_string();
        let ok = sse(&[
            r#"{"choices":[{"delta":{"content":"Hel"}}]}"#,
            r#"{"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":5,"completion_tokens":2}}"#,
            "[DONE]",
        ]);
        let (url, seen) = serve(vec![busy, ok]);
        let c = HttpClient::new("p", url, Some("sk-test".into()), 2).with_session_header(Some("x-session".into()));
        let mut streamed = String::new();
        let (msg, usage, cut) = complete(&c, &req(), &CancelToken::new(), &|| {}, |t| streamed.push_str(t)).unwrap();
        assert_eq!((msg.text(), streamed.as_str(), usage.prompt_tokens, cut), ("Hello", "Hello", 5, false));
        let heads = seen.lock().unwrap();
        assert_eq!(heads.len(), 2, "retried once after the 429");
        let h = heads[1].to_ascii_lowercase();
        assert!(h.contains("authorization: bearer sk-test") && h.contains("x-session: conv-1"), "{h}");
    }

    #[test]
    fn client_errors_are_not_retried_and_show_the_body() {
        let bad = "HTTP/1.1 400 Bad Request\r\nContent-Length: 13\r\n\r\nunknown model".to_string();
        let (url, seen) = serve(vec![bad]);
        let c = HttpClient::new("p", url, None, 1);
        let err = complete(&c, &req(), &CancelToken::new(), &|| {}, |_| {}).unwrap_err().to_string();
        assert!(err.contains("400") && err.contains("unknown model"), "{err}");
        assert_eq!(seen.lock().unwrap().len(), 1);
    }

    #[test]
    fn a_repeating_reply_is_cut_mid_stream() {
        let spam: Vec<String> =
            (0..300).map(|_| r#"{"choices":[{"delta":{"content":"Let me check that again. "}}]}"#.to_string()).collect();
        let refs: Vec<&str> = spam.iter().map(String::as_str).chain(["[DONE]"]).collect();
        let (url, _) = serve(vec![sse(&refs)]);
        let c = HttpClient::new("p", url, None, 1);
        let (msg, usage, cut) = complete(&c, &req(), &CancelToken::new(), &|| {}, |_| {}).unwrap();
        assert!(cut && msg.text().contains("cut by the supervisor") && msg.text().len() < 600);
        assert!(usage.completion_tokens > 0, "estimated for the budget");
    }
}
