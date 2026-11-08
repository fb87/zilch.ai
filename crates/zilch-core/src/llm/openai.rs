//! OpenAI-compatible streaming client (OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, ...).

use super::protocol::Protocol;
use super::sse::SseDecoder;
use super::types::{ChatRequest, LlmEvent};
use super::{EventStream, LlmClient};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use futures::StreamExt;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::Semaphore;

pub struct OpenAiClient {
    pub name: String,
    base_url: String,
    api_key: Option<String>,
    http: reqwest::Client,
    limiter: Arc<Semaphore>,
    max_retries: u32,
    headers: Vec<(String, String)>,
    session_header: Option<String>,
    protocol: Protocol,
    idle_timeout: Duration,
}

pub const USER_AGENT: &str = concat!("zilch/", env!("CARGO_PKG_VERSION"));

impl OpenAiClient {
    pub fn new(name: impl Into<String>, base_url: impl Into<String>, api_key: Option<String>, max_concurrent: usize) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            http: reqwest::Client::builder().user_agent(USER_AGENT).connect_timeout(Duration::from_secs(30)).build().expect("http client"),
            limiter: Arc::new(Semaphore::new(max_concurrent.max(1))),
            max_retries: 4,
            headers: vec![],
            session_header: None,
            protocol: Protocol::Chat,
            idle_timeout: Duration::from_secs(180),
        }
    }

    /// Extra static headers sent with every request.
    pub fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.headers = headers;
        self
    }

    /// Abort when no response headers or stream data arrive for this long (default 180s).
    pub fn with_idle_timeout(mut self, d: Duration) -> Self {
        self.idle_timeout = d;
        self
    }

    /// Send each agent's stable conversation id in this header (e.g. `x-opencode-session`).
    /// Wire protocol (Chat Completions by default).
    pub fn with_protocol(mut self, p: Protocol) -> Self {
        self.protocol = p;
        self
    }

    pub fn with_session_header(mut self, name: Option<String>) -> Self {
        self.session_header = name;
        self
    }
}

#[async_trait]
impl LlmClient for OpenAiClient {
    async fn stream(&self, req: &ChatRequest) -> Result<EventStream> {
        // Hold a provider slot for the whole lifetime of the stream.
        let permit = self.limiter.clone().acquire_owned().await?;
        let url = format!("{}{}", self.base_url, self.protocol.path());
        let body = self.protocol.body(req);
        let protocol = self.protocol;

        let mut attempt = 0;
        let resp = loop {
            let mut rb = self.http.post(&url).json(&body);
            match (self.protocol, &self.api_key) {
                (Protocol::Anthropic, Some(k)) => rb = rb.header("x-api-key", k).header("anthropic-version", "2023-06-01"),
                (Protocol::Anthropic, None) => rb = rb.header("anthropic-version", "2023-06-01"),
                (_, Some(k)) => rb = rb.bearer_auth(k),
                (_, None) => {}
            }
            for (k, v) in &self.headers {
                rb = rb.header(k, v);
            }
            if let (Some(h), Some(s)) = (&self.session_header, &req.session) {
                rb = rb.header(h, s);
            }
            let res = match tokio::time::timeout(self.idle_timeout, rb.send()).await {
                Ok(r) => r.map_err(anyhow::Error::from),
                Err(_) => Err(anyhow::anyhow!("no response headers within {}s", self.idle_timeout.as_secs())),
            };
            let (retryable, retry_after) = match &res {
                Ok(r) => {
                    let s = r.status().as_u16();
                    let after = r.headers().get("retry-after").and_then(|v| v.to_str().ok()).and_then(|v| v.trim().parse::<u64>().ok());
                    (s == 429 || s == 408 || r.status().is_server_error(), after)
                }
                Err(e) => {
                    let transient =
                        e.downcast_ref::<reqwest::Error>().is_none_or(|re| re.is_connect() || re.is_timeout() || re.is_request());
                    (transient, None)
                }
            };
            if retryable && attempt < self.max_retries {
                attempt += 1;
                // Honour Retry-After (capped), otherwise exponential backoff with jitter.
                let wait = match retry_after {
                    Some(secs) => Duration::from_secs(secs.min(60)),
                    None => Duration::from_millis(500u64 * 2u64.pow(attempt) + rand::random::<u64>() % 250),
                };
                tokio::time::sleep(wait).await;
                continue;
            }
            let r = res.with_context(|| format!("request to provider '{}' failed", self.name))?;
            if !r.status().is_success() {
                let status = r.status();
                let text = r.text().await.unwrap_or_default();
                bail!("provider '{}' returned {status}: {}", self.name, truncate(&text, 2000));
            }
            break r;
        };

        let mut bytes = resp.bytes_stream();
        let idle = self.idle_timeout;
        let stream = async_stream(move |tx| async move {
            let _permit = permit;
            let mut dec = SseDecoder::default();
            let mut parser = protocol.parser();
            loop {
                // A provider that stops sending mid-stream must not hang the agent forever.
                let chunk = match tokio::time::timeout(idle, bytes.next()).await {
                    Ok(Some(c)) => c,
                    Ok(None) => break,
                    Err(_) => {
                        let _ = tx.send(Err(anyhow::anyhow!("stream stalled: no data for {}s", idle.as_secs()))).await;
                        return;
                    }
                };
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("stream error: {e}"))).await;
                        return;
                    }
                };
                for data in dec.push(&chunk) {
                    match parser.parse(&data) {
                        Ok(None) => return,
                        Ok(Some(evs)) => {
                            for ev in evs {
                                if tx.send(Ok(ev)).await.is_err() {
                                    return;
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(e)).await;
                            return;
                        }
                    }
                }
            }
            for data in dec.finish() {
                if let Ok(Some(evs)) = parser.parse(&data) {
                    for ev in evs {
                        let _ = tx.send(Ok(ev)).await;
                    }
                }
            }
        });
        Ok(stream)
    }
}

/// Tiny channel-backed stream builder (avoids pulling in an extra crate).
fn async_stream<F, Fut>(f: F) -> EventStream
where
    F: FnOnce(tokio::sync::mpsc::Sender<Result<LlmEvent>>) -> Fut,
    Fut: std::future::Future<Output = ()> + Send + 'static,
{
    let (tx, rx) = tokio::sync::mpsc::channel(64);
    tokio::spawn(f(tx));
    Box::pin(futures::stream::unfold(rx, |mut rx| async move { rx.recv().await.map(|v| (v, rx)) }))
}

fn truncate(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}
