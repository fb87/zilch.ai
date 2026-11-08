//! OpenAI-compatible streaming client (OpenAI, OpenRouter, Ollama, vLLM, llama.cpp, ...).

use super::sse::{parse_chunk, SseDecoder};
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
}

pub const USER_AGENT: &str = concat!("zilch/", env!("CARGO_PKG_VERSION"));

impl OpenAiClient {
    pub fn new(name: impl Into<String>, base_url: impl Into<String>, api_key: Option<String>, max_concurrent: usize) -> Self {
        Self {
            name: name.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            api_key,
            http: reqwest::Client::builder()
                .user_agent(USER_AGENT)
                .connect_timeout(Duration::from_secs(30))
                .build()
                .expect("http client"),
            limiter: Arc::new(Semaphore::new(max_concurrent.max(1))),
            max_retries: 4,
            headers: vec![],
            session_header: None,
        }
    }

    /// Extra static headers sent with every request.
    pub fn with_headers(mut self, headers: Vec<(String, String)>) -> Self {
        self.headers = headers;
        self
    }

    /// Send each agent's stable conversation id in this header (e.g. `x-opencode-session`).
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
        let url = format!("{}/chat/completions", self.base_url);
        let body = req.to_body();

        let mut attempt = 0;
        let resp = loop {
            let mut rb = self.http.post(&url).json(&body);
            if let Some(k) = &self.api_key {
                rb = rb.bearer_auth(k);
            }
            for (k, v) in &self.headers {
                rb = rb.header(k, v);
            }
            if let (Some(h), Some(s)) = (&self.session_header, &req.session) {
                rb = rb.header(h, s);
            }
            let res = rb.send().await;
            let retryable = match &res {
                Ok(r) => r.status().as_u16() == 429 || r.status().is_server_error(),
                Err(e) => e.is_connect() || e.is_timeout(),
            };
            if retryable && attempt < self.max_retries {
                attempt += 1;
                let base = 500u64 * 2u64.pow(attempt);
                let jitter = rand::random::<u64>() % 250;
                tokio::time::sleep(Duration::from_millis(base + jitter)).await;
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
        let stream = async_stream(move |tx| async move {
            let _permit = permit;
            let mut dec = SseDecoder::default();
            while let Some(chunk) = bytes.next().await {
                let chunk = match chunk {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = tx.send(Err(anyhow::anyhow!("stream error: {e}"))).await;
                        return;
                    }
                };
                for data in dec.push(&chunk) {
                    match parse_chunk(&data) {
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
                if let Ok(Some(evs)) = parse_chunk(&data) {
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
