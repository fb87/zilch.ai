pub mod accumulate;
pub mod openai;
pub mod protocol;
pub mod sse;
pub mod types;

pub use types::*;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use futures::stream::BoxStream;
use futures::StreamExt;
use std::collections::HashMap;
use std::sync::Arc;

pub type EventStream = BoxStream<'static, Result<LlmEvent>>;

/// Anything that can stream a chat completion. Implement this to add a non-OpenAI backend.
#[async_trait]
pub trait LlmClient: Send + Sync {
    async fn stream(&self, req: &ChatRequest) -> Result<EventStream>;
}

/// Run a request to completion, forwarding text deltas to `on_text`.
pub async fn complete(client: &dyn LlmClient, req: &ChatRequest, mut on_text: impl FnMut(&str) + Send) -> Result<(Message, Usage)> {
    let mut stream = client.stream(req).await?;
    let mut acc = accumulate::Accumulator::default();
    while let Some(ev) = stream.next().await {
        let ev = ev?;
        if let LlmEvent::TextDelta(t) = &ev {
            on_text(t);
        }
        acc.push(&ev);
    }
    let usage = acc.usage;
    Ok((acc.finish(), usage))
}

/// Named providers; every agent bound to a provider shares its client (and its concurrency limit).
#[derive(Clone, Default)]
pub struct Providers {
    map: HashMap<String, Arc<dyn LlmClient>>,
}

impl Providers {
    pub fn insert(&mut self, name: impl Into<String>, client: Arc<dyn LlmClient>) {
        self.map.insert(name.into(), client);
    }
    pub fn get(&self, name: &str) -> Result<Arc<dyn LlmClient>> {
        self.map.get(name).cloned().ok_or_else(|| anyhow!("unknown provider '{name}'"))
    }
    pub fn contains(&self, name: &str) -> bool {
        self.map.contains_key(name)
    }
}
