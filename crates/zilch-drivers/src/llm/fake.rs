//! Scripted LLM for tests and replays: a function from the request to the events of the reply.

use super::{Backend, ChatRequest, LlmEvent};
use anyhow::Result;
use std::sync::Mutex;
use std::time::Duration;
use zilch_kernel::CancelToken;

pub type Script = dyn Fn(&ChatRequest) -> Vec<LlmEvent> + Send + Sync;

pub struct ScriptedLlm {
    script: Box<Script>,
    /// Pause before answering (makes parallelism measurable).
    pub delay: Duration,
    /// Every request seen, for assertions.
    pub requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptedLlm {
    pub fn new(script: impl Fn(&ChatRequest) -> Vec<LlmEvent> + Send + Sync + 'static) -> Self {
        Self { script: Box::new(script), delay: Duration::ZERO, requests: Mutex::new(vec![]) }
    }
    pub fn with_delay(mut self, d: Duration) -> Self {
        self.delay = d;
        self
    }
}

impl Backend for ScriptedLlm {
    fn stream(&self, req: &ChatRequest, cancel: &CancelToken, alive: &dyn Fn(), on_event: &mut dyn FnMut(LlmEvent) -> bool) -> Result<()> {
        self.requests.lock().unwrap().push(req.clone());
        let end = std::time::Instant::now() + self.delay;
        while std::time::Instant::now() < end {
            if cancel.is_cancelled() {
                anyhow::bail!("cancelled");
            }
            alive(); // a slow but healthy model: still streaming
            std::thread::sleep(Duration::from_millis(5));
        }
        for ev in (self.script)(req) {
            if !on_event(ev) {
                break;
            }
        }
        Ok(())
    }
}
