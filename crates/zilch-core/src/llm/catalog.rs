//! Model catalog: context window and USD pricing per model, discovered from the provider's
//! `GET {base}/models` (OpenCode, OpenRouter formats) and/or set in config. Drives cost tracking,
//! the dollar budget, and automatic `context_tokens`.

use super::Usage;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;
use std::time::Duration;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ModelInfo {
    pub context: Option<u64>,
    /// USD per 1M input tokens.
    pub input_per_mtok: Option<f64>,
    /// USD per 1M output tokens.
    pub output_per_mtok: Option<f64>,
}

impl ModelInfo {
    pub fn cost(&self, u: &Usage) -> Option<f64> {
        let i = self.input_per_mtok?;
        let o = self.output_per_mtok?;
        Some((u.prompt_tokens as f64 * i + u.completion_tokens as f64 * o) / 1_000_000.0)
    }
    /// Field-wise overlay: values in `o` win.
    pub fn merged(self, o: ModelInfo) -> ModelInfo {
        ModelInfo {
            context: o.context.or(self.context),
            input_per_mtok: o.input_per_mtok.or(self.input_per_mtok),
            output_per_mtok: o.output_per_mtok.or(self.output_per_mtok),
        }
    }
}

/// Parse a `/models` response (model id → info). Unknown shapes yield no info, never an error.
pub fn parse_models(v: &Value) -> HashMap<String, ModelInfo> {
    let mut out = HashMap::new();
    let Some(list) = v.get("data").and_then(Value::as_array) else { return out };
    let num = |x: &Value| x.as_f64().or_else(|| x.as_str().and_then(|s| s.parse().ok()));
    for m in list {
        let Some(id) = m.get("id").and_then(Value::as_str) else { continue };
        // OpenCode: limit.context, cost.text.{input,output}.price (USD / 1M tokens)
        let mut info = ModelInfo {
            context: m.pointer("/limit/context").and_then(Value::as_u64),
            input_per_mtok: m.pointer("/cost/text/input/price").and_then(num),
            output_per_mtok: m.pointer("/cost/text/output/price").and_then(num),
        };
        // OpenRouter: context_length, pricing.{prompt,completion} (USD / token, as strings)
        if info.context.is_none() {
            info.context = m.get("context_length").and_then(Value::as_u64);
        }
        if info.input_per_mtok.is_none() {
            info.input_per_mtok = m.pointer("/pricing/prompt").and_then(num).map(|p| p * 1_000_000.0);
        }
        if info.output_per_mtok.is_none() {
            info.output_per_mtok = m.pointer("/pricing/completion").and_then(num).map(|p| p * 1_000_000.0);
        }
        out.insert(id.to_string(), info);
    }
    out
}

/// Best-effort fetch of `{base}/models` (short timeout; errors → empty).
pub async fn fetch_models(base_url: &str, api_key: Option<&str>, headers: &[(String, String)]) -> HashMap<String, ModelInfo> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));
    let client = match reqwest::Client::builder().user_agent(super::openai::USER_AGENT).timeout(Duration::from_secs(5)).build() {
        Ok(c) => c,
        Err(_) => return HashMap::new(),
    };
    let mut rb = client.get(&url);
    if let Some(k) = api_key {
        rb = rb.bearer_auth(k);
    }
    for (k, v) in headers {
        rb = rb.header(k, v);
    }
    match rb.send().await {
        Ok(r) if r.status().is_success() => r.json::<Value>().await.map(|v| parse_models(&v)).unwrap_or_default(),
        _ => HashMap::new(),
    }
}

/// Use of one model this session (for the high-cost model warning).
#[derive(Debug, Clone, Copy)]
pub struct ModelUse {
    pub first: std::time::Instant,
    pub spent: f64,
}

/// Team-wide cost accounting (keyed by "provider/model") with an optional dollar cap.
#[derive(Debug, Default)]
pub struct CostMeter {
    pub catalog: RwLock<HashMap<String, ModelInfo>>,
    /// Spend in micro-dollars (atomic so parallel workers can record concurrently).
    spent_micros: AtomicU64,
    pub limit_usd: Option<f64>,
    /// When each priced model was first used this session and what it cost so far.
    uses: std::sync::Mutex<HashMap<String, ModelUse>>,
}

impl CostMeter {
    pub fn new(limit_usd: Option<f64>) -> Self {
        Self { limit_usd, ..Default::default() }
    }
    pub fn info(&self, key: &str) -> Option<ModelInfo> {
        self.catalog.read().unwrap().get(key).copied()
    }
    /// Record usage; returns its cost if the model's price is known.
    pub fn record(&self, key: &str, u: &Usage) -> Option<f64> {
        let c = self.info(key)?.cost(u)?;
        self.spent_micros.fetch_add((c * 1_000_000.0).round() as u64, Ordering::Relaxed);
        let mut uses = self.uses.lock().unwrap();
        uses.entry(key.to_string()).or_insert(ModelUse { first: std::time::Instant::now(), spent: 0.0 }).spent += c;
        Some(c)
    }
    /// This session's use of a model, if it was used with a known price.
    pub fn model_use(&self, key: &str) -> Option<ModelUse> {
        self.uses.lock().unwrap().get(key).copied()
    }
    #[doc(hidden)]
    /// Tests: pretend a model was first used `ago` before now.
    pub fn backdate(&self, key: &str, ago: Duration) {
        if let Some(u) = self.uses.lock().unwrap().get_mut(key) {
            u.first -= ago;
        }
    }
    pub fn spent(&self) -> f64 {
        self.spent_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0
    }
    /// Would sending `prompt_tokens` more input to `key` exceed the cap?
    pub fn would_exceed(&self, key: &str, prompt_tokens: u64) -> bool {
        let Some(limit) = self.limit_usd else { return false };
        let next = self.info(key).and_then(|i| i.input_per_mtok).map_or(0.0, |p| prompt_tokens as f64 * p / 1_000_000.0);
        self.spent() + next > limit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_opencode_and_openrouter_shapes() {
        let oc = json!({"data":[
            {"id":"beam-free","limit":{"context":1048576},"cost":{"text":{"input":{"price":0},"output":{"price":0}}}},
            {"id":"glm-5.3","limit":{"context":200000},"cost":{"text":{"input":{"price":1.4},"output":{"price":4.4}}}}
        ]});
        let m = parse_models(&oc);
        assert_eq!(m["beam-free"], ModelInfo { context: Some(1048576), input_per_mtok: Some(0.0), output_per_mtok: Some(0.0) });
        assert_eq!(m["glm-5.3"].input_per_mtok, Some(1.4));

        let or = json!({"data":[{"id":"x/y","context_length":32768,"pricing":{"prompt":"0.00000008","completion":"0.0000002"}}]});
        let i = parse_models(&or)["x/y"];
        assert_eq!(i.context, Some(32768));
        assert!((i.input_per_mtok.unwrap() - 0.08).abs() < 1e-9 && (i.output_per_mtok.unwrap() - 0.2).abs() < 1e-9);

        assert!(parse_models(&json!({"object":"list"})).is_empty());
        assert!(parse_models(&json!({"data":[{"id":"bare"}]}))["bare"].cost(&Usage::default()).is_none());
    }

    #[test]
    fn meter_tracks_spend_and_cap() {
        let m = CostMeter::new(Some(0.01));
        m.catalog
            .write()
            .unwrap()
            .insert("p/m".into(), ModelInfo { context: None, input_per_mtok: Some(2.0), output_per_mtok: Some(10.0) });
        let c = m.record("p/m", &Usage { prompt_tokens: 1000, completion_tokens: 500 }).unwrap();
        assert!((c - 0.007).abs() < 1e-9);
        assert!(!m.would_exceed("p/m", 1000)); // +0.002 → 0.009
        assert!(m.would_exceed("p/m", 2000)); // +0.004 → 0.011
        assert!(m.record("unknown/model", &Usage { prompt_tokens: 5, completion_tokens: 5 }).is_none());
        assert!(!CostMeter::new(None).would_exceed("p/m", u64::MAX));
    }
}
