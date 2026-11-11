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
    /// USD per 1M input tokens served from the provider's prompt cache (None: the input price).
    pub cached_input_per_mtok: Option<f64>,
    /// Hours when the provider charges a multiple of these prices.
    pub peak: Option<Peak>,
}

/// A price multiplier for some hours of the week (OpenCode `cost.multipliers`), in UTC.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Peak {
    /// Bit 0 = Monday … bit 6 = Sunday.
    pub days: u8,
    /// [start, end) in minutes after 00:00 UTC; the first `count` are used.
    pub windows: [(u16, u16); 4],
    pub count: u8,
    pub factor: f64,
}

impl Peak {
    /// Does the multiplier apply at `unix_secs`?
    pub fn applies(&self, unix_secs: u64) -> bool {
        let day = unix_secs / 86_400;
        let weekday = ((day + 3) % 7) as u8; // 1970-01-01 was a Thursday; Monday = 0
        let minute = ((unix_secs % 86_400) / 60) as u16;
        self.days & (1 << weekday) != 0 && self.windows[..self.count as usize].iter().any(|(a, b)| *a <= minute && minute < *b)
    }
}

impl ModelInfo {
    /// The cost of a request made now.
    pub fn cost(&self, u: &Usage) -> Option<f64> {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
        self.cost_at(u, now)
    }
    /// Fresh input at the input price, cached input at the cache price, output at the output price,
    /// times the peak multiplier when it applies.
    pub fn cost_at(&self, u: &Usage, unix_secs: u64) -> Option<f64> {
        let i = self.input_per_mtok?;
        let o = self.output_per_mtok?;
        let cached = u.cached_tokens.min(u.prompt_tokens);
        let c = self.cached_input_per_mtok.unwrap_or(i);
        let factor = self.peak.filter(|p| p.applies(unix_secs)).map_or(1.0, |p| p.factor);
        Some(((u.prompt_tokens - cached) as f64 * i + cached as f64 * c + u.completion_tokens as f64 * o) * factor / 1_000_000.0)
    }
    /// Field-wise overlay: values in `o` win.
    pub fn merged(self, o: ModelInfo) -> ModelInfo {
        ModelInfo {
            context: o.context.or(self.context),
            input_per_mtok: o.input_per_mtok.or(self.input_per_mtok),
            output_per_mtok: o.output_per_mtok.or(self.output_per_mtok),
            cached_input_per_mtok: o.cached_input_per_mtok.or(self.cached_input_per_mtok),
            peak: o.peak.or(self.peak),
        }
    }
}

/// OpenCode `cost.multipliers[0]`: `{when: {utc_days: {in: [..]}, any: [{all: [{utc_hours: {gte}}, {utc_hours: {lt}}]}]}, factor}`.
fn parse_peak(v: &Value) -> Option<Peak> {
    let m = v.as_array()?.first()?;
    let factor = m.get("factor")?.as_f64()?;
    let when = m.get("when")?;
    let names = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"];
    let days = match when.pointer("/utc_days/in").and_then(Value::as_array) {
        Some(list) => list.iter().filter_map(|d| names.iter().position(|n| Some(*n) == d.as_str())).fold(0u8, |acc, i| acc | 1 << i),
        None => 0x7f,
    };
    let minutes = |s: &str| -> Option<u16> {
        let (h, m) = s.split_once(':')?;
        Some(h.parse::<u16>().ok()? * 60 + m.parse::<u16>().ok()?)
    };
    let mut peak = Peak { days, factor, ..Default::default() };
    for alt in when.get("any").and_then(Value::as_array).into_iter().flatten().take(4) {
        let (mut start, mut end) = (0u16, 24 * 60);
        for cond in alt.get("all").and_then(Value::as_array).into_iter().flatten() {
            if let Some(g) = cond.pointer("/utc_hours/gte").and_then(Value::as_str).and_then(minutes) {
                start = g;
            }
            if let Some(l) = cond.pointer("/utc_hours/lt").and_then(Value::as_str).and_then(minutes) {
                end = l;
            }
        }
        peak.windows[peak.count as usize] = (start, end);
        peak.count += 1;
    }
    if peak.count == 0 {
        peak.windows[0] = (0, 24 * 60);
        peak.count = 1;
    }
    Some(peak)
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
            cached_input_per_mtok: m.pointer("/cost/text/input/cache/read").and_then(num),
            peak: m.pointer("/cost/multipliers").and_then(parse_peak),
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
        if info.cached_input_per_mtok.is_none() {
            info.cached_input_per_mtok = m.pointer("/pricing/input_cache_read").and_then(num).map(|p| p * 1_000_000.0);
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
        assert_eq!(
            m["beam-free"],
            ModelInfo { context: Some(1048576), input_per_mtok: Some(0.0), output_per_mtok: Some(0.0), ..Default::default() }
        );
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
        m.catalog.write().unwrap().insert(
            "p/m".into(),
            ModelInfo { context: None, input_per_mtok: Some(2.0), output_per_mtok: Some(10.0), ..Default::default() },
        );
        let c = m.record("p/m", &Usage { prompt_tokens: 1000, completion_tokens: 500, ..Default::default() }).unwrap();
        assert!((c - 0.007).abs() < 1e-9);
        assert!(!m.would_exceed("p/m", 1000)); // +0.002 → 0.009
        assert!(m.would_exceed("p/m", 2000)); // +0.004 → 0.011
        assert!(m.record("unknown/model", &Usage { prompt_tokens: 5, completion_tokens: 5, ..Default::default() }).is_none());
        assert!(!CostMeter::new(None).would_exceed("p/m", u64::MAX));
    }
}

#[cfg(test)]
mod cache_price_tests {
    use super::*;

    fn deepseek() -> ModelInfo {
        let v: Value = serde_json::from_str(
            r#"{"data":[{"id":"deepseek-v4.1-flash","cost":{"text":{"input":{"price":0.15,"cache":{"read":0.003}},"output":{"price":0.6}},
            "multipliers":[{"when":{"utc_days":{"in":["mon","tue","wed","thu","fri"]},"any":[
              {"all":[{"utc_hours":{"gte":"01:00"}},{"utc_hours":{"lt":"04:00"}}]},
              {"all":[{"utc_hours":{"gte":"06:00"}},{"utc_hours":{"lt":"10:00"}}]}]},"factor":2}]}}]}"#,
        )
        .unwrap();
        parse_models(&v)["deepseek-v4.1-flash"]
    }

    #[test]
    fn cached_tokens_and_peak_hours_are_priced_as_the_provider_lists_them() {
        let m = deepseek();
        assert_eq!((m.input_per_mtok, m.output_per_mtok, m.cached_input_per_mtok), (Some(0.15), Some(0.6), Some(0.003)));
        // 1M prompt tokens of which 900k cached, 100k out.
        let u = Usage { prompt_tokens: 1_000_000, completion_tokens: 100_000, cached_tokens: 900_000 };
        // Wednesday 2026-11-11 12:00 UTC (off-peak): 0.1×0.15 + 0.9×0.003 + 0.1×0.6 = 0.0777
        let off = 1_794_398_400;
        assert!((m.cost_at(&u, off).unwrap() - 0.0777).abs() < 1e-9, "{:?}", m.cost_at(&u, off));
        // Same day 07:30 UTC (peak): twice that.
        let peak = off - 4 * 3600 - 1800;
        assert!((m.cost_at(&u, peak).unwrap() - 0.1554).abs() < 1e-9);
        // Saturday 07:30 UTC: off-peak again.
        assert!((m.cost_at(&u, peak + 3 * 86_400).unwrap() - 0.0777).abs() < 1e-9);
        // Without a cache price, cached tokens cost the input price.
        let plain = ModelInfo { cached_input_per_mtok: None, peak: None, ..m };
        assert!((plain.cost_at(&u, off).unwrap() - (0.15 + 0.06)).abs() < 1e-9);
    }
}
