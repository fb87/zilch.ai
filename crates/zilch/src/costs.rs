//! Cost tracking for the CLI (ported from zilch): token usage and USD per agent and model, the
//! `/usage` report, and the high-cost model warning — a model whose output price is at least
//! `expensive_model_price` and that has been in use for `expensive_model_minutes` or cost
//! `expensive_model_spend` gets a pulsing segment in the status row until `/ack` or until no group
//! uses it any more (`/model … default`). The meter itself (prices, spend, caps) lives in the
//! drivers; this side only reads it.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use zilch_agents::Book;
use zilch_drivers::llm::catalog::CostMeter;
use zilch_ipc::{Group, Usage};

pub struct Costs {
    pub meter: Arc<CostMeter>,
    pub book: Arc<Book>,
    price: f64,
    minutes: u64,
    spend: f64,
    /// Tokens per (who, model): who is "main" or "subagents".
    usage: Mutex<BTreeMap<(String, String), Usage>>,
    /// Warning steps the guest acknowledged / already announced: model → (minutes step, spend step).
    acked: Mutex<HashMap<String, (u64, u64)>>,
    announced: Mutex<HashMap<String, (u64, u64)>>,
}

/// A high-cost model that has been in use for a while.
#[derive(Debug, Clone, PartialEq)]
pub struct Alert {
    pub model: String,
    pub groups: Vec<&'static str>,
    pub minutes: u64,
    pub spent: f64,
    pub step: (u64, u64),
}

impl Costs {
    pub fn new(meter: Arc<CostMeter>, book: Arc<Book>, price: f64, minutes: u64, spend: f64) -> Self {
        Self {
            meter,
            book,
            price,
            minutes: minutes.max(1),
            spend: spend.max(0.01),
            usage: Mutex::default(),
            acked: Mutex::default(),
            announced: Mutex::default(),
        }
    }

    pub fn record(&self, who: &str, model: &str, u: Usage) {
        *self.usage.lock().unwrap().entry((who.to_string(), model.to_string())).or_default() += u;
    }

    /// `/usage`: tokens and cost per agent and model, then the session total.
    pub fn report(&self) -> String {
        let usage = self.usage.lock().unwrap();
        if usage.is_empty() {
            return "No model calls yet.".into();
        }
        let mut s = format!("{:<10} {:<36} {:>10} {:>10} {:>10}\n", "AGENT", "MODEL", "PROMPT", "OUTPUT", "USD");
        for ((who, model), u) in usage.iter() {
            let cost = self.meter.info(model).and_then(|i| i.cost(u)).map_or("?".to_string(), |c| format!("{c:.4}"));
            s.push_str(&format!("{who:<10} {model:<36} {:>10} {:>10} {cost:>10}\n", u.prompt_tokens, u.completion_tokens));
        }
        let (p, c) = usage.values().fold((0, 0), |(p, c), u| (p + u.prompt_tokens, c + u.completion_tokens));
        let cap = self.meter.limit_usd.map(|l| format!(" of ${l:.2} cap")).unwrap_or_default();
        s.push_str(&format!("{:<47} {p:>10} {c:>10} {:>10}{cap}", "total", format!("{:.4}", self.meter.spent())));
        s
    }

    /// Models in use now: model key → the groups using it.
    fn in_use(&self) -> BTreeMap<String, Vec<&'static str>> {
        let mut m: BTreeMap<String, Vec<&'static str>> = BTreeMap::new();
        m.entry(self.book.binding(Group::Architecture).key()).or_default().push("main");
        m.entry(self.book.binding(Group::Development).key()).or_default().push("subagents");
        m
    }

    /// High-cost models in use past a step the guest has not acknowledged.
    pub fn alerts(&self) -> Vec<Alert> {
        if self.price <= 0.0 {
            return vec![];
        }
        let acked = self.acked.lock().unwrap();
        let mut out = Vec::new();
        for (model, groups) in self.in_use() {
            let expensive = self.meter.info(&model).and_then(|i| i.output_per_mtok).is_some_and(|o| o >= self.price);
            let Some(u) = self.meter.model_use(&model).filter(|_| expensive) else { continue };
            let minutes = u.first.elapsed().as_secs() / 60;
            let step = (minutes / self.minutes, (u.spent / self.spend) as u64);
            let seen = acked.get(&model).copied().unwrap_or((0, 0));
            if step.0 > seen.0 || step.1 > seen.1 {
                out.push(Alert { model, groups, minutes, spent: u.spent, step });
            }
        }
        out
    }

    /// `/ack`: silence the current warnings until each crosses its next step.
    pub fn ack(&self) -> usize {
        let alerts = self.alerts();
        let mut acked = self.acked.lock().unwrap();
        for a in &alerts {
            acked.insert(a.model.clone(), a.step);
        }
        alerts.len()
    }

    /// Lines to print once per new warning step.
    pub fn announcements(&self) -> Vec<String> {
        let fresh: Vec<Alert> = {
            let announced = self.announced.lock().unwrap();
            self.alerts().into_iter().filter(|a| announced.get(&a.model) != Some(&a.step)).collect()
        };
        let mut announced = self.announced.lock().unwrap();
        fresh
            .into_iter()
            .map(|a| {
                announced.insert(a.model.clone(), a.step);
                let group = if a.groups.contains(&"main") { "main" } else { "developer" };
                format!(
                    "$ high-cost model in use: {} ({}) for {} min, ${:.2} so far. Switch back with /model {group} default, or /ack to silence it until the next step.",
                    a.model,
                    a.groups.join(", "),
                    a.minutes,
                    a.spent
                )
            })
            .collect()
    }

    /// The status row segment while a warning is up (`None` otherwise).
    pub fn status_segment(&self) -> Option<String> {
        let alerts = self.alerts();
        if alerts.is_empty() {
            return None;
        }
        let parts: Vec<String> = alerts
            .iter()
            .map(|a| {
                format!("{} · {} · {} min · ${:.2}", a.model.rsplit('/').next().unwrap_or(&a.model), a.groups.join(","), a.minutes, a.spent)
            })
            .collect();
        Some(format!("$ {} — /ack", parts.join(" | ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use zilch_drivers::llm::catalog::ModelInfo;
    use zilch_ipc::ModelBinding;

    fn costs() -> Costs {
        let b = |m: &str| ModelBinding { provider: "p".into(), model: m.into(), ..Default::default() };
        let book = Arc::new(Book::new(b("cheap"), b("cheap"), vec![], vec![], std::path::Path::new("/tmp")));
        let meter = Arc::new(CostMeter::new(None));
        let price = |o: f64| ModelInfo { context: None, input_per_mtok: Some(1.0), output_per_mtok: Some(o), ..Default::default() };
        *meter.catalog.write().unwrap() = [("p/cheap".to_string(), price(0.5)), ("p/opus".to_string(), price(75.0))].into();
        Costs::new(meter, book, 10.0, 30, 2.0)
    }

    #[test]
    fn usage_is_reported_per_agent_and_model_with_cost() {
        let c = costs();
        let u = Usage { prompt_tokens: 1_000_000, completion_tokens: 0, ..Default::default() };
        c.meter.record("p/cheap", &u);
        c.record("main", "p/cheap", u);
        let r = c.report();
        assert!(r.contains("main") && r.contains("p/cheap") && r.contains("1.0000") && r.contains("total"), "{r}");
    }

    #[test]
    fn a_high_cost_model_in_use_too_long_warns_until_acknowledged_or_switched_back() {
        let c = costs();
        let small = Usage { prompt_tokens: 1000, completion_tokens: 100, ..Default::default() };
        c.book.set_binding(Group::Architecture, ModelBinding { provider: "p".into(), model: "opus".into(), ..Default::default() });
        c.meter.record("p/opus", &small);
        assert!(c.alerts().is_empty(), "just switched");
        c.meter.backdate("p/opus", Duration::from_secs(31 * 60));
        let a = c.alerts();
        assert_eq!((a[0].model.as_str(), a[0].groups.clone(), a[0].minutes), ("p/opus", vec!["main"], 31));
        assert_eq!(c.announcements().len(), 1, "announced once per step");
        assert!(c.announcements().is_empty());
        assert!(c.status_segment().unwrap().starts_with("$ opus · main · 31 min"));
        assert_eq!(c.ack(), 1);
        assert!(c.alerts().is_empty() && c.status_segment().is_none());
        c.meter.record("p/opus", &Usage { prompt_tokens: 0, completion_tokens: 30_000, ..Default::default() }); // $2.25
        assert_eq!(c.alerts().len(), 1, "spending past $2 alerts again");
        c.book.set_binding(Group::Architecture, ModelBinding { provider: "p".into(), model: "cheap".into(), ..Default::default() });
        assert!(c.alerts().is_empty(), "switched back: nothing high-cost in use");
    }
}
