//! `--debug` / `/debug`: proof for finding bottlenecks. Every model call gets its duration, time to
//! first token, tokens and cost; every tool call its duration and output size; every request the
//! time zilch itself took since the agent's previous results came back (step, prompt building,
//! scheduling); every order a breakdown of where its time and tokens went.
//!
//! The probe measures as events arrive (wall clock), whether or not debug output is on, so
//! switching `/debug` on mid-order still gives the order's full breakdown.

use crate::costs::{cached_share, fmt_tokens};
use std::collections::{BTreeMap, HashMap};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use zilch_ipc::{Pid, TicketId, Usage};

/// Who a finished call belongs to: its agent's label, its order, and whether to keep its note
/// for the debug line.
pub struct At<'a> {
    pub who: &'a str,
    pub ticket: Option<TicketId>,
    pub keep: bool,
}

#[derive(Default)]
pub struct Probe {
    s: Mutex<State>,
}

#[derive(Default)]
struct State {
    /// Calls in flight: start time.
    started: HashMap<(Pid, usize), Instant>,
    /// Calls in flight per process, and when its last batch finished.
    inflight: HashMap<Pid, u32>,
    ready: HashMap<Pid, Instant>,
    /// A model call in flight: when its first streamed token arrived.
    first_token: HashMap<Pid, Duration>,
    /// Usage of the model call that is about to finish (Usage comes just before its result).
    usage: HashMap<Pid, (Usage, Option<f64>)>,
    /// The debug note of a finished call, and the gap before a model call, until drawn.
    notes: HashMap<(Pid, usize), String>,
    gaps: HashMap<(Pid, usize), Duration>,
    orders: HashMap<TicketId, Order>,
}

/// Where one order's time and tokens went.
#[derive(Default)]
struct Order {
    /// Per agent: model time, model calls, input, cached and output tokens.
    model: BTreeMap<String, (Duration, u32, u64, u64, u64)>,
    tools: (Duration, u32),
    /// Time between an agent's results coming back and its next request (zilch's own share).
    overhead: Duration,
    /// The slowest calls: (duration, who, what).
    slowest: Vec<(Duration, String, String)>,
}

pub fn ms(d: Duration) -> String {
    let ms = d.as_millis();
    match ms {
        0..=999 => format!("{ms}ms"),
        1000..=59_999 => format!("{:.1}s", d.as_secs_f64()),
        _ => format!("{}m{:02}s", ms / 60_000, (ms / 1000) % 60),
    }
}

impl Probe {
    /// A call was issued. Returns, for a model call, how long zilch took since this process's
    /// previous results came back.
    pub fn issued(&self, pid: Pid, idx: usize, summary: &str, ticket: Option<TicketId>, keep: bool) -> Option<Duration> {
        let mut s = self.s.lock().unwrap();
        let now = Instant::now();
        s.started.insert((pid, idx), now);
        *s.inflight.entry(pid).or_default() += 1;
        if !summary.starts_with("llm ") {
            return None;
        }
        s.first_token.remove(&pid);
        let gap = s.ready.get(&pid).map(|r| now.duration_since(*r))?;
        if let Some(o) = ticket.and_then(|t| s.orders.get_mut(&t)) {
            o.overhead += gap;
        }
        if keep {
            s.gaps.insert((pid, idx), gap);
        }
        Some(gap)
    }

    /// Streamed text arrived for a process's model call.
    pub fn token(&self, pid: Pid) {
        let mut s = self.s.lock().unwrap();
        if let Some(start) = s.started.iter().filter(|((p, _), _)| *p == pid).map(|(_, t)| *t).min() {
            s.first_token.entry(pid).or_insert_with(|| start.elapsed());
        }
    }

    pub fn usage(&self, pid: Pid, u: Usage, cost: Option<f64>) {
        self.s.lock().unwrap().usage.insert(pid, (u, cost));
    }

    pub fn order_opened(&self, ticket: TicketId) {
        self.s.lock().unwrap().orders.insert(ticket, Order::default());
    }

    /// A call finished: work out its debug note (fetched with `note` when its line is drawn) and
    /// count it for its order.
    pub fn done(&self, pid: Pid, idx: usize, summary: &str, output_chars: usize, at: At<'_>) {
        let At { who, ticket, keep } = at;
        let waits = ["dispatch", "explore", "submit_plan"].iter().any(|p| summary.starts_with(p));
        let mut s = self.s.lock().unwrap();
        let now = Instant::now();
        let took = s.started.remove(&(pid, idx)).map(|t| now.duration_since(t)).unwrap_or_default();
        let left = s.inflight.entry(pid).or_default();
        *left = left.saturating_sub(1);
        if *left == 0 {
            s.ready.insert(pid, now);
        }
        let note = if summary.starts_with("llm ") {
            let first = s.first_token.remove(&pid);
            let usage = s.usage.remove(&pid);
            let mut n = ms(took);
            if let Some(f) = first {
                n.push_str(&format!(" (first token {})", ms(f)));
            }
            if let Some((u, cost)) = usage {
                n.push_str(&format!(
                    " · in {} ({}) · out {}",
                    fmt_tokens(u.prompt_tokens),
                    cached_share(u.cached_tokens, u.prompt_tokens),
                    fmt_tokens(u.completion_tokens)
                ));
                if let Some(c) = cost {
                    n.push_str(&format!(" · ${c:.4}"));
                }
                if u.completion_tokens > 0 && took.as_secs_f64() > 0.0 {
                    n.push_str(&format!(" · {:.0} tok/s", u.completion_tokens as f64 / took.as_secs_f64()));
                }
                if let Some(o) = ticket.and_then(|t| s.orders.get_mut(&t)) {
                    let e = o.model.entry(who.to_string()).or_default();
                    e.0 += took;
                    e.1 += 1;
                    e.2 += u.prompt_tokens;
                    e.3 += u.cached_tokens;
                    e.4 += u.completion_tokens;
                }
            }
            n
        } else {
            if let (false, Some(o)) = (waits, ticket.and_then(|t| s.orders.get_mut(&t))) {
                o.tools.0 += took;
                o.tools.1 += 1;
            }
            format!("{} · {} chars ≈ {} tokens", ms(took), output_chars, fmt_tokens(output_chars as u64 / 4))
        };
        // Waiting for subagents, explorers or a plan is their calls' time, not a call of its own.
        if let (false, Some(o)) = (waits, ticket.and_then(|t| s.orders.get_mut(&t))) {
            o.slowest.push((took, who.to_string(), summary.chars().take(60).collect()));
            o.slowest.sort_by_key(|a| std::cmp::Reverse(a.0));
            o.slowest.truncate(5);
        }
        if keep {
            s.notes.insert((pid, idx), note);
        }
    }

    /// zilch's own time before a model call (since the process's previous results came back).
    pub fn gap(&self, pid: Pid, idx: usize) -> Option<Duration> {
        self.s.lock().unwrap().gaps.remove(&(pid, idx))
    }

    /// The debug note of a call that just finished.
    pub fn note(&self, pid: Pid, idx: usize) -> Option<String> {
        self.s.lock().unwrap().notes.remove(&(pid, idx))
    }

    /// Where an order's time and tokens went (removes it).
    pub fn order_report(&self, ticket: TicketId, wall: Duration) -> Option<String> {
        let o = self.s.lock().unwrap().orders.remove(&ticket)?;
        let model: Duration = o.model.values().map(|m| m.0).sum();
        let calls: u32 = o.model.values().map(|m| m.1).sum();
        let mut out = vec![format!(
            "debug · wall {} · model {} in {calls} calls · tools {} in {} calls · zilch between calls {}",
            ms(wall),
            ms(model),
            ms(o.tools.0),
            o.tools.1,
            ms(o.overhead)
        )];
        out.push("  (model and tool times add up across agents working in parallel)".into());
        for (who, (t, n, i, c, x)) in &o.model {
            out.push(format!(
                "  {who}: {n} model calls, {} ({} per call) · in {} ({}) · out {}",
                ms(*t),
                ms(*t / (*n).max(1)),
                fmt_tokens(*i),
                cached_share(*c, *i),
                fmt_tokens(*x)
            ));
        }
        if !o.slowest.is_empty() {
            out.push("  slowest:".into());
            for (t, who, what) in &o.slowest {
                out.push(format!("    {} {who} {what}", ms(*t)));
            }
        }
        Some(out.join("\n"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calls_get_timings_sizes_and_an_order_breakdown() {
        let p = Probe::default();
        p.order_opened(1);
        // A model call, then a tool batch, then the next model call.
        assert_eq!(p.issued(2, 0, "llm m", Some(1), true), None, "no previous results yet");
        p.token(2);
        p.usage(2, Usage { prompt_tokens: 1000, completion_tokens: 50, cached_tokens: 800 }, Some(0.001));
        p.done(2, 0, "llm m", 0, At { who: "2 answer", ticket: Some(1), keep: true });
        let n = p.note(2, 0).unwrap();
        assert!(n.contains("(first token ") && n.contains("in 1.0k (80% cached) · out 50 · $0.0010"), "{n}");
        p.issued(2, 0, "read_file(a.rs, all)", Some(1), true);
        p.done(2, 0, "read_file(a.rs, all)", 4000, At { who: "2 answer", ticket: Some(1), keep: true });
        assert!(p.note(2, 0).unwrap().ends_with("4000 chars ≈ 1.0k tokens"));
        assert!(p.issued(2, 0, "llm m", Some(1), true).is_some(), "the gap since the tool results");
        let r = p.order_report(1, Duration::from_secs(5)).unwrap();
        assert!(r.starts_with("debug · wall 5.0s · model ") && r.contains("tools ") && r.contains("in 1 calls"), "{r}");
        assert!(r.contains("2 answer: 1 model calls") && r.contains("slowest:"), "{r}");
        assert_eq!(ms(Duration::from_millis(12)), "12ms");
        assert_eq!(ms(Duration::from_millis(7400)), "7.4s");
    }
}
