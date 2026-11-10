//! Loop protection: notices an agent repeating itself and stops it from burning tokens.
//!
//! Plain code, no LLM (a model watching a model would cost tokens on every step and can loop
//! too). Each step of a turn is observed; a loop is
//! - the same tool call (name + arguments) with the same result `limit` times in a turn, which
//!   also catches cycles such as read → failed edit → read → failed edit;
//! - the same whole step (text and calls with their results) `limit` times;
//! - `limit + 2` steps in a row where every tool call failed;
//! - a reply whose text keeps repeating itself while it streams ([`StreamGuard`]).
//!
//! The first trip earns a short nudge pushed into the history; the next one stops the turn.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

pub const DEFAULT_LOOP_LIMIT: usize = 3;

/// How a turn ended by the supervisor begins (the note is also the worker's report).
pub const STOPPED_BY_SUPERVISOR: &str = "[stopped by the supervisor:";

#[derive(Debug, Clone, PartialEq)]
pub enum Verdict {
    Ok,
    /// Push this message and let the agent continue.
    Nudge(String),
    /// End the turn; the reason is for the user (and the master, for a worker).
    Stop(String),
}

/// One tool call of a step, as the model sent it and as it came back.
pub struct CallRecord<'a> {
    pub name: &'a str,
    pub arguments: &'a str,
    pub result: &'a str,
}

/// Per-turn loop detector. `limit == 0` switches it off.
pub struct Supervisor {
    limit: usize,
    calls: HashMap<u64, usize>,
    steps: HashMap<u64, usize>,
    failing_steps: usize,
    strikes: usize,
}

fn hash_of(parts: &[&str]) -> u64 {
    let mut h = DefaultHasher::new();
    for p in parts {
        p.hash(&mut h);
    }
    h.finish()
}

/// Output with the parts that change on every run of the same thing removed, so a real loop is
/// not missed because a duration, timestamp, temp path or address differs: colour codes, times and
/// dates, durations ("finished in 3.21s"), hex addresses, temp paths, process ids, extra whitespace.
pub fn normalize_output(s: &str) -> String {
    use std::sync::OnceLock;
    static RULES: OnceLock<Vec<(regex::Regex, &'static str)>> = OnceLock::new();
    let rules = RULES.get_or_init(|| {
        [
            (r"\x1b\[[0-9;?]*[A-Za-z]", ""),
            (r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?", "<time>"),
            (r"\b\d{1,2}:\d{2}:\d{2}(\.\d+)?\b", "<time>"),
            (r"\b\d+(\.\d+)?\s?(ns|µs|us|ms|s|secs?|seconds?|m|min|minutes?)\b", "<dur>"),
            (r"\b0x[0-9a-fA-F]{4,}\b", "<addr>"),
            (r#"(/tmp|/var/folders|/private/var/folders)/[^\s:'"]+"#, "<tmp>"),
            (r"\b(?i:pid)[ :=#]*\d+", "pid <n>"),
            (r"[ \t]+", " "),
        ]
        .into_iter()
        .map(|(p, r)| (regex::Regex::new(p).expect("valid pattern"), r))
        .collect()
    });
    let mut out = s.trim().to_string();
    for (re, with) in rules {
        out = re.replace_all(&out, *with).into_owned();
    }
    out
}

/// Arguments compared by meaning, not formatting (`{"a":1, "b":2}` = `{"b":2,"a":1}`).
fn canonical_args(raw: &str) -> String {
    serde_json::from_str::<serde_json::Value>(raw).map(|v| v.to_string()).unwrap_or_else(|_| raw.trim().to_string())
}

impl Supervisor {
    pub fn new(limit: usize) -> Self {
        Self { limit, calls: HashMap::new(), steps: HashMap::new(), failing_steps: 0, strikes: 0 }
    }

    pub fn enabled(&self) -> bool {
        self.limit > 0
    }

    /// Look at one finished step (the assistant's text and the tool calls it made).
    pub fn observe(&mut self, text: &str, calls: &[CallRecord]) -> Verdict {
        if !self.enabled() {
            return Verdict::Ok;
        }
        let mut step_parts: Vec<String> = vec![normalize_output(text)];
        let mut repeated: Option<(String, usize)> = None;
        for c in calls {
            let args = canonical_args(c.arguments);
            let result = normalize_output(c.result);
            let n = self.calls.entry(hash_of(&[c.name, &args, &result])).or_default();
            *n += 1;
            if *n >= self.limit && repeated.as_ref().is_none_or(|(_, m)| *n > *m) {
                repeated = Some((c.name.to_string(), *n));
            }
            step_parts.extend([c.name.to_string(), args, result]);
        }
        let parts: Vec<&str> = step_parts.iter().map(String::as_str).collect();
        let step_n = {
            let n = self.steps.entry(hash_of(&parts)).or_default();
            *n += 1;
            *n
        };
        let all_failed = !calls.is_empty() && calls.iter().all(|c| c.result.starts_with("Error"));
        self.failing_steps = if all_failed { self.failing_steps + 1 } else { 0 };

        let reason = if let Some((name, n)) = repeated {
            Some(format!("called `{name}` with the same arguments {n} times and got the same result each time"))
        } else if step_n >= self.limit && !text.trim().is_empty() {
            Some(format!("gave the same reply {step_n} times"))
        } else if self.failing_steps >= self.limit + 2 {
            Some(format!("{} steps in a row where every tool call failed", self.failing_steps))
        } else {
            None
        };
        match reason {
            Some(r) => self.strike(r),
            None => Verdict::Ok,
        }
    }

    /// A loop was seen (by `observe` or by the stream guard): nudge first, stop the next time.
    pub fn strike(&mut self, reason: String) -> Verdict {
        self.strikes += 1;
        if self.strikes == 1 {
            // Count afresh after the warning: one more identical step is then the loop again.
            self.failing_steps = 0;
            Verdict::Nudge(reason)
        } else {
            Verdict::Stop(reason)
        }
    }
}

/// The message pushed back to the agent on its first strike.
pub fn nudge_message(reason: &str) -> String {
    format!(
        "[supervisor] You {reason}. Repeating it will not give a different outcome. Do not do it again: \
         re-read what you need, change your approach, or stop and report what is blocking you."
    )
}

/// Watches a reply while it streams and asks to cut it when its tail keeps repeating.
pub struct StreamGuard {
    tail: Vec<char>,
    since_check: usize,
}

/// The repeating tail that is looked at, and the longest period recognised within it.
const WINDOW: usize = 600;
const MAX_PERIOD: usize = 200;
const TAIL_KEEP: usize = WINDOW + MAX_PERIOD;

impl Default for StreamGuard {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamGuard {
    pub fn new() -> Self {
        Self { tail: Vec::new(), since_check: 0 }
    }

    /// Feed a streamed chunk. Returns true when the reply should be cut.
    pub fn push(&mut self, chunk: &str) -> bool {
        self.tail.extend(chunk.chars());
        self.since_check += chunk.len();
        if self.tail.len() > TAIL_KEEP * 2 {
            self.tail.drain(..self.tail.len() - TAIL_KEEP);
        }
        if self.since_check < 64 || self.tail.len() < WINDOW {
            return false;
        }
        self.since_check = 0;
        self.repeating_period().is_some()
    }

    /// The period p if the last WINDOW characters are one short piece repeated (at least 3
    /// times), ignoring pieces that are only whitespace or a single repeated character (rules,
    /// table borders).
    fn repeating_period(&self) -> Option<usize> {
        let t = &self.tail;
        let n = t.len();
        let start = n - WINDOW;
        (2..=MAX_PERIOD.min(WINDOW / 3)).find(|&p| {
            start >= p && (start..n).all(|i| t[i] == t[i - p]) && {
                let piece = &t[n - p..];
                piece.iter().any(|c| !c.is_whitespace()) && piece.iter().any(|c| *c != piece[0])
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call<'a>(name: &'a str, arguments: &'a str, result: &'a str) -> CallRecord<'a> {
        CallRecord { name, arguments, result }
    }

    #[test]
    fn the_same_call_with_the_same_result_is_nudged_then_stopped() {
        let mut s = Supervisor::new(3);
        let edit = || call("edit_file", r#"{"path":"a.rs","old_string":"x"}"#, "Error: old_string not found");
        assert_eq!(s.observe("", &[edit()]), Verdict::Ok);
        assert_eq!(s.observe("", &[call("read_file", r#"{"path":"a.rs"}"#, "fn a() {}")]), Verdict::Ok);
        // Same arguments in another order and spacing: still the same call.
        assert_eq!(
            s.observe("", &[call("edit_file", r#"{ "old_string": "x", "path": "a.rs" }"#, "Error: old_string not found")]),
            Verdict::Ok
        );
        assert!(matches!(s.observe("", &[edit()]), Verdict::Nudge(r) if r.contains("`edit_file`") && r.contains("3 times")));
        assert!(matches!(s.observe("", &[edit()]), Verdict::Stop(_)));
    }

    #[test]
    fn repeating_a_call_whose_result_changes_is_not_a_loop() {
        let mut s = Supervisor::new(3);
        for i in 0..6 {
            let out = format!("{i} tests failed");
            assert_eq!(s.observe("", &[call("bash", r#"{"command":"cargo test"}"#, &out)]), Verdict::Ok);
        }
    }

    #[test]
    fn cycles_and_repeated_replies_and_failure_streaks_are_caught() {
        // read → failed edit → read → failed edit → …
        let mut s = Supervisor::new(3);
        let read = || call("read_file", r#"{"path":"a"}"#, "same");
        let edit = || call("edit_file", r#"{"path":"a"}"#, "Error: no match");
        let mut verdicts = Vec::new();
        for _ in 0..3 {
            verdicts.push(s.observe("", &[read()]));
            verdicts.push(s.observe("", &[edit()]));
        }
        assert!(verdicts.iter().any(|v| matches!(v, Verdict::Nudge(_))), "{verdicts:?}");

        let mut s = Supervisor::new(3);
        assert_eq!(s.observe("I will now finish.", &[]), Verdict::Ok);
        assert_eq!(s.observe("I will now finish.", &[]), Verdict::Ok);
        assert!(matches!(s.observe("I will now finish.", &[]), Verdict::Nudge(r) if r.contains("same reply")));

        let mut s = Supervisor::new(3);
        let mut last = Verdict::Ok;
        for i in 0..5 {
            let args = format!(r#"{{"command":"try {i}"}}"#);
            let res = format!("Error: attempt {i} failed");
            last = s.observe("", &[call("bash", &args, &res)]);
        }
        assert!(matches!(last, Verdict::Nudge(r) if r.contains("5 steps in a row")));
    }

    #[test]
    fn zero_switches_it_off() {
        let mut s = Supervisor::new(0);
        for _ in 0..10 {
            assert_eq!(s.observe("again", &[call("x", "{}", "same")]), Verdict::Ok);
        }
    }

    #[test]
    fn a_reply_stuck_repeating_itself_is_cut_but_normal_text_is_not() {
        let mut g = StreamGuard::new();
        let mut cut = false;
        for _ in 0..200 {
            if g.push("I need to check the file again. ") {
                cut = true;
                break;
            }
        }
        assert!(cut, "a repeated sentence is cut");

        let mut g = StreamGuard::new();
        let prose = "The scheduler starts every ready TODO in parallel and records its result. ";
        for i in 0..100 {
            assert!(!g.push(&format!("{prose}Step {i} adds item {} to the list.\n", i * 7)), "varied text");
        }
        let mut g = StreamGuard::new();
        for _ in 0..100 {
            assert!(!g.push("────────────────────────────────────────"), "a long rule is not a loop");
        }
    }
}

#[cfg(test)]
mod normalize_tests {
    use super::*;

    #[test]
    fn outputs_differing_only_in_noise_count_as_the_same_loop() {
        let run = |dur: &str, tmp: &str, at: &str, pid: u32| {
            format!("\x1b[1mtest result:\x1b[0m FAILED. 0 passed; 1 failed; finished in {dur}\n  at {at} pid={pid}\n  wrote {tmp}/out.log  0x7ffe{pid:04x}")
        };
        let a = run("3.21s", "/tmp/.tmpAbC123", "2026-11-10T12:00:01Z", 4242);
        let b = run("0.98s", "/tmp/.tmpXyZ789", "2026-11-10T12:03:59Z", 5151);
        assert_eq!(normalize_output(&a), normalize_output(&b));
        assert_ne!(normalize_output("3 tests failed"), normalize_output("4 tests failed"), "real differences stay");

        let mut s = Supervisor::new(3);
        let outs = [a.clone(), b.clone(), run("12ms", "/tmp/x", "09:00:00", 1)];
        let mut verdict = Verdict::Ok;
        for o in &outs {
            verdict = s.observe("", &[CallRecord { name: "bash", arguments: r#"{"command":"cargo test"}"#, result: o }]);
        }
        assert!(matches!(verdict, Verdict::Nudge(_)), "{verdict:?}");
    }
}
