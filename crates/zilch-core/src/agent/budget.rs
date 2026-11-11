//! The per-turn model-call budget (`max_steps`), aimed at real waste only: stopping a task that was
//! nearly done throws its work away and the rerun pays the tokens again.
//!
//! - At the budget, a task still making progress (a successful write, or at least one tool result
//!   it had not seen per three model calls) gets half the budget more, up to 3× in total.
//! - A task without progress is asked to wrap up and report; it is stopped only if it is still going
//!   after a short grace (max/4, at least 5 calls).

use std::collections::HashSet;
use std::hash::{Hash, Hasher};

/// Tools whose success counts as progress (they change the project or the spec documents).
const WRITE_TOOLS: &[&str] = &["write_file", "edit_file", "write_doc", "edit_doc"];

/// What the loop should do before its next model call.
#[derive(Debug, PartialEq)]
pub enum Decision {
    Go,
    /// Still making progress: the budget grew to this many calls.
    Extended {
        limit: usize,
        writes: usize,
        fresh: usize,
    },
    /// Out of budget without progress: tell the agent to wrap up (this many calls left).
    WrapUp {
        grace: usize,
    },
    /// Still going after the wrap-up grace: stop.
    Stop {
        limit: usize,
    },
}

pub struct StepBudget {
    max: usize,
    calls: usize,
    granted: usize,
    warned: bool,
    /// Progress since the last extension: successful writes, new results, and where the window began.
    writes: usize,
    fresh: usize,
    window_start: usize,
    seen: HashSet<u64>,
}

impl StepBudget {
    pub fn new(max: usize) -> Self {
        Self { max: max.max(1), calls: 0, granted: 0, warned: false, writes: 0, fresh: 0, window_start: 0, seen: HashSet::new() }
    }

    fn grace(&self) -> usize {
        (self.max / 4).max(5)
    }

    /// Call before each model call.
    pub fn before_call(&mut self) -> Decision {
        let limit = self.max + self.granted;
        if self.warned {
            if self.calls >= limit + self.grace() {
                return Decision::Stop { limit: limit + self.grace() };
            }
        } else if self.calls >= limit {
            let window = self.calls.saturating_sub(self.window_start).max(1);
            let progressing = self.writes > 0 || self.fresh * 3 >= window;
            if progressing && limit < self.max * 3 {
                self.granted = (self.granted + (self.max / 2).max(1)).min(self.max * 2);
                let d = Decision::Extended { limit: self.max + self.granted, writes: self.writes, fresh: self.fresh };
                (self.writes, self.fresh, self.window_start) = (0, 0, self.calls);
                self.calls += 1;
                return d;
            }
            self.warned = true;
            self.calls += 1;
            return Decision::WrapUp { grace: self.grace() };
        }
        self.calls += 1;
        Decision::Go
    }

    /// Record a tool result (call after each tool call).
    pub fn record(&mut self, tool: &str, result: &str) {
        let failed = result.starts_with("Error") || result.starts_with("error:");
        if failed {
            return;
        }
        if WRITE_TOOLS.contains(&tool) {
            self.writes += 1;
        }
        let mut h = std::collections::hash_map::DefaultHasher::new();
        (tool, result).hash(&mut h);
        if self.seen.insert(h.finish()) {
            self.fresh += 1;
        }
    }
}

/// The note that asks an agent out of budget to finish.
pub fn wrap_up_message(max: usize, grace: usize) -> String {
    format!(
        "[supervisor] This task has used its budget of {max} model calls. Finish now: stop exploring, report what is \
         done and what is left (with the usual STATUS line if your role has one). You have at most {grace} more model calls."
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Up to `calls` model calls, like the agent loop: it ends at the first Stop.
    fn run(b: &mut StepBudget, calls: usize, mut each: impl FnMut(&mut StepBudget, usize)) -> Vec<Decision> {
        let mut out = vec![];
        for i in 0..calls {
            let d = b.before_call();
            let stop = matches!(d, Decision::Stop { .. });
            if d != Decision::Go {
                out.push(d);
            }
            if stop {
                break;
            }
            each(b, i);
        }
        out
    }

    #[test]
    fn a_task_making_progress_is_extended_up_to_three_times_the_budget() {
        let mut b = StepBudget::new(10);
        // Every call reads something new.
        let d = run(&mut b, 40, |b, i| b.record("read_file", &format!("content {i}")));
        let extended: Vec<usize> =
            d.iter().filter_map(|d| if let Decision::Extended { limit, .. } = d { Some(*limit) } else { None }).collect();
        assert_eq!(extended, [15, 20, 25, 30]);
        // At 3× it is asked to wrap up, then stopped after the grace (5 calls).
        assert_eq!(d[4], Decision::WrapUp { grace: 5 });
        assert_eq!(d[5], Decision::Stop { limit: 35 });
    }

    #[test]
    fn a_task_going_in_circles_is_asked_to_wrap_up_then_stopped() {
        let mut b = StepBudget::new(10);
        let d = run(&mut b, 20, |b, _| b.record("read_file", "the same"));
        assert_eq!(d, [Decision::WrapUp { grace: 5 }, Decision::Stop { limit: 15 }]);
    }

    #[test]
    fn a_successful_write_counts_as_progress_a_failed_one_does_not() {
        let mut b = StepBudget::new(8);
        let d = run(&mut b, 9, |b, i| if i == 7 { b.record("edit_file", "ok") } else { b.record("grep", "x") });
        assert!(matches!(d[0], Decision::Extended { limit: 12, writes: 1, .. }), "{d:?}");
        let mut b = StepBudget::new(8);
        let d = run(&mut b, 9, |b, _| b.record("edit_file", "Error: stale edit"));
        assert_eq!(d, [Decision::WrapUp { grace: 5 }]);
    }
}
