//! Questions from agents (tool approvals, plan reviews) in the REPL.
//!
//! With two lanes running in parallel the keyboard has a single owner: the line editor. So
//! agents don't read stdin themselves; they queue a question here, it is printed above the
//! prompt, and the user's next input line answers it. Questions are answered in order.

use crate::render::{SharedScreen, BOLD, CYAN, DIM, RESET, YELLOW};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use zilch_core::agent::{ApprovalRequest, Decision};
use zilch_core::team::todo::Todo;
use zilch_core::{Approver, PlanDecision, PlanReviewer};

enum Ask {
    Approval { text: String, tx: oneshot::Sender<Decision> },
    Plan { text: String, tx: oneshot::Sender<PlanDecision> },
}

pub struct Inbox {
    queue: Mutex<VecDeque<Ask>>,
    screen: SharedScreen,
}

impl Inbox {
    pub fn new(screen: SharedScreen) -> Arc<Self> {
        Arc::new(Self { queue: Mutex::new(VecDeque::new()), screen })
    }

    pub fn pending(&self) -> bool {
        !self.queue.lock().unwrap().is_empty()
    }

    /// Short prompt hint while a question is waiting for an answer.
    pub fn prompt(&self) -> Option<&'static str> {
        match self.queue.lock().unwrap().front()? {
            Ask::Approval { .. } => Some("allow? [y/n/a] › "),
            Ask::Plan { .. } => Some("plan? [A]pprove / [r]eject / feedback › "),
        }
    }

    async fn push(&self, ask: Ask) {
        let text = match &ask {
            Ask::Approval { text, .. } | Ask::Plan { text, .. } => text.clone(),
        };
        let first = {
            let mut q = self.queue.lock().unwrap();
            q.push_back(ask);
            q.len() == 1
        };
        // Only the question at the front is shown; the next one appears once it is answered.
        if first {
            self.screen.lock().await.line(&text);
        }
    }

    /// Answer the oldest question with the user's input line. Returns false if none was pending.
    pub async fn answer(&self, line: &str) -> bool {
        let (ask, next) = {
            let mut q = self.queue.lock().unwrap();
            let Some(ask) = q.pop_front() else { return false };
            let next = q.front().map(|a| match a {
                Ask::Approval { text, .. } | Ask::Plan { text, .. } => text.clone(),
            });
            (ask, next)
        };
        let l = line.trim();
        match ask {
            Ask::Approval { tx, .. } => {
                let d = match l.to_lowercase().as_str() {
                    "y" | "yes" => Decision::Allow,
                    "a" | "always" => Decision::AlwaysForTool,
                    _ => Decision::Deny,
                };
                let _ = tx.send(d);
            }
            Ask::Plan { tx, .. } => {
                let d = match l.to_lowercase().as_str() {
                    "" | "a" | "y" | "yes" | "approve" => PlanDecision::Approve,
                    "r" | "n" | "no" | "reject" => PlanDecision::Reject(String::new()),
                    _ => PlanDecision::Feedback(l.to_string()),
                };
                let _ = tx.send(d);
            }
        }
        if let Some(text) = next {
            self.screen.lock().await.line(&text);
        }
        true
    }
}

pub struct ReplApprover(pub Arc<Inbox>);

#[async_trait]
impl Approver for ReplApprover {
    async fn approve(&self, req: &ApprovalRequest) -> Decision {
        let (tx, rx) = oneshot::channel();
        let text = {
            let s = self.0.screen.lock().await;
            let (y, b, d, r) = (s.c(YELLOW), s.c(BOLD), s.c(DIM), s.c(RESET));
            format!(
                "{y}?{r} {b}{}{r} wants to run {b}{}{r}: {}\n  {d}answer y (allow) / n (deny) / a (always allow {}){r}",
                req.agent, req.tool, req.summary, req.tool
            )
        };
        self.0.push(Ask::Approval { text, tx }).await;
        rx.await.unwrap_or(Decision::Deny)
    }
}

pub struct ReplReviewer(pub Arc<Inbox>);

#[async_trait]
impl PlanReviewer for ReplReviewer {
    async fn review(&self, summary: &str, todos: &[Todo], waves: &[Vec<String>]) -> PlanDecision {
        let (tx, rx) = oneshot::channel();
        let text = {
            let s = self.0.screen.lock().await;
            render_plan(summary, todos, waves, (s.c(CYAN), s.c(BOLD), s.c(DIM), s.c(RESET), s.c(YELLOW)))
        };
        self.0.push(Ask::Plan { text, tx }).await;
        rx.await.unwrap_or_else(|_| PlanDecision::Reject("no answer".into()))
    }
}

/// The plan as shown to the user: TODOs grouped into waves that run in parallel.
pub fn render_plan(summary: &str, todos: &[Todo], waves: &[Vec<String>], colors: (&str, &str, &str, &str, &str)) -> String {
    let (c, b, d, r, y) = colors;
    let mut s = format!("{c}━━ Proposed plan ━━{r}\n{summary}\n");
    for (i, wave) in waves.iter().enumerate() {
        s.push_str(&format!("\n{d}wave {} — runs in parallel{r}\n", i + 1));
        for id in wave {
            if let Some(t) = todos.iter().find(|t| &t.id == id) {
                let files = if t.files.is_empty() { "any file".into() } else { t.files.join(", ") };
                let model = t.model.as_ref().map(|m| format!(" · model {m}")).unwrap_or_default();
                s.push_str(&format!("  ☐ {b}[{}]{r} {}  {d}({} · {}{}){r}\n", t.id, t.title, t.worker, files, model));
                for line in t.task.lines().take(3) {
                    s.push_str(&format!("      {d}{line}{r}\n"));
                }
            }
        }
    }
    s.push_str(&format!("\n{y}?{r} answer: Enter = approve · r = reject · anything else = feedback to revise"));
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::Screen;

    #[tokio::test]
    async fn questions_are_answered_in_order_by_input_lines() {
        let screen = Screen::new(Box::new(std::io::sink()), -1, false, false, true);
        let inbox = Inbox::new(screen);
        let req = |tool: &str| ApprovalRequest { agent: "coder#1".into(), tool: tool.into(), summary: "x".into() };
        let (a1, a2) = (ReplApprover(inbox.clone()), ReplApprover(inbox.clone()));
        let h1 = tokio::spawn(async move { a1.approve(&req("bash")).await });
        let h2 = tokio::spawn(async move { a2.approve(&req("write_file")).await });
        while inbox.queue.lock().unwrap().len() < 2 {
            tokio::task::yield_now().await;
        }
        assert_eq!(inbox.prompt(), Some("allow? [y/n/a] › "));
        assert!(inbox.answer("y").await);
        assert!(inbox.answer("nope").await);
        assert!(!inbox.answer("y").await, "nothing left");
        let mut got = vec![h1.await.unwrap(), h2.await.unwrap()];
        got.sort_by_key(|d| *d == Decision::Deny);
        assert_eq!(got, vec![Decision::Allow, Decision::Deny]);

        let rev = ReplReviewer(inbox.clone());
        let h = tokio::spawn(async move { rev.review("s", &[], &[]).await });
        while !inbox.pending() {
            tokio::task::yield_now().await;
        }
        inbox.answer("please split task 2").await;
        assert_eq!(h.await.unwrap(), PlanDecision::Feedback("please split task 2".into()));
    }
}
