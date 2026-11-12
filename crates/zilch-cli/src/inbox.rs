//! Questions from agents (tool approvals, plan reviews) in the REPL.
//!
//! With two lanes running in parallel the keyboard has a single owner: the line editor. So
//! agents don't read stdin themselves; they queue a question here, it is printed above the
//! prompt, and the user's next input line answers it. Questions are answered in order.
//!
//! The question at the front is also pinned in the status row (one line, with its lane) until it
//! is answered, so output from the other lane or from workers cannot scroll it out of sight.

use crate::render::{SharedScreen, BOLD, CYAN, DIM, RESET, YELLOW};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tokio::sync::oneshot;
use zilch_core::agent::{ApprovalRequest, Decision};
use zilch_core::team::todo::Todo;
use zilch_core::{Approver, PlanDecision, PlanReviewer};

enum Reply {
    Approval(oneshot::Sender<Decision>),
    Plan(oneshot::Sender<PlanDecision>),
}

struct Ask {
    /// The full question, printed above the prompt.
    text: String,
    /// One line for the status row.
    short: String,
    reply: Reply,
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
        match self.queue.lock().unwrap().front()?.reply {
            Reply::Approval(_) => Some("allow? [y/n/a] › "),
            Reply::Plan(_) => Some("plan? [A]pprove / [r]eject / feedback › "),
        }
    }

    fn front(&self) -> Option<(String, String)> {
        self.queue.lock().unwrap().front().map(|a| (a.text.clone(), a.short.clone()))
    }

    /// Print the front question and pin it in the status row (or unpin when none is left).
    async fn show_front(&self) {
        let front = self.front();
        let mut s = self.screen.lock().await;
        match front {
            Some((text, short)) => {
                s.line(&text);
                s.set_question(Some(short));
            }
            None => s.set_question(None),
        }
    }

    /// Print the waiting question again (a line was typed before its answer prompt appeared).
    pub async fn reshow(&self) {
        self.show_front().await;
    }

    async fn push(&self, ask: Ask) {
        let first = {
            let mut q = self.queue.lock().unwrap();
            q.push_back(ask);
            q.len() == 1
        };
        // Only the question at the front is shown; the next one appears once it is answered.
        if first {
            self.show_front().await;
        }
    }

    /// Is `line` nothing but an answer to the waiting question (empty, y / n / a, approve, reject …)?
    /// Such a line answers it even when it was typed at the lane's prompt: a question arriving while
    /// the line editor already waits does not change the prompt on screen, and an empty line or a
    /// lone "y" cannot be an order.
    pub fn is_answer(&self, line: &str) -> bool {
        let l = line.trim().to_lowercase();
        match self.queue.lock().unwrap().front().map(|a| &a.reply) {
            Some(Reply::Approval(_)) => matches!(l.as_str(), "y" | "yes" | "n" | "no" | "a" | "always"),
            Some(Reply::Plan(_)) => matches!(l.as_str(), "" | "a" | "y" | "yes" | "approve" | "r" | "n" | "no" | "reject"),
            None => false,
        }
    }

    /// Answer the oldest question with the user's input line. Returns false if none was pending.
    pub async fn answer(&self, line: &str) -> bool {
        let Some(ask) = self.queue.lock().unwrap().pop_front() else { return false };
        let l = line.trim();
        match ask.reply {
            Reply::Approval(tx) => {
                let d = match l.to_lowercase().as_str() {
                    "y" | "yes" => Decision::Allow,
                    "a" | "always" => Decision::AlwaysForTool,
                    _ => Decision::Deny,
                };
                let _ = tx.send(d);
            }
            Reply::Plan(tx) => {
                let d = match l.to_lowercase().as_str() {
                    "" | "a" | "y" | "yes" | "approve" => PlanDecision::Approve,
                    "r" | "n" | "no" | "reject" => PlanDecision::Reject(String::new()),
                    _ => PlanDecision::Feedback(l.to_string()),
                };
                let _ = tx.send(d);
            }
        }
        self.show_front().await;
        true
    }
}

pub struct ReplApprover(pub Arc<Inbox>);

#[async_trait]
impl Approver for ReplApprover {
    async fn approve(&self, req: &ApprovalRequest) -> Decision {
        let (tx, rx) = oneshot::channel();
        let (text, short) = {
            let s = self.0.screen.lock().await;
            let (y, b, d, r) = (s.c(YELLOW), s.c(BOLD), s.c(DIM), s.c(RESET));
            let lane = s.lane_label(&req.agent);
            let text = format!(
                "{y}?{r} {b}{lane}{r} · {b}{}{r} wants to run {b}{}{r}: {}\n  {d}answer y (allow) / n (deny) / a (always allow {}){r}",
                req.agent, req.tool, req.summary, req.tool
            );
            (text, format!("{lane} · {} wants {}: {} — y/n/a", req.agent, req.tool, req.summary))
        };
        self.0.push(Ask { text, short, reply: Reply::Approval(tx) }).await;
        rx.await.unwrap_or(Decision::Deny)
    }
}

pub struct ReplReviewer(pub Arc<Inbox>);

#[async_trait]
impl PlanReviewer for ReplReviewer {
    async fn review(&self, summary: &str, todos: &[Todo], waves: &[Vec<String>]) -> PlanDecision {
        let (tx, rx) = oneshot::channel();
        let (text, short) = {
            let s = self.0.screen.lock().await;
            // Plans come from the code lane's master.
            let lane = s.lane_label("master");
            let (b, r) = (s.c(BOLD), s.c(RESET));
            let plan = render_plan(summary, todos, waves, (s.c(CYAN), s.c(BOLD), s.c(DIM), s.c(RESET), s.c(YELLOW)));
            let n = todos.len();
            (
                format!("{b}{lane}{r} lane asks for a plan review:\n{plan}"),
                format!("{lane} · plan review, {n} TODO(s) — Enter approve / r reject / feedback"),
            )
        };
        self.0.push(Ask { text, short, reply: Reply::Plan(tx) }).await;
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

    #[tokio::test]
    async fn answer_words_answer_a_question_that_arrived_while_the_lane_prompt_was_up() {
        let screen = Screen::new(Box::new(std::io::sink()), -1, false, false, true);
        let inbox = Inbox::new(screen);
        assert!(!inbox.is_answer(""), "nothing waiting");
        let rev = ReplReviewer(inbox.clone());
        let h = tokio::spawn(async move { rev.review("s", &[], &[]).await });
        while !inbox.pending() {
            tokio::task::yield_now().await;
        }
        // A plain Enter, y or r answers the plan; a sentence is not taken as an answer this way.
        for w in ["", "y", "Approve", "r", " no "] {
            assert!(inbox.is_answer(w), "{w:?}");
        }
        assert!(!inbox.is_answer("add a --verbose flag"));
        inbox.answer("").await;
        assert_eq!(h.await.unwrap(), PlanDecision::Approve);
        // For a command approval an empty line is not an answer (it would deny).
        let a = ReplApprover(inbox.clone());
        let req = ApprovalRequest { agent: "coder#1".into(), tool: "bash".into(), summary: "x".into() };
        let h = tokio::spawn(async move { a.approve(&req).await });
        while !inbox.pending() {
            tokio::task::yield_now().await;
        }
        assert!(!inbox.is_answer("") && inbox.is_answer("y") && inbox.is_answer("a"));
        inbox.answer("y").await;
        assert_eq!(h.await.unwrap(), Decision::Allow);
    }

    #[tokio::test]
    async fn the_waiting_question_is_pinned_in_the_status_row_with_its_lane() {
        let screen = Screen::new(Box::new(std::io::sink()), -1, false, false, true);
        let bar = Arc::new(crate::status::StatusBar::default());
        screen.lock().await.status_bar = Some(bar.clone());
        let inbox = Inbox::new(screen.clone());
        let ask = |agent: &str, tool: &str| {
            let a = ReplApprover(inbox.clone());
            let req = ApprovalRequest { agent: agent.into(), tool: tool.into(), summary: "cargo test".into() };
            tokio::spawn(async move { a.approve(&req).await })
        };
        let h1 = ask("coder#1", "bash");
        while !inbox.pending() {
            tokio::task::yield_now().await;
        }
        assert_eq!(bar.text(), "? code · coder#1 wants bash: cargo test — y/n/a ");
        let h2 = ask("spec", "write_doc");
        while inbox.queue.lock().unwrap().len() < 2 {
            tokio::task::yield_now().await;
        }
        // Other activity does not replace the pinned question.
        screen.lock().await.update_status();
        assert!(bar.text().starts_with("? code · coder#1"), "{}", bar.text());
        inbox.answer("y").await;
        assert!(bar.text().starts_with("? spec · spec wants write_doc"), "the next question is pinned: {}", bar.text());
        inbox.answer("n").await;
        assert_eq!(bar.text(), "▸ code: idle  │  spec: idle", "unpinned once answered");
        assert_eq!((h1.await.unwrap(), h2.await.unwrap()), (Decision::Allow, Decision::Deny));
    }
}
