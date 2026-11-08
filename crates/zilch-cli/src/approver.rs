//! Interactive approval of mutating tool calls and of the master's TODO plan.
//! Both hold the screen lock while asking, so parallel workers queue their prompts.

use crate::render::{SharedScreen, BOLD, CYAN, DIM, RESET, YELLOW};
use async_trait::async_trait;
use zilch_core::agent::{ApprovalRequest, Decision};
use zilch_core::team::todo::Todo;
use zilch_core::{Approver, PlanDecision, PlanReviewer};

async fn read_line() -> Option<String> {
    tokio::task::spawn_blocking(|| {
        let mut s = String::new();
        match std::io::stdin().read_line(&mut s) {
            Ok(0) | Err(_) => None,
            Ok(_) => Some(s.trim().to_string()),
        }
    })
    .await
    .ok()
    .flatten()
}

pub struct TerminalApprover(pub SharedScreen);

#[async_trait]
impl Approver for TerminalApprover {
    async fn approve(&self, req: &ApprovalRequest) -> Decision {
        let mut screen = self.0.lock().await;
        screen.clear_panel();
        let (y, b, d, r) = (screen.c(YELLOW), screen.c(BOLD), screen.c(DIM), screen.c(RESET));
        screen.write_raw(&format!(
            "\n{y}?{r} {b}{}{r} wants to run {b}{}{r}: {}\n  {d}[y]es / [n]o / [a]lways allow {}{r} › ",
            req.agent, req.tool, req.summary, req.tool
        ));
        let answer = read_line().await.unwrap_or_default().to_lowercase();
        screen.line("");
        match answer.as_str() {
            "y" | "yes" => Decision::Allow,
            "a" | "always" => Decision::AlwaysForTool,
            _ => Decision::Deny,
        }
    }
}

pub struct TerminalReviewer(pub SharedScreen);

#[async_trait]
impl PlanReviewer for TerminalReviewer {
    async fn review(&self, summary: &str, todos: &[Todo], waves: &[Vec<String>]) -> PlanDecision {
        let mut screen = self.0.lock().await;
        screen.clear_panel();
        let (c, b, d, r, y) = (screen.c(CYAN), screen.c(BOLD), screen.c(DIM), screen.c(RESET), screen.c(YELLOW));
        let mut s = format!("\n{c}━━ Proposed plan ━━{r}\n{summary}\n");
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
        s.push_str(&format!("\n{y}?{r} [A]pprove (enter) / [f]eedback / [r]eject — or just type feedback › "));
        screen.write_raw(&s);
        // EOF (no interactive input, e.g. piped or CI) must never approve.
        let Some(answer) = read_line().await else {
            screen.line("(no input — plan rejected; use --yolo or approval = \"auto\" for unattended runs)");
            return PlanDecision::Reject("no interactive input available".into());
        };
        let decision = match answer.to_lowercase().as_str() {
            "a" | "y" | "yes" | "approve" | "" => PlanDecision::Approve,
            "r" | "n" | "no" | "reject" => {
                screen.write_raw(&format!("{d}reason (optional) ›{r} "));
                PlanDecision::Reject(read_line().await.unwrap_or_default())
            }
            "f" | "feedback" => {
                screen.write_raw(&format!("{d}feedback ›{r} "));
                PlanDecision::Feedback(read_line().await.unwrap_or_default())
            }
            // Anything else typed is treated as feedback directly.
            _ => PlanDecision::Feedback(answer),
        };
        screen.line("");
        decision
    }
}
