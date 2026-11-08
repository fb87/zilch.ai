//! Master coordination tools: todo_write, propose_plan, run_todos.

use super::{arg_opt_str, Tool, ToolCtx};
use crate::agent::ApprovalMode;
use crate::team::plan_state::PlanDecision;
use crate::team::todo::TodoInput;
use crate::team::{scheduler, TeamShared};
use anyhow::{bail, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

pub struct TodoWrite(pub Arc<TeamShared>);

#[async_trait]
impl Tool for TodoWrite {
    fn name(&self) -> &str {
        "todo_write"
    }
    fn description(&self) -> String {
        "Create, update, cancel or remove TODO items (upsert by id). Each item is one unit of work for one worker. \
         Keep items independent with disjoint `files` so they run in parallel; use `depends_on` only when required. \
         Set status \"cancelled\" to drop an item, or \"pending\" to retry a failed/blocked one."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{"items":{"type":"array","items":{"type":"object","properties":{
            "id":{"type":"string","description":"Short stable id, e.g. \"t1\""},
            "title":{"type":"string"},
            "task":{"type":"string","description":"Complete, self-contained brief for the worker (it cannot see this conversation)"},
            "worker":{"type":"string","description":"Worker profile (default \"coder\") or an existing instance id like \"coder#2\""},
            "files":{"type":"array","items":{"type":"string"},"description":"Globs of files this item may modify"},
            "depends_on":{"type":"array","items":{"type":"string"}},
            "model":{"type":"string","description":"Optional model override for this item"},
            "provider":{"type":"string"},
            "status":{"type":"string","enum":["pending","cancelled"]},
            "remove":{"type":"boolean"}
        },"required":["id"]}}},"required":["items"]})
    }
    fn summarize(&self, args: &Value) -> String {
        let n = args.get("items").and_then(Value::as_array).map_or(0, Vec::len);
        format!("{n} item(s)")
    }
    async fn call(&self, args: Value, _ctx: &ToolCtx) -> Result<String> {
        let items: Vec<TodoInput> =
            serde_json::from_value(args.get("items").cloned().unwrap_or(Value::Null)).map_err(|e| anyhow::anyhow!("invalid items: {e}"))?;
        for i in &items {
            if let Some(w) = &i.worker {
                if !w.contains('#') && self.0.cfg.workers().all(|p| &p.name != w) {
                    bail!("unknown worker '{w}' for item '{}'", i.id);
                }
            }
        }
        let needs = {
            let mut l = self.0.todos.lock().unwrap();
            // Validate on a copy so a bad batch leaves the list untouched.
            let mut next = l.clone();
            let needs = next.apply(items, "coder")?;
            *l = next;
            needs
        };
        self.0.todos_changed();
        let l = self.0.todos.lock().unwrap();
        let mut out = format!("TODO list updated:\n{}", l.render_compact());
        if !needs.is_empty() {
            out.push_str(&format!("\nNeeds approval: {}. Call propose_plan before run_todos.", needs.join(", ")));
        }
        if let Ok(w) = l.waves() {
            let w: Vec<String> = w.iter().map(|x| format!("[{}]", x.join(" "))).collect();
            out.push_str(&format!("\nParallel waves: {}", w.join(" → ")));
        }
        Ok(out)
    }
}

pub struct ProposePlan(pub Arc<TeamShared>);

#[async_trait]
impl Tool for ProposePlan {
    fn name(&self) -> &str {
        "propose_plan"
    }
    fn description(&self) -> String {
        "Submit the current TODO list to the user for approval. Returns the user's decision \
         (approved / feedback to revise / rejected). Required before run_todos."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{"summary":{"type":"string","description":"2-5 sentence overview of the approach"}},"required":["summary"]})
    }
    async fn call(&self, args: Value, _ctx: &ToolCtx) -> Result<String> {
        let summary = arg_opt_str(&args, "summary").unwrap_or("");
        let (open, waves) = {
            let l = self.0.todos.lock().unwrap();
            let open: Vec<_> = l.items.iter().filter(|t| !t.status.is_terminal()).cloned().collect();
            (open, l.waves()?)
        };
        if open.is_empty() {
            bail!("the TODO list has no open items; add some with todo_write first");
        }
        let decision = if self.0.gate.mode == ApprovalMode::Auto {
            PlanDecision::Approve
        } else {
            self.0.reviewer.review(summary, &open, &waves).await
        };
        Ok(match decision {
            PlanDecision::Approve => {
                self.0.todos.lock().unwrap().approve_all_open();
                self.0.todos_changed();
                "The user APPROVED the plan. Call run_todos now.".into()
            }
            PlanDecision::Feedback(f) => {
                format!("The user wants changes before approving:\n{f}\n\nRevise the TODOs with todo_write, then call propose_plan again.")
            }
            PlanDecision::Reject(r) => format!(
                "The user REJECTED the plan{}. Do not run it. Ask the user how they want to proceed.",
                if r.is_empty() { String::new() } else { format!(": {r}") }
            ),
        })
    }
}

pub struct RunTodos(pub Arc<TeamShared>);

#[async_trait]
impl Tool for RunTodos {
    fn name(&self) -> &str {
        "run_todos"
    }
    fn description(&self) -> String {
        "Execute approved pending TODOs (all of them, or the given ids). Independent items run in parallel on \
         separate worker instances; dependencies and overlapping file scopes are respected. Returns each item's \
         status and the worker's report."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{"ids":{"type":"array","items":{"type":"string"}}}})
    }
    fn summarize(&self, args: &Value) -> String {
        match args.get("ids").and_then(Value::as_array) {
            Some(ids) => ids.iter().filter_map(Value::as_str).collect::<Vec<_>>().join(", "),
            None => "all pending".into(),
        }
    }
    async fn call(&self, args: Value, _ctx: &ToolCtx) -> Result<String> {
        let ids = args.get("ids").and_then(Value::as_array).map(|a| a.iter().filter_map(Value::as_str).map(str::to_string).collect());
        scheduler::run(&self.0, ids).await
    }
}
