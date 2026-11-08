//! `delegate`: send one task to a worker. Without a todo_id it is read-only exploration;
//! with a todo_id it is a follow-up on an approved TODO (write access within that TODO's files).

use super::{arg_bool, arg_opt_str, arg_str, Scope, Tool, ToolCtx};
use crate::agent::EventKind;
use crate::team::todo::TodoStatus;
use crate::team::{report_ok, TeamShared};
use anyhow::{anyhow, bail, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::sync::Arc;

pub struct Delegate(pub Arc<TeamShared>);

#[async_trait]
impl Tool for Delegate {
    fn name(&self) -> &str {
        "delegate"
    }
    fn description(&self) -> String {
        let workers = self.0.worker_profiles().iter().map(|(n, d)| format!("{n} ({d})")).collect::<Vec<_>>().join("; ");
        format!(
            "Send one task to a worker and wait for its report. Several delegate calls in one turn run in parallel. \
             Without todo_id: READ-ONLY exploration (worker gets read-only tools; default worker \"explorer\"). \
             With todo_id: follow-up work on that approved TODO with write access to its files (defaults to the \
             instance that ran it, keeping its context). Worker profiles: {workers}. You can also target a live \
             instance id like \"coder#2\"."
        )
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "task":{"type":"string","description":"Complete, self-contained instructions"},
            "worker":{"type":"string","description":"Profile name or instance id"},
            "todo_id":{"type":"string"},
            "fresh":{"type":"boolean","description":"Reset the instance's history first"}
        },"required":["task"]})
    }
    fn summarize(&self, args: &Value) -> String {
        let w = args.get("worker").and_then(Value::as_str).unwrap_or("explorer");
        let t: String = args.get("task").and_then(Value::as_str).unwrap_or("").chars().take(80).collect();
        format!("{w}: {t}")
    }
    async fn call(&self, args: Value, _ctx: &ToolCtx) -> Result<String> {
        let sh = &self.0;
        let task = arg_str(&args, "task")?;
        let fresh = arg_bool(&args, "fresh");
        let todo_id = arg_opt_str(&args, "todo_id");

        let (worker, scope, binding, readonly) = match todo_id {
            None => (arg_opt_str(&args, "worker").unwrap_or("explorer").to_string(), None, None, true),
            Some(id) => {
                let t = sh.todos.lock().unwrap().get(id).cloned().ok_or_else(|| anyhow!("unknown TODO '{id}'"))?;
                if !t.approved {
                    bail!("TODO '{id}' is not approved; call propose_plan first");
                }
                if t.status == TodoStatus::InProgress {
                    bail!("TODO '{id}' is currently running");
                }
                let worker = arg_opt_str(&args, "worker").map(str::to_string).or(t.instance.clone()).unwrap_or(t.worker.clone());
                let binding = if worker.contains('#') {
                    None
                } else {
                    let p = sh.cfg.workers().find(|p| p.name == worker).ok_or_else(|| anyhow!("unknown worker '{worker}'"))?;
                    Some(sh.binding_for(p, t.model.as_deref(), t.provider.as_deref())?)
                };
                let scope = if t.files.is_empty() { None } else { Some(Scope::new(&t.files)?) };
                (worker, scope, binding, false)
            }
        };

        let _slot = sh.pool.slots.clone().acquire_owned().await?;
        let (inst, mut agent) = sh.acquire(&worker, binding, readonly).await?;
        sh.events.emit("master", EventKind::DelegateStart { worker: inst.id.clone(), task: task.chars().take(120).collect() });
        if let Some(id) = todo_id {
            sh.set_todo_status(id, TodoStatus::InProgress, Some(inst.id.clone()), None);
        }
        // Read-only workers must not burn their step budget on something they can't do.
        let brief = if readonly {
            format!(
                "{task}\n\n(Note: this is a READ-ONLY assignment — you have no shell and cannot modify files. If the request \
                 requires running commands or editing, do not attempt workarounds: answer what you can from reading, then \
                 end with STATUS: failed and explain what needs a todo_id/write access.)"
            )
        } else {
            task.to_string()
        };
        let res = sh.run_task(&mut agent, &brief, scope, fresh).await;
        drop(agent); // release the instance so the checkpoint records its new history
        let ok = res.as_ref().is_ok_and(|r| report_ok(r));
        sh.events.emit("master", EventKind::DelegateEnd { worker: inst.id.clone(), ok });
        if let Some(id) = todo_id {
            let (st, rep) = match &res {
                Ok(r) => (if ok { TodoStatus::Done } else { TodoStatus::Failed }, r.clone()),
                Err(e) => (TodoStatus::Failed, format!("Error: {e:#}")),
            };
            sh.set_todo_status(id, st, Some(inst.id.clone()), Some(rep));
        }
        sh.checkpoint();
        Ok(format!("[report from {}]\n{}", inst.id, res?))
    }
}
