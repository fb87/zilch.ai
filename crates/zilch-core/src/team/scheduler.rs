//! Runs approved TODOs as a dependency graph, in parallel, with file-scope conflict avoidance.

use super::todo::{scopes_overlap, Todo, TodoStatus};
use super::{report_ok, TeamShared};
use crate::tools::{truncate_output, Scope};
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use tokio::task::JoinSet;

type Outcome = (String, Result<(String, String)>); // (todo id, Ok((instance, report)))

/// Execute the given TODO ids (or every approved pending item). Returns a digest for the master.
pub async fn run(shared: &Arc<TeamShared>, ids: Option<Vec<String>>) -> Result<String> {
    let targets: Vec<String> = {
        let l = shared.todos.lock().unwrap();
        if let Some(ids) = &ids {
            for id in ids {
                if l.get(id).is_none() {
                    bail!("unknown TODO id '{id}'");
                }
            }
        }
        let selected: Vec<&Todo> = l
            .items
            .iter()
            .filter(|t| ids.as_ref().is_none_or(|ids| ids.contains(&t.id)))
            .filter(|t| t.status == TodoStatus::Pending)
            .collect();
        let unapproved: Vec<&str> = selected.iter().filter(|t| !t.approved).map(|t| t.id.as_str()).collect();
        if !unapproved.is_empty() {
            bail!("these TODOs are not approved yet: {}. Call propose_plan first.", unapproved.join(", "));
        }
        selected.iter().map(|t| t.id.clone()).collect()
    };
    if targets.is_empty() {
        return Ok("Nothing to run: no approved pending TODOs (failed/blocked items must be reset to \"pending\" with todo_write to retry).".into());
    }
    let target_set: HashSet<String> = targets.iter().cloned().collect();
    let cancel = shared.cancel.current();
    let mut running: HashMap<String, Vec<String>> = HashMap::new();
    let mut set: JoinSet<Outcome> = JoinSet::new();

    loop {
        // Propagate failures and pick everything that can start now.
        let to_start: Vec<Todo> = {
            let mut l = shared.todos.lock().unwrap();
            let status: HashMap<String, TodoStatus> = l.items.iter().map(|t| (t.id.clone(), t.status.clone())).collect();
            let mut newly_blocked = Vec::new();
            for t in l.items.iter_mut().filter(|t| target_set.contains(&t.id) && t.status == TodoStatus::Pending) {
                if t.depends_on.iter().any(|d| matches!(status.get(d), Some(TodoStatus::Failed | TodoStatus::Blocked))) {
                    t.status = TodoStatus::Blocked;
                    newly_blocked.push(t.id.clone());
                }
            }
            let mut picked: Vec<Todo> = Vec::new();
            for t in l.items.iter().filter(|t| target_set.contains(&t.id) && t.status == TodoStatus::Pending && !running.contains_key(&t.id)) {
                let deps_done = t.depends_on.iter().all(|d| matches!(status.get(d), Some(TodoStatus::Done | TodoStatus::Cancelled)));
                let conflict = running.values().any(|f| scopes_overlap(f, &t.files)) || picked.iter().any(|p| scopes_overlap(&p.files, &t.files));
                if deps_done && !conflict {
                    picked.push(t.clone());
                }
            }
            drop(l);
            for id in newly_blocked {
                shared.set_todo_status(&id, TodoStatus::Blocked, None, Some("Blocked: a dependency failed.".into()));
            }
            picked
        };

        for t in to_start {
            running.insert(t.id.clone(), t.files.clone());
            shared.set_todo_status(&t.id, TodoStatus::InProgress, None, None);
            let sh = shared.clone();
            set.spawn(async move {
                let id = t.id.clone();
                (id, run_one(sh, t).await)
            });
        }

        if set.is_empty() {
            break;
        }
        let joined = tokio::select! {
            j = set.join_next() => j,
            _ = cancel.cancelled() => {
                set.abort_all();
                for id in running.keys() {
                    shared.set_todo_status(id, TodoStatus::Pending, None, None);
                }
                bail!("cancelled");
            }
        };
        let Some(joined) = joined else { break };
        let (id, outcome) = match joined {
            Ok(o) => o,
            Err(e) => bail!("worker task panicked: {e}"),
        };
        running.remove(&id);
        match outcome {
            Ok((instance, report)) => {
                let st = if report_ok(&report) { TodoStatus::Done } else { TodoStatus::Failed };
                shared.set_todo_status(&id, st, Some(instance), Some(report));
            }
            Err(e) => shared.set_todo_status(&id, TodoStatus::Failed, None, Some(format!("Error: {e:#}"))),
        }
    }

    // Digest for the master.
    let l = shared.todos.lock().unwrap();
    let mut out = String::from("Execution finished.\n");
    for id in &targets {
        let Some(t) = l.get(id) else { continue };
        out.push_str(&format!("\n### [{}] {} — {:?}{}\n", t.id, t.title, t.status, t.instance.as_ref().map(|i| format!(" ({i})")).unwrap_or_default()));
        match (&t.status, &t.report) {
            (TodoStatus::Pending, _) => out.push_str(&format!("Not run: waiting on {}\n", t.depends_on.join(", "))),
            (_, Some(r)) => out.push_str(&truncate_output(r.trim(), 4000)),
            _ => {}
        }
        out.push('\n');
    }
    Ok(out)
}

async fn run_one(sh: Arc<TeamShared>, t: Todo) -> Result<(String, String)> {
    let _slot = sh.pool.slots.clone().acquire_owned().await?;
    let binding = if t.worker.contains('#') {
        None
    } else {
        let profile = sh.cfg.workers().find(|p| p.name == t.worker).ok_or_else(|| anyhow::anyhow!("unknown worker '{}'", t.worker))?;
        Some(sh.binding_for(profile, t.model.as_deref(), t.provider.as_deref())?)
    };
    let (inst, mut agent) = sh.acquire(&t.worker, binding, false).await?;
    sh.set_todo_status(&t.id, TodoStatus::InProgress, Some(inst.id.clone()), None);
    let scope = if t.files.is_empty() { None } else { Some(Scope::new(&t.files)?) };
    let brief = format!("TODO [{}] {}\n\n{}\n\nFiles in scope: {}", t.id, t.title, t.task, if t.files.is_empty() { "(not restricted)".into() } else { t.files.join(", ") });
    let report = sh.run_task(&mut agent, &brief, scope, false).await?;
    Ok((inst.id.clone(), report))
}
