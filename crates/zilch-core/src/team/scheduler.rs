//! Runs approved TODOs as a dependency graph, in parallel, with file-scope conflict avoidance.

use super::todo::{scopes_overlap, Todo, TodoStatus};
use super::{report_ok, TeamShared};
use crate::agent::EventKind;
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
        return Ok(
            "Nothing to run: no approved pending TODOs (failed/blocked items must be reset to \"pending\" with todo_write to retry)."
                .into(),
        );
    }
    let target_set: HashSet<String> = targets.iter().cloned().collect();
    let cancel = shared.cancel.current();
    let mut running: HashMap<String, Vec<String>> = HashMap::new();
    let mut keys: HashMap<String, u64> = HashMap::new();
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
            for t in
                l.items.iter().filter(|t| target_set.contains(&t.id) && t.status == TodoStatus::Pending && !running.contains_key(&t.id))
            {
                let deps_done = t.depends_on.iter().all(|d| matches!(status.get(d), Some(TodoStatus::Done | TodoStatus::Cancelled)));
                let conflict =
                    running.values().any(|f| scopes_overlap(f, &t.files)) || picked.iter().any(|p| scopes_overlap(&p.files, &t.files));
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
            let key = task_key(&t);
            if let Some(refusal) = refuse_repeat(shared, key) {
                // Loop protection: this exact task keeps failing; running it again is waste.
                shared.events.emit(
                    "master",
                    EventKind::Notice(format!("⚠ supervisor: [{}] not run — the same task already failed {} times", t.id, refusal.0)),
                );
                shared.set_todo_status(&t.id, TodoStatus::Failed, None, Some(refusal.1));
                continue;
            }
            keys.insert(t.id.clone(), key);
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
        let key = keys.remove(&id);
        let failure = match outcome {
            Ok((instance, report)) => {
                let ok = report_ok(&report);
                let st = if ok { TodoStatus::Done } else { TodoStatus::Failed };
                shared.set_todo_status(&id, st, Some(instance), Some(report.clone()));
                (!ok).then_some(report)
            }
            Err(e) => {
                let report = format!("Error: {e:#}");
                shared.set_todo_status(&id, TodoStatus::Failed, None, Some(report.clone()));
                (!report.contains("cancelled")).then_some(report)
            }
        };
        if let (Some(key), Some(report)) = (key, failure) {
            let mut f = shared.task_failures.lock().unwrap();
            let e = f.entry(key).or_insert((0, String::new()));
            e.0 += 1;
            e.1 = report;
        }
        // Persist progress now: a crash later in this run must not lose finished TODOs.
        shared.checkpoint();
    }

    // Digest for the master.
    let l = shared.todos.lock().unwrap();
    let mut out = String::from("Execution finished.\n");
    for id in &targets {
        let Some(t) = l.get(id) else { continue };
        out.push_str(&format!(
            "\n### [{}] {} — {:?}{}\n",
            t.id,
            t.title,
            t.status,
            t.instance.as_ref().map(|i| format!(" ({i})")).unwrap_or_default()
        ));
        match (&t.status, &t.report) {
            (TodoStatus::Pending, _) => out.push_str(&format!("Not run: waiting on {}\n", t.depends_on.join(", "))),
            (_, Some(r)) => out.push_str(&truncate_output(r.trim(), 4000)),
            _ => {}
        }
        out.push('\n');
    }
    Ok(out)
}

/// Identity of a task for loop protection: the worker and its instructions, whitespace-normalized.
/// A TODO re-run under a new id is the same task; one whose instructions changed is not.
pub fn task_key(t: &Todo) -> u64 {
    use std::hash::{Hash, Hasher};
    let worker = t.worker.split('#').next().unwrap_or(&t.worker);
    let task = t.task.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut h = std::collections::hash_map::DefaultHasher::new();
    (worker, task, &t.files).hash(&mut h);
    h.finish()
}

/// If this task already failed `loop_limit - 1` times unchanged, the (count, report) to fail it
/// with instead of running it again.
fn refuse_repeat(shared: &TeamShared, key: u64) -> Option<(usize, String)> {
    let limit = shared.cfg.loop_limit;
    let f = shared.task_failures.lock().unwrap();
    let (n, last) = f.get(&key)?;
    (limit > 0 && *n + 1 >= limit).then(|| {
        (
            *n,
            format!(
                "[stopped by the supervisor: not run — this exact task already failed {n} times. Running it unchanged \
                 will fail again: change the instructions (another approach, a smaller step), or report the blocker \
                 to the user.]\nSTATUS: failed\n\nLast failure:\n{}",
                truncate_output(last.trim(), 2000)
            ),
        )
    })
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
    let brief = format!(
        "TODO [{}] {}\n\n{}\n\nFiles in scope: {}",
        t.id,
        t.title,
        t.task,
        if t.files.is_empty() { "(not restricted)".into() } else { t.files.join(", ") }
    );
    let report = sh.run_task(&mut agent, &brief, scope, false).await?;
    Ok((inst.id.clone(), report))
}
