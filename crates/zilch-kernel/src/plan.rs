//! The Architect's plan as a task DAG: validation, parallel waves, and which tasks may start.
//! (Adapted from zilch's TODO list: same cycle check and the same conservative overlap test.)

use std::collections::{HashMap, HashSet};
use zilch_ipc::PlanTask;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskStatus {
    Pending,
    Running,
    Done,
    Failed,
    /// A dependency failed.
    Blocked,
}

impl TaskStatus {
    pub fn is_terminal(self) -> bool {
        matches!(self, TaskStatus::Done | TaskStatus::Failed | TaskStatus::Blocked)
    }
}

/// Unique ids, known dependencies, no cycles, and something to do in every task.
pub fn validate(tasks: &[PlanTask]) -> Result<(), String> {
    if tasks.is_empty() {
        return Err("the plan has no tasks".into());
    }
    let ids: HashSet<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
    if ids.len() != tasks.len() {
        return Err("duplicate task ids".into());
    }
    for t in tasks {
        if t.id.trim().is_empty() || t.task.trim().is_empty() {
            return Err(format!("task '{}' needs an id and instructions", t.id));
        }
        for d in &t.depends_on {
            if !ids.contains(d.as_str()) {
                return Err(format!("'{}' depends on unknown task '{d}'", t.id));
            }
        }
    }
    waves(tasks).map(|_| ())
}

/// Topological layers: every task in a wave can run in parallel (scope conflicts permitting).
pub fn waves(tasks: &[PlanTask]) -> Result<Vec<Vec<String>>, String> {
    let deps: HashMap<&str, Vec<&str>> = tasks.iter().map(|t| (t.id.as_str(), t.depends_on.iter().map(String::as_str).collect())).collect();
    let mut done: HashSet<&str> = HashSet::new();
    let mut remaining: Vec<&str> = tasks.iter().map(|t| t.id.as_str()).collect();
    let mut out = Vec::new();
    while !remaining.is_empty() {
        let (ready, rest): (Vec<&str>, Vec<&str>) = remaining.iter().partition(|id| deps[*id].iter().all(|d| done.contains(d)));
        if ready.is_empty() {
            return Err(format!("dependency cycle among: {}", rest.join(", ")));
        }
        done.extend(ready.iter().copied());
        out.push(ready.iter().map(|s| s.to_string()).collect());
        remaining = rest;
    }
    Ok(out)
}

/// Does an owned-path pattern cover `path`? A pattern without glob characters names a file or a
/// directory: `src/x.rs`, `zilch_agent/` and `zilch_agent` all cover the path itself and everything
/// below it (models write directories either way). Anything else is a glob (`src/**/*.rs`).
pub fn path_matches(pattern: &str, path: &str) -> bool {
    let pattern = pattern.trim_start_matches("./");
    let path = path.trim_start_matches("./");
    if pattern.is_empty() {
        return false;
    }
    if !pattern.contains(['*', '?', '[', '{']) {
        let dir = pattern.trim_end_matches('/');
        return path == dir || path.strip_prefix(dir).is_some_and(|rest| rest.starts_with('/'));
    }
    // Usual glob meaning: `*` stays within one directory, `**` crosses directories.
    globset::GlobBuilder::new(pattern).literal_separator(true).build().map(|g| g.compile_matcher().is_match(path)).unwrap_or(false)
}

/// Conservative overlap test between two write scopes ("anything" overlaps everything): one
/// pattern covers the other, or two globs could match the same file.
pub fn scopes_overlap(a: &[String], b: &[String]) -> bool {
    if a.is_empty() || b.is_empty() {
        return true;
    }
    let is_glob = |p: &str| p.contains(['*', '?', '[', '{']);
    a.iter().any(|x| {
        b.iter().any(|y| {
            // Two globs that do not cover each other may still match a common file: assume they do.
            x == y || path_matches(x, y) || path_matches(y, x) || (is_glob(x) && is_glob(y))
        })
    })
}

/// Tasks that can start now: pending, dependencies done, and no write-scope overlap with a
/// running task or with another task picked in the same round.
pub fn startable(tasks: &[PlanTask], status: &[TaskStatus]) -> Vec<usize> {
    let st: HashMap<&str, TaskStatus> = tasks.iter().zip(status).map(|(t, s)| (t.id.as_str(), *s)).collect();
    let mut busy: Vec<&[String]> =
        tasks.iter().zip(status).filter(|(_, s)| **s == TaskStatus::Running).map(|(t, _)| t.owned_paths.as_slice()).collect();
    let mut picked = Vec::new();
    for (i, t) in tasks.iter().enumerate() {
        if status[i] != TaskStatus::Pending || !t.depends_on.iter().all(|d| st.get(d.as_str()) == Some(&TaskStatus::Done)) {
            continue;
        }
        if busy.iter().any(|b| scopes_overlap(b, &t.owned_paths)) {
            continue;
        }
        busy.push(&t.owned_paths);
        picked.push(i);
    }
    picked
}

/// Mark pending tasks whose dependency failed or is blocked. Returns the indexes changed.
pub fn propagate_failures(tasks: &[PlanTask], status: &mut [TaskStatus]) -> Vec<usize> {
    let mut changed = Vec::new();
    loop {
        let st: HashMap<String, TaskStatus> = tasks.iter().zip(status.iter()).map(|(t, s)| (t.id.clone(), *s)).collect();
        let mut any = false;
        for (i, t) in tasks.iter().enumerate() {
            if status[i] == TaskStatus::Pending
                && t.depends_on.iter().any(|d| matches!(st.get(d), Some(TaskStatus::Failed | TaskStatus::Blocked)))
            {
                status[i] = TaskStatus::Blocked;
                changed.push(i);
                any = true;
            }
        }
        if !any {
            return changed;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub fn task(id: &str, deps: &[&str], paths: &[&str]) -> PlanTask {
        PlanTask {
            id: id.into(),
            title: id.into(),
            task: format!("do {id}"),
            depends_on: deps.iter().map(|s| s.to_string()).collect(),
            owned_paths: paths.iter().map(|s| s.to_string()).collect(),
            acceptance: vec![],
            priority: None,
            worker: None,
        }
    }

    #[test]
    fn validation_catches_cycles_unknown_deps_and_duplicates() {
        assert!(validate(&[task("a", &[], &["a"]), task("b", &["a"], &["b"])]).is_ok());
        assert!(validate(&[task("a", &["b"], &[]), task("b", &["a"], &[])]).unwrap_err().contains("cycle"));
        assert!(validate(&[task("a", &["zz"], &[])]).unwrap_err().contains("unknown"));
        assert!(validate(&[task("a", &[], &[]), task("a", &[], &[])]).unwrap_err().contains("duplicate"));
        assert!(validate(&[]).is_err());
    }

    #[test]
    fn independent_disjoint_tasks_start_together_overlapping_ones_wait() {
        let t = [task("a", &[], &["src/a.rs"]), task("b", &[], &["src/b.rs"]), task("c", &[], &["src/**"]), task("d", &["a", "b"], &["x"])];
        let mut st = vec![TaskStatus::Pending; 4];
        assert_eq!(startable(&t, &st), vec![0, 1], "c overlaps both, d waits for its dependencies");
        st[0] = TaskStatus::Done;
        st[1] = TaskStatus::Running;
        assert_eq!(startable(&t, &st), Vec::<usize>::new(), "c still overlaps the running b");
        st[1] = TaskStatus::Done;
        assert_eq!(startable(&t, &st), vec![2, 3]);
    }

    #[test]
    fn a_failure_blocks_everything_downstream() {
        let t = [task("a", &[], &[]), task("b", &["a"], &[]), task("c", &["b"], &[]), task("d", &[], &[])];
        let mut st = vec![TaskStatus::Failed, TaskStatus::Pending, TaskStatus::Pending, TaskStatus::Pending];
        assert_eq!(propagate_failures(&t, &mut st), vec![1, 2]);
        assert_eq!(st[3], TaskStatus::Pending);
    }
}

#[cfg(test)]
mod path_tests {
    use super::*;

    #[test]
    fn directories_cover_what_is_below_them_however_they_are_written() {
        for dir in ["zilch_agent/", "zilch_agent", "./zilch_agent/"] {
            assert!(path_matches(dir, "zilch_agent/config.py"), "{dir}");
            assert!(path_matches(dir, "zilch_agent/sub/x.py"), "{dir}");
            assert!(!path_matches(dir, "zilch_agent_old/x.py"), "a name prefix is not a directory: {dir}");
            assert!(!path_matches(dir, "other/zilch_agent/x.py"), "{dir}");
        }
        assert!(path_matches("src/x.rs", "src/x.rs") && !path_matches("src/x.rs", "src/y.rs"));
        assert!(path_matches("src/**/*.rs", "src/a/b.rs") && !path_matches("src/*.rs", "src/a/b.rs"));
    }

    #[test]
    fn directory_scopes_overlap_their_files() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert!(scopes_overlap(&s(&["zilch_agent/"]), &s(&["zilch_agent/config.py"])));
        assert!(scopes_overlap(&s(&["zilch_agent"]), &s(&["zilch_agent/**"])));
        assert!(!scopes_overlap(&s(&["zilch_agent/"]), &s(&["tests/"])));
        assert!(!scopes_overlap(&s(&["src/a.rs"]), &s(&["src/b.rs"])));
        assert!(scopes_overlap(&s(&["src/*.rs"]), &s(&["src/a.rs"])));
    }
}
