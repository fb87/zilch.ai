//! The master's TODO list: plan, work queue, progress tracker and record of results.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TodoStatus {
    Pending,
    InProgress,
    Done,
    Failed,
    Blocked,
    Cancelled,
}

impl TodoStatus {
    pub fn is_terminal(&self) -> bool {
        matches!(self, Self::Done | Self::Cancelled)
    }
    pub fn icon(&self) -> &'static str {
        match self {
            Self::Pending => "☐",
            Self::InProgress => "◐",
            Self::Done => "✓",
            Self::Failed => "✗",
            Self::Blocked => "⊘",
            Self::Cancelled => "–",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Todo {
    pub id: String,
    pub title: String,
    /// Self-contained instructions for the worker.
    pub task: String,
    /// Worker profile ("coder") or a specific instance ("coder#2").
    pub worker: String,
    /// Globs of files this item may modify. Empty = may touch anything (runs alone).
    pub files: Vec<String>,
    pub depends_on: Vec<String>,
    /// Optional per-item model override.
    pub model: Option<String>,
    pub provider: Option<String>,
    pub status: TodoStatus,
    /// True once the user approved it (directly or via an approved scope).
    pub approved: bool,
    pub instance: Option<String>,
    pub report: Option<String>,
}

/// Fields the master can set through `todo_write`.
#[derive(Debug, Clone, Deserialize)]
pub struct TodoInput {
    pub id: String,
    pub title: Option<String>,
    pub task: Option<String>,
    pub worker: Option<String>,
    pub files: Option<Vec<String>>,
    pub depends_on: Option<Vec<String>>,
    pub model: Option<String>,
    pub provider: Option<String>,
    /// Only "cancelled" or "pending" (to retry a failed/blocked item) may be set by the master.
    pub status: Option<TodoStatus>,
    #[serde(default)]
    pub remove: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TodoList {
    pub items: Vec<Todo>,
    /// Union of file globs from every approved item (used to auto-approve in-scope follow-ups).
    pub approved_scope: Vec<String>,
}

impl TodoList {
    pub fn get(&self, id: &str) -> Option<&Todo> {
        self.items.iter().find(|t| t.id == id)
    }
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Todo> {
        self.items.iter_mut().find(|t| t.id == id)
    }

    /// Apply master edits. Returns ids whose approval was revoked or that need approval.
    pub fn apply(&mut self, inputs: Vec<TodoInput>, default_worker: &str) -> Result<Vec<String>> {
        let mut needs_approval = Vec::new();
        for inp in inputs {
            if inp.remove {
                if let Some(t) = self.get(&inp.id) {
                    if t.status == TodoStatus::InProgress {
                        bail!("cannot remove '{}' while it is in progress", inp.id);
                    }
                }
                self.items.retain(|t| t.id != inp.id);
                continue;
            }
            let scope = self.approved_scope.clone();
            match self.get_mut(&inp.id) {
                Some(t) => {
                    if t.status == TodoStatus::InProgress {
                        bail!("cannot edit '{}' while it is in progress", t.id);
                    }
                    let mut changed_work = false;
                    if let Some(v) = inp.title {
                        t.title = v;
                    }
                    if let Some(v) = inp.task {
                        changed_work |= v != t.task;
                        t.task = v;
                    }
                    if let Some(v) = inp.worker {
                        t.worker = v;
                    }
                    if let Some(v) = inp.files {
                        changed_work |= v != t.files;
                        t.files = v;
                    }
                    if let Some(v) = inp.depends_on {
                        t.depends_on = v;
                    }
                    if inp.model.is_some() {
                        t.model = inp.model;
                    }
                    if inp.provider.is_some() {
                        t.provider = inp.provider;
                    }
                    match inp.status {
                        Some(TodoStatus::Cancelled) => t.status = TodoStatus::Cancelled,
                        Some(TodoStatus::Pending) => {
                            t.status = TodoStatus::Pending;
                            t.report = None;
                        }
                        Some(s) => bail!("status '{s:?}' cannot be set directly; it is managed by the scheduler"),
                        None => {}
                    }
                    if changed_work && t.status == TodoStatus::Done {
                        t.status = TodoStatus::Pending;
                    }
                    if changed_work && t.approved && !within(&t.files, &scope) {
                        t.approved = false;
                    }
                    if !t.approved && !t.status.is_terminal() {
                        needs_approval.push(t.id.clone());
                    }
                }
                None => {
                    let files = inp.files.unwrap_or_default();
                    let approved = !scope.is_empty() && within(&files, &scope);
                    let t = Todo {
                        title: inp.title.unwrap_or_else(|| inp.id.clone()),
                        task: inp.task.ok_or_else(|| anyhow::anyhow!("new item '{}' needs a task", inp.id))?,
                        worker: inp.worker.unwrap_or_else(|| default_worker.to_string()),
                        files,
                        depends_on: inp.depends_on.unwrap_or_default(),
                        model: inp.model,
                        provider: inp.provider,
                        status: TodoStatus::Pending,
                        approved,
                        instance: None,
                        report: None,
                        id: inp.id,
                    };
                    if !approved {
                        needs_approval.push(t.id.clone());
                    }
                    self.items.push(t);
                }
            }
        }
        self.validate()?;
        Ok(needs_approval)
    }

    /// No unknown dependencies, no cycles, unique ids.
    pub fn validate(&self) -> Result<()> {
        let ids: HashSet<&str> = self.items.iter().map(|t| t.id.as_str()).collect();
        if ids.len() != self.items.len() {
            bail!("duplicate TODO ids");
        }
        for t in &self.items {
            for d in &t.depends_on {
                if !ids.contains(d.as_str()) {
                    bail!("'{}' depends on unknown item '{d}'", t.id);
                }
            }
        }
        self.waves().map(|_| ())
    }

    /// Topological layers of non-terminal items (for display and cycle detection).
    pub fn waves(&self) -> Result<Vec<Vec<String>>> {
        let deps: HashMap<&str, Vec<&str>> =
            self.items.iter().map(|t| (t.id.as_str(), t.depends_on.iter().map(String::as_str).collect())).collect();
        let mut done: HashSet<&str> = self.items.iter().filter(|t| t.status.is_terminal()).map(|t| t.id.as_str()).collect();
        let mut remaining: Vec<&str> = self.items.iter().filter(|t| !t.status.is_terminal()).map(|t| t.id.as_str()).collect();
        let mut waves = Vec::new();
        while !remaining.is_empty() {
            let (ready, rest): (Vec<&str>, Vec<&str>) = remaining.iter().partition(|id| deps[*id].iter().all(|d| done.contains(d)));
            if ready.is_empty() {
                bail!("dependency cycle among: {}", rest.join(", "));
            }
            done.extend(ready.iter().copied());
            waves.push(ready.iter().map(|s| s.to_string()).collect());
            remaining = rest;
        }
        Ok(waves)
    }

    pub fn approve_all_open(&mut self) {
        for t in self.items.iter_mut().filter(|t| !t.status.is_terminal()) {
            t.approved = true;
            for f in &t.files {
                if !self.approved_scope.contains(f) {
                    self.approved_scope.push(f.clone());
                }
            }
        }
    }

    pub fn unapproved_open(&self) -> Vec<&Todo> {
        self.items.iter().filter(|t| !t.approved && !t.status.is_terminal()).collect()
    }

    /// Compact rendering injected into the master's context each turn.
    pub fn render_compact(&self) -> String {
        let mut s = String::new();
        for t in &self.items {
            s.push_str(&format!(
                "{} [{}] {} — {:?}{}{}{}\n",
                t.status.icon(),
                t.id,
                t.title,
                t.status,
                t.instance.as_ref().map(|i| format!(" ({i})")).unwrap_or_default(),
                if t.approved { "" } else { " [needs approval]" },
                if t.depends_on.is_empty() { String::new() } else { format!(" after {}", t.depends_on.join(",")) },
            ));
        }
        s
    }
}

/// Every pattern in `files` is covered by `scope` (exact pattern, or matched by a scope glob).
fn within(files: &[String], scope: &[String]) -> bool {
    if files.is_empty() {
        return false; // "touches anything" is never implicitly in scope
    }
    files.iter().all(|f| scope.iter().any(|s| s == f || globset::Glob::new(s).map(|g| g.compile_matcher().is_match(f)).unwrap_or(false)))
}

/// Conservative overlap test between two file scopes.
pub fn scopes_overlap(a: &[String], b: &[String]) -> bool {
    if a.is_empty() || b.is_empty() {
        return true;
    }
    let m = |p: &str, s: &str| globset::Glob::new(p).map(|g| g.compile_matcher().is_match(s)).unwrap_or(true);
    a.iter().any(|x| b.iter().any(|y| x == y || m(x, y) || m(y, x)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inp(id: &str, files: &[&str], deps: &[&str]) -> TodoInput {
        TodoInput {
            id: id.into(),
            title: None,
            task: Some(format!("do {id}")),
            worker: None,
            files: Some(files.iter().map(|s| s.to_string()).collect()),
            depends_on: Some(deps.iter().map(|s| s.to_string()).collect()),
            model: None,
            provider: None,
            status: None,
            remove: false,
        }
    }

    #[test]
    fn waves_and_cycles() {
        let mut l = TodoList::default();
        l.apply(vec![inp("a", &["a.rs"], &[]), inp("b", &["b.rs"], &[]), inp("c", &["c.rs"], &["a", "b"])], "coder").unwrap();
        assert_eq!(l.waves().unwrap(), vec![vec!["a".to_string(), "b".into()], vec!["c".into()]]);
        let err = l.apply(vec![TodoInput { depends_on: Some(vec!["c".into()]), task: None, ..inp("a", &["a.rs"], &[]) }], "coder");
        assert!(err.unwrap_err().to_string().contains("cycle"));
    }

    #[test]
    fn approval_scope_rules() {
        let mut l = TodoList::default();
        let need = l.apply(vec![inp("a", &["src/**"], &[])], "coder").unwrap();
        assert_eq!(need, vec!["a"]);
        l.approve_all_open();
        // In-scope follow-up is auto-approved; out-of-scope needs approval.
        let need = l.apply(vec![inp("fix", &["src/lib.rs"], &[]), inp("docs", &["README.md"], &[])], "coder").unwrap();
        assert_eq!(need, vec!["docs"]);
        assert!(l.get("fix").unwrap().approved);
    }

    #[test]
    fn overlap() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert!(scopes_overlap(&s(&["src/**"]), &s(&["src/main.rs"])));
        assert!(!scopes_overlap(&s(&["src/a.rs"]), &s(&["src/b.rs"])));
        assert!(scopes_overlap(&s(&[]), &s(&["x"])));
    }
}
