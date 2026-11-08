//! Session persistence: save the team's state after every turn and resume it later.
//!
//! Stored outside the repository (`<data dir>/zilch/sessions/<id>.json`, mode 0600) because it
//! contains prompts, code and tool output. Writes are atomic (temp file + rename).

use super::todo::{TodoList, TodoStatus};
use crate::llm::{Message, ModelBinding, Role, Usage};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

pub const FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentState {
    pub history: Vec<Message>,
    pub session: String,
    #[serde(default)]
    pub usage: Usage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerState {
    pub id: String,
    pub profile: String,
    pub binding: ModelBinding,
    pub readonly: bool,
    pub agent: AgentState,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub version: u32,
    pub id: String,
    pub cwd: PathBuf,
    pub created_ms: u64,
    pub updated_ms: u64,
    /// First user request, for listings.
    pub title: String,
    pub master: AgentState,
    pub workers: Vec<WorkerState>,
    pub todos: TodoList,
    pub counters: HashMap<String, usize>,
}

impl SessionState {
    /// Make a loaded state safe to continue: nothing can still be "in progress", and a history
    /// that ends in an unanswered tool call (crash mid-turn) is trimmed back to a valid point.
    pub fn sanitize(&mut self) {
        for t in self.todos.items.iter_mut().filter(|t| t.status == TodoStatus::InProgress) {
            t.status = TodoStatus::Pending;
        }
        trim_dangling(&mut self.master.history);
        for w in &mut self.workers {
            trim_dangling(&mut w.agent.history);
        }
    }
}

/// Drop a trailing assistant tool call whose results are missing (and any partial results).
pub(crate) fn trim_dangling(h: &mut Vec<Message>) {
    loop {
        let Some(last_asst) = h.iter().rposition(|m| m.role == Role::Assistant) else { return };
        let calls: Vec<&str> = h[last_asst].tool_calls.iter().map(|c| c.id.as_str()).collect();
        if calls.is_empty() {
            return;
        }
        let answered = h[last_asst + 1..].iter().filter(|m| m.role == Role::Tool).count();
        if answered >= calls.len() {
            return;
        }
        h.truncate(last_asst);
    }
}

pub fn now_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

pub fn new_session_id() -> String {
    format!("{}-{:04x}", now_ms(), rand::random::<u16>())
}

/// Summary row for listings.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub id: String,
    pub cwd: PathBuf,
    pub updated_ms: u64,
    pub title: String,
    pub turns: usize,
    pub todos_open: usize,
}

#[derive(Debug, Clone)]
pub struct SessionStore {
    pub dir: PathBuf,
}

impl SessionStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }

    /// `<data dir>/zilch/sessions`.
    pub fn default_store() -> Option<Self> {
        dirs::data_dir().map(|d| Self::new(d.join("zilch").join("sessions")))
    }

    fn path(&self, id: &str) -> Result<PathBuf> {
        if id.is_empty() || !id.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_') {
            bail!("invalid session id '{id}'");
        }
        Ok(self.dir.join(format!("{id}.json")))
    }

    pub fn save(&self, state: &SessionState) -> Result<PathBuf> {
        std::fs::create_dir_all(&self.dir)?;
        #[cfg(unix)]
        {
            // Session titles and ids are private too: owner-only directory.
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(&self.dir, std::fs::Permissions::from_mode(0o700));
        }
        let path = self.path(&state.id)?;
        let tmp = path.with_extension(format!("json.tmp-{}", std::process::id()));
        {
            use std::io::Write;
            let mut opts = std::fs::OpenOptions::new();
            opts.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                opts.mode(0o600);
            }
            let mut f = opts.open(&tmp).with_context(|| format!("writing {}", tmp.display()))?;
            f.write_all(&serde_json::to_vec(state)?)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &path)?;
        Ok(path)
    }

    pub fn load(&self, id: &str) -> Result<SessionState> {
        let path = self.path(id)?;
        let bytes = std::fs::read(&path).with_context(|| format!("no session '{id}' ({})", path.display()))?;
        let mut s: SessionState = serde_json::from_slice(&bytes).with_context(|| format!("corrupt session file {}", path.display()))?;
        if s.version > FORMAT_VERSION {
            bail!("session '{id}' was written by a newer zilch (format {}); please upgrade", s.version);
        }
        s.sanitize();
        Ok(s)
    }

    /// Sessions for `cwd` (or all when None), newest first.
    pub fn list(&self, cwd: Option<&Path>) -> Vec<SessionInfo> {
        let Ok(rd) = std::fs::read_dir(&self.dir) else { return vec![] };
        let mut out: Vec<SessionInfo> = rd
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().is_some_and(|x| x == "json"))
            .filter_map(|e| serde_json::from_slice::<SessionState>(&std::fs::read(e.path()).ok()?).ok())
            .filter(|s| cwd.is_none_or(|c| s.cwd == c))
            .map(|s| SessionInfo {
                turns: s.master.history.iter().filter(|m| m.role == Role::User).count(),
                todos_open: s.todos.items.iter().filter(|t| !t.status.is_terminal()).count(),
                id: s.id,
                cwd: s.cwd,
                updated_ms: s.updated_ms,
                title: s.title,
            })
            .collect();
        out.sort_by_key(|s| std::cmp::Reverse(s.updated_ms));
        out
    }

    pub fn latest(&self, cwd: &Path) -> Option<SessionInfo> {
        self.list(Some(cwd)).into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{FunctionCall, ToolCall};

    fn state(id: &str, cwd: &str, updated: u64) -> SessionState {
        SessionState {
            version: FORMAT_VERSION,
            id: id.into(),
            cwd: cwd.into(),
            created_ms: 1,
            updated_ms: updated,
            title: format!("task {id}"),
            master: AgentState {
                history: vec![Message::user("hello"), Message::assistant("hi")],
                session: "s".into(),
                usage: Usage::default(),
            },
            workers: vec![],
            todos: TodoList::default(),
            counters: HashMap::new(),
        }
    }

    #[test]
    fn save_load_list_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());
        store.save(&state("a-1", "/p", 10)).unwrap();
        store.save(&state("b-2", "/p", 20)).unwrap();
        store.save(&state("c-3", "/other", 30)).unwrap();
        let l = store.list(Some(Path::new("/p")));
        assert_eq!(l.iter().map(|s| s.id.as_str()).collect::<Vec<_>>(), vec!["b-2", "a-1"]);
        assert_eq!(store.latest(Path::new("/p")).unwrap().id, "b-2");
        assert_eq!(store.load("a-1").unwrap().master.history.len(), 2);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mode = std::fs::metadata(dir.path().join("a-1.json")).unwrap().permissions().mode();
            assert_eq!(mode & 0o777, 0o600);
            let dmode = std::fs::metadata(dir.path()).unwrap().permissions().mode();
            assert_eq!(dmode & 0o777, 0o700, "sessions dir must be owner-only");
        }
        assert!(store.load("../etc/passwd").is_err(), "ids are validated");
        assert!(store.load("missing").is_err());
    }

    #[test]
    fn sanitize_fixes_crashed_state() {
        let mut s = state("x", "/p", 1);
        let call = |id: &str| Message {
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall {
                id: id.into(),
                kind: "function".into(),
                function: FunctionCall { name: "bash".into(), arguments: "{}".into() },
            }],
            tool_call_id: None,
            reasoning_content: None,
        };
        s.master.history.push(Message::user("next"));
        s.master.history.push(call("c1"));
        s.master.history.push(Message::tool("c1", "ok"));
        s.master.history.push(call("c2")); // crashed before the result was recorded
        s.todos.apply(vec![serde_json::from_value(serde_json::json!({"id":"t","task":"x"})).unwrap()], "coder").unwrap();
        s.todos.items[0].status = TodoStatus::InProgress;
        s.sanitize();
        assert_eq!(s.master.history.last().unwrap().role, Role::Tool, "dangling call trimmed");
        assert_eq!(s.todos.items[0].status, TodoStatus::Pending);
    }

    #[test]
    fn corrupt_and_future_files_are_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path());
        std::fs::write(dir.path().join("bad.json"), "{not json").unwrap();
        assert!(store.load("bad").unwrap_err().to_string().contains("corrupt"));
        let mut s = state("future", "/p", 1);
        s.version = FORMAT_VERSION + 1;
        store.save(&s).unwrap();
        assert!(store.load("future").unwrap_err().to_string().contains("newer zilch"));
        // Listing skips unreadable files instead of failing.
        assert_eq!(store.list(None).len(), 1);
    }
}
