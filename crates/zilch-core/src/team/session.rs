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
    /// A brief, human-readable recap of the session (no model call): what was asked, the last
    /// answer, TODO progress and kept workers. Shown when resuming so the user can pick the next
    /// step quickly.
    pub fn recap(&self) -> String {
        let one_line = |s: &str, n: usize| {
            let flat = s.split_whitespace().collect::<Vec<_>>().join(" ");
            if flat.chars().count() > n {
                format!("{}…", flat.chars().take(n).collect::<String>())
            } else {
                flat
            }
        };
        let user_msgs: Vec<&Message> =
            self.master.history.iter().filter(|m| m.role == Role::User && !m.text().starts_with("<conversation-summary>")).collect();
        let interrupted =
            self.master.history.last().is_some_and(|m| m.text().contains("still in progress when the session was last saved"));
        let last_answer = self
            .master
            .history
            .iter()
            .rev()
            .find(|m| {
                m.role == Role::Assistant
                    && m.tool_calls.is_empty()
                    && !m.text().is_empty()
                    && !m.text().starts_with("[This request was still in progress")
            })
            .map(|m| m.text().to_string());
        let age = (now_ms().saturating_sub(self.updated_ms)) / 1000;
        let ago = match age {
            a if a < 120 => format!("{a}s ago"),
            a if a < 7200 => format!("{}m ago", a / 60),
            a if a < 172_800 => format!("{}h ago", a / 3600),
            a => format!("{}d ago", a / 86_400),
        };
        let mut s = format!("↺ Resumed session {} · last active {ago} · {} turn(s)\n", self.id, user_msgs.len());
        if !self.title.is_empty() {
            s.push_str(&format!("  Started with: {}\n", one_line(&self.title, 100)));
        }
        if let Some(last) = user_msgs.last() {
            if user_msgs.len() > 1 {
                s.push_str(&format!("  Last request: {}\n", one_line(last.text(), 100)));
            }
        }
        if interrupted {
            s.push_str("  ⚠ The last request was interrupted before it finished (see TODOs below).\n");
        } else if let Some(a) = last_answer {
            s.push_str(&format!("  Last answer:  {}\n", one_line(&a, 160)));
        }
        let items = &self.todos.items;
        if !items.is_empty() {
            let count = |st: TodoStatus| items.iter().filter(|t| t.status == st).count();
            let mut parts = Vec::new();
            for (st, label) in [
                (TodoStatus::Done, "done"),
                (TodoStatus::Failed, "failed"),
                (TodoStatus::Blocked, "blocked"),
                (TodoStatus::Pending, "pending"),
                (TodoStatus::InProgress, "in progress"),
                (TodoStatus::Cancelled, "cancelled"),
            ] {
                let n = count(st);
                if n > 0 {
                    parts.push(format!("{n} {label}"));
                }
            }
            s.push_str(&format!("  TODOs: {}\n", parts.join(" · ")));
            // Unfinished items first (they are what needs deciding), then what was done.
            let ordered = items.iter().filter(|t| !t.status.is_terminal()).chain(items.iter().filter(|t| t.status == TodoStatus::Done));
            for (shown, t) in ordered.enumerate() {
                if shown == 8 {
                    s.push_str(&format!("    … and {} more (/todos)\n", items.len() - shown));
                    break;
                }
                let note = match (&t.status, &t.report) {
                    (TodoStatus::Failed | TodoStatus::Blocked, Some(r)) => format!(" — {}", one_line(r, 70)),
                    _ => String::new(),
                };
                s.push_str(&format!("    {} [{}] {}{note}\n", t.status.icon(), t.id, one_line(&t.title, 60)));
            }
        }
        if !self.workers.is_empty() {
            let w: Vec<String> = self.workers.iter().map(|w| format!("{} ({} msgs)", w.id, w.agent.history.len())).collect();
            s.push_str(&format!("  Workers kept: {}\n", w.join(", ")));
        }
        s.trim_end().to_string()
    }

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
    fn recap_summarizes_progress_and_flags_interruptions() {
        let mut s = state("r-1", "/p", now_ms() - 7_200_000);
        s.title = "make a and b".into();
        s.master.history = vec![
            Message::user("make a and b"),
            Message::assistant("Did a; dropped b for now."),
            Message::user("now   add\n tests"),
            Message::assistant("Tests added for parse.py; the CLI test still fails."),
        ];
        let todo = |id: &str, title: &str| {
            serde_json::from_value::<super::super::todo::TodoInput>(serde_json::json!({"id": id, "title": title, "task": "x"})).unwrap()
        };
        s.todos.apply(vec![todo("a", "Create a.txt"), todo("c", "Add tests"), todo("d", "Write docs")], "coder").unwrap();
        s.todos.items[0].status = TodoStatus::Done;
        s.todos.items[1].status = TodoStatus::Failed;
        s.todos.items[1].report = Some("STATUS: failed\nSUMMARY: CLI test asserts wrong padding".into());
        s.workers.push(WorkerState {
            id: "coder#1".into(),
            profile: "coder".into(),
            binding: Default::default(),
            readonly: false,
            agent: AgentState { history: vec![Message::user("x"); 8], session: "s".into(), usage: Usage::default() },
        });
        let r = s.recap();
        assert!(r.starts_with("↺ Resumed session r-1 · last active 2h ago · 2 turn(s)"), "{r}");
        assert!(r.contains("Started with: make a and b"));
        assert!(r.contains("Last request: now add tests"), "whitespace flattened: {r}");
        assert!(r.contains("Last answer:  Tests added for parse.py"));
        assert!(r.contains("TODOs: 1 done · 1 failed · 1 pending"), "{r}");
        // Unfinished items are listed before done ones; failures show why.
        assert!(r.find("[c] Add tests").unwrap() < r.find("[a] Create a.txt").unwrap());
        assert!(r.contains("✗ [c] Add tests — STATUS: failed SUMMARY: CLI test asserts wrong padding"), "{r}");
        assert!(r.contains("Workers kept: coder#1 (8 msgs)"));

        // A checkpoint saved mid-turn is reported as interrupted, not as a normal answer.
        s.master.history.push(Message::user("refactor everything"));
        s.master.history.push(Message::assistant("[This request was still in progress when the session was last saved. ...]"));
        let r = s.recap();
        assert!(r.contains("⚠ The last request was interrupted") && !r.contains("Last answer"), "{r}");
        assert!(r.contains("Last request: refactor everything"));
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
