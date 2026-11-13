//! Tools: synchronous ports of zilch's tools (same behaviour, same tests), run by the kernel's
//! driver pools. Capability checks happen in the kernel before a tool runs; the path checks here
//! (workspace confinement, `.git/` protection, stale-edit detection) still apply on top.

pub mod bash;
pub mod edit;
pub mod fs;
pub mod git;
pub mod repomap_tool;
pub mod sandbox;
#[cfg(target_os = "linux")]
pub mod sandbox_linux;
pub mod search;
pub mod spec_tools;
pub mod web;

use anyhow::Result;
use serde_json::Value;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex};
use zilch_ipc::ToolSpec;
use zilch_kernel::{Access, CancelToken};

pub trait Tool: Send + Sync {
    fn name(&self) -> &str;
    fn description(&self) -> String;
    /// JSON Schema for the arguments object.
    fn schema(&self) -> Value;
    /// File-writing tools: the path a call would modify (for the kernel's capability check).
    fn target_path(&self, _args: &Value) -> Option<String> {
        None
    }
    /// Runs commands (goes to the exec pool and needs the exec capability).
    fn exec(&self) -> bool {
        false
    }
    /// One-line human summary of a call.
    fn summarize(&self, args: &Value) -> String {
        let s = args.to_string();
        if s.len() > 120 {
            format!("{}…", &s[..s.floor_char_boundary(120)])
        } else {
            s
        }
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String>;
}

/// Safety policy applied by the built-in tools.
#[derive(Debug, Clone)]
pub struct ToolPolicy {
    /// Confine file tools to the workspace and forbid writes inside `.git/`.
    pub sandbox: bool,
    pub secret_env: Vec<String>,
    pub env_passthrough: Vec<String>,
    pub bash_sandbox: sandbox::BashSandbox,
    pub bash: sandbox::SandboxSpec,
    /// Directories the read tools may also read (never write): fetched outside sources.
    pub read_roots: Vec<PathBuf>,
}

impl Default for ToolPolicy {
    fn default() -> Self {
        Self {
            sandbox: true,
            secret_env: vec![],
            env_passthrough: vec![],
            bash_sandbox: sandbox::BashSandbox::Auto,
            bash: sandbox::SandboxSpec { network: true, protect_git: true, ..Default::default() },
            read_roots: vec![],
        }
    }
}

impl ToolPolicy {
    /// Should `name` be removed from the environment of commands the model runs?
    pub fn is_secret_env(&self, name: &str) -> bool {
        if self.env_passthrough.iter().any(|p| p == name) {
            return false;
        }
        let up = name.to_ascii_uppercase();
        self.secret_env.iter().any(|s| s == name)
            || ["_API_KEY", "_APIKEY", "_TOKEN", "_SECRET", "_PASSWORD", "_PASSWD", "_CREDENTIALS", "_PRIVATE_KEY"]
                .iter()
                .any(|s| up.ends_with(s))
            || ["AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN", "GH_TOKEN", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"].contains(&up.as_str())
    }
}

/// Per-path locks so parallel processes never interleave writes to one file.
#[derive(Clone, Default)]
pub struct FileLocks(Arc<(Mutex<HashSet<PathBuf>>, Condvar)>);

pub struct FileGuard {
    locks: FileLocks,
    path: PathBuf,
}

impl FileLocks {
    pub fn lock(&self, path: &Path) -> FileGuard {
        let (m, cv) = &*self.0;
        let mut held = m.lock().unwrap();
        while held.contains(path) {
            held = cv.wait(held).unwrap();
        }
        held.insert(path.to_path_buf());
        FileGuard { locks: self.clone(), path: path.to_path_buf() }
    }
}

impl Drop for FileGuard {
    fn drop(&mut self) {
        let (m, cv) = &*self.locks.0;
        m.lock().unwrap().remove(&self.path);
        cv.notify_all();
    }
}

/// What a tool call runs with: one process's view of the workspace.
pub struct ToolCtx {
    pub agent_id: String,
    pub cwd: PathBuf,
    pub policy: Arc<ToolPolicy>,
    pub cancel: CancelToken,
    pub locks: FileLocks,
    /// Content hash of every file this process read or wrote (stale-edit detection).
    pub read_hashes: Arc<Mutex<HashMap<PathBuf, String>>>,
    /// What this process was shown that is still in its context: path → (content hash, line ranges);
    /// also identical grep/glob calls (see `repeat_note`). Kept per process by the driver set and
    /// forgotten when its history shrinks (compaction, restart, dropped reasoning).
    pub shown: Shown,
    /// The workspace's repository map (None: repo_map / find_definition scan per call).
    pub repo_map: Option<Arc<Mutex<crate::repomap::RepoMap>>>,
    /// Live output of long commands (latest line), for the process table.
    pub progress: Option<Arc<dyn Fn(String) + Send + Sync>>,
    /// Heartbeat to the kernel while a long call is still working (see `DriverCtx::alive`).
    pub alive: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl ToolCtx {
    pub fn new(agent_id: impl Into<String>, cwd: PathBuf) -> Self {
        Self {
            agent_id: agent_id.into(),
            cwd,
            policy: Arc::new(ToolPolicy::default()),
            cancel: CancelToken::new(),
            locks: FileLocks::default(),
            read_hashes: Arc::default(),
            shown: Arc::default(),
            repo_map: None,
            progress: None,
            alive: None,
        }
    }

    /// Tell the kernel this call is still working (cheap; rate-limited by the kernel).
    pub fn alive(&self) {
        if let Some(a) = &self.alive {
            a();
        }
    }

    pub fn with_policy(mut self, policy: Arc<ToolPolicy>) -> Self {
        self.policy = policy;
        self
    }

    /// Lexical resolution against the workspace (no policy checks). Prefer [`Self::resolve_checked`].
    pub fn resolve(&self, p: &str) -> PathBuf {
        let path = Path::new(p);
        let abs = if path.is_absolute() { path.to_path_buf() } else { self.cwd.join(path) };
        normalize(&abs)
    }

    /// Resolve a model-supplied path and enforce the sandbox: the real location (after following
    /// symlinks of every existing ancestor) must be inside the workspace; writes into `.git/` are
    /// refused.
    /// Reads may also go below a read root (`ToolPolicy::read_roots`), judged by the real path.
    pub fn resolve_checked(&self, p: &str, write: bool) -> Result<PathBuf> {
        match resolve_checked(&self.cwd, self.policy.sandbox, p, write) {
            Err(e) if !write => self.read_root(p).map(|(_, real)| real).ok_or(e),
            r => r,
        }
    }

    /// The read root holding `p` (an existing path, symlinks resolved) and `p`'s real path.
    pub fn read_root(&self, p: &str) -> Option<(PathBuf, PathBuf)> {
        let path = Path::new(p);
        if !path.is_absolute() {
            return None;
        }
        let real = normalize(path).canonicalize().ok()?;
        self.policy.read_roots.iter().filter_map(|r| r.canonicalize().ok()).find(|r| real.starts_with(r)).map(|r| (r, real))
    }

    pub fn relative(&self, abs: &Path) -> String {
        abs.strip_prefix(&self.cwd).unwrap_or(abs).to_string_lossy().into_owned()
    }
}

pub fn resolve_checked(cwd: &Path, sandbox: bool, p: &str, write: bool) -> Result<PathBuf> {
    let path = Path::new(p);
    let abs = normalize(&if path.is_absolute() { path.to_path_buf() } else { cwd.join(path) });
    if !sandbox {
        return Ok(abs);
    }
    let root = cwd.canonicalize().unwrap_or_else(|_| cwd.to_path_buf());
    // Canonicalize the deepest existing ancestor, then re-append the not-yet-existing tail.
    let mut existing = abs.clone();
    let mut tail = Vec::new();
    while !existing.exists() {
        match (existing.file_name().map(|s| s.to_os_string()), existing.parent()) {
            (Some(name), Some(parent)) => {
                tail.push(name);
                existing = parent.to_path_buf();
            }
            _ => break,
        }
    }
    let mut real = existing.canonicalize().unwrap_or(existing);
    for part in tail.iter().rev() {
        real.push(part);
    }
    if !real.starts_with(&root) {
        anyhow::bail!(
            "access denied: '{p}' is outside the workspace ({}). Only files inside the project can be accessed{}",
            root.display(),
            if write {
                "; don't work around this with shell commands (their /tmp is private and discarded): report that the \
                 path cannot be written and suggest one inside the workspace."
            } else {
                "."
            }
        );
    }
    if write && real.strip_prefix(&root).ok().and_then(|r| r.components().next()).is_some_and(|c| c.as_os_str() == ".git") {
        anyhow::bail!("access denied: writing inside .git/ is not allowed");
    }
    Ok(abs)
}

/// Lexically normalize `.` and `..` (works for files that don't exist yet).
pub fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

#[derive(Clone, Default)]
pub struct ToolRegistry {
    tools: BTreeMap<String, Arc<dyn Tool>>,
}

impl ToolRegistry {
    pub fn register(&mut self, t: Arc<dyn Tool>) {
        self.tools.insert(t.name().to_string(), t);
    }
    pub fn get(&self, name: &str) -> Option<&Arc<dyn Tool>> {
        self.tools.get(name)
    }
    pub fn names(&self) -> Vec<String> {
        self.tools.keys().cloned().collect()
    }
    /// Specs of the named tools, for an LLM request.
    pub fn specs(&self, names: &[&str]) -> Vec<ToolSpec> {
        self.tools.values().filter(|t| names.contains(&t.name())).map(|t| ToolSpec::new(t.name(), t.description(), t.schema())).collect()
    }

    /// What a call would touch, relative to the workspace (for the kernel's capability check).
    pub fn classify(&self, cwd: &Path, sandbox: bool, name: &str, args: &Value) -> Access {
        let Some(tool) = self.get(name) else {
            return Access::Invalid(format!("unknown tool '{name}'. Available: {}", self.names().join(", ")));
        };
        if tool.exec() {
            return Access::Exec;
        }
        match tool.target_path(args) {
            None => Access::Read,
            Some(p) => match resolve_checked(cwd, sandbox, &p, true) {
                Ok(abs) => Access::Write(abs.strip_prefix(cwd).unwrap_or(&abs).to_string_lossy().into_owned()),
                Err(e) => Access::Invalid(e.to_string()),
            },
        }
    }
}

/// Every built-in tool.
pub fn all_tools() -> ToolRegistry {
    let mut r = ToolRegistry::default();
    r.register(Arc::new(fs::ReadFile));
    r.register(Arc::new(fs::WriteFile));
    r.register(Arc::new(edit::EditFile));
    r.register(Arc::new(search::GlobTool));
    r.register(Arc::new(search::GrepTool));
    r.register(Arc::new(repomap_tool::RepoMapTool));
    r.register(Arc::new(repomap_tool::FindDefinition));
    r.register(Arc::new(git::GitTool));
    r.register(Arc::new(bash::Bash));
    r
}

/// Run a command to completion with a timeout and cancellation (stdout and stderr captured).
pub fn output_with_timeout(
    mut cmd: std::process::Command,
    timeout: std::time::Duration,
    cancel: &CancelToken,
    alive: &dyn Fn(),
) -> Result<std::process::Output> {
    use std::io::Read;
    use std::process::Stdio;
    let mut child = cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).spawn()?;
    let drain = |r: Option<Box<dyn Read + Send>>| {
        std::thread::spawn(move || {
            let mut buf = Vec::new();
            if let Some(mut r) = r {
                let _ = r.read_to_end(&mut buf);
            }
            buf
        })
    };
    let out = drain(child.stdout.take().map(|s| Box::new(s) as Box<dyn Read + Send>));
    let err = drain(child.stderr.take().map(|s| Box::new(s) as Box<dyn Read + Send>));
    let start = std::time::Instant::now();
    let status = loop {
        if let Some(s) = child.try_wait()? {
            break s;
        }
        alive();
        if cancel.is_cancelled() || start.elapsed() > timeout {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!(if cancel.is_cancelled() { "cancelled".to_string() } else { format!("timed out after {}s", timeout.as_secs()) });
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    };
    Ok(std::process::Output { status, stdout: out.join().unwrap_or_default(), stderr: err.join().unwrap_or_default() })
}

/// Keep the head and tail of long outputs.
pub fn truncate_output(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let head = s.floor_char_boundary(max * 2 / 3);
    let tail = s.ceil_char_boundary(s.len() - max / 3);
    format!("{}\n… [{} bytes truncated] …\n{}", &s[..head], tail - head, &s[tail..])
}

pub fn sha256(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    Sha256::digest(data).iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn arg_str<'a>(args: &'a Value, key: &str) -> Result<&'a str> {
    args.get(key).and_then(Value::as_str).ok_or_else(|| anyhow::anyhow!("missing string argument '{key}'"))
}
pub(crate) fn arg_opt_str<'a>(args: &'a Value, key: &str) -> Option<&'a str> {
    args.get(key).and_then(Value::as_str)
}
pub(crate) fn arg_opt_u64(args: &Value, key: &str) -> Option<u64> {
    args.get(key).and_then(|v| v.as_u64().or_else(|| v.as_str().and_then(|s| s.parse().ok())))
}
pub(crate) fn arg_bool(args: &Value, key: &str) -> bool {
    args.get(key).and_then(Value::as_bool).unwrap_or(false)
}

/// Today's date (UTC) as YYYY-MM-DD without a date crate (Howard Hinnant's civil_from_days).
pub fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let z = (secs / 86_400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{day:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn same_path_writes_are_serialized() {
        let locks = FileLocks::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let hs: Vec<_> = (0..3)
            .map(|i| {
                let (locks, log) = (locks.clone(), log.clone());
                std::thread::spawn(move || {
                    let _g = locks.lock(Path::new("/x"));
                    log.lock().unwrap().push(format!("start{i}"));
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    log.lock().unwrap().push(format!("end{i}"));
                })
            })
            .collect();
        for h in hs {
            h.join().unwrap();
        }
        let log = log.lock().unwrap();
        for pair in log.chunks(2) {
            assert_eq!(pair[0].replace("start", ""), pair[1].replace("end", ""));
        }
    }

    #[test]
    fn calls_are_classified_for_the_capability_check() {
        let dir = tempfile::tempdir().unwrap();
        let r = all_tools();
        let d = dir.path();
        assert_eq!(r.classify(d, true, "read_file", &json!({"path":"a"})), Access::Read);
        assert_eq!(r.classify(d, true, "write_file", &json!({"path":"./src/../src/a.rs","content":""})), Access::Write("src/a.rs".into()));
        assert_eq!(r.classify(d, true, "edit_file", &json!({"path":"b.rs"})), Access::Write("b.rs".into()));
        assert_eq!(r.classify(d, true, "bash", &json!({"command":"ls"})), Access::Exec);
        assert!(matches!(r.classify(d, true, "write_file", &json!({"path":"../x"})), Access::Invalid(e) if e.contains("outside")));
        assert!(matches!(r.classify(d, true, "teleport", &json!({})), Access::Invalid(e) if e.contains("unknown tool")));
    }
}

/// A file's content hash and the line ranges of it an agent has been shown.
pub type ShownRanges = (String, Vec<(usize, usize)>);
/// One process's shown memory (see `ToolCtx::shown`).
pub type Shown = Arc<Mutex<HashMap<PathBuf, ShownRanges>>>;

/// For a tool whose output depends only on its arguments and the files (grep, glob): a short note
/// when this exact call returned exactly this output before and that output is still in the agent's
/// context, so it is not sent twice. Shares `ToolCtx::shown` (under a key no real path can have), so it
/// is forgotten at the same moments: compaction, reset, a rolled-back turn.
pub fn repeat_note(ctx: &ToolCtx, tool: &str, args: &Value, output: &str) -> Option<String> {
    if output.len() < 200 {
        return None; // a note would not be shorter
    }
    let key = PathBuf::from(format!("\u{0}{tool}\u{0}{args}"));
    let hash = sha256(output.as_bytes());
    let mut shown = ctx.shown.lock().unwrap();
    if shown.get(&key).is_some_and(|(h, _)| *h == hash) {
        return Some(format!(
            "[same result as your earlier identical {tool} call, still above in this conversation: nothing changed. Use that output.]"
        ));
    }
    shown.insert(key, (hash, vec![]));
    None
}

/// Command output as the model should see it: progress noise out, everything else in. Removes ANSI
/// escapes and carriage-return redraws (progress bars), drops lines that are reliably noise (build
/// progress like `Compiling x`, passing-test lines like `test a ... ok`), counts what was dropped,
/// and collapses runs of identical lines. Errors, warnings, failures, summaries and anything not
/// recognised stay. Short outputs are returned as they are.
pub fn compress_output(s: &str) -> String {
    use std::sync::OnceLock;
    static ANSI: OnceLock<regex::Regex> = OnceLock::new();
    static NOISE: OnceLock<Vec<(&'static str, regex::Regex)>> = OnceLock::new();
    if s.len() < 2_000 {
        return s.to_string();
    }
    let ansi = ANSI.get_or_init(|| regex::Regex::new(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b\][^\x07]*\x07").expect("valid"));
    let noise = NOISE.get_or_init(|| {
        let r = |s: &str| regex::Regex::new(s).expect("valid");
        vec![
            ("build progress", r(r"^\s+(Compiling|Checking|Downloaded|Downloading|Fresh|Updating|Locking|Adding|Blocking|Documenting|Installing|Unpacking|Packaging|Verifying|Archiving|Removing)\s")),
            ("package downloads", r(r"^(Collecting|Downloading|Requirement already satisfied|Using cached|Obtaining)\b")),
            ("passing tests", r(r"^test \S.* \.\.\. (ok|ignored)$|^\S+::\S+ PASSED|^--- PASS: |^=== (RUN|PAUSE|CONT)\s|^\s+✓ |^ok \d+ ")),
            ("progress", r(r"^\s*\d{1,3}(\.\d+)?%\s*$|^\s*\[[=#>\- ]{5,}\]\s*(\d{1,3}%)?\s*$|^[\w/.\-]+\.py [.sx]+\s*(\[\s*\d+%\])?$")),
        ]
    });
    let mut dropped: std::collections::BTreeMap<&str, usize> = Default::default();
    let mut out: Vec<String> = Vec::new();
    let mut repeat = 0usize;
    for raw in s.lines() {
        // A progress bar redraws itself with \r: only its last state counts.
        let line = raw.rsplit('\r').next().unwrap_or(raw);
        let line = ansi.replace_all(line, "");
        if let Some((kind, _)) = noise.iter().find(|(_, re)| re.is_match(&line)) {
            *dropped.entry(kind).or_default() += 1;
            continue;
        }
        // Runs of blank lines collapse to one, silently.
        if line.trim().is_empty() && out.last().is_some_and(|l| l.trim().is_empty()) {
            continue;
        }
        if out.last().is_some_and(|l| *l == line && !line.trim().is_empty()) {
            repeat += 1;
            continue;
        }
        if repeat > 0 {
            out.push(format!("[previous line repeated {repeat} more times]"));
            repeat = 0;
        }
        out.push(line.into_owned());
    }
    if repeat > 0 {
        out.push(format!("[previous line repeated {repeat} more times]"));
    }
    if !dropped.is_empty() {
        let what: Vec<String> = dropped.iter().map(|(k, n)| format!("{n} {k}")).collect();
        out.push(format!("[omitted as noise: {} lines]", what.join(", ")));
    }
    out.join("\n")
}

#[cfg(test)]
mod compress_tests {
    use super::*;

    #[test]
    fn build_and_test_noise_goes_errors_and_summaries_stay() {
        let mut log = String::new();
        for i in 0..120 {
            log.push_str(&format!("   Compiling crate{i} v0.1.{i}\n"));
        }
        log.push_str("warning: unused variable: `x`\n --> src/lib.rs:3:9\n");
        for i in 0..200 {
            log.push_str(&format!("test tests::case_{i} ... ok\n"));
        }
        log.push_str("test tests::broken ... FAILED\n\nfailures:\n\n---- tests::broken stdout ----\nthread 'main' panicked at src/lib.rs:9:5\n\n\n\n");
        log.push_str("\x1b[32mdownloading\x1b[0m\r 10%\r 55%\r100% done\n");
        log.push_str("test result: FAILED. 200 passed; 1 failed\n");
        let out = compress_output(&log);
        for keep in [
            "warning: unused variable",
            "--> src/lib.rs:3:9",
            "test tests::broken ... FAILED",
            "panicked at src/lib.rs:9:5",
            "test result: FAILED. 200 passed; 1 failed",
            "100% done",
        ] {
            assert!(out.contains(keep), "kept: {keep}\n{out}");
        }
        assert!(!out.contains("Compiling crate7") && !out.contains("case_42 ... ok") && !out.contains("\x1b["));
        assert!(out.contains("[omitted as noise: 120 build progress, 200 passing tests lines]"), "{out}");
        assert!(!out.contains("\n\n\n"), "blank runs collapse");
        assert!(out.len() < log.len() / 5, "{} of {}", out.len(), log.len());
    }

    #[test]
    fn short_output_is_untouched_and_repeats_collapse() {
        assert_eq!(compress_output("   Compiling x v1\nok\n"), "   Compiling x v1\nok\n");
        let long = "retrying connection\n".repeat(300);
        let out = compress_output(&long);
        assert_eq!(out, "retrying connection\n[previous line repeated 299 more times]");
    }
}
