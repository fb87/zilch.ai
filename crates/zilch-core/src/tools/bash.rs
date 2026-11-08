use super::{arg_opt_u64, arg_str, truncate_output, Tool, ToolCtx};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;

pub struct Bash;

/// Kills the command's whole process group when dropped (timeout, cancellation, or panic),
/// so background children (`npm run dev &`, test runners) never outlive the tool call.
struct GroupGuard {
    pgid: Option<i32>,
}

impl Drop for GroupGuard {
    fn drop(&mut self) {
        #[cfg(unix)]
        if let Some(pgid) = self.pgid {
            // SAFETY: plain syscall on a process group we created; errors (already gone) are ignored.
            unsafe {
                libc::killpg(pgid, libc::SIGKILL);
            }
        }
    }
}

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> String {
        "Run a shell command (bash -c) in the working directory. Returns combined stdout/stderr and exit code. \
         Default timeout 120s, max 600s. Non-interactive only. Secrets (API keys, tokens) are not available \
         in the environment."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "command":{"type":"string"},
            "timeout_secs":{"type":"integer"}
        },"required":["command"]})
    }
    fn mutating(&self) -> bool {
        true
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("command").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let cmd = arg_str(&args, "command")?;
        let timeout = Duration::from_secs(arg_opt_u64(&args, "timeout_secs").unwrap_or(120).clamp(1, 600));
        let mut command = tokio::process::Command::new("bash");
        command
            .arg("-c")
            .arg(format!("{cmd} 2>&1"))
            .current_dir(&ctx.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);
        // Never hand provider keys or other secrets to commands the model writes.
        for (name, _) in std::env::vars_os() {
            if let Some(n) = name.to_str() {
                if ctx.policy.is_secret_env(n) {
                    command.env_remove(n);
                }
            }
        }
        #[cfg(unix)]
        command.process_group(0);
        let mut child = command.spawn()?;
        let guard = GroupGuard { pgid: child.id().map(|p| p as i32) };

        // Read output concurrently so a chatty command never blocks on a full pipe.
        let mut stdout = child.stdout.take().expect("stdout");
        let reader = tokio::spawn(async move {
            let mut buf = Vec::new();
            let _ = stdout.read_to_end(&mut buf).await;
            buf
        });
        let status = tokio::time::timeout(timeout, child.wait()).await;
        // Shell finished (or timed out): kill the whole group so background children that still
        // hold the pipe open can't keep us waiting or outlive the call.
        drop(guard);
        let buf = tokio::time::timeout(Duration::from_secs(5), reader).await.ok().and_then(Result::ok).unwrap_or_default();
        let out = truncate_output(&String::from_utf8_lossy(&buf), 30_000);
        Ok(match status {
            Ok(Ok(s)) => format!("{out}\n[exit code: {}]", s.code().map_or("signal".into(), |c| c.to_string())),
            Ok(Err(e)) => format!("{out}\n[error: {e}]"),
            Err(_) => format!("{out}\n[timed out after {}s; process group killed]", timeout.as_secs()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
    use crate::team::file_locks::FileLocks;
    use crate::tools::CancelCell;
    use std::sync::Arc;
    use std::time::Instant;

    fn ctx(dir: &std::path::Path) -> ToolCtx {
        let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
        ToolCtx::new("t", dir.to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default())
    }

    #[tokio::test]
    async fn secrets_are_scrubbed_from_the_environment() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("ZILCH_TEST_API_KEY", "sk-secret-123");
        std::env::set_var("ZILCH_TEST_PLAIN", "visible");
        let out =
            Bash.call(json!({"command": "echo key=[$ZILCH_TEST_API_KEY] plain=[$ZILCH_TEST_PLAIN]"}), &ctx(dir.path())).await.unwrap();
        assert!(out.contains("key=[]") && out.contains("plain=[visible]"), "{out}");
    }

    #[tokio::test]
    async fn background_children_do_not_block_or_survive() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("still_alive");
        let t = Instant::now();
        // A background child holding stdout open used to block until the timeout.
        let cmd = format!("(sleep 3; touch {}) & echo started", marker.display());
        let out = Bash.call(json!({"command": cmd}), &ctx(dir.path())).await.unwrap();
        assert!(out.contains("started") && out.contains("exit code: 0"), "{out}");
        assert!(t.elapsed() < Duration::from_secs(2), "blocked for {:?}", t.elapsed());
        tokio::time::sleep(Duration::from_millis(3500)).await;
        assert!(!marker.exists(), "background child outlived the tool call");
    }

    #[tokio::test]
    async fn timeout_kills_the_group() {
        let dir = tempfile::tempdir().unwrap();
        let t = Instant::now();
        let out = Bash.call(json!({"command": "sleep 30 & sleep 30", "timeout_secs": 1}), &ctx(dir.path())).await.unwrap();
        assert!(out.contains("timed out after 1s"), "{out}");
        assert!(t.elapsed() < Duration::from_secs(4));
    }

    #[test]
    fn secret_heuristic() {
        let p = crate::tools::ToolPolicy { env_passthrough: vec!["GITHUB_TOKEN".into()], ..Default::default() };
        for s in ["OPENCODE_API_KEY", "OPENROUTER_API_KEY", "SLACK_TOKEN", "DB_PASSWORD", "AWS_SECRET_ACCESS_KEY"] {
            assert!(p.is_secret_env(s), "{s}");
        }
        for s in ["PATH", "HOME", "GITHUB_TOKEN", "TOKENIZERS_PARALLELISM"] {
            assert!(!p.is_secret_env(s), "{s}");
        }
    }
}
