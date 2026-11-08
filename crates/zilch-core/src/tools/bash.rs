use super::sandbox::{bwrap_args, bwrap_available, BashSandbox};
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
        let sandboxed = match ctx.policy.bash_sandbox {
            BashSandbox::Off => false,
            BashSandbox::Auto => bwrap_available(),
            BashSandbox::Bwrap => {
                if !bwrap_available() {
                    anyhow::bail!("bash sandbox is required (bash_sandbox = \"bwrap\") but bubblewrap is not available or not permitted on this machine");
                }
                true
            }
        };
        let mut command = if sandboxed {
            let workspace = ctx.cwd.canonicalize().unwrap_or_else(|_| ctx.cwd.clone());
            let mut c = tokio::process::Command::new("bwrap");
            c.args(bwrap_args(&workspace, &ctx.policy.bash)).arg("--").arg("bash");
            c
        } else {
            tokio::process::Command::new("bash")
        };
        command
            .arg("-c")
            // `exec 2>&1` merges stderr for EVERY command; a trailing `2>&1` only covered the last
            // one, silently dropping e.g. compiler errors in `cargo build; echo done`.
            .arg(format!("exec 2>&1\n{cmd}"))
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
    async fn stderr_of_every_command_is_captured() {
        // Regression: a trailing `2>&1` only applied to the last command of a list.
        let dir = tempfile::tempdir().unwrap();
        let c =
            ctx(dir.path()).with_policy(Arc::new(crate::tools::ToolPolicy { bash_sandbox: BashSandbox::Off, ..Default::default() }), None);
        let out = Bash.call(json!({"command": "echo first-error >&2; ls /definitely/missing; echo done"}), &c).await.unwrap();
        assert!(out.contains("first-error") && out.contains("No such file") && out.contains("done"), "{out}");
    }

    #[tokio::test]
    async fn timeout_kills_the_group() {
        let dir = tempfile::tempdir().unwrap();
        let t = Instant::now();
        let out = Bash.call(json!({"command": "sleep 30 & sleep 30", "timeout_secs": 1}), &ctx(dir.path())).await.unwrap();
        assert!(out.contains("timed out after 1s"), "{out}");
        assert!(t.elapsed() < Duration::from_secs(4));
    }

    fn sandboxed_ctx(dir: &std::path::Path, network: bool, hide: Vec<String>) -> ToolCtx {
        let policy = crate::tools::ToolPolicy {
            bash_sandbox: BashSandbox::Bwrap,
            bash: crate::tools::sandbox::SandboxSpec { network, hide, writable: vec![], overlay: None },
            ..Default::default()
        };
        ctx(dir).with_policy(Arc::new(policy), None)
    }

    #[tokio::test]
    async fn bwrap_confines_commands() {
        if !bwrap_available() {
            eprintln!("skipping: bubblewrap not available");
            return;
        }
        let outer = tempfile::tempdir().unwrap();
        let ws = outer.path().join("ws");
        let vault = outer.path().join("vault");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&vault).unwrap();
        std::fs::write(vault.join("id_rsa"), "PRIVATE-KEY-DATA").unwrap();
        std::fs::write(outer.path().join("notes.txt"), "readable").unwrap(); // on host /tmp
        let c = sandboxed_ctx(&ws, false, vec![vault.display().to_string()]);
        let run = |cmd: String| {
            let c = &c;
            async move { Bash.call(json!({"command": cmd}), c).await.unwrap() }
        };

        // Hidden secrets are invisible; other host files stay readable (toolchains, sources).
        let out = run(format!("cat {}/id_rsa; ls -A {} | wc -l", vault.display(), vault.display())).await;
        assert!(!out.contains("PRIVATE-KEY-DATA") && out.lines().any(|l| l.trim() == "0"), "{out}");
        assert!(out.contains("No such file"), "stderr of earlier commands must be captured: {out}");
        // Host files outside /tmp stay readable (toolchains, sources); host /tmp is private.
        let manifest = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
        assert!(run(format!("cat {manifest}")).await.contains("zilch-core"));
        let out = run(format!("cat {}/notes.txt", outer.path().display())).await;
        assert!(!out.contains("readable"), "host /tmp must not be visible: {out}");

        // Host filesystem is read-only; nothing written outside the workspace reaches the host.
        let probe = concat!(env!("CARGO_MANIFEST_DIR"), "/.zilch-sandbox-probe");
        let out = run(format!("touch {probe}; touch {}/evil; echo done", outer.path().display())).await;
        let leaked = std::path::Path::new(probe).exists();
        let _ = std::fs::remove_file(probe);
        assert!(out.contains("Read-only file system") && !leaked, "{out}");
        assert!(!outer.path().join("evil").exists(), "write escaped to host /tmp");
        let out = run("echo hi > made.txt && f=$(mktemp) && echo ok > $f && cat $f".into()).await;
        assert!(out.contains("ok") && ws.join("made.txt").exists(), "{out}");

        // No network: only loopback in the namespace.
        let out = run("tail -n +3 /proc/net/dev | grep -v ' lo:' | wc -l".into()).await;
        assert!(out.trim_start().starts_with('0'), "network interfaces visible: {out}");

        // Own PID namespace: host processes are invisible.
        let out = run("ls /proc | grep -c '^[0-9]'".into()).await;
        let n: usize = out.lines().next().unwrap().trim().parse().unwrap();
        assert!(n < 10, "saw {n} processes: {out}");
    }

    #[tokio::test]
    async fn bwrap_network_can_be_enabled_and_children_are_killed() {
        if !bwrap_available() {
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let c = sandboxed_ctx(dir.path(), true, vec![]);
        let out = Bash.call(json!({"command": "tail -n +3 /proc/net/dev | wc -l"}), &c).await.unwrap();
        assert!(!out.trim_start().starts_with('0'), "{out}");
        let marker = dir.path().join("alive");
        let out = Bash.call(json!({"command": format!("(sleep 2; touch {}) & echo bg", marker.display())}), &c).await.unwrap();
        assert!(out.contains("bg"));
        tokio::time::sleep(Duration::from_millis(2500)).await;
        assert!(!marker.exists(), "sandboxed background child outlived the call");
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
