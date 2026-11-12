//! `bash`: run a command in the workspace, inside zilch's OS sandbox when available.
//! Same behaviour as zilch's async tool, on `std::process`: own process group (killed on finish,
//! timeout or cancel, so background children never outlive the call), stderr merged, secrets
//! removed from the environment, output capped.

#[cfg(test)]
use super::sandbox::native_available;
use super::sandbox::{self, BashSandbox};
use super::{arg_opt_u64, arg_str, compress_output, truncate_output, Tool, ToolCtx};
use anyhow::Result;
use serde_json::{json, Value};
use std::io::Read;
use std::process::Stdio;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

pub struct Bash;

/// Kills the command's whole process group when dropped (timeout, cancellation, or panic).
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
    fn exec(&self) -> bool {
        true
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("command").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let cmd = arg_str(&args, "command")?;
        let timeout = Duration::from_secs(arg_opt_u64(&args, "timeout_secs").unwrap_or(120).clamp(1, 600));
        let sandboxed = match ctx.policy.bash_sandbox {
            BashSandbox::Off => false,
            BashSandbox::Auto => sandbox::available(),
            BashSandbox::Required => {
                if !sandbox::available() {
                    anyhow::bail!("bash sandbox is required (bash_sandbox = \"required\") but no OS sandbox works on this machine");
                }
                true
            }
        };
        let mut command = if sandboxed {
            let workspace = ctx.cwd.canonicalize().unwrap_or_else(|_| ctx.cwd.clone());
            let (program, args) = sandbox::wrapper(&workspace, &ctx.policy.bash);
            let mut c = std::process::Command::new(program);
            c.args(args).arg("bash");
            c
        } else {
            std::process::Command::new("bash")
        };
        // `exec 2>&1` merges stderr for EVERY command of the script, not just the last one.
        command
            .arg("-c")
            .arg(format!("exec 2>&1\n{cmd}"))
            .current_dir(&ctx.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null());
        for (name, _) in std::env::vars_os() {
            if let Some(n) = name.to_str() {
                if ctx.policy.is_secret_env(n) {
                    command.env_remove(n);
                }
            }
        }
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        let mut child = command.spawn()?;
        let guard = GroupGuard { pgid: Some(child.id() as i32) };

        // The reader appends into a shared buffer, so we never depend on the pipe closing (a
        // background child can keep it open even after its group was signalled).
        let mut stdout = child.stdout.take().expect("stdout");
        let collected = Arc::new(Mutex::new(Vec::<u8>::new()));
        let sink = collected.clone();
        let progress = ctx.progress.clone();
        std::thread::spawn(move || {
            let mut chunk = [0u8; 8192];
            let mut last_emit = Instant::now();
            loop {
                let n = match stdout.read(&mut chunk) {
                    Ok(0) | Err(_) => break,
                    Ok(n) => n,
                };
                let line = {
                    let mut buf = sink.lock().unwrap();
                    buf.extend_from_slice(&chunk[..n]);
                    // Live progress: latest non-empty output line, at most ~4×/s.
                    (last_emit.elapsed() >= Duration::from_millis(250)).then(|| {
                        last_emit = Instant::now();
                        let tail = &buf[buf.len().saturating_sub(4096)..];
                        String::from_utf8_lossy(tail)
                            .lines()
                            .rev()
                            .map(str::trim)
                            .find(|l| !l.is_empty())
                            .map(|l| l.chars().take(120).collect::<String>())
                    })
                };
                if let (Some(Some(line)), Some(p)) = (line, &progress) {
                    p(line);
                }
            }
        });
        let start = Instant::now();
        let outcome = loop {
            match child.try_wait() {
                Ok(Some(s)) => break Ok(s),
                Err(e) => break Err(format!("error: {e}")),
                Ok(None) => {}
            }
            if ctx.cancel.is_cancelled() {
                break Err("cancelled; process group killed".into());
            }
            ctx.alive();
            if start.elapsed() > timeout {
                break Err(format!("timed out after {}s; process group killed", timeout.as_secs()));
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        // Kill the whole group so background children can't outlive the call, reap the shell,
        // then give the reader a short grace period to drain what is left.
        drop(guard);
        let _ = child.kill();
        let _ = child.wait();
        std::thread::sleep(Duration::from_millis(20));
        let buf = std::mem::take(&mut *collected.lock().unwrap());
        let out = truncate_output(&compress_output(&String::from_utf8_lossy(&buf)), 30_000);
        Ok(match outcome {
            Ok(s) => format!("{out}\n[exit code: {}]", s.code().map_or("signal".into(), |c| c.to_string())),
            Err(e) => format!("{out}\n[{e}]"),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::ToolPolicy;

    fn ctx(dir: &std::path::Path) -> ToolCtx {
        ToolCtx::new("t", dir.to_path_buf())
    }

    #[test]
    fn secrets_are_scrubbed_from_the_environment() {
        let dir = tempfile::tempdir().unwrap();
        std::env::set_var("ZILCH_TEST_API_KEY", "sk-secret-123");
        std::env::set_var("ZILCH_TEST_PLAIN", "visible");
        let out = Bash.call(json!({"command": "echo key=[$ZILCH_TEST_API_KEY] plain=[$ZILCH_TEST_PLAIN]"}), &ctx(dir.path())).unwrap();
        assert!(out.contains("key=[]") && out.contains("plain=[visible]"), "{out}");
    }

    #[test]
    fn background_children_do_not_block_or_survive() {
        let dir = tempfile::tempdir().unwrap();
        let marker = dir.path().join("still_alive");
        let _ = sandbox::available();
        let t = Instant::now();
        let cmd = format!("(sleep 3; touch {}) & echo started", marker.display());
        let out = Bash.call(json!({"command": cmd}), &ctx(dir.path())).unwrap();
        assert!(out.contains("started") && out.contains("exit code: 0"), "{out}");
        assert!(t.elapsed() < Duration::from_secs(2), "blocked for {:?}; output: {out}", t.elapsed());
        std::thread::sleep(Duration::from_millis(3500));
        assert!(!marker.exists(), "background child outlived the tool call");
    }

    #[test]
    fn stderr_of_every_command_is_captured() {
        let dir = tempfile::tempdir().unwrap();
        let c = ctx(dir.path()).with_policy(Arc::new(ToolPolicy { bash_sandbox: BashSandbox::Off, ..Default::default() }));
        let out = Bash.call(json!({"command": "echo first-error >&2; ls /definitely/missing; echo done"}), &c).unwrap();
        assert!(out.contains("first-error") && out.contains("No such file") && out.contains("done"), "{out}");
    }

    #[test]
    fn timeout_and_cancel_kill_the_group() {
        let dir = tempfile::tempdir().unwrap();
        let t = Instant::now();
        let out = Bash.call(json!({"command": "sleep 30 & sleep 30", "timeout_secs": 1}), &ctx(dir.path())).unwrap();
        assert!(out.contains("timed out after 1s"), "{out}");
        assert!(t.elapsed() < Duration::from_secs(4));

        let c = ctx(dir.path());
        let cancel = c.cancel.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(300));
            cancel.cancel();
        });
        let t = Instant::now();
        let out = Bash.call(json!({"command": "echo begun; sleep 30"}), &c).unwrap();
        assert!(out.contains("begun") && out.contains("cancelled"), "{out}");
        assert!(t.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn running_commands_report_progress() {
        let dir = tempfile::tempdir().unwrap();
        let lines = Arc::new(Mutex::new(Vec::new()));
        let sink = lines.clone();
        let mut c = ctx(dir.path());
        c.progress = Some(Arc::new(move |l| sink.lock().unwrap().push(l)));
        let out = Bash.call(json!({"command": "for i in 1 2 3 4; do echo step$i; sleep 0.3; done"}), &c).unwrap();
        assert!(out.contains("step4"));
        let lines = lines.lock().unwrap();
        assert!(lines.iter().any(|l| l.starts_with("step")), "{lines:?}");
    }

    #[test]
    fn sandbox_confines_commands() {
        if !native_available() {
            eprintln!("skipping: native sandbox not available");
            return;
        }
        let outer = tempfile::tempdir().unwrap();
        let ws = outer.path().join("ws");
        let vault = outer.path().join("vault");
        std::fs::create_dir_all(&ws).unwrap();
        std::fs::create_dir_all(&vault).unwrap();
        std::fs::write(vault.join("id_rsa"), "PRIVATE-KEY-DATA").unwrap();
        let policy = ToolPolicy {
            bash_sandbox: BashSandbox::Required,
            bash: sandbox::SandboxSpec {
                network: false,
                hide: vec![vault.display().to_string()],
                writable: vec![],
                overlay: None,
                protect_git: true,
            },
            ..Default::default()
        };
        let c = ctx(&ws).with_policy(Arc::new(policy));
        let run = |cmd: String| Bash.call(json!({"command": cmd}), &c).unwrap();
        let out = run(format!("cat {}/id_rsa; ls -A {} | wc -l", vault.display(), vault.display()));
        assert!(!out.contains("PRIVATE-KEY-DATA") && out.lines().any(|l| l.trim() == "0"), "{out}");
        let probe = concat!(env!("CARGO_MANIFEST_DIR"), "/.zilch-sandbox-probe");
        let out = run(format!("touch {probe}; touch {}/evil; echo done", outer.path().display()));
        let leaked = std::path::Path::new(probe).exists();
        let _ = std::fs::remove_file(probe);
        assert!(out.contains("Read-only file system") && !leaked, "{out}");
        assert!(!outer.path().join("evil").exists(), "write escaped to host /tmp");
        let out = run("echo hi > made.txt && echo ok".into());
        assert!(out.contains("ok") && ws.join("made.txt").exists(), "{out}");
        let out = run("tail -n +3 /proc/net/dev | grep -v ' lo:' | wc -l".into());
        assert!(out.trim_start().starts_with('0'), "network interfaces visible: {out}");
    }
}
