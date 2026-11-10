//! Read-only git: status, diff, log, show. Gives agents without `bash` (the master, the spec
//! lane, explorers) a view of what changed.
//!
//! The command line is built here from a fixed set of options, never from free-form arguments
//! (`git diff --output=FILE` would write files). Repository config cannot run programs either:
//! external diff drivers, textconv filters, fsmonitor hooks and pagers are switched off, and
//! `status` takes no index lock (GIT_OPTIONAL_LOCKS=0), so nothing in `.git` is written.

use super::{arg_bool, arg_opt_str, arg_opt_u64, arg_str, truncate_output, Tool, ToolCtx};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::time::Duration;

const MAX_OUTPUT: usize = 30_000;
const TIMEOUT: Duration = Duration::from_secs(30);

pub struct GitTool;

/// The git argument list for a request (without the safety `-c` options).
fn build_args(args: &Value, path: Option<&str>) -> Result<Vec<String>> {
    let command = arg_str(args, "command")?;
    let rev = arg_opt_str(args, "rev").filter(|r| !r.trim().is_empty());
    if let Some(r) = rev {
        // A revision is a name or range, never an option.
        if r.starts_with('-') || r.chars().any(char::is_whitespace) {
            bail!("invalid rev {r:?}");
        }
    }
    let mut a: Vec<String> = match command {
        "status" => vec!["status".into(), "--short".into(), "--branch".into()],
        "diff" => {
            let mut v = vec!["diff".into(), "--no-ext-diff".into(), "--no-textconv".into(), "--no-color".into()];
            if arg_bool(args, "staged") {
                v.push("--cached".into());
            }
            if arg_bool(args, "stat") {
                v.push("--stat".into());
            }
            v.extend(rev.map(str::to_string));
            v
        }
        "log" => {
            let n = arg_opt_u64(args, "limit").unwrap_or(20).clamp(1, 200);
            let mut v = vec!["log".into(), "--no-color".into(), "--date=short".into(), "--format=%h %ad %an  %s".into(), format!("-n{n}")];
            v.extend(rev.map(str::to_string));
            v
        }
        "show" => {
            let mut v = vec!["show".into(), "--no-ext-diff".into(), "--no-textconv".into(), "--no-color".into()];
            if arg_bool(args, "stat") {
                v.push("--stat".into());
            }
            v.push(rev.unwrap_or("HEAD").to_string());
            v
        }
        other => bail!("unknown git command {other:?} (status, diff, log, show)"),
    };
    if let Some(p) = path {
        a.push("--".into());
        a.push(p.to_string());
    }
    Ok(a)
}

impl Tool for GitTool {
    fn name(&self) -> &str {
        "git"
    }
    fn description(&self) -> String {
        "Read-only git. `status`: changed files and branch. `diff`: uncommitted changes (`staged` for the index, \
         `rev` to compare with a commit or range like `main..HEAD`, `stat` for a summary). `log`: recent commits \
         (`limit`, default 20). `show`: one commit (`rev`, default HEAD). `path` limits any of them to a file or directory."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "command":{"type":"string","enum":["status","diff","log","show"]},
            "path":{"type":"string","description":"Limit to this file or directory"},
            "rev":{"type":"string","description":"Commit, branch or range (diff, log, show)"},
            "staged":{"type":"boolean","default":false},
            "stat":{"type":"boolean","default":false,"description":"Summary of changed files instead of the patch"},
            "limit":{"type":"integer","description":"log: number of commits (default 20)"}
        },"required":["command"]})
    }
    fn summarize(&self, args: &Value) -> String {
        let cmd = args.get("command").and_then(Value::as_str).unwrap_or("?");
        let extra = [arg_opt_str(args, "rev"), arg_opt_str(args, "path")].into_iter().flatten().collect::<Vec<_>>().join(" ");
        if extra.is_empty() {
            cmd.to_string()
        } else {
            format!("{cmd} {extra}")
        }
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        // The path goes through the same checks as read_file (inside the workspace, not hidden).
        let path = match arg_opt_str(&args, "path").filter(|p| !p.trim().is_empty()) {
            Some(p) => Some(ctx.relative(&ctx.resolve_checked(p, false)?)),
            None => None,
        };
        let git_args = build_args(&args, path.as_deref())?;
        let mut cmd = std::process::Command::new("git");
        cmd.arg("--no-pager");
        for c in ["core.fsmonitor=false", "core.hooksPath=/dev/null", "core.pager=cat", "diff.external=", "color.ui=false"] {
            cmd.args(["-c", c]);
        }
        cmd.args(&git_args)
            .current_dir(&ctx.cwd)
            .env("GIT_OPTIONAL_LOCKS", "0")
            .env("GIT_TERMINAL_PROMPT", "0")
            .env_remove("GIT_EXTERNAL_DIFF")
            .stdin(std::process::Stdio::null());
        for (k, _) in std::env::vars() {
            if ctx.policy.is_secret_env(&k) {
                cmd.env_remove(k);
            }
        }
        let out = super::output_with_timeout(cmd, TIMEOUT, &ctx.cancel).context("could not run git (is it installed?)")?;
        let stdout = String::from_utf8_lossy(&out.stdout);
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            bail!("git {} failed: {}", git_args[0], err.trim());
        }
        if stdout.trim().is_empty() {
            return Ok(match git_args[0].as_str() {
                "diff" => "No differences.".into(),
                "log" => "No commits.".into(),
                _ => "(no output)".into(),
            });
        }
        Ok(truncate_output(&stdout, MAX_OUTPUT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arguments_are_built_from_fixed_options_only() {
        let a = build_args(&json!({"command":"diff","staged":true,"rev":"main..HEAD"}), Some("src/lib.rs")).unwrap();
        assert_eq!(a, ["diff", "--no-ext-diff", "--no-textconv", "--no-color", "--cached", "main..HEAD", "--", "src/lib.rs"]);
        let a = build_args(&json!({"command":"log","limit":5000}), None).unwrap();
        assert!(a.contains(&"-n200".to_string()), "limit is capped: {a:?}");
        assert_eq!(build_args(&json!({"command":"show"}), None).unwrap().last().unwrap(), "HEAD");
        // Options smuggled in as a revision, and commands outside the read-only set, are refused.
        assert!(build_args(&json!({"command":"diff","rev":"--output=/tmp/x"}), None).is_err());
        assert!(build_args(&json!({"command":"diff","rev":"HEAD --output=x"}), None).is_err());
        assert!(build_args(&json!({"command":"push"}), None).is_err());
        assert!(build_args(&json!({"command":"checkout"}), None).is_err());
    }

    fn sh(dir: &std::path::Path, cmd: &str) {
        let ok = std::process::Command::new("sh").arg("-c").arg(cmd).current_dir(dir).status().unwrap().success();
        assert!(ok, "{cmd}");
    }

    #[test]
    fn reads_a_real_repository_without_running_repo_configured_programs() {
        if std::process::Command::new("git").arg("--version").output().is_err() {
            eprintln!("git not installed; skipping");
            return;
        }
        let dir = tempfile::tempdir().unwrap();
        let d = dir.path();
        sh(d, "git init -q -b main && git config user.email t@t && git config user.name T && printf 'one\n' > a.txt && git add a.txt && git commit -qm first");
        // A hostile repository config: an external diff driver, a textconv filter and an
        // fsmonitor hook that would each leave a marker file if git ran them.
        sh(d, "printf '#!/bin/sh\ntouch \"$(dirname \"$0\")/RAN\"\n' > evil.sh && chmod +x evil.sh");
        sh(d, &format!("git config diff.external {0}/evil.sh && git config core.fsmonitor {0}/evil.sh && git config diff.evil.textconv {0}/evil.sh && echo 'a.txt diff=evil' > .gitattributes", d.display()));
        sh(d, "printf 'two\n' >> a.txt");
        let ctx = ToolCtx::new("master", d.to_path_buf());

        let status = GitTool.call(json!({"command":"status"}), &ctx).unwrap();
        assert!(status.contains("## main") && status.contains(" M a.txt"), "{status}");
        let diff = GitTool.call(json!({"command":"diff","path":"a.txt"}), &ctx).unwrap();
        assert!(diff.contains("+two"), "{diff}");
        let stat = GitTool.call(json!({"command":"diff","stat":true}), &ctx).unwrap();
        assert!(stat.contains("a.txt") && stat.contains("1 insertion"), "{stat}");
        let log = GitTool.call(json!({"command":"log"}), &ctx).unwrap();
        assert!(log.contains("first"), "{log}");
        let show = GitTool.call(json!({"command":"show","stat":true}), &ctx).unwrap();
        assert!(show.contains("first") && show.contains("a.txt"), "{show}");
        assert!(!d.join("RAN").exists(), "a repo-configured program was executed");
        // Outside the workspace is refused like read_file.
        assert!(GitTool.call(json!({"command":"diff","path":"/etc"}), &ctx).is_err());
    }
}
