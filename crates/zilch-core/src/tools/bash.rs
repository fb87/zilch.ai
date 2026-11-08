use super::{arg_opt_u64, arg_str, truncate_output, Tool, ToolCtx};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::{json, Value};
use std::process::Stdio;
use std::time::Duration;
use tokio::io::AsyncReadExt;

pub struct Bash;

#[async_trait]
impl Tool for Bash {
    fn name(&self) -> &str {
        "bash"
    }
    fn description(&self) -> String {
        "Run a shell command (bash -c) in the working directory. Returns combined stdout/stderr and exit code. \
         Default timeout 120s, max 600s. Non-interactive only."
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
        let mut child = tokio::process::Command::new("bash")
            .arg("-c")
            .arg(format!("{cmd} 2>&1"))
            .current_dir(&ctx.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true)
            .spawn()?;
        let mut stdout = child.stdout.take().expect("stdout");
        let mut buf = Vec::new();
        let res = tokio::time::timeout(timeout, async {
            stdout.read_to_end(&mut buf).await?;
            child.wait().await
        })
        .await;
        let text = String::from_utf8_lossy(&buf);
        let out = truncate_output(&text, 30_000);
        Ok(match res {
            Ok(Ok(status)) => format!("{out}\n[exit code: {}]", status.code().map_or("signal".into(), |c| c.to_string())),
            Ok(Err(e)) => format!("{out}\n[error: {e}]"),
            Err(_) => format!("{out}\n[timed out after {}s; process killed]", timeout.as_secs()),
        })
    }
}
