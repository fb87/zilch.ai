//! Editor integration over a tiny IPC: a Unix socket at `<workspace>/.zilch/zilch.sock` speaking
//! newline-delimited JSON. Editors (see `editor/nvim`) send a selection plus a comment, which is
//! submitted to a lane exactly as if typed at the prompt.
//!
//! Requests (one JSON object per line), each answered with one JSON line:
//!   {"type":"send", "text": "...", "lane": "code"|"spec"|null,
//!    "file": "src/x.rs", "start": 10, "end": 20, "selection": "..."}
//!     → {"ok": true, "lane": "code"} | {"ok": false, "error": "..."}
//!   {"type":"status"}
//!     → {"ok": true, "active": "code", "code": "idle", "spec": "working", "docs": "..."}

use crate::lanes::{Dispatcher, Lane};
use crate::render::{CYAN, DIM, RESET};
use serde::Deserialize;
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};

pub fn socket_path(workspace: &Path) -> PathBuf {
    workspace.join(".zilch").join("zilch.sock")
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum Request {
    Send {
        text: String,
        #[serde(default)]
        lane: Option<String>,
        #[serde(default)]
        file: Option<String>,
        #[serde(default)]
        start: Option<u64>,
        #[serde(default)]
        end: Option<u64>,
        #[serde(default)]
        selection: Option<String>,
    },
    Status,
}

/// The message the agent receives for an editor request: location, the selected code, then the
/// user's comment.
pub fn compose(text: &str, file: Option<&str>, start: Option<u64>, end: Option<u64>, selection: Option<&str>) -> String {
    let mut out = String::new();
    if let Some(f) = file {
        let range = match (start, end) {
            (Some(s), Some(e)) if s != e => format!(":{s}-{e}"),
            (Some(s), _) => format!(":{s}"),
            _ => String::new(),
        };
        out.push_str(&format!("[From the editor] {f}{range}\n"));
    }
    if let Some(sel) = selection.filter(|s| !s.trim().is_empty()) {
        let lang = file.and_then(|f| Path::new(f).extension()).map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        out.push_str(&format!("```{lang}\n{}\n```\n", sel.trim_end_matches('\n')));
    }
    out.push_str(text.trim());
    out
}

/// Removes the socket file when the REPL exits.
pub struct IpcGuard(PathBuf);

impl Drop for IpcGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Start listening. Returns `Ok(None)` if another live zilch already owns this workspace's socket.
pub async fn serve(dispatcher: Arc<Dispatcher>, workspace: &Path) -> std::io::Result<Option<IpcGuard>> {
    let path = socket_path(workspace);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if path.exists() {
        if UnixStream::connect(&path).await.is_ok() {
            return Ok(None); // another zilch is serving this workspace
        }
        std::fs::remove_file(&path)?; // stale socket from a crash
    }
    let listener = UnixListener::bind(&path)?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    }
    tokio::spawn(async move {
        while let Ok((stream, _)) = listener.accept().await {
            let d = dispatcher.clone();
            tokio::spawn(async move {
                let _ = handle_conn(d, stream).await;
            });
        }
    });
    Ok(Some(IpcGuard(path)))
}

async fn handle_conn(d: Arc<Dispatcher>, stream: UnixStream) -> std::io::Result<()> {
    let (r, mut w) = stream.into_split();
    let mut lines = BufReader::new(r).lines();
    while let Some(line) = lines.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Request>(&line) {
            Ok(req) => handle(&d, req).await,
            Err(e) => json!({"ok": false, "error": format!("bad request: {e}")}),
        };
        w.write_all(format!("{reply}\n").as_bytes()).await?;
    }
    Ok(())
}

async fn handle(d: &Arc<Dispatcher>, req: Request) -> Value {
    match req {
        Request::Status => {
            let state = |l: Lane| if d.busy(l) { "working" } else { "idle" };
            json!({
                "ok": true,
                "active": d.active().name(),
                "code": state(Lane::Code),
                "spec": state(Lane::Spec),
                "docs": d.team.spec_status().describe(&d.team.shared.spec_store.dir),
            })
        }
        Request::Send { text, lane, file, start, end, selection } => {
            if text.trim().is_empty() && selection.as_deref().is_none_or(|s| s.trim().is_empty()) {
                return json!({"ok": false, "error": "empty message"});
            }
            let lane = match lane.as_deref() {
                None | Some("") => d.active(),
                Some(l) => match Lane::parse(l) {
                    Some(l) => l,
                    None => return json!({"ok": false, "error": format!("unknown lane '{l}' (code | spec)")}),
                },
            };
            let message = compose(&text, file.as_deref(), start, end, selection.as_deref());
            // Echo what arrived so the REPL shows the conversation's full context.
            {
                let mut s = d.screen.lock().await;
                let loc =
                    file.as_deref().map(|f| format!(" {f}{}", start.map(|s| format!(":{s}")).unwrap_or_default())).unwrap_or_default();
                let first = text.lines().next().unwrap_or("").chars().take(100).collect::<String>();
                let msg = format!("{}◆ editor → {} lane{}:{} {first}", s.c(CYAN), lane.name(), loc, s.c(RESET));
                s.line(&msg);
                if lane != d.active() {
                    let note = format!("{}  (running in the background; /{} to follow it){}", s.c(DIM), lane.name(), s.c(RESET));
                    s.line(&note);
                }
            }
            match d.submit(lane, message) {
                Ok(()) => json!({"ok": true, "lane": lane.name()}),
                Err(e) => json!({"ok": false, "error": e}),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn composes_location_selection_and_comment() {
        let m =
            compose("Why is this rounding here?", Some("ledger/currency.py"), Some(14), Some(16), Some("    usd = amount / RATES[src]\n"));
        assert_eq!(m, "[From the editor] ledger/currency.py:14-16\n```py\n    usd = amount / RATES[src]\n```\nWhy is this rounding here?");
        assert_eq!(compose("hi", None, None, None, None), "hi");
        assert_eq!(compose("x", Some("a.rs"), Some(3), Some(3), Some("  ")), "[From the editor] a.rs:3\nx");
    }

    #[test]
    fn parses_requests() {
        let r: Request =
            serde_json::from_str(r#"{"type":"send","text":"t","lane":"spec","file":"f","start":1,"end":2,"selection":"s"}"#).unwrap();
        assert!(matches!(r, Request::Send { lane: Some(ref l), start: Some(1), .. } if l == "spec"));
        assert!(matches!(serde_json::from_str::<Request>(r#"{"type":"status"}"#).unwrap(), Request::Status));
        assert!(serde_json::from_str::<Request>(r#"{"type":"nope"}"#).is_err());
    }
}
