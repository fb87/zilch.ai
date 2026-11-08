//! Editor integration over a tiny IPC: a local socket speaking newline-delimited JSON. Editors
//! (see `editor/nvim`) send a selection plus a comment, which is submitted to a lane exactly as if
//! typed at the prompt.
//!
//! Where the socket lives:
//! - Linux: an abstract socket (`@zilch-<uid>-<hash>`). It has no file on disk, so there is
//!   nothing to clean up or leave stale: the kernel drops the name when zilch exits, even on a crash.
//! - Other Unix (macOS): a socket file inside a private (0700, owner-checked) runtime directory.
//!
//! Either way every connection's peer credentials are checked and only the user running zilch is
//! served (abstract sockets carry no file permissions, so this check is the access control).
//! Editors discover the endpoint through `<workspace>/.zilch/ipc` (one line: `@name` or a path).
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

fn uid() -> u32 {
    unsafe { libc::getuid() }
}

fn workspace_hash(workspace: &Path) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    workspace.hash(&mut h);
    format!("{:016x}", h.finish())
}

/// The endpoint editors connect to for this workspace: `@name` for a Linux abstract socket,
/// otherwise a filesystem path.
pub fn endpoint(workspace: &Path) -> String {
    if cfg!(target_os = "linux") {
        format!("@zilch-{}-{}", uid(), workspace_hash(workspace))
    } else {
        socket_file(workspace).display().to_string()
    }
}

/// Non-Linux socket file: `<private dir>/<hash>.sock`, short enough for sockaddr_un (~104 bytes).
pub fn socket_file(workspace: &Path) -> PathBuf {
    const MAX: usize = 100;
    let name = format!("{}.sock", workspace_hash(workspace));
    let u = uid();
    [std::env::var_os("XDG_RUNTIME_DIR").map(|d| PathBuf::from(d).join("zilch")), Some(PathBuf::from(format!("/tmp/zilch-{u}")))]
        .into_iter()
        .flatten()
        .map(|d| d.join(&name))
        .find(|p| p.as_os_str().len() <= MAX)
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/zilch-{u}")).join(&name))
}

/// Create (or adopt) a directory only we can enter. It must be a real directory (not a symlink)
/// owned by us; it is forced to 0700. A directory planted by another user is refused.
#[cfg_attr(target_os = "linux", allow(dead_code))]
pub fn private_dir(dir: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    if let Some(parent) = dir.parent() {
        std::fs::create_dir_all(parent)?;
    }
    match std::fs::create_dir(dir) {
        Ok(()) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e),
    }
    let md = std::fs::symlink_metadata(dir)?;
    if !md.file_type().is_dir() || md.uid() != uid() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!("{} is not a directory owned by this user", dir.display()),
        ));
    }
    std::fs::set_permissions(dir, std::fs::Permissions::from_mode(0o700))
}

/// `<workspace>/.zilch/ipc`: one line with the endpoint, read by editor plugins.
pub fn pointer_path(workspace: &Path) -> PathBuf {
    workspace.join(".zilch").join("ipc")
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

/// Removes the pointer file (and the socket file, where there is one) when the REPL exits.
pub struct IpcGuard {
    pub endpoint: String,
    socket: Option<PathBuf>,
    pointer: PathBuf,
}

impl Drop for IpcGuard {
    fn drop(&mut self) {
        if let Some(s) = &self.socket {
            let _ = std::fs::remove_file(s);
        }
        // Only remove the pointer while it still names us.
        if std::fs::read_to_string(&self.pointer).is_ok_and(|p| p.trim() == self.endpoint) {
            let _ = std::fs::remove_file(&self.pointer);
        }
    }
}

type Bound = Option<(UnixListener, Option<PathBuf>)>;

#[cfg(target_os = "linux")]
fn bind(_workspace: &Path, endpoint: &str) -> std::io::Result<Bound> {
    use std::os::linux::net::SocketAddrExt;
    let addr = std::os::unix::net::SocketAddr::from_abstract_name(endpoint.trim_start_matches('@').as_bytes())?;
    match std::os::unix::net::UnixListener::bind_addr(&addr) {
        Ok(l) => {
            l.set_nonblocking(true)?;
            Ok(Some((UnixListener::from_std(l)?, None)))
        }
        // The name is held by a live process: another zilch serves this workspace.
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => Ok(None),
        Err(e) => Err(e),
    }
}

#[cfg(not(target_os = "linux"))]
fn bind(workspace: &Path, _endpoint: &str) -> std::io::Result<Bound> {
    use std::os::unix::fs::PermissionsExt;
    let path = socket_file(workspace);
    // The private directory closes the window between bind() and chmod(): nobody else can even
    // reach the socket file.
    private_dir(path.parent().expect("socket file has a parent"))?;
    if path.exists() {
        if std::os::unix::net::UnixStream::connect(&path).is_ok() {
            return Ok(None); // another zilch is serving this workspace
        }
        std::fs::remove_file(&path)?; // stale socket from a crash
    }
    let listener = UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    Ok(Some((listener, Some(path))))
}

/// Only the user running zilch may talk to it.
fn peer_allowed(stream: &UnixStream) -> bool {
    peer_is(stream, uid())
}

fn peer_is(stream: &UnixStream, expected: u32) -> bool {
    stream.peer_cred().is_ok_and(|c| c.uid() == expected)
}

/// Start listening. Returns `Ok(None)` if another live zilch already serves this workspace.
pub async fn serve(dispatcher: Arc<Dispatcher>, workspace: &Path) -> std::io::Result<Option<IpcGuard>> {
    let endpoint = endpoint(workspace);
    let Some((listener, socket)) = bind(workspace, &endpoint)? else {
        return Ok(None);
    };
    let pointer = pointer_path(workspace);
    if let Some(dir) = pointer.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let guard = IpcGuard { endpoint: endpoint.clone(), socket, pointer: pointer.clone() };
    std::fs::write(&pointer, format!("{endpoint}\n"))?;
    tokio::spawn(async move {
        while let Ok((mut stream, _)) = listener.accept().await {
            if !peer_allowed(&stream) {
                let _ = stream.write_all(b"{\"ok\":false,\"error\":\"permission denied\"}\n").await;
                continue;
            }
            let d = dispatcher.clone();
            tokio::spawn(async move {
                let _ = handle_conn(d, stream).await;
            });
        }
    });
    Ok(Some(guard))
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

    #[tokio::test]
    async fn one_listener_per_workspace_and_it_vanishes_on_drop() {
        let ws = tempfile::tempdir().unwrap();
        let ep = endpoint(ws.path());
        let (listener, socket) = bind(ws.path(), &ep).unwrap().expect("first bind");
        assert!(bind(ws.path(), &ep).unwrap().is_none(), "second zilch must not take over");
        if cfg!(target_os = "linux") {
            assert!(ep.starts_with(&format!("@zilch-{}-", uid())));
            assert!(socket.is_none(), "abstract socket: no file on disk");
        }
        drop(listener);
        if let Some(p) = &socket {
            let _ = std::fs::remove_file(p);
        }
        // Released (Linux: by the kernel, with no file to clean up) → bindable again.
        assert!(bind(ws.path(), &ep).unwrap().is_some());
    }

    #[tokio::test]
    async fn same_user_peer_is_allowed() {
        let ws = tempfile::tempdir().unwrap();
        let ep = endpoint(ws.path());
        let (listener, _socket) = bind(ws.path(), &ep).unwrap().unwrap();
        let addr = listener.local_addr().unwrap();
        let client = tokio::spawn(async move {
            #[cfg(target_os = "linux")]
            {
                use std::os::linux::net::SocketAddrExt;
                let name = addr.as_abstract_name().unwrap().to_vec();
                let a = std::os::unix::net::SocketAddr::from_abstract_name(name).unwrap();
                std::os::unix::net::UnixStream::connect_addr(&a).unwrap()
            }
            #[cfg(not(target_os = "linux"))]
            std::os::unix::net::UnixStream::connect(addr.as_pathname().unwrap()).unwrap()
        });
        let (stream, _) = listener.accept().await.unwrap();
        assert!(peer_allowed(&stream));
        assert!(!peer_is(&stream, uid().wrapping_add(1)), "another uid is refused");
        drop(client.await.unwrap());
    }

    #[test]
    fn private_dir_refuses_symlinks_and_tightens_mode() {
        use std::os::unix::fs::PermissionsExt;
        let t = tempfile::tempdir().unwrap();
        let d = t.path().join("a/b");
        private_dir(&d).unwrap();
        assert_eq!(std::fs::metadata(&d).unwrap().permissions().mode() & 0o777, 0o700);
        std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
        private_dir(&d).unwrap();
        assert_eq!(std::fs::metadata(&d).unwrap().permissions().mode() & 0o777, 0o700);
        let link = t.path().join("link");
        std::os::unix::fs::symlink(&d, &link).unwrap();
        assert!(private_dir(&link).is_err(), "a symlink (possibly planted) must be refused");
    }
}
