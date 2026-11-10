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

use serde::Deserialize;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::Arc;

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
    match UnixListener::bind_addr(&addr) {
        Ok(l) => Ok(Some((l, None))),
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
        if UnixStream::connect(&path).is_ok() {
            return Ok(None); // another zilch is serving this workspace
        }
        std::fs::remove_file(&path)?; // stale socket from a crash
    }
    let listener = UnixListener::bind(&path)?;
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600))?;
    Ok(Some((listener, Some(path))))
}

/// Only the user running zilch-os may talk to it (peer credentials, not file permissions).
fn peer_allowed(stream: &UnixStream) -> bool {
    peer_uid(stream) == Some(uid())
}

#[cfg(target_os = "linux")]
fn peer_uid(stream: &UnixStream) -> Option<u32> {
    use std::os::unix::io::AsRawFd;
    let mut cred = libc::ucred { pid: 0, uid: 0, gid: 0 };
    let mut len = std::mem::size_of::<libc::ucred>() as libc::socklen_t;
    // SAFETY: getsockopt fills `cred` (of the size we pass) for a valid socket descriptor.
    let r = unsafe {
        libc::getsockopt(stream.as_raw_fd(), libc::SOL_SOCKET, libc::SO_PEERCRED, &mut cred as *mut _ as *mut libc::c_void, &mut len)
    };
    (r == 0).then_some(cred.uid)
}

#[cfg(not(target_os = "linux"))]
fn peer_uid(stream: &UnixStream) -> Option<u32> {
    use std::os::unix::io::AsRawFd;
    let (mut uid, mut gid) = (0, 0);
    // SAFETY: getpeereid writes the peer's ids for a valid connected socket.
    let r = unsafe { libc::getpeereid(stream.as_raw_fd(), &mut uid, &mut gid) };
    (r == 0).then_some(uid)
}

/// Submit `message` to a lane (None = the active one); Ok = the lane's name.
pub type SubmitFn = dyn Fn(Option<&str>, String) -> Result<String, String> + Send + Sync;

/// What the link needs from the REPL: submit a message to a lane (None = the active lane;
/// returns the lane's name or an error), and report status as JSON.
pub struct Hooks {
    pub submit: Box<SubmitFn>,
    pub status: Box<dyn Fn() -> Value + Send + Sync>,
}

/// Start listening on a thread. `Ok(None)` if another live zilch already serves this workspace.
pub fn serve(hooks: Arc<Hooks>, workspace: &Path) -> std::io::Result<Option<IpcGuard>> {
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
    std::thread::Builder::new().name("zilch-os-ipc".into()).spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut stream) = stream else { continue };
            if !peer_allowed(&stream) {
                let _ = stream.write_all(b"{\"ok\":false,\"error\":\"permission denied\"}\n");
                continue;
            }
            let h = hooks.clone();
            std::thread::spawn(move || {
                let _ = handle_conn(&h, stream);
            });
        }
    })?;
    Ok(Some(guard))
}

fn handle_conn(h: &Hooks, stream: UnixStream) -> std::io::Result<()> {
    let mut w = stream.try_clone()?;
    for line in BufReader::new(stream).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let reply = match serde_json::from_str::<Request>(&line) {
            Ok(req) => handle(h, req),
            Err(e) => json!({"ok": false, "error": format!("bad request: {e}")}),
        };
        w.write_all(format!("{reply}\n").as_bytes())?;
    }
    Ok(())
}

fn handle(h: &Hooks, req: Request) -> Value {
    match req {
        Request::Status => (h.status)(),
        Request::Send { text, lane, file, start, end, selection } => {
            if text.trim().is_empty() && selection.as_deref().is_none_or(|s| s.trim().is_empty()) {
                return json!({"ok": false, "error": "empty message"});
            }
            let message = compose(&text, file.as_deref(), start, end, selection.as_deref());
            match (h.submit)(lane.as_deref().filter(|l| !l.is_empty()), message) {
                Ok(lane) => json!({"ok": true, "lane": lane}),
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

    #[test]
    fn one_listener_per_workspace_and_it_vanishes_on_drop() {
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

    #[test]
    fn same_user_peer_is_allowed() {
        let ws = tempfile::tempdir().unwrap();
        let ep = endpoint(ws.path());
        let (listener, _socket) = bind(ws.path(), &ep).unwrap().unwrap();
        let addr = listener.local_addr().unwrap();
        let client = std::thread::spawn(move || {
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
        let (stream, _) = listener.accept().unwrap();
        assert!(peer_allowed(&stream));
        assert_eq!(peer_uid(&stream), Some(uid()), "the peer's uid is read from the socket");
        drop(client.join().unwrap());
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

#[cfg(test)]
mod round_trip {
    use super::*;
    use std::sync::Mutex;

    fn connect(ep: &str) -> UnixStream {
        #[cfg(target_os = "linux")]
        {
            use std::os::linux::net::SocketAddrExt;
            let a = std::os::unix::net::SocketAddr::from_abstract_name(ep.trim_start_matches('@').as_bytes()).unwrap();
            UnixStream::connect_addr(&a).unwrap()
        }
        #[cfg(not(target_os = "linux"))]
        UnixStream::connect(ep).unwrap()
    }

    #[test]
    fn an_editor_request_reaches_the_lane_and_status_answers() {
        let ws = tempfile::tempdir().unwrap();
        type Got = Arc<Mutex<Vec<(Option<String>, String)>>>;
        let got: Got = Arc::default();
        let sink = got.clone();
        let hooks = Arc::new(Hooks {
            submit: Box::new(move |lane, msg| {
                sink.lock().unwrap().push((lane.map(str::to_string), msg));
                Ok(lane.unwrap_or("code").to_string())
            }),
            status: Box::new(|| json!({"ok": true, "active": "code"})),
        });
        let guard = serve(hooks, ws.path()).unwrap().expect("listening");
        assert_eq!(std::fs::read_to_string(pointer_path(ws.path())).unwrap().trim(), guard.endpoint, "editors find it");
        let mut s = connect(&guard.endpoint);
        let mut r = BufReader::new(s.try_clone().unwrap());
        let mut ask = |req: &str| {
            s.write_all(format!("{req}\n").as_bytes()).unwrap();
            let mut line = String::new();
            r.read_line(&mut line).unwrap();
            serde_json::from_str::<Value>(&line).unwrap()
        };
        let reply = ask(r#"{"type":"send","text":"why?","lane":"spec","file":"a.rs","start":3,"end":4,"selection":"let x = 1;"}"#);
        assert_eq!(reply, json!({"ok": true, "lane": "spec"}));
        assert_eq!(ask(r#"{"type":"send","text":"  "}"#)["error"], "empty message");
        assert_eq!(ask(r#"{"type":"status"}"#)["active"], "code");
        let got = got.lock().unwrap();
        assert_eq!(got[0].0.as_deref(), Some("spec"));
        assert_eq!(got[0].1, "[From the editor] a.rs:3-4\n```rs\nlet x = 1;\n```\nwhy?");
        drop(guard);
        assert!(!pointer_path(ws.path()).exists(), "the pointer goes away with zilch-os");
    }
}
