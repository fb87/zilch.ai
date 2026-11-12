//! `zilch daemon`: one process per user hosting many sessions (each its own kernel, workspace,
//! config and costs), reached over a per-user local socket (peer uid checked, no TCP).
//!
//! Protocol: one JSON object per line. A connection either makes one request (`new`, `list`,
//! `stop`, `shutdown`) and gets one answer, or `attach`es to a session and becomes a stream: the
//! daemon sends the session's output (`out`), its status (`state`, on change) and replies to the
//! client's requests (`reply`); the client sends typed lines (`input`), `interrupt`, `order`,
//! `answer`, `totals` and finally `detach`. Detaching leaves the session running; questions then
//! wait for the next client (and run the `on_question` hook).

use crate::session::{Link, Local, Out, Session, SessionOpts, UiState};
use crate::{ipc, set_width};
use anyhow::{anyhow, bail, Context};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use zilch_ipc::TicketId;

/// Client → daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Up {
    New {
        opts: SessionOpts,
        name: Option<String>,
    },
    List,
    Stop {
        id: String,
    },
    Shutdown,
    /// Attach to `id`, else to the newest session of `workspace`, else create one from `create`.
    Attach {
        id: Option<String>,
        workspace: Option<PathBuf>,
        create: Option<SessionOpts>,
        replay: usize,
        width: usize,
    },
    Input {
        rid: u64,
        line: String,
        asked: Option<(u64, TicketId)>,
        status_row: bool,
        width: usize,
    },
    Interrupt {
        rid: u64,
    },
    Order {
        text: String,
    },
    Answer {
        qid: u64,
        text: String,
    },
    Totals {
        rid: u64,
    },
    Detach,
}

/// One session in a listing.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Info {
    pub id: String,
    pub workspace: PathBuf,
    /// "idle" | "working" | "waiting for you" | "ended"
    pub state: String,
    pub open_orders: usize,
    pub spent: f64,
    pub attached: usize,
    pub age_secs: u64,
}

/// Daemon → client.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "t", rename_all = "snake_case")]
pub enum Down {
    Ok {
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        banner: Option<String>,
        #[serde(default)]
        sessions: Vec<Info>,
    },
    Err {
        error: String,
    },
    Out {
        out: Out,
    },
    State {
        state: UiState,
    },
    Reply {
        rid: u64,
        text: String,
    },
}

struct Entry {
    id: String,
    workspace: PathBuf,
    link: Arc<Local>,
    created: Instant,
}

#[derive(Default)]
struct Daemon {
    sessions: Mutex<BTreeMap<String, Arc<Entry>>>,
}

impl Daemon {
    fn info(&self) -> Vec<Info> {
        self.sessions
            .lock()
            .unwrap()
            .values()
            .map(|e| {
                let st = e.link.state();
                let guard = e.link.0.lock().unwrap();
                let (spent, open) = guard.as_ref().map(|s| (s.summary().0, s.summary().1)).unwrap_or((0.0, 0));
                let state = if guard.is_none() {
                    "ended"
                } else if st.question.is_some() {
                    "waiting for you"
                } else if st.busy || open > 0 {
                    "working"
                } else {
                    "idle"
                };
                Info {
                    id: e.id.clone(),
                    workspace: e.workspace.clone(),
                    state: state.into(),
                    open_orders: open,
                    spent,
                    attached: e.link.1.listeners(),
                    age_secs: e.created.elapsed().as_secs(),
                }
            })
            .collect()
    }

    /// Open a session; its id is `name` or `<workspace name>-<n>`.
    fn create(&self, opts: &SessionOpts, name: Option<&str>) -> anyhow::Result<Arc<Entry>> {
        if let Some(n) = name {
            if n.is_empty() || !n.chars().all(|c| c.is_ascii_alphanumeric() || "-_.".contains(c)) {
                bail!("session names are letters, digits, '-', '_' and '.'");
            }
            if self.sessions.lock().unwrap().contains_key(n) {
                bail!("a session named '{n}' already exists");
            }
        }
        let session = Session::open(opts)?;
        let mut all = self.sessions.lock().unwrap();
        let id = match name {
            Some(n) => n.to_string(),
            None => {
                let base: String = session
                    .cwd
                    .file_name()
                    .map(|s| s.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' { c } else { '_' }).collect())
                    .unwrap_or_else(|| "session".into());
                (1..).map(|n| format!("{base}-{n}")).find(|id| !all.contains_key(id)).expect("a free id")
            }
        };
        let entry = Arc::new(Entry {
            id: id.clone(),
            workspace: session.cwd.clone(),
            link: Arc::new(Local::new(session)),
            created: Instant::now(),
        });
        all.insert(id, entry.clone());
        Ok(entry)
    }

    fn find(&self, id: Option<&str>, workspace: Option<&PathBuf>) -> Option<Arc<Entry>> {
        let all = self.sessions.lock().unwrap();
        if let Some(id) = id {
            return all.get(id).cloned();
        }
        let ws = workspace?.canonicalize().ok()?;
        all.values().filter(|e| e.workspace == ws && !e.link.closed()).max_by_key(|e| std::cmp::Reverse(e.created.elapsed())).cloned()
    }

    fn stop(&self, id: &str, reason: &str) -> anyhow::Result<()> {
        let e = self.sessions.lock().unwrap().remove(id).ok_or_else(|| anyhow!("no session '{id}' (see zilch ls)"))?;
        let totals = e.link.stop(reason);
        eprintln!("[zilch daemon] session {id} stopped ({reason}){}", totals.first().map(|t| format!(": {t}")).unwrap_or_default());
        Ok(())
    }
}

fn send(w: &Mutex<UnixStream>, d: &Down) -> std::io::Result<()> {
    let mut line = serde_json::to_string(d).map_err(std::io::Error::other)?;
    line.push('\n');
    w.lock().unwrap().write_all(line.as_bytes())
}

/// `zilch daemon run`: serve until `shutdown`. Returns false if a daemon already runs.
pub fn run() -> anyhow::Result<bool> {
    let endpoint = ipc::daemon_endpoint();
    let Some(listener) = ipc::bind_endpoint(&endpoint)? else { return Ok(false) };
    eprintln!("[zilch daemon] {} listening on {endpoint} (pid {})", env!("CARGO_PKG_VERSION"), std::process::id());
    let daemon = Arc::new(Daemon::default());
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        if !ipc::peer_allowed(&stream) {
            let _ = stream.write_all(b"{\"t\":\"err\",\"error\":\"permission denied\"}\n");
            continue;
        }
        let d = daemon.clone();
        std::thread::spawn(move || {
            if let Err(e) = connection(&d, stream) {
                eprintln!("[zilch daemon] connection: {e:#}");
            }
        });
    }
    Ok(true)
}

fn connection(d: &Arc<Daemon>, stream: UnixStream) -> anyhow::Result<()> {
    let w = Arc::new(Mutex::new(stream.try_clone()?));
    let mut lines = BufReader::new(stream.try_clone()?).lines();
    let Some(first) = lines.next() else { return Ok(()) };
    let up: Up = match serde_json::from_str(&first?) {
        Ok(u) => u,
        Err(e) => {
            send(&w, &Down::Err { error: format!("bad request: {e}") })?;
            return Ok(());
        }
    };
    let ok = |id: Option<String>, banner: Option<String>| Down::Ok { id, banner, sessions: vec![] };
    let reply = match up {
        Up::New { opts, name } => match d.create(&opts, name.as_deref()) {
            Ok(e) => ok(Some(e.id.clone()), e.link.0.lock().unwrap().as_ref().map(|s| s.banner.clone())),
            Err(e) => Down::Err { error: format!("{e:#}") },
        },
        Up::List => Down::Ok { id: None, banner: None, sessions: d.info() },
        Up::Stop { id } => match d.stop(&id, "stopped by the guest") {
            Ok(()) => ok(Some(id), None),
            Err(e) => Down::Err { error: e.to_string() },
        },
        Up::Shutdown => {
            let ids: Vec<String> = d.sessions.lock().unwrap().keys().cloned().collect();
            for id in ids {
                let _ = d.stop(&id, "the daemon shut down");
            }
            send(&w, &ok(None, None))?;
            eprintln!("[zilch daemon] shut down");
            std::process::exit(0);
        }
        Up::Attach { id, workspace, create, replay, width } => {
            set_width(width);
            let entry = match d.find(id.as_deref(), workspace.as_ref()) {
                Some(e) => Ok(e),
                None => match create {
                    Some(opts) => d.create(&opts, id.as_deref()),
                    None => Err(anyhow!("no session {}(see zilch ls)", id.map(|i| format!("'{i}' ")).unwrap_or_default())),
                },
            };
            match entry {
                Ok(e) => return attached(&e, &w, &mut lines, &stream, replay),
                Err(e) => Down::Err { error: format!("{e:#}") },
            }
        }
        _ => Down::Err { error: "attach first".into() },
    };
    send(&w, &reply)?;
    Ok(())
}

/// Serve an attached client until it detaches or disconnects.
fn attached(
    e: &Arc<Entry>,
    w: &Arc<Mutex<UnixStream>>,
    lines: &mut std::io::Lines<BufReader<UnixStream>>,
    stream: &UnixStream,
    replay: usize,
) -> anyhow::Result<()> {
    let banner = e.link.0.lock().unwrap().as_ref().map(|s| s.banner.clone());
    send(w, &Down::Ok { id: Some(e.id.clone()), banner, sessions: vec![] })?;
    let _attached = e.link.1.attach();
    let done = Arc::new(AtomicBool::new(false));
    // Output to the client.
    let rx = e.link.subscribe(replay);
    // The replay goes out before anything this client asks is answered.
    while let Ok(out) = rx.try_recv() {
        send(w, &Down::Out { out })?;
    }
    let (w2, done2) = (w.clone(), done.clone());
    std::thread::spawn(move || {
        for out in rx.iter() {
            let end = matches!(out, Out::Ended { .. });
            if done2.load(Ordering::SeqCst) || send(&w2, &Down::Out { out }).is_err() || end {
                break;
            }
        }
    });
    // The status row, when it changes.
    let (w3, done3, link) = (w.clone(), done.clone(), e.link.clone());
    std::thread::spawn(move || {
        let mut last = UiState { status: "\0".into(), ..Default::default() };
        while !done3.load(Ordering::SeqCst) {
            let st = link.state();
            if st != last {
                if send(&w3, &Down::State { state: st.clone() }).is_err() {
                    break;
                }
                last = st;
            }
            std::thread::sleep(Duration::from_millis(100));
        }
    });
    for line in lines.by_ref() {
        let Ok(line) = line else { break };
        let up: Up = match serde_json::from_str(&line) {
            Ok(u) => u,
            Err(_) => continue,
        };
        match up {
            Up::Input { rid, line, asked, status_row, width } => {
                set_width(width);
                let text = e.link.input(&line, asked, status_row);
                send(w, &Down::Reply { rid, text })?;
            }
            Up::Interrupt { rid } => send(w, &Down::Reply { rid, text: e.link.interrupt() })?,
            Up::Order { text } => e.link.order(&text),
            Up::Answer { qid, text } => e.link.answer(qid, &text),
            Up::Totals { rid } => send(w, &Down::Reply { rid, text: e.link.totals().join("\n") })?,
            Up::Detach => break,
            _ => {}
        }
    }
    done.store(true, Ordering::SeqCst);
    let _ = stream.shutdown(std::net::Shutdown::Both);
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Client side

/// One request, one answer (`new`, `list`, `stop`, `shutdown`).
pub fn request(up: &Up) -> anyhow::Result<Down> {
    let endpoint = ipc::daemon_endpoint();
    let mut s = ipc::connect(&endpoint).map_err(|_| anyhow!("no zilch daemon is running (start it with: zilch daemon start)"))?;
    s.write_all(format!("{}\n", serde_json::to_string(up)?).as_bytes())?;
    let mut line = String::new();
    BufReader::new(s).read_line(&mut line)?;
    match serde_json::from_str::<Down>(&line).context("bad answer from the daemon")? {
        Down::Err { error } => bail!("{error}"),
        d => Ok(d),
    }
}

pub fn running() -> bool {
    ipc::connect(&ipc::daemon_endpoint()).is_ok()
}

/// A session in the daemon, as a [`Link`].
pub struct Remote {
    w: Mutex<UnixStream>,
    state: Mutex<UiState>,
    replies: Mutex<HashMap<u64, Sender<String>>>,
    next: AtomicU64,
    subs: Mutex<Vec<Sender<Out>>>,
    /// Output that arrived before anyone subscribed (the replayed backlog).
    pending: Mutex<Vec<Out>>,
    closed: AtomicBool,
}

impl Remote {
    /// Attach; returns the link, the session's id and its banner.
    pub fn attach(up: &Up) -> anyhow::Result<(Arc<Remote>, String, String)> {
        let endpoint = ipc::daemon_endpoint();
        let mut s = ipc::connect(&endpoint).map_err(|_| anyhow!("no zilch daemon is running (start it with: zilch daemon start)"))?;
        s.write_all(format!("{}\n", serde_json::to_string(up)?).as_bytes())?;
        let mut reader = BufReader::new(s.try_clone()?);
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let (id, banner) = match serde_json::from_str::<Down>(&line).context("bad answer from the daemon")? {
            Down::Ok { id, banner, .. } => (id.unwrap_or_default(), banner.unwrap_or_default()),
            Down::Err { error } => bail!("{error}"),
            other => bail!("unexpected answer from the daemon: {other:?}"),
        };
        let r = Arc::new(Remote {
            w: Mutex::new(s),
            state: Mutex::default(),
            replies: Mutex::default(),
            next: AtomicU64::new(1),
            subs: Mutex::default(),
            pending: Mutex::default(),
            closed: AtomicBool::new(false),
        });
        let r2 = r.clone();
        std::thread::Builder::new().name("zilch-remote".into()).spawn(move || {
            for line in reader.lines() {
                let Ok(line) = line else { break };
                match serde_json::from_str::<Down>(&line) {
                    Ok(Down::Out { out }) => {
                        if matches!(out, Out::Ended { .. }) {
                            r2.closed.store(true, Ordering::SeqCst);
                        }
                        r2.publish(out);
                    }
                    Ok(Down::State { state }) => *r2.state.lock().unwrap() = state,
                    Ok(Down::Reply { rid, text }) => {
                        if let Some(tx) = r2.replies.lock().unwrap().remove(&rid) {
                            let _ = tx.send(text);
                        }
                    }
                    _ => {}
                }
            }
            if !r2.closed.swap(true, Ordering::SeqCst) {
                r2.publish(Out::Ended { reason: "lost the connection to the daemon".into() });
            }
            r2.replies.lock().unwrap().clear(); // wakes waiting calls
        })?;
        Ok((r, id, banner))
    }

    fn publish(&self, out: Out) {
        let mut subs = self.subs.lock().unwrap();
        if subs.is_empty() {
            self.pending.lock().unwrap().push(out);
            return;
        }
        subs.retain(|s| s.send(out.clone()).is_ok());
    }

    fn send(&self, up: &Up) {
        if let Ok(line) = serde_json::to_string(up) {
            let _ = self.w.lock().unwrap().write_all(format!("{line}\n").as_bytes());
        }
    }

    /// Send a request and wait for its reply.
    fn call(&self, f: impl FnOnce(u64) -> Up) -> String {
        let rid = self.next.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = std::sync::mpsc::channel();
        self.replies.lock().unwrap().insert(rid, tx);
        self.send(&f(rid));
        rx.recv_timeout(Duration::from_secs(120)).unwrap_or_default()
    }

    /// Leave on purpose (no "lost the connection" notice when the daemon then closes it).
    pub fn detach(&self) {
        self.closed.store(true, Ordering::SeqCst);
        self.send(&Up::Detach);
    }
}

impl Link for Remote {
    fn input(&self, line: &str, asked: Option<(u64, TicketId)>, status_row: bool) -> String {
        let width = crate::terminal_width();
        self.call(|rid| Up::Input { rid, line: line.to_string(), asked, status_row, width })
    }
    fn interrupt(&self) -> String {
        self.call(|rid| Up::Interrupt { rid })
    }
    fn state(&self) -> UiState {
        self.state.lock().unwrap().clone()
    }
    fn subscribe(&self, _replay: usize) -> Receiver<Out> {
        // The replay was chosen at attach; what arrived since waits in `pending`.
        let (tx, rx) = std::sync::mpsc::channel();
        let mut subs = self.subs.lock().unwrap();
        for out in self.pending.lock().unwrap().drain(..) {
            let _ = tx.send(out);
        }
        subs.push(tx);
        rx
    }
    fn order(&self, text: &str) {
        self.send(&Up::Order { text: text.to_string() });
    }
    fn answer(&self, qid: u64, text: &str) {
        self.send(&Up::Answer { qid, text: text.to_string() });
    }
    fn totals(&self) -> Vec<String> {
        self.call(|rid| Up::Totals { rid }).lines().map(str::to_string).collect()
    }
    fn closed(&self) -> bool {
        self.closed.load(Ordering::SeqCst)
    }
}

/// `zilch daemon start`: run the daemon in the background (its own session, output to a log).
pub fn start() -> anyhow::Result<()> {
    if running() {
        println!("zilch daemon already running ({})", ipc::daemon_endpoint());
        return Ok(());
    }
    let log = dirs::data_dir().unwrap_or_else(std::env::temp_dir).join("zilch").join("logs").join("daemon.log");
    if let Some(dir) = log.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let out = std::fs::OpenOptions::new().create(true).append(true).open(&log)?;
    let mut cmd = std::process::Command::new(std::env::current_exe()?);
    cmd.args(["daemon", "run"]).stdin(std::process::Stdio::null()).stdout(out.try_clone()?).stderr(out);
    {
        use std::os::unix::process::CommandExt;
        // SAFETY: setsid() is async-signal-safe; the child leaves our terminal's session.
        unsafe {
            cmd.pre_exec(|| {
                libc::setsid();
                Ok(())
            });
        }
    }
    let child = cmd.spawn()?;
    let deadline = Instant::now() + Duration::from_secs(5);
    while !running() {
        if Instant::now() > deadline {
            bail!("the daemon did not start; see {}", log.display());
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    println!("zilch daemon started (pid {}, log {})", child.id(), log.display());
    Ok(())
}

/// `zilch ls`.
pub fn list_text(sessions: &[Info]) -> String {
    if sessions.is_empty() {
        return "No sessions (zilch new, or zilch attach in a workspace).".into();
    }
    let mut s = format!("{:<20} {:<16} {:>5} {:>9} {:>8} {:>6}  {}\n", "ID", "STATE", "OPEN", "SPENT", "ATTACHED", "AGE", "WORKSPACE");
    for i in sessions {
        let age = match i.age_secs {
            a if a < 3600 => format!("{}m", a / 60),
            a if a < 86_400 => format!("{}h", a / 3600),
            a => format!("{}d", a / 86_400),
        };
        s.push_str(&format!(
            "{:<20} {:<16} {:>5} {:>9} {:>8} {:>6}  {}\n",
            i.id,
            i.state,
            i.open_orders,
            format!("${:.4}", i.spent),
            i.attached,
            age,
            i.workspace.display()
        ));
    }
    s.trim_end().to_string()
}
