//! The daemon end to end: the real binary against a scripted OpenAI-compatible server.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Output, Stdio};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Answers every chat request with the same text (streamed), recording each request body.
fn model_server(reply: &'static str) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = seen.clone();
    std::thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { continue };
            let log = log.clone();
            std::thread::spawn(move || {
                let mut r = BufReader::new(s.try_clone().unwrap());
                let mut len = 0usize;
                loop {
                    let mut line = String::new();
                    if r.read_line(&mut line).unwrap_or(0) == 0 {
                        return;
                    }
                    if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                        len = v.trim().parse().unwrap();
                    }
                    if line == "\r\n" {
                        break;
                    }
                }
                let mut body = vec![0; len];
                r.read_exact(&mut body).unwrap();
                log.lock().unwrap().push(String::from_utf8_lossy(&body).into_owned());
                let chunk = serde_json::json!({"choices":[{"delta":{"reasoning_content": "Thinking it over.\n", "content": reply}}],"usage":{"prompt_tokens":10,"completion_tokens":3}});
                let body = format!("data: {chunk}\n\ndata: [DONE]\n\n");
                let resp = format!(
                    "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = s.write_all(resp.as_bytes());
            });
        }
    });
    (url, seen)
}

struct Env {
    home: tempfile::TempDir,
    socket: PathBuf,
}

impl Env {
    fn new(url: &str) -> Env {
        let home = tempfile::tempdir().unwrap();
        let cfg = home.path().join("config/zilch");
        std::fs::create_dir_all(&cfg).unwrap();
        std::fs::write(
            cfg.join("config.toml"),
            format!(
                "[providers.fake]\nbase_url = \"{url}\"\napi_key_env = \"FAKE_KEY\"\ndiscover = false\n\n[defaults]\nprovider = \"fake\"\nmodel = \"m\"\n"
            ),
        )
        .unwrap();
        let run = home.path().join("run");
        std::fs::create_dir_all(&run).unwrap();
        let socket = run.join("d.sock");
        Env { home, socket }
    }

    fn cmd(&self, dir: &Path) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_zilch"));
        c.current_dir(dir)
            .env("HOME", self.home.path())
            .env("XDG_CONFIG_HOME", self.home.path().join("config"))
            .env("XDG_DATA_HOME", self.home.path().join("data"))
            .env("ZILCH_DAEMON_SOCKET", &self.socket)
            .env("FAKE_KEY", "k")
            .env_remove("ZILCH_MODEL")
            .env_remove("ZILCH_PROVIDER")
            .env_remove("OPENAI_BASE_URL")
            .stdin(Stdio::null());
        c
    }

    fn run(&self, dir: &Path, args: &[&str]) -> Output {
        let out = self.cmd(dir).args(args).output().unwrap();
        out
    }
}

fn text(o: &Output) -> String {
    format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
}

fn start_daemon(env: &Env, dir: &Path) -> Child {
    let child = env.cmd(dir).args(["daemon", "run"]).stdout(Stdio::null()).stderr(Stdio::null()).spawn().unwrap();
    let deadline = Instant::now() + Duration::from_secs(10);
    while UnixStream::connect(&env.socket).is_err() {
        assert!(Instant::now() < deadline, "the daemon did not start");
        std::thread::sleep(Duration::from_millis(50));
    }
    child
}

#[test]
fn daemon_sessions_outlive_their_clients() {
    let (url, seen) = model_server("The answer is 42.");
    let env = Env::new(&url);
    let ws = tempfile::tempdir().unwrap();
    let mut daemon = start_daemon(&env, ws.path());
    // A second daemon on the same socket refuses to start.
    let again = env.run(ws.path(), &["daemon", "run"]);
    assert!(!again.status.success() && text(&again).contains("already runs"), "{}", text(&again));

    let new = env.run(ws.path(), &["new", "--name", "t1", "--no-log"]);
    assert!(new.status.success(), "{}", text(&new));
    assert!(text(&new).starts_with("t1\n"), "{}", text(&new));
    let dup = env.run(ws.path(), &["new", "--name", "t1"]);
    assert!(!dup.status.success() && text(&dup).contains("already exists"), "{}", text(&dup));

    // One order into the session; the client exits once it is served, the session stays.
    let p = env.run(ws.path(), &["-p", "what is the answer?", "--session", "t1"]);
    assert!(p.status.success(), "{}", text(&p));
    let out = text(&p);
    assert!(out.contains("The answer is 42.") && out.contains("order #1 served"), "{out}");
    assert!(!out.contains('\x1b'), "no colour codes into a pipe: {out:?}");
    let ls = text(&env.run(ws.path(), &["ls"]));
    assert!(ls.contains("t1") && ls.contains("idle") && ls.contains(&ws.path().canonicalize().unwrap().display().to_string()), "{ls}");

    // A second client: the same main agent, which remembers the first order.
    let p2 = env.run(ws.path(), &["-p", "and again?", "--session", "t1"]);
    assert!(p2.status.success() && text(&p2).contains("order #2 served"), "{}", text(&p2));
    let last = seen.lock().unwrap().iter().rev().find(|b| b.contains("and again?")).cloned().unwrap();
    assert!(last.contains("what is the answer?"), "the session kept its conversation: {last}");

    // Attach over the socket: replayed output, a command's reply, then detach.
    let mut s = UnixStream::connect(&env.socket).unwrap();
    s.write_all(b"{\"op\":\"attach\",\"id\":\"t1\",\"replay\":50,\"width\":80}\n").unwrap();
    let mut r = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    r.read_line(&mut line).unwrap();
    assert!(line.contains("\"t\":\"ok\"") && line.contains("\"id\":\"t1\""), "{line}");
    s.write_all(b"{\"op\":\"input\",\"rid\":7,\"line\":\"/ps\",\"asked\":null,\"status_row\":false,\"width\":80}\n").unwrap();
    let mut got_replay = false;
    let reply = loop {
        line.clear();
        assert!(r.read_line(&mut line).unwrap() > 0, "connection closed");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();
        if v["t"] == "out" && v["out"]["text"].as_str().is_some_and(|t| t.contains("order #2 served")) {
            got_replay = true;
        }
        if v["t"] == "reply" && v["rid"] == 7 {
            break v["text"].as_str().unwrap().to_string();
        }
    };
    assert!(got_replay, "attach replays recent output");
    assert!(reply.contains("PID") || reply.contains("No processes"), "{reply}");
    s.write_all(b"{\"op\":\"detach\"}\n").unwrap();

    // Stop the session, then the daemon.
    assert!(env.run(ws.path(), &["stop", "t1"]).status.success());
    assert!(text(&env.run(ws.path(), &["ls"])).contains("No sessions"));
    let p3 = env.run(ws.path(), &["stop", "t1"]);
    assert!(!p3.status.success() && text(&p3).contains("no session"), "{}", text(&p3));
    assert!(env.run(ws.path(), &["daemon", "stop"]).status.success());
    let deadline = Instant::now() + Duration::from_secs(10);
    while daemon.try_wait().unwrap().is_none() {
        assert!(Instant::now() < deadline, "the daemon did not exit");
        std::thread::sleep(Duration::from_millis(50));
    }
    let gone = env.run(ws.path(), &["ls"]);
    assert!(!gone.status.success() && text(&gone).contains("no zilch daemon"), "{}", text(&gone));
}

#[test]
fn p_with_a_new_session_name_creates_it_for_this_directory() {
    let (url, _) = model_server("Done.");
    let env = Env::new(&url);
    let ws = tempfile::tempdir().unwrap();
    let mut daemon = start_daemon(&env, ws.path());
    let p = env.run(ws.path(), &["-p", "hi", "--session", "fresh", "--no-log"]);
    assert!(p.status.success() && text(&p).contains("Done."), "{}", text(&p));
    assert!(text(&env.run(ws.path(), &["ls"])).contains("fresh"));
    let _ = env.run(ws.path(), &["daemon", "stop"]);
    let _ = daemon.wait();
}

#[test]
fn verbose_shows_the_models_streamed_reasoning_and_text_quiet_does_not() {
    let (url, _) = model_server("The answer is 42.");
    let env = Env::new(&url);
    let ws = tempfile::tempdir().unwrap();
    let quiet = text(&env.run(ws.path(), &["-p", "what is it?", "--no-log", "--no-session"]));
    assert!(quiet.contains("The answer is 42.") && !quiet.contains('┊') && !quiet.contains("│ The answer"), "{quiet}");
    let verbose = text(&env.run(ws.path(), &["-p", "what is it?", "--no-log", "--no-session", "-v"]));
    assert!(verbose.contains("1 ┊ Thinking it over."), "the reasoning, per process: {verbose}");
    assert!(verbose.contains("1 │ The answer is 42."), "the reply text as it streamed: {verbose}");
    assert!(verbose.contains("▸ llm "), "model calls too: {verbose}");
}

#[test]
fn a_served_order_says_how_long_it_took_and_what_it_used() {
    let (url, _) = model_server("Done.");
    let env = Env::new(&url);
    let ws = tempfile::tempdir().unwrap();
    let out = text(&env.run(ws.path(), &["-p", "hi", "--no-log", "--no-session", "-v"]));
    // The fake server answers 10 in / 3 out per request; its model has no known price.
    let served = out.lines().find(|l| l.contains("order #1 served")).unwrap_or_default().to_string();
    assert!(served.contains("s · 1 request · 10 in (") && served.contains("· 3 out · price unknown"), "{served}");
    assert!(!out.contains("▸ recv") && !out.contains("✓ serve"), "no idle bookkeeping lines: {out}");
}
