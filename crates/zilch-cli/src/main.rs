mod approver;
mod inbox;
mod ipc;
mod lanes;
mod log;
mod markdown;
mod render;
mod status;
mod wrap;

/// The REPL line editor; its helper draws the live status bar above the input line.
type LineEditor = rustyline::Editor<status::StatusHelper, rustyline::history::DefaultHistory>;

use anyhow::{bail, Result};
use clap::Parser;
use lanes::{Dispatcher, Lane};
use render::{Screen, SharedScreen, BOLD, CYAN, DIM, GREEN, RED, RESET};
use std::io::{IsTerminal, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use zilch_core::team::session;
use zilch_core::team::todo::TodoStatus;
use zilch_core::*;

/// Exit code for a run stopped by the user (like SIGINT).
const EXIT_CANCELLED: i32 = 130;

#[derive(Parser, Debug)]
#[command(name = "zilch", version, about = "A minimal, parallel, TODO-driven multi-agent coding assistant")]
struct Cli {
    /// Run a single prompt non-interactively and print the final answer.
    #[arg(short, long)]
    prompt: Option<String>,
    /// Model for the master.
    #[arg(long)]
    model: Option<String>,
    /// Model for a worker profile, e.g. `--worker-model coder=gpt-4.1-mini` (repeatable).
    #[arg(long = "worker-model", value_name = "PROFILE=MODEL")]
    worker_models: Vec<String>,
    /// Maximum number of workers running at the same time.
    #[arg(long)]
    workers: Option<usize>,
    /// Approval mode: ask | plan | auto.
    #[arg(long, value_parser = parse_mode)]
    approval: Option<ApprovalMode>,
    /// Approve everything automatically (plans and actions). Same as --approval auto.
    #[arg(long)]
    yolo: bool,
    /// Stop when the whole team has used this many tokens (prompt + completion).
    #[arg(long, value_name = "TOKENS")]
    max_tokens: Option<u64>,
    /// Directory for the spec lane's documents (spec.md, design.md, plan.md, versions/); default specs/.
    #[arg(long, value_name = "DIR")]
    spec_dir: Option<String>,
    /// Stop before the team's spend would exceed this many USD (needs known prices).
    #[arg(long, value_name = "USD")]
    max_cost: Option<f64>,
    /// Write the JSONL session log here (default: <data dir>/zilch/logs/).
    #[arg(long, value_name = "FILE")]
    log: Option<std::path::PathBuf>,
    /// Don't write a session log.
    #[arg(long)]
    no_log: bool,
    /// Continue the most recent session in this directory.
    #[arg(short = 'c', long = "continue")]
    r#continue: bool,
    /// Resume a specific session by id (see --sessions).
    #[arg(long, value_name = "ID")]
    resume: Option<String>,
    /// List saved sessions for this directory, then exit.
    #[arg(long)]
    sessions: bool,
    /// Don't save this session.
    #[arg(long)]
    no_session: bool,
    /// Lane for -p: "code" (default, plan + implement) or "spec" (requirements interview).
    #[arg(long, value_parser = ["code", "spec"], default_value = "code")]
    lane: String,
    /// Check configuration and connectivity of every configured model, then exit.
    #[arg(long)]
    doctor: bool,
    /// Show worker output and tool calls.
    #[arg(short, long)]
    verbose: bool,
    /// Working directory (defaults to the current directory).
    #[arg(short = 'C', long)]
    cwd: Option<std::path::PathBuf>,
}

fn parse_mode(s: &str) -> Result<ApprovalMode, String> {
    match s {
        "ask" => Ok(ApprovalMode::Ask),
        "plan" => Ok(ApprovalMode::Plan),
        "auto" => Ok(ApprovalMode::Auto),
        _ => Err(format!("invalid approval mode '{s}' (ask | plan | auto)")),
    }
}

fn main() {
    // Sandbox helper mode: must be handled before any threads exist (unshare(CLONE_NEWUSER)
    // refuses multithreaded processes), i.e. before the async runtime starts.
    let mut args = std::env::args();
    if args.nth(1).as_deref() == Some("__sandbox") {
        #[cfg(target_os = "linux")]
        zilch_core::tools::sandbox_linux::helper_main(args);
        #[cfg(not(target_os = "linux"))]
        std::process::exit(126);
    }
    // The sandbox re-executes this same binary, so a single `zilch` is all a user needs.
    if let Ok(exe) = std::env::current_exe() {
        zilch_core::tools::sandbox::set_helper(exe, vec!["__sandbox".into()]);
    }
    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().expect("tokio runtime");
    match rt.block_on(real_main()) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
    }
}

/// Read piped stdin without hanging when stdin is an open pipe that never sends anything
/// (CI runners, editors): wait briefly for the first byte or EOF, otherwise ignore stdin.
fn read_piped_stdin() -> Option<String> {
    if std::io::stdin().is_terminal() {
        return None;
    }
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let mut first = [0u8; 1];
        let mut stdin = std::io::stdin().lock();
        match stdin.read(&mut first) {
            Ok(0) | Err(_) => {
                let _ = tx.send(None);
            }
            Ok(_) => {
                let mut rest = Vec::new();
                let _ = stdin.read_to_end(&mut rest);
                let mut all = vec![first[0]];
                all.extend(rest);
                let _ = tx.send(Some(String::from_utf8_lossy(&all).into_owned()));
            }
        }
    });
    // Data that is actually piped in starts arriving immediately; a silent open pipe doesn't.
    match rx.recv_timeout(Duration::from_millis(250)) {
        Ok(v) => v,
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            // Something is writing slowly (e.g. `slow_cmd | zilch -p`): wait for it to finish.
            if std::env::var_os("ZILCH_STDIN_WAIT").is_some() {
                rx.recv().ok().flatten()
            } else {
                None
            }
        }
        Err(_) => None,
    }
}

async fn real_main() -> Result<i32> {
    let cli = Cli::parse();
    let cwd = match &cli.cwd {
        Some(p) => std::fs::canonicalize(p)?,
        None => std::env::current_dir()?.canonicalize()?,
    };
    let mut worker_models = Vec::new();
    for wm in &cli.worker_models {
        let Some((p, m)) = wm.split_once('=') else { bail!("--worker-model expects PROFILE=MODEL, got '{wm}'") };
        worker_models.push((p.to_string(), m.to_string()));
    }
    let ov = Overrides {
        master_model: cli.model.clone(),
        worker_models,
        max_workers: cli.workers,
        approval: if cli.yolo { Some(ApprovalMode::Auto) } else { cli.approval },
        max_session_tokens: cli.max_tokens,
        max_session_cost: cli.max_cost,
        spec_dir: cli.spec_dir.clone(),
    };
    let cfg = Config::load(&cwd, &ov)?;
    let mut cfg = cfg;
    // Prices and context windows from the providers' model lists (best effort, ~5s max).
    let (catalog, model_warnings) = cfg.discover_checked().await;
    for w in &model_warnings {
        eprintln!(
            "{}warning:{} {w}",
            if std::io::stderr().is_terminal() { "\x1b[33m" } else { "" },
            if std::io::stderr().is_terminal() { "\x1b[0m" } else { "" }
        );
    }
    cfg.apply_catalog(&catalog);
    let providers = cfg.build_providers()?;
    if cli.sessions {
        return list_sessions(&cwd);
    }
    if cli.doctor {
        return doctor(&cfg, &providers, &catalog).await;
    }

    let one_shot = cli.prompt.is_some();
    let tty = std::io::stdout().is_terminal();
    // In -p mode progress goes to stderr and the final answer to stdout. In the REPL, output is
    // printed above the live input line (both lanes can produce output while the user types).
    let mut editor: Option<LineEditor> = if one_shot { None } else { Some(LineEditor::new()?) };
    let out: Box<dyn std::io::Write + Send> = match editor.as_mut() {
        None => Box::new(std::io::stderr()),
        Some(rl) => Box::new(render::PrinterOut::new(Box::new(rl.create_external_printer()?))),
    };
    let color = if one_shot { std::io::stderr().is_terminal() } else { tty } && std::env::var_os("NO_COLOR").is_none();
    let screen = Screen::new(out, if one_shot { 2 } else { 1 }, color, cli.verbose, !one_shot);
    if !one_shot {
        let mut s = screen.lock().await;
        s.live_panel = false; // cursor movement would corrupt the line being typed
        s.set_markdown_indent(cfg.markdown_indent);
        s.lanes = true;
        // Live status above the input line (needs color/escapes; otherwise heartbeat lines).
        if color {
            if let Some(rl) = editor.as_mut() {
                let bar = Arc::new(status::StatusBar::default());
                rl.set_helper(Some(status::StatusHelper(bar.clone())));
                status::spawn_refresher(bar.clone(), Box::new(rl.create_external_printer()?));
                s.status_bar = Some(bar);
                s.update_status();
            }
        }
    }

    let mut event_log = None;
    if !cli.no_log {
        if let Some(path) = cli.log.clone().or_else(log::EventLog::default_path) {
            match log::EventLog::open(&path) {
                Ok(mut l) => {
                    l.record("session_start", serde_json::json!({"cwd": cwd, "version": env!("CARGO_PKG_VERSION"), "one_shot": one_shot}));
                    event_log = Some(l);
                }
                Err(e) => eprintln!("warning: cannot open session log {}: {e}", path.display()),
            }
        }
    }
    let log_path = event_log.as_ref().map(|l| l.path.clone());

    let (sink, rx) = EventSink::new();
    let renderer = render::spawn_renderer(screen.clone(), rx, event_log);
    // -p: agents may prompt on the terminal directly. REPL: the line editor owns the keyboard, so
    // questions go through the inbox and are answered by the next input line.
    let inbox = inbox::Inbox::new(screen.clone());
    let (approver, reviewer): (Arc<dyn Approver>, Arc<dyn PlanReviewer>) = if one_shot {
        (Arc::new(approver::TerminalApprover(screen.clone())), Arc::new(approver::TerminalReviewer(screen.clone())))
    } else {
        (Arc::new(inbox::ReplApprover(inbox.clone())), Arc::new(inbox::ReplReviewer(inbox.clone())))
    };
    let mut team = Team::new(cfg, providers, approver, reviewer, sink)?;
    let store = if cli.no_session { None } else { session::SessionStore::default_store() };
    if let Some(s) = store.clone() {
        team = team.with_store(s);
    }
    team.set_catalog(catalog);
    let team = Arc::new(team);
    // Resume a previous session if asked (the recap is shown so the user can pick the next step).
    let mut recap: Option<String> = None;
    let resume_id = match (&cli.resume, cli.r#continue) {
        (Some(id), _) => Some(id.clone()),
        (None, true) => match store.as_ref().and_then(|s| s.latest(&cwd)) {
            Some(info) => Some(info.id),
            None => bail!("no previous session for {}", cwd.display()),
        },
        _ => None,
    };
    if let Some(id) = resume_id {
        let Some(store) = &store else { bail!("--resume needs session storage (remove --no-session)") };
        let state = store.load(&id)?;
        let summary = state.recap();
        for w in team.resume(state).await? {
            eprintln!("warning: {w}");
        }
        recap = Some(summary);
    }

    // Ctrl-C cancels the running turn (master, scheduler and every worker); pressing it twice
    // within 2s force-quits.
    {
        let team = team.clone();
        let screen = screen.clone();
        let last = Arc::new(AtomicU64::new(0));
        let start = Instant::now();
        tokio::spawn(async move {
            while tokio::signal::ctrl_c().await.is_ok() {
                let now = start.elapsed().as_millis() as u64;
                let prev = last.swap(now, Ordering::SeqCst);
                if prev != 0 && now - prev < 2_000 {
                    eprintln!("\nforce quit");
                    std::process::exit(EXIT_CANCELLED);
                }
                team.cancel();
                if let Ok(mut s) = screen.try_lock() {
                    s.line("⏹ cancelling… (press Ctrl-C again to force quit)");
                }
            }
        });
    }

    if let Some(mut prompt) = cli.prompt {
        if let Some(r) = &recap {
            // Progress output goes to stderr in -p mode; stdout stays reserved for the answer.
            eprintln!("{r}\n");
        }
        if let Some(piped) = read_piped_stdin().filter(|p| !p.trim().is_empty()) {
            prompt = format!("{prompt}\n\n<stdin>\n{piped}\n</stdin>");
        }
        let res = if cli.lane == "spec" { team.run_spec(&prompt).await } else { team.run(&prompt).await };
        // The Ctrl-C task keeps the event sink alive, so give the renderer a moment instead of joining it.
        tokio::time::sleep(Duration::from_millis(50)).await;
        renderer.abort();
        return Ok(match res {
            Ok(answer) => {
                println!("{}", render::wrap_for_tty(1, &answer, team.shared.cfg.markdown_indent));
                if !cli.no_session {
                    eprintln!("session {} — continue with: zilch -c -p \"...\"", team.session_id());
                }
                0
            }
            Err(e) if e.to_string().contains("cancelled") => {
                eprintln!("cancelled");
                EXIT_CANCELLED
            }
            Err(e) => {
                eprintln!("error: {e:#}");
                if let Some(p) = &log_path {
                    eprintln!("session log: {}", p.display());
                }
                1
            }
        });
    }

    repl(team, screen, log_path, recap, editor.expect("line editor exists in REPL mode"), inbox).await?;
    Ok(0)
}

/// `zilch --doctor`: show the resolved configuration and ping every distinct (provider, model).
async fn doctor(
    cfg: &Config,
    providers: &zilch_core::llm::Providers,
    catalog: &std::collections::HashMap<String, zilch_core::llm::catalog::ModelInfo>,
) -> Result<i32> {
    println!("workspace   {}", cfg.cwd.display());
    if cfg.sources.is_empty() {
        let central =
            zilch_core::config::user_config_dirs().into_iter().map(|d| d.join("config.toml").display().to_string()).collect::<Vec<_>>();
        println!("config      none found (built-in defaults + env); looked in .zilch/config.toml, {}", central.join(", "));
    }
    for (i, s) in cfg.sources.iter().enumerate() {
        println!("{}{}", if i == 0 { "config      " } else { "            " }, s.display());
    }
    println!(
        "approval    {:?} · file sandbox {} · max workers {} · token cap {} · cost cap {}",
        cfg.approval,
        if cfg.sandbox { "on" } else { "OFF" },
        cfg.max_workers,
        cfg.max_session_tokens.map_or("none".into(), |t| t.to_string()),
        cfg.max_session_cost.map_or("none".into(), |c| format!("${c}"))
    );
    let (confined, why) = zilch_core::tools::sandbox::status(cfg.bash_sandbox, cfg.bash_network);
    println!("bash        {} {why}", if confined { "✓" } else { "⚠" });
    println!("\nproviders");
    for (name, p) in &cfg.providers {
        let key = match (&p.api_key, &p.api_key_env) {
            (Some(_), _) => "inline key".to_string(),
            (None, Some(v)) => format!("${v} {}", if std::env::var(v).is_ok_and(|k| !k.is_empty()) { "set" } else { "NOT set" }),
            _ => "no key".into(),
        };
        println!("  {name:<12} {}  ({key})", p.base_url.as_deref().unwrap_or("?"));
    }
    println!("\nagents");
    for p in &cfg.profiles {
        let info = catalog.get(&p.binding.key()).copied().unwrap_or_default();
        let price = match (info.input_per_mtok, info.output_per_mtok) {
            (Some(i), Some(o)) if i == 0.0 && o == 0.0 => "free".to_string(),
            (Some(i), Some(o)) => format!("${i}/${o} per 1M"),
            _ => "price unknown".into(),
        };
        println!("  {:<12} {:<40} {:<18} ctx {:>7}  tools: {}", p.name, p.binding.key(), price, p.context_tokens, p.tools.join(","));
    }
    println!("\nconnectivity (tiny request per distinct model)");
    let mut seen = std::collections::BTreeSet::new();
    let mut failures = 0;
    for p in &cfg.profiles {
        if !seen.insert(p.binding.key()) {
            continue;
        }
        let client = providers.get(&p.binding.provider)?;
        let mut binding = p.binding.clone();
        binding.max_tokens = Some(binding.max_tokens.unwrap_or(64).min(64));
        let req = zilch_core::llm::ChatRequest {
            binding,
            messages: vec![zilch_core::llm::Message::user("Reply with the single word: OK")],
            tools: vec![],
            session: Some(zilch_core::agent::run::new_session_id("doctor")),
        };
        let t = Instant::now();
        match tokio::time::timeout(Duration::from_secs(60), zilch_core::llm::complete(client.as_ref(), &req, |_| {})).await {
            Ok(Ok(_)) => println!("  ✓ {:<40} {:>6} ms", p.binding.key(), t.elapsed().as_millis()),
            Ok(Err(e)) => {
                failures += 1;
                println!("  ✗ {:<40} {e:#}", p.binding.key());
            }
            Err(_) => {
                failures += 1;
                println!("  ✗ {:<40} timed out after 60s", p.binding.key());
            }
        }
    }
    println!("\n{}", if failures == 0 { "all good".to_string() } else { format!("{failures} problem(s) found") });
    Ok(if failures == 0 { 0 } else { 1 })
}

async fn repl(
    team: Arc<Team>,
    screen: SharedScreen,
    log_path: Option<std::path::PathBuf>,
    recap: Option<String>,
    mut rl: LineEditor,
    inbox: Arc<inbox::Inbox>,
) -> Result<()> {
    banner(&team, &screen).await;
    if let Some(r) = recap {
        let mut s = screen.lock().await;
        let (c, d, x) = (s.c(CYAN), s.c(DIM), s.c(RESET));
        let mut lines = r.lines();
        let head = lines.next().unwrap_or_default().to_string();
        let body: Vec<String> = lines.map(|l| format!("{d}{l}{x}")).collect();
        s.line(&format!("\n{c}{head}{x}\n{}\n", body.join("\n")));
    }
    let history = dirs::data_dir().map(|d| d.join("zilch").join("history.txt"));
    if let Some(h) = &history {
        let _ = std::fs::create_dir_all(h.parent().unwrap());
        let _ = rl.load_history(h);
    }

    let lanes = Dispatcher::new(team.clone(), screen.clone(), log_path);
    // Editor integration (Neovim plugin): selections + comments arrive over a local socket.
    let _ipc = match ipc::serve(lanes.clone(), &team.shared.cfg.cwd).await {
        Ok(Some(guard)) => {
            let mut s = screen.lock().await;
            let msg = format!("{}editor link: {} (Neovim: :ZilchSend){}", s.c(DIM), guard.endpoint, s.c(RESET));
            s.line(&msg);
            Some(guard)
        }
        Ok(None) => {
            screen.lock().await.line("editor link: another zilch already serves this workspace; not listening");
            None
        }
        Err(e) => {
            screen.lock().await.line(&format!("editor link unavailable: {e}"));
            None
        }
    };

    let bar = screen.lock().await.status_bar.clone();
    loop {
        // The prompt itself only names the lane; live activity is drawn in the status row above
        // it (the leading "\n" reserves that row, see status.rs).
        let row = if bar.is_some() { "\n" } else { "" };
        // A question shown in the prompt is answered by this line; one that arrives while the
        // user is already typing is not (that line was meant for the lane).
        let asked = inbox.prompt();
        let prompt = match (asked, &bar) {
            (Some(q), _) => format!("{row}{q}"),
            (None, Some(_)) => format!("{row}{} › ", lanes.active().icon()),
            (None, None) => format!("[{}] › ", lanes.active().name()),
        };
        if let Some(b) = &bar {
            b.reading.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let read = tokio::task::block_in_place(|| rl.readline(&prompt));
        if let Some(b) = &bar {
            b.reading.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let line = match read {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                // Ctrl-C cancels the active lane's running turn (the other lane keeps going).
                let active = lanes.active();
                if lanes.busy(active) {
                    lanes.cancel(active);
                    screen.lock().await.line(&format!("⏹ cancelling the {} lane…", active.name()));
                } else {
                    screen.lock().await.line("(Ctrl-D to quit)");
                }
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        // A pending question (approval / plan review) takes the line as its answer when its
        // prompt was showing. Slash commands still run (e.g. /code to look before answering).
        if inbox.pending() && !line.trim_start().starts_with('/') {
            if asked.is_some() {
                inbox.answer(&line).await;
                continue;
            }
            // Typed before the question appeared: run it normally and show the question again.
            inbox.reshow().await;
        }
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(input);

        match input {
            "/spec" | "/code" => {
                let active = if input == "/spec" { Lane::Spec } else { Lane::Code };
                lanes.set_active(active);
                let mut s = screen.lock().await;
                s.switch_lane(active.root());
                let msg = match active {
                    Lane::Spec => format!(
                        "{}→ spec lane: requirements engineering. {}{}",
                        s.c(CYAN),
                        team.spec_status().describe(&team.shared.spec_store.dir),
                        s.c(RESET)
                    ),
                    Lane::Code => {
                        format!("{}→ code lane: planning and implementation (works from plan.md/design.md).{}", s.c(CYAN), s.c(RESET))
                    }
                };
                s.line(&msg);
                continue;
            }
            "/status" => {
                let state = |l: Lane| if lanes.busy(l) { "working" } else { "idle" };
                let msg = format!(
                    "lanes: code {} · spec {} (active: {})\n{}",
                    state(Lane::Code),
                    state(Lane::Spec),
                    lanes.active().name(),
                    team.spec_status().describe(&team.shared.spec_store.dir)
                );
                screen.lock().await.line(&msg);
                continue;
            }
            _ if input.starts_with('/') => {
                if !command(&team, &screen, input).await {
                    break;
                }
                continue;
            }
            _ => {}
        }
        // Runs in the background: the prompt comes back immediately, so the user can switch lanes
        // and keep working while this one runs.
        if let Err(e) = lanes.submit(lanes.active(), input.to_string()) {
            screen.lock().await.line(&format!("{}{}", e[..1].to_uppercase(), &e[1..]));
        }
    }

    lanes.shutdown().await;
    if let Some(h) = &history {
        let _ = rl.save_history(h);
    }
    if team.save().await.ok().flatten().is_some() {
        let s = screen.lock().await;
        println!("{}resume this session with: zilch --resume {}{}", s.c(DIM), team.session_id(), s.c(RESET));
    }
    Ok(())
}

async fn banner(team: &Team, screen: &SharedScreen) {
    let cfg = &team.shared.cfg;
    let mut s = screen.lock().await;
    let (c, b, d, r) = (s.c(CYAN), s.c(BOLD), s.c(DIM), s.c(RESET));
    let workers: Vec<String> = cfg.workers().map(|p| format!("{} {d}({}){r}", p.name, p.binding.key())).collect();
    let msg = format!(
        "{c}{b}zilch{r} {d}· {}{r}\n  master  {d}{}{r}\n  workers {}\n  {d}max {} parallel · approval: {:?} · session {} · /help for commands{r}{}",
        cfg.cwd.display(),
        cfg.master().binding.key(),
        workers.join(", "),
        cfg.max_workers,
        cfg.approval,
        team.session_id(),
        match zilch_core::tools::sandbox::status(cfg.bash_sandbox, cfg.bash_network) {
            (true, _) => String::new(),
            (false, why) => format!("\n  {}⚠ bash sandbox {why}{r}", s.c(render::YELLOW)),
        },
    );
    s.line(&msg);
}

/// `/about`: what this zilch is (from the package metadata) and which config it runs with.
fn about(cfg: &Config) -> String {
    let repo = env!("CARGO_PKG_REPOSITORY");
    let mut lines = vec![
        format!("zilch {}", env!("CARGO_PKG_VERSION")),
        format!("  {}", env!("CARGO_PKG_DESCRIPTION")),
        String::new(),
        format!("  repository  {repo}"),
        format!("  docs        {repo}#readme"),
        format!("  author      {}", env!("CARGO_PKG_AUTHORS").replace(':', ", ")),
        format!("  license     {}", env!("CARGO_PKG_LICENSE")),
        format!("  platform    {}/{}", std::env::consts::OS, std::env::consts::ARCH),
        format!("  workspace   {}", cfg.cwd.display()),
    ];
    if cfg.sources.is_empty() {
        lines.push("  config      built-in defaults + environment (no config file found)".into());
    }
    for (i, s) in cfg.sources.iter().enumerate() {
        lines.push(format!("{}{}", if i == 0 { "  config      " } else { "              " }, s.display()));
    }
    lines.join("\n")
}

/// Returns false to exit.
async fn command(team: &Team, screen: &SharedScreen, input: &str) -> bool {
    let (cmd, arg) = input.split_once(' ').map(|(c, a)| (c, a.trim())).unwrap_or((input, ""));
    let out = match cmd {
        "/exit" | "/quit" => return false,
        "/help" => "\
/todos            show the TODO list with status and reports
/team             show worker instances
/models           show the model of every agent
/prompt <agent>   print the composed system prompt (master, coder, coder#2, …)
/usage            token usage per agent and model
/reset <instance> drop a worker instance (its history and context)
/spec             switch to the spec lane (requirements → spec, design, plan); keeps running in parallel
/code             switch to the code lane (implements from plan.md/design.md)
/status           what each lane is doing + spec document versions
/recap            what this session has done so far (requests, last answer, TODOs, workers)
/sessions         list saved sessions for this directory
/clear            start a fresh session (the current one stays resumable)
/about            version, repository, author, license, platform and config in use
/redraw           clear the screen and re-render the last answer at the current width
/exit             quit"
            .to_string(),
        "/about" => about(&team.shared.cfg),
        "/redraw" => {
            screen.lock().await.redraw();
            return true;
        }
        "/todos" => {
            let todos = team.todos();
            if todos.is_empty() {
                "No TODOs.".into()
            } else {
                let s = screen.lock().await;
                todos
                    .iter()
                    .map(|t| {
                        let color = match t.status {
                            TodoStatus::Done => s.c(GREEN),
                            TodoStatus::Failed | TodoStatus::Blocked => s.c(RED),
                            _ => "",
                        };
                        let mut line = format!(
                            "{color}{}{} [{}] {} {}({} · {}{}{}){}",
                            t.status.icon(),
                            s.c(RESET),
                            t.id,
                            t.title,
                            s.c(DIM),
                            t.instance.as_deref().unwrap_or(&t.worker),
                            if t.files.is_empty() { "any file".into() } else { t.files.join(", ") },
                            if t.depends_on.is_empty() { String::new() } else { format!(" · after {}", t.depends_on.join(",")) },
                            if t.approved { "" } else { " · needs approval" },
                            s.c(RESET)
                        );
                        if let Some(r) = &t.report {
                            for l in r.lines().take(6) {
                                line.push_str(&format!("\n    {}{l}{}", s.c(DIM), s.c(RESET)));
                            }
                        }
                        line
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        "/team" => {
            let list = team.instances();
            if list.is_empty() {
                "No worker instances yet (they are spawned on demand).".into()
            } else {
                list.iter()
                    .map(|i| {
                        format!(
                            "{} {:<12} {:<34} {}{} msgs",
                            if i.busy { "◐" } else { "○" },
                            i.id,
                            i.model,
                            if i.readonly { "read-only · " } else { "" },
                            i.messages
                        )
                    })
                    .collect::<Vec<_>>()
                    .join("\n")
            }
        }
        "/models" => team.models().iter().map(|(a, m)| format!("{a:<22} {m}")).collect::<Vec<_>>().join("\n"),
        "/prompt" => {
            let name = if arg.is_empty() { "master" } else { arg };
            team.prompt_of(name).await.unwrap_or_else(|| format!("No agent or profile named '{name}'."))
        }
        "/usage" => screen.lock().await.usage_report(),
        "/reset" => {
            if team.reset_instance(arg) {
                format!("Dropped {arg}.")
            } else {
                format!("No instance '{arg}'. See /team.")
            }
        }
        "/recap" => team.snapshot().await.recap().replacen("↺ Resumed session", "↺ Session", 1),
        "/sessions" => {
            let _ = list_sessions(&team.shared.cfg.cwd);
            format!("current session: {}", team.session_id())
        }
        "/clear" => {
            team.clear().await;
            "Cleared master history, workers and TODOs.".into()
        }
        _ => format!("Unknown command {cmd}. Try /help."),
    };
    screen.lock().await.line(&out);
    true
}

/// `zilch --sessions`: saved sessions for this directory, newest first.
fn list_sessions(cwd: &std::path::Path) -> Result<i32> {
    let Some(store) = session::SessionStore::default_store() else { bail!("no data directory for session storage") };
    let list = store.list(Some(cwd));
    if list.is_empty() {
        println!("No saved sessions for {}.", cwd.display());
        return Ok(0);
    }
    let now = session::now_ms();
    for s in list {
        let age = (now.saturating_sub(s.updated_ms)) / 1000;
        let ago = match age {
            a if a < 120 => format!("{a}s ago"),
            a if a < 7200 => format!("{}m ago", a / 60),
            a if a < 172_800 => format!("{}h ago", a / 3600),
            a => format!("{}d ago", a / 86_400),
        };
        let open = if s.todos_open > 0 { format!(" · {} open TODOs", s.todos_open) } else { String::new() };
        println!("{}  {:>8}  {} turns{open}  {}", s.id, ago, s.turns, s.title.replace('\n', " "));
    }
    println!("\nresume with: zilch --resume <id>   (or zilch -c for the latest)");
    Ok(0)
}
