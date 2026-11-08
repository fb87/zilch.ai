mod approver;
mod log;
mod render;

use anyhow::{bail, Result};
use clap::Parser;
use render::{Screen, SharedScreen, BOLD, CYAN, DIM, GREEN, RED, RESET};
use std::io::{IsTerminal, Read};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
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
    /// Write the JSONL session log here (default: <data dir>/zilch/logs/).
    #[arg(long, value_name = "FILE")]
    log: Option<std::path::PathBuf>,
    /// Don't write a session log.
    #[arg(long)]
    no_log: bool,
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

#[tokio::main]
async fn main() {
    match real_main().await {
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
    };
    let cfg = Config::load(&cwd, &ov)?;
    let providers = cfg.build_providers()?;
    if cli.doctor {
        return doctor(&cfg, &providers).await;
    }

    let one_shot = cli.prompt.is_some();
    let tty = std::io::stdout().is_terminal();
    // In -p mode progress goes to stderr and the final answer to stdout.
    let out: Box<dyn std::io::Write + Send> = if one_shot { Box::new(std::io::stderr()) } else { Box::new(std::io::stdout()) };
    let color = if one_shot { std::io::stderr().is_terminal() } else { tty } && std::env::var_os("NO_COLOR").is_none();
    let screen = Screen::new(out, color, cli.verbose, !one_shot);

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
    let approver: Arc<dyn Approver> = Arc::new(approver::TerminalApprover(screen.clone()));
    let reviewer: Arc<dyn PlanReviewer> = Arc::new(approver::TerminalReviewer(screen.clone()));
    let team = Arc::new(Team::new(cfg, providers, approver, reviewer, sink)?);

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
        if let Some(piped) = read_piped_stdin().filter(|p| !p.trim().is_empty()) {
            prompt = format!("{prompt}\n\n<stdin>\n{piped}\n</stdin>");
        }
        let res = team.run(&prompt).await;
        // The Ctrl-C task keeps the event sink alive, so give the renderer a moment instead of joining it.
        tokio::time::sleep(Duration::from_millis(50)).await;
        renderer.abort();
        return Ok(match res {
            Ok(answer) => {
                println!("{answer}");
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

    repl(team, screen, log_path).await?;
    Ok(0)
}

/// `zilch --doctor`: show the resolved configuration and ping every distinct (provider, model).
async fn doctor(cfg: &Config, providers: &zilch_core::llm::Providers) -> Result<i32> {
    println!("workspace   {}", cfg.cwd.display());
    println!(
        "approval    {:?} · sandbox {} · max workers {} · token cap {}",
        cfg.approval,
        if cfg.sandbox { "on" } else { "OFF" },
        cfg.max_workers,
        cfg.max_session_tokens.map_or("none".into(), |t| t.to_string())
    );
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
        println!("  {:<12} {:<40} tools: {}", p.name, p.binding.key(), p.tools.join(","));
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

async fn repl(team: Arc<Team>, screen: SharedScreen, log_path: Option<std::path::PathBuf>) -> Result<()> {
    banner(&team, &screen).await;
    let mut rl = rustyline::DefaultEditor::new()?;
    let history = dirs::data_dir().map(|d| d.join("zilch").join("history.txt"));
    if let Some(h) = &history {
        let _ = std::fs::create_dir_all(h.parent().unwrap());
        let _ = rl.load_history(h);
    }
    loop {
        let line = tokio::task::block_in_place(|| rl.readline("› "));
        let line = match line {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => continue,
            Err(rustyline::error::ReadlineError::Eof) => break,
            Err(e) => return Err(e.into()),
        };
        let input = line.trim();
        if input.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(input);
        if input.starts_with('/') {
            if !command(&team, &screen, input).await {
                break;
            }
            continue;
        }
        let (before_in, before_out) = screen.lock().await.total_tokens();
        let res = team.run(input).await;
        // Let the renderer drain pending events before printing the footer.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        let mut s = screen.lock().await;
        let (i, o) = s.total_tokens();
        let msg = match res {
            Ok(_) => format!("{}↳ tokens: +{} in / +{} out{}", s.c(DIM), i - before_in, o - before_out, s.c(RESET)),
            Err(e) => {
                let hint =
                    log_path.as_ref().map(|p| format!("\n{}session log: {}{}", s.c(DIM), p.display(), s.c(RESET))).unwrap_or_default();
                format!("{}error:{} {e:#}{hint}", s.c(RED), s.c(RESET))
            }
        };
        s.line(&msg);
    }
    if let Some(h) = &history {
        let _ = rl.save_history(h);
    }
    Ok(())
}

async fn banner(team: &Team, screen: &SharedScreen) {
    let cfg = &team.shared.cfg;
    let mut s = screen.lock().await;
    let (c, b, d, r) = (s.c(CYAN), s.c(BOLD), s.c(DIM), s.c(RESET));
    let workers: Vec<String> = cfg.workers().map(|p| format!("{} {d}({}){r}", p.name, p.binding.key())).collect();
    let msg = format!(
        "{c}{b}zilch{r} {d}· {}{r}\n  master  {d}{}{r}\n  workers {}\n  {d}max {} parallel · approval: {:?} · /help for commands{r}",
        cfg.cwd.display(),
        cfg.master().binding.key(),
        workers.join(", "),
        cfg.max_workers,
        cfg.approval,
    );
    s.line(&msg);
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
/clear            forget everything: master history, workers, TODOs
/exit             quit"
            .to_string(),
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
        "/clear" => {
            team.clear().await;
            "Cleared master history, workers and TODOs.".into()
        }
        _ => format!("Unknown command {cmd}. Try /help."),
    };
    screen.lock().await.line(&out);
    true
}
