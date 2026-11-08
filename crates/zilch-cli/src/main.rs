mod approver;
mod render;

use anyhow::{bail, Result};
use clap::Parser;
use render::{Screen, SharedScreen, BOLD, CYAN, DIM, GREEN, RED, RESET};
use std::io::{IsTerminal, Read};
use std::sync::Arc;
use zilch_core::team::todo::TodoStatus;
use zilch_core::*;

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
    if let Err(e) = real_main().await {
        eprintln!("error: {e:#}");
        std::process::exit(1);
    }
}

async fn real_main() -> Result<()> {
    let cli = Cli::parse();
    let cwd = match &cli.cwd {
        Some(p) => std::fs::canonicalize(p)?,
        None => std::env::current_dir()?,
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
    };
    let cfg = Config::load(&cwd, &ov)?;
    let providers = cfg.build_providers()?;

    let one_shot = cli.prompt.is_some();
    let tty = std::io::stdout().is_terminal();
    // In -p mode progress goes to stderr and the final answer to stdout.
    let out: Box<dyn std::io::Write + Send> = if one_shot { Box::new(std::io::stderr()) } else { Box::new(std::io::stdout()) };
    let color = if one_shot { std::io::stderr().is_terminal() } else { tty } && std::env::var_os("NO_COLOR").is_none();
    let screen = Screen::new(out, color, cli.verbose, !one_shot);

    let (sink, rx) = EventSink::new();
    let renderer = render::spawn_renderer(screen.clone(), rx);
    let approver: Arc<dyn Approver> = Arc::new(approver::TerminalApprover(screen.clone()));
    let reviewer: Arc<dyn PlanReviewer> = Arc::new(approver::TerminalReviewer(screen.clone()));
    let team = Arc::new(Team::new(cfg, providers, approver, reviewer, sink)?);

    // Ctrl-C cancels the running turn (master, scheduler and every worker).
    {
        let team = team.clone();
        tokio::spawn(async move {
            while tokio::signal::ctrl_c().await.is_ok() {
                team.cancel();
            }
        });
    }

    if let Some(mut prompt) = cli.prompt {
        if !std::io::stdin().is_terminal() {
            let mut piped = String::new();
            std::io::stdin().read_to_string(&mut piped)?;
            if !piped.trim().is_empty() {
                prompt = format!("{prompt}\n\n<stdin>\n{piped}\n</stdin>");
            }
        }
        let res = team.run(&prompt).await;
        // The Ctrl-C task keeps the event sink alive, so give the renderer a moment instead of joining it.
        tokio::time::sleep(std::time::Duration::from_millis(30)).await;
        renderer.abort();
        println!("{}", res?);
        return Ok(());
    }

    repl(team, screen).await
}

async fn repl(team: Arc<Team>, screen: SharedScreen) -> Result<()> {
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
            Err(e) => format!("{}error:{} {e:#}", s.c(RED), s.c(RESET)),
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
