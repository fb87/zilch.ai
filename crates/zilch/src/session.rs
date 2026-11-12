//! One zilch session: a workspace's config, drivers, agents and kernel, its view of the events,
//! and what a typed line does. It runs inside the REPL process (plain `zilch`) or inside the
//! daemon (`zilch daemon`), and is driven the same way in both through [`Link`]: lines in, rendered
//! output and status out.

use crate::{
    about, ipc, model_cmd, models_list, open_log, ps, recap, sessions_list, spec_status, terminal_width, View, CYAN, DIM, HELP, RESET,
    YELLOW,
};
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use zilch_agents::Book;
use zilch_drivers::config::{Config, Overrides};
use zilch_drivers::{DriverSet, ARCHITECT_TOOLS, DEVELOPER_TOOLS};
use zilch_ipc::{EventBus, Group, KernelEvent, Pid, TicketId};
use zilch_kernel::{Cmd, KernelConfig, KernelHandle};

/// How to open a session (from the command line, or sent to the daemon by a client).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SessionOpts {
    pub workspace: PathBuf,
    pub model: Option<String>,
    pub architect_model: Option<String>,
    pub developer_model: Option<String>,
    pub web: bool,
    /// "ask" | "plan" | "auto" (None: the config's).
    pub approval: Option<String>,
    pub workers: Option<usize>,
    pub log: Option<PathBuf>,
    pub no_log: bool,
    pub r#continue: bool,
    pub resume: Option<String>,
    pub no_session: bool,
    pub verbose: bool,
    /// ANSI colours in the rendered output.
    pub color: bool,
    /// Serve the editor link for this workspace.
    pub editor_link: bool,
}

/// What a front end shows besides the output: the status row and the prompt.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct UiState {
    pub status: String,
    pub busy: bool,
    /// The oldest question waiting: (qid, ticket, is a command).
    pub question: Option<(u64, TicketId, bool)>,
    /// Typing into the spec lane.
    pub spec: bool,
    /// For completion.
    pub pids: Vec<Pid>,
    pub models: Vec<String>,
}

/// Output for the front ends.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "t")]
pub enum Out {
    /// A rendered line (or several).
    Line {
        text: String,
    },
    Opened {
        ticket: TicketId,
        text: String,
    },
    /// A question was asked; `ask`: it waits for an answer (not auto-approved).
    Question {
        qid: u64,
        ticket: TicketId,
        command: bool,
        ask: bool,
    },
    Served {
        ticket: TicketId,
        ok: bool,
    },
    /// The session ended (stopped, or the daemon shut down).
    Ended {
        reason: String,
    },
}

/// How many rendered lines a session keeps for clients that attach later.
const BACKLOG: usize = 2_000;

/// Fans a session's output out to its subscribers and keeps a backlog of rendered lines.
#[derive(Default)]
pub struct Hub {
    inner: Mutex<HubInner>,
    attached: std::sync::atomic::AtomicUsize,
}

#[derive(Default)]
struct HubInner {
    backlog: VecDeque<String>,
    subs: Vec<Sender<Out>>,
}

impl Hub {
    pub fn publish(&self, out: Out) {
        let mut h = self.inner.lock().unwrap();
        if let Out::Line { text } = &out {
            h.backlog.push_back(text.clone());
            while h.backlog.len() > BACKLOG {
                h.backlog.pop_front();
            }
        }
        h.subs.retain(|s| s.send(out.clone()).is_ok());
    }

    /// A new subscriber, first sent the last `replay` rendered lines.
    pub fn subscribe(&self, replay: usize) -> Receiver<Out> {
        let (tx, rx) = std::sync::mpsc::channel();
        let mut h = self.inner.lock().unwrap();
        let skip = h.backlog.len().saturating_sub(replay);
        for text in h.backlog.iter().skip(skip) {
            let _ = tx.send(Out::Line { text: text.clone() });
        }
        h.subs.push(tx);
        rx
    }

    /// A client is attached while the guard lives (the `on_question` hook runs only when none is).
    pub fn attach(self: &Arc<Self>) -> Attached {
        self.attached.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Attached(self.clone())
    }

    pub fn listeners(&self) -> usize {
        self.attached.load(std::sync::atomic::Ordering::SeqCst)
    }
}

pub struct Attached(Arc<Hub>);

impl Drop for Attached {
    fn drop(&mut self) {
        self.0.attached.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
    }
}

pub struct Session {
    pub cwd: PathBuf,
    pub cfg: Config,
    pub banner: String,
    pub view: Arc<View>,
    pub hub: Arc<Hub>,
    kernel: KernelHandle,
    drivers: Arc<DriverSet>,
    book: Arc<Book>,
    meter: Arc<zilch_drivers::llm::catalog::CostMeter>,
    providers: Vec<String>,
    configured: (zilch_ipc::ModelBinding, zilch_ipc::ModelBinding),
    sessions: Option<PathBuf>,
    session_path: Option<PathBuf>,
    _ipc: Option<ipc::IpcGuard>,
}

impl Session {
    pub fn open(opts: &SessionOpts) -> anyhow::Result<Session> {
        let cwd = opts.workspace.canonicalize().unwrap_or_else(|_| opts.workspace.clone());
        if !cwd.is_dir() {
            anyhow::bail!("workspace {} is not a directory", cwd.display());
        }
        let ov = Overrides {
            model: opts.model.clone(),
            architect_model: opts.architect_model.clone(),
            developer_model: opts.developer_model.clone(),
            web: opts.web.then_some(true),
        };
        let cfg = Config::load(&cwd, &ov)?;
        // Prices and context sizes from the providers (best effort, a few seconds at most), the
        // session caps, and a client for every usable provider (so /model can switch).
        let meter = Arc::new(zilch_drivers::llm::catalog::CostMeter::new(cfg.max_session_cost));
        *meter.catalog.write().unwrap() = cfg.discover_catalog();
        let mut drivers = DriverSet::new(&cwd, cfg.policy.clone(), cfg.backends())
            .with_costs(meter.clone(), cfg.max_session_tokens)
            .with_spec_dir(cfg.spec_dir.clone())
            .with_web(&cfg.web);
        let providers = drivers.providers();
        let configured = (cfg.architect.clone(), cfg.developer.clone());
        // Sessions: the main agent's conversation is saved after every served order; -c continues
        // the most recent one of this directory.
        let sessions =
            (!opts.no_session).then(zilch_drivers::session::default_root).flatten().map(|r| zilch_drivers::session::dir_for(&r, &cwd));
        let mut resumed: Option<(PathBuf, Vec<zilch_ipc::Message>)> = None;
        let mut session_note = String::new();
        let mut session_path: Option<PathBuf> = None;
        if let Some(dir) = &sessions {
            let earlier = match &opts.resume {
                Some(id) => {
                    let p = dir.join(format!("{id}.json"));
                    if !p.exists() {
                        anyhow::bail!("no session '{id}' in {} (see zilch --sessions)", cwd.display());
                    }
                    Some(p)
                }
                None => opts.r#continue.then(|| zilch_drivers::session::latest(dir)).flatten(),
            };
            let loaded = earlier.as_ref().and_then(|p| zilch_drivers::session::load(p)).and_then(|v| zilch_agents::snapshot_messages(&v));
            let path = match (earlier, loaded) {
                (Some(p), Some(msgs)) => {
                    resumed = Some((p.clone(), msgs));
                    p
                }
                _ => {
                    if opts.r#continue || opts.resume.is_some() {
                        session_note = "no earlier session in this directory; starting a new one".into();
                    }
                    zilch_drivers::session::new_path(dir)
                }
            };
            session_path = Some(path.clone());
            drivers = drivers.with_session_file(path);
        }
        let mut book =
            Book::new(cfg.architect.clone(), cfg.developer.clone(), drivers.specs(ARCHITECT_TOOLS), drivers.specs(DEVELOPER_TOOLS), &cwd)
                .with_spec(drivers.specs(zilch_drivers::SPEC_TOOLS), &cfg.spec_dir);
        if let Some(t) = cfg.main_context_tokens {
            book.main_context_tokens = t;
        }
        book.explorer_tools = drivers.specs(zilch_drivers::EXPLORER_TOOLS);
        if cfg.repo_map {
            *book.repo_overview.write().unwrap() = drivers.repo_overview(10_000);
        }
        // Prompt overrides and custom subagent profiles from the roles.
        for (g, o) in &cfg.prompts {
            book.prompts.insert(*g, (o.text.clone(), o.replace));
        }
        for p in &cfg.profiles {
            let tools: Vec<&str> = p.tools.iter().map(String::as_str).collect();
            book.profiles.push(zilch_agents::SubagentProfile {
                name: p.name.clone(),
                description: p.description.clone(),
                binding: p.binding.clone(),
                tools: drivers.specs(&tools),
                prompt: p.prompt.as_ref().map(|o| (o.text.clone(), o.replace)),
            });
        }
        let mut first_ticket = 1;
        if let Some((path, msgs)) = &resumed {
            let orders: Vec<&str> = msgs.iter().map(|m| m.text()).filter(|t| t.starts_with("Order #")).collect();
            first_ticket = orders.iter().filter_map(|t| t[7..].split(':').next()?.parse::<u64>().ok()).max().unwrap_or(0) + 1;
            let last: String = orders.last().map(|t| t.split_once(": ").map(|x| x.1).unwrap_or(t)).unwrap_or("").chars().take(70).collect();
            let id = path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            session_note = format!("↺ continuing session {id}: {} order(s), last: {last}", orders.len());
            *book.resume.lock().unwrap() = Some(msgs.clone());
            // The spec lane's conversation too, when the session had one.
            *book.resume_spec.lock().unwrap() = zilch_drivers::session::load_spec(path).and_then(|v| zilch_agents::snapshot_messages(&v));
        }
        let book = Arc::new(book);
        let bus = Arc::new(EventBus::new());
        let log = open_log(opts.log.clone(), opts.no_log);
        if let Some((_, file)) = &log {
            bus.set_log(Box::new(file.try_clone()?));
        }
        let events = bus.subscribe(4096);
        let approval = match opts.approval.as_deref() {
            Some(a @ ("ask" | "plan" | "auto")) => a.to_string(),
            Some(other) => anyhow::bail!("invalid --approval '{other}' (ask | plan | auto)"),
            None => cfg.approval.clone(),
        };
        let defaults = KernelConfig::default();
        let kcfg = KernelConfig {
            workers: opts.workers.unwrap_or(defaults.workers),
            io_threads: cfg.io_threads,
            loop_limit: cfg.loop_limit.unwrap_or(defaults.loop_limit),
            max_steps: cfg.max_steps,
            max_tokens: cfg.max_task_tokens,
            group_tokens: cfg.group_tokens.clone(),
            first_ticket,
            auto_approve: approval == "auto",
            ask_commands: approval == "ask",
            ..defaults
        };
        let view = Arc::new(View {
            color: opts.color,
            verbose: opts.verbose,
            procs: Mutex::default(),
            questions: Mutex::default(),
            commands: Mutex::default(),
            open: Mutex::default(),
            live: Mutex::default(),
            note: Mutex::default(),
            on_served: Default::default(),
            started: Instant::now(),
            costs: Arc::new(crate::costs::Costs::new(
                meter.clone(),
                book.clone(),
                cfg.expensive_model_price,
                cfg.expensive_model_minutes,
                cfg.expensive_model_spend,
            )),
            last_summary: Mutex::default(),
            orders: Mutex::default(),
            lane: Mutex::new(Group::Architecture),
            lanes_of: Mutex::default(),
        });
        // A continued session's earlier orders count for /recap.
        if let Some((_, msgs)) = &resumed {
            let mut orders = view.orders.lock().unwrap();
            for t in msgs.iter().map(|m| m.text()).filter(|t| t.starts_with("Order #")) {
                let (num, text) = t[7..].split_once(": ").unwrap_or(("0", t));
                orders.push((num.parse().unwrap_or(0), text.to_string(), Some("(earlier session)".into())));
            }
        }
        let (d, r) = (view.c(DIM), view.c(RESET));
        let (_, sandbox) = zilch_drivers::tools::sandbox::status(cfg.policy.bash_sandbox, cfg.policy.bash.network);
        let mut banner = format!(
            "{}zilch {}{} {d}· main {} · subagents {} · {} workers{r}\n{d}workspace: {}{r}\n{d}commands: {sandbox}{r}{}",
            view.c(crate::BOLD),
            env!("CARGO_PKG_VERSION"),
            r,
            cfg.architect.key(),
            cfg.developer.key(),
            kcfg.workers,
            cwd.display(),
            log.as_ref().map(|(p, _)| format!("\n{d}event log: {}{r}", p.display())).unwrap_or_default()
        );
        if !session_note.is_empty() {
            banner = format!("{banner}\n{}{session_note}{r}", view.c(CYAN));
        }
        let drivers = Arc::new(drivers);
        if cfg.repo_map {
            // Between orders: the next planner / answerer / spec agent sees the current tree (the
            // text changes only when the tree did, so their cached prompt prefix otherwise stays
            // valid).
            let (d, b) = (drivers.clone(), book.clone());
            let _ = view.on_served.set(Box::new(move || {
                let fresh = d.repo_overview(10_000);
                let mut current = b.repo_overview.write().unwrap();
                if *current != fresh {
                    *current = fresh;
                }
            }));
        }
        let kernel = zilch_kernel::start(kcfg, drivers.clone(), book.clone(), bus);
        let hub = Arc::new(Hub::default());
        // The event pump: track and render every event, publish the result.
        {
            let (v, h, ask) = (view.clone(), hub.clone(), approval != "auto");
            let hook = cfg.on_question.clone().map(|c| (c, cwd.clone()));
            std::thread::Builder::new().name("zilch-view".into()).spawn(move || {
                for env in events.iter() {
                    v.track(&env.event);
                    if let Some(s) = v.render(&env.event) {
                        h.publish(Out::Line { text: s });
                    }
                    for a in v.costs.announcements() {
                        h.publish(Out::Line { text: format!("{}{a}{}", v.c(YELLOW), v.c(RESET)) });
                    }
                    match &env.event {
                        KernelEvent::TicketOpened { ticket, text, .. } => h.publish(Out::Opened { ticket: *ticket, text: text.clone() }),
                        KernelEvent::Question { qid, ticket, command, text, .. } => {
                            h.publish(Out::Question { qid: *qid, ticket: *ticket, command: *command, ask });
                            if let (true, Some((cmd, ws))) = (ask, &hook) {
                                if h.listeners() == 0 {
                                    run_hook(cmd, ws, *ticket, text);
                                }
                            }
                        }
                        KernelEvent::TicketServed { ticket, ok, .. } => h.publish(Out::Served { ticket: *ticket, ok: *ok }),
                        _ => {}
                    }
                }
            })?;
        }
        // Editor link (Neovim: :ZilchSend): a selection plus a comment arrives over a local socket
        // and becomes an order in the requested lane (or the active one).
        let mut guard = None;
        if opts.editor_link {
            let commander = kernel.commander();
            let (v_submit, v_status, h_submit, spec_dir) = (view.clone(), view.clone(), hub.clone(), cfg.spec_dir.clone());
            let hooks = Arc::new(ipc::Hooks {
                submit: Box::new(move |lane, msg| {
                    let lane = match lane {
                        None => *v_submit.lane.lock().unwrap(),
                        Some("code") => Group::Architecture,
                        Some("spec") => Group::Spec,
                        Some(l) => return Err(format!("unknown lane '{l}' (code | spec)")),
                    };
                    let name = if lane == Group::Spec { "spec" } else { "code" };
                    let first: String = msg.lines().last().unwrap_or("").chars().take(100).collect();
                    let loc =
                        msg.lines().next().and_then(|l| l.strip_prefix("[From the editor] ")).map(|l| format!(" {l}")).unwrap_or_default();
                    h_submit.publish(Out::Line {
                        text: format!("{}◆ editor → {name} lane{loc}:{} {first}", v_submit.c(CYAN), v_submit.c(RESET)),
                    });
                    commander.send(Cmd::OrderTo(lane, msg));
                    Ok(name.to_string())
                }),
                status: Box::new(move || {
                    let live = v_status.live.lock().unwrap();
                    let state = |g: Group| {
                        let busy = live.values().any(|l| l.group == g && (l.state == "running" || l.state == "blocked(io)"));
                        if busy {
                            "working"
                        } else {
                            "idle"
                        }
                    };
                    let active = if *v_status.lane.lock().unwrap() == Group::Spec { "spec" } else { "code" };
                    serde_json::json!({"ok": true, "active": active, "code": state(Group::Architecture), "spec": state(Group::Spec), "docs": spec_status(&spec_dir)})
                }),
            });
            match ipc::serve(hooks, &cwd) {
                Ok(Some(g)) => {
                    banner.push_str(&format!("\n{d}editor link: {} (Neovim: :ZilchSend){r}", g.endpoint));
                    guard = Some(g);
                }
                Ok(None) => banner.push_str(&format!("\n{d}editor link: another zilch already serves this workspace; not listening{r}")),
                Err(e) => banner.push_str(&format!("\neditor link unavailable: {e}")),
            }
        }
        Ok(Session {
            cwd,
            cfg,
            banner,
            view,
            hub,
            kernel,
            drivers,
            book,
            meter,
            providers,
            configured,
            sessions,
            session_path,
            _ipc: guard,
        })
    }

    /// Place an order in the code lane (as `-p` does).
    pub fn order(&self, text: &str) {
        self.kernel.send(Cmd::Order(text.to_string()));
    }

    pub fn answer(&self, qid: u64, text: &str) {
        self.view.questions.lock().unwrap().retain(|(q, _)| *q != qid);
        self.view.commands.lock().unwrap().remove(&qid);
        self.kernel.send(Cmd::Answer { qid, text: text.to_string() });
    }

    /// Ctrl-C: cancel the newest open order.
    pub fn interrupt(&self) -> String {
        match self.view.latest_open() {
            Some(t) => {
                self.kernel.send(Cmd::KillTicket(t));
                format!("⏹ cancelling order #{t}…")
            }
            None => "(Ctrl-D to quit)".into(),
        }
    }

    pub fn state(&self) -> UiState {
        let (status, busy) = self.view.status_line();
        let question = self.view.question().map(|(q, t)| (q, t, self.view.commands.lock().unwrap().contains_key(&q)));
        UiState {
            status,
            busy,
            question,
            spec: *self.view.lane.lock().unwrap() == Group::Spec,
            pids: self.view.live.lock().unwrap().keys().copied().collect(),
            models: self.meter.catalog.read().unwrap().keys().filter_map(|k| k.split_once('/').map(|(_, m)| m.to_string())).collect(),
        }
    }

    /// Spend so far, open orders and the session's id, for listings.
    pub fn summary(&self) -> (f64, usize, Option<String>) {
        let id = self.session_path.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned());
        (self.meter.spent(), self.view.open.lock().unwrap().len(), id)
    }

    /// The session's totals (printed when it ends).
    pub fn totals(&self) -> Vec<String> {
        [self.view.costs.session_summary(), self.view.costs.parts_report()].into_iter().flatten().collect()
    }

    pub fn shutdown(self) -> Vec<String> {
        let t = self.totals();
        self.kernel.shutdown();
        t
    }

    /// What a typed line does. `asked`: the question the prompt showed when the line was typed;
    /// `status_row`: the front end shows the status row (short notes go there, not to the output).
    pub fn handle(&mut self, line: &str, asked: Option<(u64, TicketId)>, status_row: bool) -> String {
        let view = self.view.clone();
        let mut out = String::new();
        macro_rules! say {
            ($($a:tt)*) => {{ out.push_str(&format!($($a)*)); out.push('\n'); }};
        }
        let note = |out: &mut String, msg: &str| {
            if status_row {
                view.note(msg);
            } else {
                out.push_str(&format!("({msg})\n"));
            }
        };
        let input = line.trim();
        // A question that arrived while this prompt was already drawn: a lone y / n / a is its
        // answer (almost never meant as an order); anything longer still needs the question's
        // prompt.
        let short_answer = matches!(input.to_lowercase().as_str(), "y" | "yes" | "n" | "no" | "a" | "always");
        let asked = asked.or_else(|| view.question().filter(|_| short_answer));
        // The order may have ended (cancelled, killed) after this prompt was drawn.
        let asked = asked.filter(|(_, t)| view.open.lock().unwrap().contains(t));
        if let (Some(_), true) = (asked, input.is_empty()) {
            // Approval must be explicit: a stray Enter does not approve the plan.
            note(&mut out, "still waiting for your answer: y, n, or (plans) feedback / (commands) a");
            return out;
        }
        if let (Some((qid, _)), false) = (asked, input.starts_with('/') || input.starts_with('@')) {
            self.answer(qid, input);
            return out;
        }
        if input.is_empty() {
            // Nothing to order: say so instead of silently doing nothing.
            let msg = if view.question().is_some() { "empty input ignored — a plan is waiting" } else { "empty input ignored" };
            note(&mut out, msg);
            return out;
        }
        if let Some(rest) = input.strip_prefix('@') {
            let (who, text) = rest.split_once(char::is_whitespace).map(|(w, t)| (w, t.trim())).unwrap_or((rest, ""));
            let msg = match zilch_ipc::Target::parse(who) {
                None => format!("@{who}: not a pid or group (architect, developer)"),
                Some(_) if text.is_empty() => format!("@{who}: nothing to send"),
                Some(target) => match self.kernel.tell(target, text) {
                    Ok(how) => {
                        say!("  {}✉ you → @{who}: {text}{}", view.c(DIM), view.c(RESET));
                        format!("✉ {how}")
                    }
                    Err(e) => format!("@{who}: {e}"),
                },
            };
            note(&mut out, &msg);
            return out;
        }
        let (cmd, arg) = input.split_once(' ').map(|(c, a)| (c, a.trim())).unwrap_or((input, ""));
        match cmd {
            "/help" => say!("{HELP}"),
            "/ps" => say!("{}", ps(&self.kernel)),
            "/kill" => match arg.parse::<Pid>() {
                Ok(pid) => self.kernel.send(Cmd::Kill(pid)),
                Err(_) => say!("usage: /kill <pid>  (see /ps)"),
            },
            "/cancel" => match view.latest_open() {
                Some(t) => self.kernel.send(Cmd::KillTicket(t)),
                None => say!("No open order."),
            },
            "/spec" | "/code" => {
                let lane = if cmd == "/spec" { Group::Spec } else { Group::Architecture };
                *view.lane.lock().unwrap() = lane;
                let msg = match lane {
                    Group::Spec => {
                        format!("→ spec lane: requirements engineering (spec, design, plan). {}", spec_status(&self.cfg.spec_dir))
                    }
                    _ => "→ code lane: planning and implementation (works from the spec lane's design.md and plan.md).".to_string(),
                };
                say!("{}{msg}{}", view.c(CYAN), view.c(RESET));
            }
            "/status" => {
                let lane = if *view.lane.lock().unwrap() == Group::Spec { "spec" } else { "code" };
                say!("typing into: {lane} lane\n{}\n{}", ps(&self.kernel).lines().next().unwrap_or(""), spec_status(&self.cfg.spec_dir));
            }
            "/usage" => say!("{}", view.costs.report()),
            "/sessions" => match &self.sessions {
                Some(dir) => say!("{}", sessions_list(dir, self.session_path.as_ref())),
                None => say!("Sessions are off (--no-session)."),
            },
            "/recap" => say!("{}", recap(&view, self.session_path.as_ref())),
            "/clear" => {
                // A fresh session: the main agent (and any order it is on) stops; the next order
                // starts a new one with an empty context, saved to a new session file.
                let residents: Vec<Pid> = view
                    .live
                    .lock()
                    .unwrap()
                    .iter()
                    .filter(|(_, l)| matches!(l.group, Group::Architecture | Group::Spec))
                    .map(|(p, _)| *p)
                    .collect();
                for pid in residents {
                    self.kernel.send(Cmd::Kill(pid));
                }
                let old = self.session_path.as_ref().and_then(|p| p.file_stem()).map(|s| s.to_string_lossy().into_owned());
                self.session_path = self.sessions.as_ref().map(|d| zilch_drivers::session::new_path(d));
                self.drivers.set_session_file(self.session_path.clone());
                view.orders.lock().unwrap().clear();
                *view.last_summary.lock().unwrap() = None;
                // A resumed conversation not yet handed to a main agent must not reach the new one.
                *self.book.resume.lock().unwrap() = None;
                *self.book.resume_spec.lock().unwrap() = None;
                let resume = old.map(|id| format!(" (the previous one stays resumable: --resume {id})")).unwrap_or_default();
                say!("Fresh session: the next order starts a new main agent with an empty context{resume}.");
            }
            "/models" => say!("{}", models_list(&self.book, &self.meter, &self.configured)),
            "/model" => say!("{}", model_cmd(arg, &self.book, &self.meter, &self.providers, &self.configured)),
            "/ack" => {
                let msg = if view.costs.ack() == 0 {
                    "no high-cost model warning to acknowledge"
                } else {
                    "high-cost model warning silenced until its next step"
                };
                note(&mut out, msg);
            }
            "/prompt" => match arg {
                a if a.starts_with("dev") => say!("{}", self.book.system_prompt(Group::Development)),
                a if a.starts_with("plan") => say!("{}", self.book.subagent_prompt("plan").unwrap_or_default()),
                a if a.starts_with("answer") => say!("{}", self.book.subagent_prompt("answer").unwrap_or_default()),
                a if a.starts_with("spec") => say!("{}", self.book.system_prompt(Group::Spec)),
                _ => say!("{}", self.book.system_prompt(Group::Architecture)),
            },
            "/about" => say!("{}", about(&self.cwd, &self.cfg.sources, self.session_path.as_ref())),
            "/redraw" => {
                if view.color {
                    out.push_str("\x1b[H\x1b[2J");
                }
                match view.last_summary.lock().unwrap().clone() {
                    Some(s) => {
                        let width = terminal_width().saturating_sub(2).max(20);
                        say!("{}── last answer, redrawn at {width} columns ──{}", view.c(DIM), view.c(RESET));
                        say!("{}", crate::markdown::Markdown::new(view.color).with_indent(2).text(s.trim(), width));
                    }
                    None => say!("(no answer yet)"),
                }
            }
            c if c.starts_with('/') => say!("Unknown command {c}. /help lists them."),
            _ => self.kernel.send(Cmd::OrderTo(*view.lane.lock().unwrap(), input.to_string())),
        }
        out
    }
}

/// `on_question`: tell someone a question waits while no client is attached (detached, no wait).
fn run_hook(cmd: &str, workspace: &std::path::Path, ticket: TicketId, text: &str) {
    let first: String = text.lines().find(|l| !l.trim().is_empty()).unwrap_or("").chars().take(200).collect();
    let _ = std::process::Command::new("sh")
        .arg("-c")
        .arg(cmd)
        .current_dir(workspace)
        .env("ZILCH_WORKSPACE", workspace)
        .env("ZILCH_TICKET", ticket.to_string())
        .env("ZILCH_QUESTION", first)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();
}

/// What a front end drives: an in-process session or one in the daemon.
pub trait Link: Send + Sync {
    /// A typed line; returns its output (may be empty).
    fn input(&self, line: &str, asked: Option<(u64, TicketId)>, status_row: bool) -> String;
    fn interrupt(&self) -> String;
    fn state(&self) -> UiState;
    /// Output from now on, after the last `replay` lines.
    fn subscribe(&self, replay: usize) -> Receiver<Out>;
    fn order(&self, text: &str);
    fn answer(&self, qid: u64, text: &str);
    /// The session's totals (the session line, input by part).
    fn totals(&self) -> Vec<String>;
    /// The session is gone (the front end should leave).
    fn closed(&self) -> bool {
        false
    }
}

/// A session in this process (None once stopped).
pub struct Local(pub Mutex<Option<Session>>, pub Arc<Hub>);

impl Local {
    pub fn new(s: Session) -> Self {
        let hub = s.hub.clone();
        Local(Mutex::new(Some(s)), hub)
    }

    fn with<T: Default>(&self, f: impl FnOnce(&mut Session) -> T) -> T {
        self.0.lock().unwrap().as_mut().map(f).unwrap_or_default()
    }

    /// End the session: its clients are told, its kernel stops. Returns its totals.
    pub fn stop(&self, reason: &str) -> Vec<String> {
        let Some(s) = self.0.lock().unwrap().take() else { return vec![] };
        self.1.publish(Out::Ended { reason: reason.to_string() });
        s.shutdown()
    }
}

impl Link for Local {
    fn input(&self, line: &str, asked: Option<(u64, TicketId)>, status_row: bool) -> String {
        self.with(|s| s.handle(line, asked, status_row))
    }
    fn interrupt(&self) -> String {
        self.with(|s| s.interrupt())
    }
    fn state(&self) -> UiState {
        self.with(|s| s.state())
    }
    fn subscribe(&self, replay: usize) -> Receiver<Out> {
        self.1.subscribe(replay)
    }
    fn order(&self, text: &str) {
        self.with(|s| s.order(text))
    }
    fn answer(&self, qid: u64, text: &str) {
        self.with(|s| s.answer(qid, text))
    }
    fn totals(&self) -> Vec<String> {
        self.with(|s| s.totals())
    }
    fn closed(&self) -> bool {
        self.0.lock().unwrap().is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_hub_replays_recent_lines_and_counts_attached_clients() {
        let hub = Arc::new(Hub::default());
        for i in 0..5 {
            hub.publish(Out::Line { text: format!("l{i}") });
        }
        hub.publish(Out::Served { ticket: 1, ok: true }); // not kept in the backlog
        let rx = hub.subscribe(2);
        hub.publish(Out::Line { text: "live".into() });
        let got: Vec<Out> = rx.try_iter().collect();
        let lines: Vec<String> = got.iter().filter_map(|o| if let Out::Line { text } = o { Some(text.clone()) } else { None }).collect();
        assert_eq!(lines, ["l3", "l4", "live"]);
        assert_eq!(hub.listeners(), 0);
        let a = hub.attach();
        let b = hub.attach();
        assert_eq!(hub.listeners(), 2);
        drop(a);
        assert_eq!(hub.listeners(), 1);
        drop(b);
        // A dropped subscriber is forgotten on the next publish.
        drop(rx);
        hub.publish(Out::Line { text: "x".into() });
        assert!(hub.inner.lock().unwrap().subs.is_empty());
    }

    #[test]
    fn the_backlog_is_bounded() {
        let hub = Hub::default();
        for i in 0..BACKLOG + 10 {
            hub.publish(Out::Line { text: i.to_string() });
        }
        let rx = hub.subscribe(usize::MAX);
        let first = rx.try_iter().next();
        assert_eq!(first, Some(Out::Line { text: "10".into() }));
    }
}
