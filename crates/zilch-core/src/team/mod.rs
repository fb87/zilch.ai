//! The team: one master + a pool of parallel worker instances, coordinated through a TODO list.

pub mod file_locks;
pub mod plan_state;
pub mod pool;
pub mod scheduler;
pub mod session;
pub mod todo;

use crate::agent::{Agent, ApprovalGate, Approver, EventKind, EventSink, TurnHooks};
use crate::config::{Config, READONLY_TOOLS};
use crate::llm::{Message, ModelBinding, Providers};
use crate::profile::Profile;
use crate::prompt::{self, PromptEnv};
use crate::tools::spec_tools::{Doc, EditDoc, ReleaseSpec, SpecStore, WriteDoc};
use crate::tools::{self, CancelCell, Scope, TokenBudget, ToolCtx, ToolPolicy, ToolRegistry};
use anyhow::{anyhow, bail, Result};
use file_locks::FileLocks;
use plan_state::PlanReviewer;
use pool::{Instance, WorkerPool};
use std::sync::{Arc, Mutex};
use todo::{Todo, TodoList, TodoStatus};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

/// State shared by the master's coordination tools, the scheduler and the pool.
pub struct TeamShared {
    pub cfg: Config,
    pub providers: Providers,
    pub todos: Mutex<TodoList>,
    pub pool: WorkerPool,
    pub reviewer: Arc<dyn PlanReviewer>,
    pub gate: Arc<ApprovalGate>,
    pub events: EventSink,
    pub cancel: CancelCell,
    pub file_locks: FileLocks,
    pub policy: Arc<ToolPolicy>,
    pub budget: Option<Arc<TokenBudget>>,
    pub meter: Arc<crate::llm::catalog::CostMeter>,
    persist: Mutex<Persist>,
    worker_tools: ToolRegistry,
    /// The spec lane's documents (spec.md, design.md, plan.md, versions/).
    pub spec_store: Arc<SpecStore>,
    /// Team-level loop protection: failed runs per unchanged task (worker + instructions), with
    /// the last failure report. See scheduler::task_key.
    pub task_failures: Mutex<std::collections::HashMap<u64, (usize, String)>>,
    /// Runtime model switches (/model): profile name ("master", "spec", "coder", …) → binding.
    pub model_overrides: Mutex<std::collections::HashMap<String, ModelBinding>>,
    /// High-cost model warnings the user acknowledged: model key → (minutes step, spend step).
    cost_acked: Mutex<std::collections::HashMap<String, (u64, u64)>>,
}

impl TeamShared {
    pub fn todos_changed(&self) {
        let snap = self.todos.lock().unwrap().items.clone();
        self.events.emit("master", EventKind::TodosChanged(snap));
    }

    pub fn set_todo_status(&self, id: &str, status: TodoStatus, instance: Option<String>, report: Option<String>) {
        {
            let mut l = self.todos.lock().unwrap();
            if let Some(t) = l.get_mut(id) {
                t.status = status.clone();
                if instance.is_some() {
                    t.instance = instance.clone();
                }
                if report.is_some() {
                    t.report = report;
                }
            }
        }
        self.events.emit("master", EventKind::TodoStatus { id: id.to_string(), status, instance });
        self.todos_changed();
    }

    /// Save a mid-turn checkpoint (called after each TODO / delegation finishes) so a crash during
    /// a long run_todos loses at most the work in flight. Busy workers are recorded with their last
    /// known state; in-progress TODOs are reset to pending on load.
    pub fn checkpoint(&self) {
        let (store, master) = {
            let p = self.persist.lock().unwrap();
            (p.store.clone(), p.master_checkpoint.clone())
        };
        let (Some(store), Some(master)) = (store, master) else { return };
        let mut workers = Vec::new();
        for inst in self.pool.list() {
            let state = match inst.agent.try_lock() {
                Ok(a) => Some(session::WorkerState {
                    id: inst.id.clone(),
                    profile: inst.profile.clone(),
                    binding: inst.binding.clone(),
                    readonly: inst.readonly,
                    agent: session::AgentState { history: a.history.clone(), session: a.session.clone(), usage: a.usage },
                }),
                Err(_) => self.persist.lock().unwrap().worker_cache.get(&inst.id).cloned(),
            };
            if let Some(s) = state {
                self.persist.lock().unwrap().worker_cache.insert(s.id.clone(), s.clone());
                workers.push(s);
            }
        }
        let p = self.persist.lock().unwrap();
        let state = session::SessionState {
            version: session::FORMAT_VERSION,
            id: p.id.clone(),
            cwd: self.cfg.cwd.clone(),
            created_ms: p.created_ms,
            updated_ms: session::now_ms(),
            title: p.title.clone(),
            master,
            // Latest known spec-lane state (refreshed at the start and end of every spec turn).
            spec: p.spec_checkpoint.clone().filter(|s| !s.history.is_empty()),
            workers,
            todos: self.todos.lock().unwrap().clone(),
            counters: self.pool.counters(),
        };
        drop(p);
        if let Err(e) = store.save(&state) {
            self.events.emit("master", EventKind::Notice(format!("warning: checkpoint failed: {e:#}")));
        }
    }

    pub fn worker_profiles(&self) -> Vec<(String, String)> {
        self.cfg.workers().map(|p| (p.name.clone(), p.description.clone())).collect()
    }

    fn worker_profile(&self, name: &str) -> Result<&Profile> {
        self.cfg.workers().find(|p| p.name == name).ok_or_else(|| {
            anyhow!("unknown worker '{name}'. Available: {}", self.cfg.workers().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "))
        })
    }

    /// The binding a profile runs with now: its /model switch if any, else its configured one.
    pub fn effective_binding(&self, profile: &Profile) -> ModelBinding {
        self.model_overrides.lock().unwrap().get(&profile.name).cloned().unwrap_or_else(|| profile.binding.clone())
    }

    /// Resolve a TODO's model override against the profile's binding.
    pub fn binding_for(&self, profile: &Profile, model: Option<&str>, provider: Option<&str>) -> Result<ModelBinding> {
        let mut b = self.effective_binding(profile);
        if let Some(p) = provider {
            if !self.providers.contains(p) {
                bail!("unknown provider '{p}'");
            }
            b.provider = p.to_string();
        }
        if let Some(m) = model {
            b.model = m.to_string();
        }
        Ok(b)
    }

    /// Spawn a new worker instance: system prompt and model are fixed here, at spawn time.
    fn spawn(&self, profile: &Profile, binding: ModelBinding, readonly: bool) -> Result<Arc<Instance>> {
        let id = self.pool.next_id(&profile.name);
        self.spawn_as(id, profile, binding, readonly)
    }

    /// Spawn with a given instance id (used when restoring a saved session).
    fn spawn_as(&self, id: String, profile: &Profile, binding: ModelBinding, readonly: bool) -> Result<Arc<Instance>> {
        let mut reg = self.worker_tools.subset(&profile.tools);
        if readonly {
            reg = reg.filter(|t| !t.mutating());
        }
        let names = reg.names();
        let system = prompt::build(profile, &PromptEnv { cwd: &self.cfg.cwd, agent_id: &id, tools: &names, workers: &[] });
        let llm = self.providers.get(&binding.provider)?;
        let ctx = Arc::new(
            ToolCtx::new(&id, self.cfg.cwd.clone(), self.gate.clone(), self.events.clone(), self.cancel.clone(), self.file_locks.clone())
                .with_policy(self.policy.clone(), self.budget.clone())
                .with_meter(self.meter.clone()),
        );
        self.events.emit(&id, EventKind::InstanceSpawned { id: id.clone(), model: binding.key() });
        let agent = Agent {
            id: id.clone(),
            profile: profile.name.clone(),
            llm,
            binding: binding.clone(),
            system,
            tools: reg,
            history: vec![],
            max_steps: profile.max_steps,
            loop_limit: self.cfg.loop_limit,
            context_tokens: profile.context_tokens,
            ctx,
            hooks: None,
            usage: Default::default(),
            session: crate::agent::run::new_session_id(&id),
        };
        let inst = Arc::new(Instance { id, profile: profile.name.clone(), binding, readonly, agent: Arc::new(AsyncMutex::new(agent)) });
        self.pool.add(inst.clone());
        Ok(inst)
    }

    /// Get an instance for `worker` (a profile name or a specific instance id) and lock it.
    pub async fn acquire(
        &self,
        worker: &str,
        binding: Option<ModelBinding>,
        readonly: bool,
    ) -> Result<(Arc<Instance>, OwnedMutexGuard<Agent>)> {
        if worker.contains('#') {
            let inst = self.pool.find(worker).ok_or_else(|| anyhow!("no worker instance '{worker}'"))?;
            if readonly && !inst.readonly {
                bail!("'{worker}' has write tools; read-only delegation must target a profile (e.g. 'explorer') or pass a todo_id");
            }
            if !readonly && inst.readonly {
                bail!("'{worker}' is a read-only instance; use its profile name '{}' for this TODO", inst.profile);
            }
            let g = inst.agent.clone().lock_owned().await;
            return Ok((inst, g));
        }
        let profile = self.worker_profile(worker)?;
        let binding = binding.unwrap_or_else(|| self.effective_binding(profile));
        if let Some(found) = self.pool.take_idle(&profile.name, &binding, readonly) {
            return Ok(found);
        }
        let inst = self.spawn(profile, binding, readonly)?;
        let g = inst.agent.clone().lock_owned().await;
        Ok((inst, g))
    }

    /// Run one task on a locked instance with an optional approved write scope.
    pub async fn run_task(&self, agent: &mut Agent, task: &str, scope: Option<Scope>, fresh: bool) -> Result<String> {
        if fresh {
            agent.reset();
        }
        *agent.ctx.scope.lock().unwrap() = scope;
        let res = agent.run(task).await;
        *agent.ctx.scope.lock().unwrap() = None;
        res
    }
}

/// Did the worker report success? (Workers end with `STATUS: done|failed`.)
pub fn report_ok(report: &str) -> bool {
    !report.lines().any(|l| {
        let l = l.trim().to_ascii_lowercase();
        l.starts_with("status:") && l.contains("failed")
    }) && !report.starts_with("[stopped after reaching max_steps")
}

struct MasterHooks {
    shared: Arc<TeamShared>,
    /// Last seen modification time of plan.md, to tell the master when the spec lane updated it.
    plan_seen: Mutex<Option<std::time::SystemTime>>,
}

impl MasterHooks {
    /// What the code lane should know about the spec lane's output: it implements from plan.md
    /// and design.md (never spec.md), so report their versions, staleness and fresh updates.
    fn plan_context(&self) -> Option<String> {
        let store = &self.shared.spec_store;
        let st = store.status();
        if !st.spec_exists && st.plan_based_on.is_none() {
            return None;
        }
        let mut s = format!(
            "Specification documents (maintained by the user and the spec lane, in parallel with you): {}\n\
             When the user asks to build from the spec or the plan, implement from {} and {} (never from spec.md); otherwise ignore them.",
            st.describe(&store.dir),
            store.path(Doc::Plan).display(),
            store.path(Doc::Design).display()
        );
        if st.plan_is_stale() || st.design_is_stale() {
            s.push_str(
                "\nWARNING: plan/design are based on an older spec version; tell the user before implementing parts that may have changed.",
            );
        }
        let mtime = std::fs::metadata(store.path(Doc::Plan)).and_then(|m| m.modified()).ok();
        let mut seen = self.plan_seen.lock().unwrap();
        if seen.is_some() && mtime != *seen {
            s.push_str(
                "\nNOTE: plan.md was updated since you last looked; if you are implementing it, re-read it and adjust the TODO list.",
            );
        }
        *seen = mtime;
        Some(s)
    }
}

impl TurnHooks for MasterHooks {
    fn context(&self) -> Option<String> {
        let plan = self.plan_context();
        let l = self.shared.todos.lock().unwrap();
        let todos =
            (!l.items.is_empty()).then(|| format!("Current TODO list (state: {:?}):\n{}", plan_state::plan_state(&l), l.render_compact()));
        match (plan, todos) {
            (None, None) => None,
            (a, b) => Some([a, b].into_iter().flatten().collect::<Vec<_>>().join("\n\n")),
        }
    }
    fn on_finish(&self) -> Option<String> {
        let l = self.shared.todos.lock().unwrap();
        let open: Vec<String> = l.items.iter().filter(|t| !t.status.is_terminal()).map(|t| format!("{} ({:?})", t.id, t.status)).collect();
        if open.is_empty() {
            return None;
        }
        Some(format!(
            "<system-reminder>These TODOs are not finished: {}. Before ending your turn, resolve them: run them \
             (propose_plan/run_todos), retry or fix failures, cancel items that are no longer needed (todo_write with \
             status \"cancelled\"), or explicitly tell the user what is left and why.</system-reminder>",
            open.join(", ")
        ))
    }
}

/// Instance info for frontends.
#[derive(Debug, Clone)]
pub struct InstanceInfo {
    pub id: String,
    pub profile: String,
    pub model: String,
    pub readonly: bool,
    pub busy: bool,
    pub messages: usize,
}

/// A high-cost model that has been in use for a while (see `Team::cost_alerts`).
#[derive(Debug, Clone, PartialEq)]
pub struct CostAlert {
    /// "provider/model"
    pub model: String,
    /// The profiles currently bound to it.
    pub agents: Vec<String>,
    pub minutes: u64,
    pub spent: f64,
    /// (minutes, spend) thresholds crossed; a new step alerts again after /ack.
    pub step: (u64, u64),
}

pub struct Team {
    pub shared: Arc<TeamShared>,
    master: AsyncMutex<Agent>,
    master_system: String,
    /// The /spec lane: its own conversation, model, prompt and cancellation.
    spec: AsyncMutex<Agent>,
    spec_system: String,
    spec_cancel: CancelCell,
}

/// Spec lane context: the documents' current versions every turn.
struct SpecHooks(Arc<SpecStore>);

impl TurnHooks for SpecHooks {
    fn context(&self) -> Option<String> {
        Some(format!("Spec documents: {}", self.0.status().describe(&self.0.dir)))
    }
}

/// Where and as what this team's session is saved (shared so the scheduler can checkpoint).
struct Persist {
    store: Option<session::SessionStore>,
    id: String,
    created_ms: u64,
    title: String,
    /// Master state to write in mid-turn checkpoints (history up to the current request).
    master_checkpoint: Option<session::AgentState>,
    /// Spec lane state from before its current turn (used when saving while it is busy).
    spec_checkpoint: Option<session::AgentState>,
    /// Last known state of each worker, used when it is busy during a checkpoint.
    worker_cache: std::collections::HashMap<String, session::WorkerState>,
}

impl Team {
    pub fn new(
        cfg: Config,
        providers: Providers,
        approver: Arc<dyn Approver>,
        reviewer: Arc<dyn PlanReviewer>,
        events: EventSink,
    ) -> Result<Self> {
        let gate = Arc::new(ApprovalGate::new(cfg.approval, approver));
        // Every provider key variable is a secret, plus the generic secret-name heuristic.
        let policy = Arc::new(ToolPolicy {
            sandbox: cfg.sandbox,
            secret_env: cfg.providers.values().filter_map(|p| p.api_key_env.clone()).collect(),
            env_passthrough: cfg.bash_env_passthrough.clone(),
            bash_sandbox: cfg.bash_sandbox,
            bash: crate::tools::sandbox::SandboxSpec {
                network: cfg.bash_network,
                hide: cfg.bash_hide.clone(),
                writable: cfg.bash_writable.clone(),
                overlay: cfg.bash_overlay.clone(),
                protect_git: cfg.bash_protect_git,
            },
        });
        let shared = Arc::new(TeamShared {
            pool: WorkerPool::new(cfg.max_workers),
            providers,
            todos: Mutex::new(TodoList::default()),
            reviewer,
            gate: gate.clone(),
            events: events.clone(),
            cancel: CancelCell::default(),
            file_locks: FileLocks::default(),
            persist: Mutex::new(Persist {
                store: None,
                id: session::new_session_id(),
                created_ms: session::now_ms(),
                title: String::new(),
                master_checkpoint: None,
                spec_checkpoint: None,
                worker_cache: Default::default(),
            }),
            policy,
            budget: cfg.max_session_tokens.map(|l| Arc::new(TokenBudget::new(l))),
            meter: Arc::new(crate::llm::catalog::CostMeter::new(cfg.max_session_cost)),
            worker_tools: tools::worker_tools(),
            spec_store: Arc::new(SpecStore::new(cfg.spec_dir.clone())),
            task_failures: Mutex::default(),
            model_overrides: Mutex::default(),
            cost_acked: Mutex::default(),
            cfg,
        });

        // Master: read-only tools + coordination tools, nothing mutating (enforced again here).
        let mut reg = tools::worker_tools().filter(|t| !t.mutating() && READONLY_TOOLS.contains(&t.name()));
        reg.register(Arc::new(tools::todo_tools::TodoWrite(shared.clone())));
        reg.register(Arc::new(tools::todo_tools::ProposePlan(shared.clone())));
        reg.register(Arc::new(tools::todo_tools::RunTodos(shared.clone())));
        reg.register(Arc::new(tools::delegate::Delegate(shared.clone())));
        let profile = shared.cfg.master().clone();
        let reg = reg.subset(&profile.tools);
        let names = reg.names();
        let workers = shared.worker_profiles();
        let system = prompt::build(&profile, &PromptEnv { cwd: &shared.cfg.cwd, agent_id: "master", tools: &names, workers: &workers });
        let ctx = Arc::new(
            ToolCtx::new("master", shared.cfg.cwd.clone(), gate, events, shared.cancel.clone(), shared.file_locks.clone())
                .with_policy(shared.policy.clone(), shared.budget.clone())
                .with_meter(shared.meter.clone()),
        );
        let master = Agent {
            id: "master".into(),
            profile: "master".into(),
            llm: shared.providers.get(&profile.binding.provider)?,
            binding: profile.binding.clone(),
            system: system.clone(),
            tools: reg,
            history: vec![],
            max_steps: profile.max_steps,
            loop_limit: shared.cfg.loop_limit,
            context_tokens: profile.context_tokens,
            ctx,
            hooks: Some(Arc::new(MasterHooks { shared: shared.clone(), plan_seen: Mutex::new(None) })),
            usage: Default::default(),
            session: crate::agent::run::new_session_id("master"),
        };

        // Spec lane: a separate conversation (requirements engineering) that runs in parallel with
        // the code lane. It reads the code and writes only the spec directory's documents.
        let sprof = shared.cfg.profile("spec").cloned().ok_or_else(|| anyhow!("missing spec profile"))?;
        let mut sreg = tools::worker_tools().filter(|t| !t.mutating() && READONLY_TOOLS.contains(&t.name()));
        sreg.register(Arc::new(WriteDoc(shared.spec_store.clone())));
        sreg.register(Arc::new(EditDoc(shared.spec_store.clone())));
        sreg.register(Arc::new(ReleaseSpec(shared.spec_store.clone())));
        let sreg = sreg.subset(&sprof.tools);
        let snames = sreg.names();
        let spec_system = prompt::build(&sprof, &PromptEnv { cwd: &shared.cfg.cwd, agent_id: "spec", tools: &snames, workers: &[] });
        let spec_cancel = CancelCell::default();
        let sctx = Arc::new(
            ToolCtx::new(
                "spec",
                shared.cfg.cwd.clone(),
                shared.gate.clone(),
                shared.events.clone(),
                spec_cancel.clone(),
                shared.file_locks.clone(),
            )
            .with_policy(shared.policy.clone(), shared.budget.clone())
            .with_meter(shared.meter.clone()),
        );
        let spec = Agent {
            id: "spec".into(),
            profile: "spec".into(),
            llm: shared.providers.get(&sprof.binding.provider)?,
            binding: sprof.binding.clone(),
            system: spec_system.clone(),
            tools: sreg,
            history: vec![],
            max_steps: sprof.max_steps,
            loop_limit: shared.cfg.loop_limit,
            context_tokens: sprof.context_tokens,
            ctx: sctx,
            hooks: Some(Arc::new(SpecHooks(shared.spec_store.clone()))),
            usage: Default::default(),
            session: crate::agent::run::new_session_id("spec"),
        };
        Ok(Self { shared, master: AsyncMutex::new(master), master_system: system, spec: AsyncMutex::new(spec), spec_system, spec_cancel })
    }

    /// Save the session to `store` after every turn.
    pub fn with_store(self, store: session::SessionStore) -> Self {
        self.shared.persist.lock().unwrap().store = Some(store);
        self
    }

    /// Install discovered/configured model info (prices, context) for cost tracking.
    pub fn set_catalog(&self, cat: std::collections::HashMap<String, crate::llm::catalog::ModelInfo>) {
        *self.shared.meter.catalog.write().unwrap() = cat;
    }

    /// Total USD spent this process (only models with known prices count).
    pub fn spent_usd(&self) -> f64 {
        self.shared.meter.spent()
    }

    pub fn session_id(&self) -> String {
        self.shared.persist.lock().unwrap().id.clone()
    }

    /// The single entry point for the user.
    pub async fn run(&self, input: &str) -> Result<String> {
        self.shared.cancel.reset();
        {
            let mut p = self.shared.persist.lock().unwrap();
            if p.title.is_empty() {
                p.title = input.chars().take(120).collect();
            }
        }
        let res = {
            let mut m = self.master.lock().await;
            self.apply_model_switch(&mut m, "master")?;
            // What a mid-turn checkpoint records for the master: everything before this request,
            // the request itself, and a note so a resumed master knows the turn was interrupted.
            {
                let mut history = m.history.clone();
                history.push(Message::user(input));
                history.push(Message::assistant(
                    "[This request was still in progress when the session was last saved. The TODO list shows what was \
                     completed; check it and continue from there.]",
                ));
                self.shared.persist.lock().unwrap().master_checkpoint =
                    Some(session::AgentState { history, session: m.session.clone(), usage: m.usage });
            }
            let res = m.run(input).await;
            if res.is_err() {
                // Leave history consistent: drop an assistant tool call whose results are missing.
                session::trim_dangling(&mut m.history);
                // Any item left in progress by a cancelled run goes back to pending.
                let mut l = self.shared.todos.lock().unwrap();
                for t in l.items.iter_mut().filter(|t| t.status == TodoStatus::InProgress) {
                    t.status = TodoStatus::Pending;
                }
            }
            res
        };
        // Persist after every turn — success, failure or cancellation — so nothing is lost.
        if let Err(e) = self.save().await {
            self.shared.events.emit("master", EventKind::Notice(format!("warning: could not save session: {e:#}")));
        }
        res
    }

    /// Capture the full team state (call between turns, when no worker is running).
    pub async fn snapshot(&self) -> session::SessionState {
        // Never wait on a busy lane: the other lane may be mid-turn for minutes. Busy agents are
        // recorded from the state captured at the start of their current turn.
        let agent_state = |a: &Agent| session::AgentState { history: a.history.clone(), session: a.session.clone(), usage: a.usage };
        let (master_cp, spec_cp) = {
            let p = self.shared.persist.lock().unwrap();
            (p.master_checkpoint.clone(), p.spec_checkpoint.clone())
        };
        let empty = || session::AgentState { history: vec![], session: String::new(), usage: Default::default() };
        let master = match self.master.try_lock() {
            Ok(m) => agent_state(&m),
            Err(_) => master_cp.unwrap_or_else(empty),
        };
        let spec = match self.spec.try_lock() {
            Ok(s) => Some(agent_state(&s)),
            Err(_) => spec_cp,
        }
        .filter(|s| !s.history.is_empty());
        let mut workers = Vec::new();
        for inst in self.shared.pool.list() {
            let state = match inst.agent.try_lock() {
                Ok(a) => Some(session::WorkerState {
                    id: inst.id.clone(),
                    profile: inst.profile.clone(),
                    binding: inst.binding.clone(),
                    readonly: inst.readonly,
                    agent: agent_state(&a),
                }),
                Err(_) => self.shared.persist.lock().unwrap().worker_cache.get(&inst.id).cloned(),
            };
            if let Some(s) = state {
                self.shared.persist.lock().unwrap().worker_cache.insert(s.id.clone(), s.clone());
                workers.push(s);
            }
        }
        let p = self.shared.persist.lock().unwrap();
        session::SessionState {
            version: session::FORMAT_VERSION,
            id: p.id.clone(),
            cwd: self.shared.cfg.cwd.clone(),
            created_ms: p.created_ms,
            updated_ms: session::now_ms(),
            title: p.title.clone(),
            master,
            spec,
            workers,
            todos: self.shared.todos.lock().unwrap().clone(),
            counters: self.shared.pool.counters(),
        }
    }

    /// Save now (no-op without a store). Returns the file written.
    pub async fn save(&self) -> Result<Option<std::path::PathBuf>> {
        let Some(store) = self.shared.persist.lock().unwrap().store.clone() else { return Ok(None) };
        let state = self.snapshot().await;
        if state.master.history.is_empty() && state.spec.is_none() {
            return Ok(None); // nothing worth saving yet
        }
        store.save(&state).map(Some)
    }

    /// Continue a saved session. Returns warnings (e.g. workers whose profile no longer exists).
    pub async fn resume(&self, mut state: session::SessionState) -> Result<Vec<String>> {
        if state.cwd != self.shared.cfg.cwd {
            bail!(
                "session {} belongs to {}, not {}; run zilch from that directory (or with -C)",
                state.id,
                state.cwd.display(),
                self.shared.cfg.cwd.display()
            );
        }
        state.sanitize();
        let mut warnings = Vec::new();
        {
            let mut m = self.master.lock().await;
            m.history = state.master.history;
            m.session = state.master.session;
            m.usage = state.master.usage;
        }
        if let Some(s) = state.spec {
            let mut a = self.spec.lock().await;
            a.history = s.history;
            a.session = s.session;
            a.usage = s.usage;
        }
        self.shared.pool.clear();
        self.shared.pool.set_counters(state.counters);
        for w in state.workers {
            let Some(profile) = self.shared.cfg.workers().find(|p| p.name == w.profile).cloned() else {
                warnings.push(format!("dropped {}: worker profile '{}' no longer exists", w.id, w.profile));
                continue;
            };
            if !self.shared.providers.contains(&w.binding.provider) {
                warnings.push(format!("dropped {}: provider '{}' is no longer configured", w.id, w.binding.provider));
                continue;
            }
            // Instances keep the model they were spawned with; the system prompt is rebuilt.
            let inst = self.shared.spawn_as(w.id.clone(), &profile, w.binding, w.readonly)?;
            let mut a = inst.agent.lock().await;
            a.history = w.agent.history;
            a.session = w.agent.session;
            a.usage = w.agent.usage;
            // File hashes are not persisted: workers must re-read files before editing them.
        }
        *self.shared.todos.lock().unwrap() = state.todos;
        {
            let mut p = self.shared.persist.lock().unwrap();
            p.id = state.id;
            p.created_ms = state.created_ms;
            p.title = state.title;
        }
        self.shared.todos_changed();
        Ok(warnings)
    }

    /// Cancel the code lane (master, scheduler and every worker). The spec lane is unaffected.
    pub fn cancel(&self) {
        self.shared.cancel.cancel();
    }

    /// Cancel the spec lane only.
    pub fn cancel_spec(&self) {
        self.spec_cancel.cancel();
    }

    /// One turn of the /spec lane (requirements engineering). Runs independently of — and
    /// possibly concurrently with — the code lane; saves the session afterwards.
    pub async fn run_spec(&self, input: &str) -> Result<String> {
        self.spec_cancel.reset();
        {
            let mut p = self.shared.persist.lock().unwrap();
            if p.title.is_empty() {
                p.title = input.chars().take(120).collect();
            }
        }
        let res = {
            let mut s = self.spec.lock().await;
            self.apply_model_switch(&mut s, "spec")?;
            // Recorded for saves that happen while this turn is still running.
            self.shared.persist.lock().unwrap().spec_checkpoint =
                Some(session::AgentState { history: s.history.clone(), session: s.session.clone(), usage: s.usage });
            let res = s.run(input).await;
            if res.is_err() {
                session::trim_dangling(&mut s.history);
            }
            // Refresh the cache with the finished turn, so a later code-lane checkpoint (which
            // cannot lock this lane) never writes back an older spec conversation.
            self.shared.persist.lock().unwrap().spec_checkpoint =
                Some(session::AgentState { history: s.history.clone(), session: s.session.clone(), usage: s.usage });
            res
        };
        if let Err(e) = self.save().await {
            self.shared.events.emit("spec", EventKind::Notice(format!("warning: could not save session: {e:#}")));
        }
        res
    }

    /// Versions of the spec documents (spec, design, plan) right now.
    pub fn spec_status(&self) -> crate::tools::spec_tools::SpecStatus {
        self.shared.spec_store.status()
    }

    /// Forget everything: both lanes' conversations, workers, TODOs. Starts a new session (the old
    /// one stays on disk and can still be resumed). Spec documents on disk are kept.
    pub async fn clear(&self) {
        self.master.lock().await.reset();
        self.spec.lock().await.reset();
        self.shared.pool.clear();
        *self.shared.todos.lock().unwrap() = TodoList::default();
        {
            let mut p = self.shared.persist.lock().unwrap();
            p.id = session::new_session_id();
            p.created_ms = session::now_ms();
            p.title.clear();
        }
        self.shared.todos_changed();
    }

    pub fn reset_instance(&self, id: &str) -> bool {
        self.shared.pool.remove(id)
    }

    pub fn todos(&self) -> Vec<Todo> {
        self.shared.todos.lock().unwrap().items.clone()
    }

    pub fn instances(&self) -> Vec<InstanceInfo> {
        self.shared
            .pool
            .list()
            .iter()
            .map(|i| InstanceInfo {
                id: i.id.clone(),
                profile: i.profile.clone(),
                model: i.binding.key(),
                readonly: i.readonly,
                busy: i.busy(),
                messages: i.agent.try_lock().map(|a| a.history.len()).unwrap_or(0),
            })
            .collect()
    }

    /// (agent/profile, provider/model) for the master, every profile and every live instance.
    pub fn models(&self) -> Vec<(String, String)> {
        let mut v: Vec<(String, String)> = self
            .shared
            .cfg
            .profiles
            .iter()
            .map(|p| {
                let now = self.shared.effective_binding(p);
                let note = if now != p.binding { format!("  (switched; configured: {})", p.binding.key()) } else { String::new() };
                (format!("{} (profile)", p.name), format!("{}{note}", now.key()))
            })
            .collect();
        v.extend(self.shared.pool.list().iter().map(|i| (i.id.clone(), i.binding.key())));
        v
    }

    /// /model: switch the model of the master, the spec lane or a worker profile for the rest of
    /// the session (`model` None = back to the configured one). The master and spec lane pick it
    /// up at their next turn; new worker tasks use it, running ones finish on their model.
    pub fn set_model(&self, name: &str, model: Option<&str>, provider: Option<&str>) -> Result<String> {
        if name.contains('#') {
            bail!("'{name}' is an instance; its model is fixed. Switch its profile '{}' instead", name.split('#').next().unwrap_or(name));
        }
        let profile = self.shared.cfg.profile(name).ok_or_else(|| {
            let names: Vec<&str> = self.shared.cfg.profiles.iter().map(|p| p.name.as_str()).collect();
            anyhow!("unknown agent '{name}'. Available: {}", names.join(", "))
        })?;
        let Some(model) = model else {
            self.shared.model_overrides.lock().unwrap().remove(name);
            return Ok(format!("{name} → {} (configured model)", profile.binding.key()));
        };
        let b = self.shared.binding_for(profile, Some(model), provider)?;
        // A model the provider does not list would fail with a 400 in the middle of a task.
        let listed: Vec<String> = {
            let prefix = format!("{}/", b.provider);
            self.shared.meter.catalog.read().unwrap().keys().filter(|k| k.starts_with(&prefix)).cloned().collect()
        };
        if !listed.is_empty() && !listed.contains(&b.key()) {
            bail!("'{model}' is not in provider '{}''s model list (/models in the provider docs, or `zilch --doctor`)", b.provider);
        }
        let price = match self.shared.meter.info(&b.key()) {
            Some(crate::llm::catalog::ModelInfo { input_per_mtok: Some(i), output_per_mtok: Some(o), .. }) => {
                format!(" (${i}/${o} per 1M tokens in/out)")
            }
            _ => String::new(),
        };
        let key = b.key();
        if b == profile.binding {
            self.shared.model_overrides.lock().unwrap().remove(name);
        } else {
            self.shared.model_overrides.lock().unwrap().insert(name.to_string(), b);
        }
        let when = match name {
            "master" | "spec" => "from its next turn",
            _ => "for new tasks (running ones finish on their model)",
        };
        Ok(format!("{name} → {key}{price}, {when}"))
    }

    /// Give a lane agent the model /model chose for it (called at the start of its turn).
    fn apply_model_switch(&self, agent: &mut Agent, profile: &str) -> Result<()> {
        let Some(p) = self.shared.cfg.profile(profile) else { return Ok(()) };
        let b = self.shared.effective_binding(p);
        if agent.binding != b {
            agent.llm = self.shared.providers.get(&b.provider)?;
            // Compaction follows the new model's context window (the configured one when back).
            agent.context_tokens = match self.shared.meter.info(&b.key()).and_then(|i| i.context) {
                Some(ctx) if b != p.binding => p.context_for(Some(ctx)),
                _ => p.context_tokens,
            };
            agent.binding = b;
        }
        Ok(())
    }

    /// High-cost models in use that crossed a time or spend step the user has not acknowledged.
    pub fn cost_alerts(&self) -> Vec<CostAlert> {
        let cfg = &self.shared.cfg;
        if cfg.expensive_model_price <= 0.0 {
            return vec![];
        }
        let mut by_model: std::collections::BTreeMap<String, Vec<String>> = Default::default();
        for p in &cfg.profiles {
            by_model.entry(self.shared.effective_binding(p).key()).or_default().push(p.name.clone());
        }
        let acked = self.shared.cost_acked.lock().unwrap();
        let mut out = Vec::new();
        for (model, agents) in by_model {
            let expensive = self.shared.meter.info(&model).and_then(|i| i.output_per_mtok).is_some_and(|o| o >= cfg.expensive_model_price);
            let Some(u) = self.shared.meter.model_use(&model).filter(|_| expensive) else { continue };
            let minutes = u.first.elapsed().as_secs() / 60;
            let step = (minutes / cfg.expensive_model_minutes, (u.spent / cfg.expensive_model_spend) as u64);
            let seen = acked.get(&model).copied().unwrap_or((0, 0));
            if step.0 > seen.0 || step.1 > seen.1 {
                out.push(CostAlert { model, agents, minutes, spent: u.spent, step });
            }
        }
        out
    }

    /// /ack: silence the current high-cost warnings until each crosses its next step.
    pub fn ack_cost_alerts(&self) -> usize {
        let alerts = self.cost_alerts();
        let mut acked = self.shared.cost_acked.lock().unwrap();
        for a in &alerts {
            acked.insert(a.model.clone(), a.step);
        }
        alerts.len()
    }

    /// The fully composed system prompt of the master, a live instance, or a profile (as it would be spawned).
    pub async fn prompt_of(&self, name: &str) -> Option<String> {
        if name == "master" {
            return Some(self.master_system.clone());
        }
        if name == "spec" {
            return Some(self.spec_system.clone());
        }
        if let Some(i) = self.shared.pool.find(name) {
            return Some(i.agent.lock().await.system.clone());
        }
        let p = self.shared.cfg.profile(name)?;
        let names = self.shared.worker_tools.subset(&p.tools).names();
        Some(prompt::build(p, &PromptEnv { cwd: &self.shared.cfg.cwd, agent_id: &format!("{name}#N"), tools: &names, workers: &[] }))
    }

    pub async fn master_tool_names(&self) -> Vec<String> {
        self.master.lock().await.tools.names()
    }
}
