//! The team: one master + a pool of parallel worker instances, coordinated through a TODO list.

pub mod file_locks;
pub mod plan_state;
pub mod pool;
pub mod scheduler;
pub mod todo;

use crate::agent::{Agent, ApprovalGate, Approver, EventKind, EventSink, TurnHooks};
use crate::config::{Config, READONLY_TOOLS};
use crate::llm::{ModelBinding, Providers};
use crate::profile::Profile;
use crate::prompt::{self, PromptEnv};
use crate::tools::{self, CancelCell, Scope, ToolCtx, ToolRegistry};
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
    worker_tools: ToolRegistry,
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

    pub fn worker_profiles(&self) -> Vec<(String, String)> {
        self.cfg.workers().map(|p| (p.name.clone(), p.description.clone())).collect()
    }

    fn worker_profile(&self, name: &str) -> Result<&Profile> {
        self.cfg.workers().find(|p| p.name == name).ok_or_else(|| {
            anyhow!("unknown worker '{name}'. Available: {}", self.cfg.workers().map(|p| p.name.as_str()).collect::<Vec<_>>().join(", "))
        })
    }

    /// Resolve a TODO's model override against the profile's binding.
    pub fn binding_for(&self, profile: &Profile, model: Option<&str>, provider: Option<&str>) -> Result<ModelBinding> {
        let mut b = profile.binding.clone();
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
        let mut reg = self.worker_tools.subset(&profile.tools);
        if readonly {
            reg = reg.filter(|t| !t.mutating());
        }
        let names = reg.names();
        let system = prompt::build(profile, &PromptEnv { cwd: &self.cfg.cwd, agent_id: &id, tools: &names, workers: &[] });
        let llm = self.providers.get(&binding.provider)?;
        let ctx = Arc::new(ToolCtx::new(&id, self.cfg.cwd.clone(), self.gate.clone(), self.events.clone(), self.cancel.clone(), self.file_locks.clone()));
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
    pub async fn acquire(&self, worker: &str, binding: Option<ModelBinding>, readonly: bool) -> Result<(Arc<Instance>, OwnedMutexGuard<Agent>)> {
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
        let binding = binding.unwrap_or_else(|| profile.binding.clone());
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

struct MasterHooks(Arc<TeamShared>);

impl TurnHooks for MasterHooks {
    fn context(&self) -> Option<String> {
        let l = self.0.todos.lock().unwrap();
        if l.items.is_empty() {
            return None;
        }
        Some(format!("Current TODO list (state: {:?}):\n{}", plan_state::plan_state(&l), l.render_compact()))
    }
    fn on_finish(&self) -> Option<String> {
        let l = self.0.todos.lock().unwrap();
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

pub struct Team {
    pub shared: Arc<TeamShared>,
    master: AsyncMutex<Agent>,
    master_system: String,
}

impl Team {
    pub fn new(cfg: Config, providers: Providers, approver: Arc<dyn Approver>, reviewer: Arc<dyn PlanReviewer>, events: EventSink) -> Result<Self> {
        let gate = Arc::new(ApprovalGate::new(cfg.approval, approver));
        let shared = Arc::new(TeamShared {
            pool: WorkerPool::new(cfg.max_workers),
            providers,
            todos: Mutex::new(TodoList::default()),
            reviewer,
            gate: gate.clone(),
            events: events.clone(),
            cancel: CancelCell::default(),
            file_locks: FileLocks::default(),
            worker_tools: tools::worker_tools(),
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
        let ctx = Arc::new(ToolCtx::new("master", shared.cfg.cwd.clone(), gate, events, shared.cancel.clone(), shared.file_locks.clone()));
        let master = Agent {
            id: "master".into(),
            profile: "master".into(),
            llm: shared.providers.get(&profile.binding.provider)?,
            binding: profile.binding.clone(),
            system: system.clone(),
            tools: reg,
            history: vec![],
            max_steps: profile.max_steps,
            ctx,
            hooks: Some(Arc::new(MasterHooks(shared.clone()))),
            usage: Default::default(),
            session: crate::agent::run::new_session_id("master"),
        };
        Ok(Self { shared, master: AsyncMutex::new(master), master_system: system })
    }

    /// The single entry point for the user.
    pub async fn run(&self, input: &str) -> Result<String> {
        self.shared.cancel.reset();
        let mut m = self.master.lock().await;
        let res = m.run(input).await;
        if res.is_err() {
            // Leave history consistent: drop a dangling assistant tool-call turn if cancelled mid-way.
            while m.history.last().is_some_and(|msg| !msg.tool_calls.is_empty()) {
                m.history.pop();
            }
            // Any item left in progress by a cancelled run goes back to pending.
            let mut l = self.shared.todos.lock().unwrap();
            for t in l.items.iter_mut().filter(|t| t.status == TodoStatus::InProgress) {
                t.status = TodoStatus::Pending;
            }
        }
        res
    }

    pub fn cancel(&self) {
        self.shared.cancel.cancel();
    }

    /// Forget everything: master history, workers, TODOs.
    pub async fn clear(&self) {
        self.master.lock().await.reset();
        self.shared.pool.clear();
        *self.shared.todos.lock().unwrap() = TodoList::default();
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
        let mut v: Vec<(String, String)> = self.shared.cfg.profiles.iter().map(|p| (format!("{} (profile)", p.name), p.binding.key())).collect();
        v.extend(self.shared.pool.list().iter().map(|i| (i.id.clone(), i.binding.key())));
        v
    }

    /// The fully composed system prompt of the master, a live instance, or a profile (as it would be spawned).
    pub async fn prompt_of(&self, name: &str) -> Option<String> {
        if name == "master" {
            return Some(self.master_system.clone());
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
