//! Worker instances spawned from profiles. Each instance keeps its own history, system prompt
//! and model binding for its whole lifetime.

use crate::agent::Agent;
use crate::llm::ModelBinding;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard, Semaphore};

pub struct Instance {
    pub id: String,
    pub profile: String,
    pub binding: ModelBinding,
    /// Spawned with read-only tools (exploration delegate).
    pub readonly: bool,
    pub agent: Arc<AsyncMutex<Agent>>,
}

impl Instance {
    pub fn busy(&self) -> bool {
        self.agent.try_lock().is_err()
    }
}

pub struct WorkerPool {
    pub instances: Mutex<Vec<Arc<Instance>>>,
    counters: Mutex<HashMap<String, usize>>,
    /// Bounds how many workers run at the same time.
    pub slots: Arc<Semaphore>,
}

impl WorkerPool {
    pub fn new(max_workers: usize) -> Self {
        Self { instances: Mutex::new(Vec::new()), counters: Mutex::new(HashMap::new()), slots: Arc::new(Semaphore::new(max_workers)) }
    }

    pub fn next_id(&self, profile: &str) -> String {
        let mut c = self.counters.lock().unwrap();
        let n = c.entry(profile.to_string()).or_insert(0);
        *n += 1;
        format!("{profile}#{n}")
    }

    pub fn find(&self, id: &str) -> Option<Arc<Instance>> {
        self.instances.lock().unwrap().iter().find(|i| i.id == id).cloned()
    }

    /// An idle instance with exactly this profile/binding/readonly key, locked for use.
    pub fn take_idle(&self, profile: &str, binding: &ModelBinding, readonly: bool) -> Option<(Arc<Instance>, OwnedMutexGuard<Agent>)> {
        let list = self.instances.lock().unwrap();
        for inst in list.iter() {
            if inst.profile == profile && inst.binding == *binding && inst.readonly == readonly {
                if let Ok(g) = inst.agent.clone().try_lock_owned() {
                    return Some((inst.clone(), g));
                }
            }
        }
        None
    }

    pub fn add(&self, inst: Arc<Instance>) {
        self.instances.lock().unwrap().push(inst);
    }

    pub fn remove(&self, id: &str) -> bool {
        let mut l = self.instances.lock().unwrap();
        let before = l.len();
        l.retain(|i| i.id != id);
        before != l.len()
    }

    pub fn clear(&self) {
        self.instances.lock().unwrap().clear();
        self.counters.lock().unwrap().clear();
    }

    pub fn list(&self) -> Vec<Arc<Instance>> {
        self.instances.lock().unwrap().clone()
    }
}
