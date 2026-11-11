//! The two interaction lanes and the dispatcher that runs their turns in the background.
//! Shared by the REPL and the IPC server (editor integrations), so a message from Neovim is
//! handled exactly like one typed at the prompt.

use crate::render::{SharedScreen, DIM, RED, RESET};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tokio::task::JoinHandle;
use zilch_core::Team;

/// Switching lanes only changes which one receives input; both keep running.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Lane {
    Code,
    Spec,
}

impl Lane {
    /// Agent id of the lane's root agent (matches the renderer's lane routing).
    pub fn root(self) -> &'static str {
        match self {
            Lane::Code => "master",
            Lane::Spec => "spec",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Lane::Code => "code",
            Lane::Spec => "spec",
        }
    }
    /// Single-column symbol shown for the lane on a terminal (status bar, prompt).
    pub fn icon(self) -> &'static str {
        crate::render::lane_icon(self.root())
    }
    pub fn other(self) -> Lane {
        match self {
            Lane::Code => Lane::Spec,
            Lane::Spec => Lane::Code,
        }
    }
    pub fn parse(s: &str) -> Option<Lane> {
        match s {
            "code" => Some(Lane::Code),
            "spec" => Some(Lane::Spec),
            _ => None,
        }
    }
}

pub struct Dispatcher {
    pub team: Arc<Team>,
    pub screen: SharedScreen,
    log_path: Option<PathBuf>,
    tasks: Mutex<HashMap<&'static str, JoinHandle<()>>>,
    active: Mutex<Lane>,
}

impl Dispatcher {
    pub fn new(team: Arc<Team>, screen: SharedScreen, log_path: Option<PathBuf>) -> Arc<Self> {
        Arc::new(Self { team, screen, log_path, tasks: Mutex::new(HashMap::new()), active: Mutex::new(Lane::Code) })
    }

    pub fn active(&self) -> Lane {
        *self.active.lock().unwrap()
    }
    pub fn set_active(&self, lane: Lane) {
        *self.active.lock().unwrap() = lane;
    }

    pub fn busy(&self, lane: Lane) -> bool {
        self.tasks.lock().unwrap().get(lane.name()).is_some_and(|h| !h.is_finished())
    }

    pub fn cancel(&self, lane: Lane) {
        match lane {
            Lane::Code => self.team.cancel(),
            Lane::Spec => self.team.cancel_spec(),
        }
    }

    /// Start a turn on `lane` in the background. Fails (with a user-facing message) if that
    /// lane is still working on a previous message.
    pub fn submit(self: &Arc<Self>, lane: Lane, text: String) -> Result<(), String> {
        let mut tasks = self.tasks.lock().unwrap();
        if tasks.get(lane.name()).is_some_and(|h| !h.is_finished()) {
            return Err(format!(
                "the {} lane is still working on a previous message (wait for it, use the /{} lane, or cancel it)",
                lane.name(),
                lane.other().name()
            ));
        }
        let me = self.clone();
        let handle = tokio::spawn(async move {
            let (before_in, before_out, before_cached) = {
                let s = me.screen.lock().await;
                let (i, o) = s.total_tokens();
                (i, o, s.total_cached())
            };
            let res = match lane {
                Lane::Code => me.team.run(&text).await,
                Lane::Spec => me.team.run_spec(&text).await,
            };
            // Between requests: compact the master's finished work in the background.
            if lane == Lane::Code {
                me.team.after_turn();
            }
            // Let the renderer drain pending events before printing the footer.
            tokio::time::sleep(std::time::Duration::from_millis(30)).await;
            let mut s = me.screen.lock().await;
            let (i, o) = s.total_tokens();
            let cached = s.total_cached() - before_cached;
            let cached = if cached > 0 { format!(" ({cached} cached)") } else { String::new() };
            let msg = match res {
                Ok(_) if s.active == lane.root() => {
                    let cost = s.total_cost().map(|c| format!(" · session ${c:.4}")).unwrap_or_default();
                    format!(
                        "{}↳ {} · +{} in{cached} / +{} out tokens{cost}{}",
                        s.c(DIM),
                        if s.color { lane.icon() } else { lane.name() },
                        i - before_in,
                        o - before_out,
                        s.c(RESET)
                    )
                }
                Ok(_) => return, // background lane: the renderer already announced it
                Err(e) => {
                    let hint = me
                        .log_path
                        .as_ref()
                        .map(|p| format!("\n{}session log: {}{}", s.c(DIM), p.display(), s.c(RESET)))
                        .unwrap_or_default();
                    format!("{}error ({} lane):{} {e:#}{hint}", s.c(RED), lane.name(), s.c(RESET))
                }
            };
            s.line(&msg);
        });
        tasks.insert(lane.name(), handle);
        Ok(())
    }

    /// Stop running turns (on exit) and wait briefly for them to wind down.
    pub async fn shutdown(&self) {
        for lane in [Lane::Code, Lane::Spec] {
            if self.busy(lane) {
                self.cancel(lane);
            }
        }
        let handles: Vec<_> = self.tasks.lock().unwrap().drain().map(|(_, h)| h).collect();
        for h in handles {
            let _ = tokio::time::timeout(std::time::Duration::from_secs(5), h).await;
        }
    }
}
