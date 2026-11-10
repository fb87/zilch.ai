//! Cancellation tokens that form a tree: ticket → process → one batch of syscalls.
//! Cancelling a node cancels everything below it; drivers poll `is_cancelled` between chunks.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

#[derive(Debug, Default)]
struct Node {
    flag: AtomicBool,
    parent: Option<CancelToken>,
}

#[derive(Debug, Clone, Default)]
pub struct CancelToken(Arc<Node>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    /// A token cancelled together with this one (but not the other way round).
    pub fn child(&self) -> Self {
        Self(Arc::new(Node { flag: AtomicBool::new(false), parent: Some(self.clone()) }))
    }
    pub fn cancel(&self) {
        self.0.flag.store(true, Ordering::SeqCst);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.flag.load(Ordering::SeqCst) || self.0.parent.as_ref().is_some_and(|p| p.is_cancelled())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancelling_a_parent_reaches_every_descendant_only() {
        let ticket = CancelToken::new();
        let (a, b) = (ticket.child(), ticket.child());
        let batch = a.child();
        batch.cancel();
        assert!(batch.is_cancelled() && !a.is_cancelled(), "a batch cancel leaves the process running");
        ticket.cancel();
        assert!(a.is_cancelled() && b.is_cancelled());
    }
}
