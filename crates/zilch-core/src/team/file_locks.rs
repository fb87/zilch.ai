//! Per-path async locks so parallel workers never interleave writes to the same file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard};

#[derive(Clone, Default)]
pub struct FileLocks(Arc<Mutex<HashMap<PathBuf, Arc<AsyncMutex<()>>>>>);

impl FileLocks {
    pub async fn lock(&self, path: &Path) -> OwnedMutexGuard<()> {
        let m = self.0.lock().unwrap().entry(path.to_path_buf()).or_default().clone();
        m.lock_owned().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[tokio::test]
    async fn same_path_serializes() {
        let locks = FileLocks::default();
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut hs = vec![];
        for i in 0..3 {
            let (locks, log) = (locks.clone(), log.clone());
            hs.push(tokio::spawn(async move {
                let _g = locks.lock(Path::new("/x")).await;
                log.lock().unwrap().push(format!("start{i}"));
                tokio::time::sleep(Duration::from_millis(20)).await;
                log.lock().unwrap().push(format!("end{i}"));
            }));
        }
        for h in hs {
            h.await.unwrap();
        }
        let log = log.lock().unwrap();
        for pair in log.chunks(2) {
            assert_eq!(pair[0].replace("start", ""), pair[1].replace("end", ""));
        }
    }
}
