//! Session files: the main agent's conversation, saved each time it serves an order, so
//! `zilch-os -c` can continue where the last session in this directory left off.
//! Layout: `<data dir>/zilch/os-sessions/<workspace key>/<id>.json` (mode 0600).

use serde_json::{json, Value};
use std::path::{Path, PathBuf};

/// Sessions of one workspace live in their own directory.
pub fn dir_for(root: &Path, cwd: &Path) -> PathBuf {
    let key: String = cwd.to_string_lossy().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '_' }).collect();
    root.join(key.trim_matches('_'))
}

/// `<data dir>/zilch/os-sessions`.
pub fn default_root() -> Option<PathBuf> {
    dirs::data_dir().map(|d| d.join("zilch").join("os-sessions"))
}

/// A fresh session file path (named by start time).
pub fn new_path(dir: &Path) -> PathBuf {
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    dir.join(format!("{ms}.json"))
}

/// The most recently saved session of this workspace.
pub fn latest(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .max_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok())
}

/// The snapshot stored in a session file.
pub fn load(path: &Path) -> Option<Value> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    v.get("snapshot").cloned()
}

/// Write atomically (temp file + rename), readable only by the user.
pub fn save(path: &Path, cwd: &Path, snapshot: &Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let body = json!({ "version": 1, "cwd": cwd, "updated_ms": ms, "snapshot": snapshot });
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&body).unwrap_or_default())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(tmp, path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sessions_are_saved_per_workspace_and_the_latest_one_is_found() {
        let root = tempfile::tempdir().unwrap();
        let (a, b) = (dir_for(root.path(), Path::new("/work/a")), dir_for(root.path(), Path::new("/work/b")));
        assert_ne!(a, b);
        assert!(latest(&a).is_none());
        let first = a.join("1.json");
        save(&first, Path::new("/work/a"), &json!({"messages": ["one"]})).unwrap();
        std::thread::sleep(std::time::Duration::from_millis(20));
        let second = a.join("2.json");
        save(&second, Path::new("/work/a"), &json!({"messages": ["two"]})).unwrap();
        assert_eq!(latest(&a), Some(second.clone()));
        assert_eq!(load(&second).unwrap()["messages"][0], "two");
        assert!(latest(&b).is_none(), "another workspace has its own sessions");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&second).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
}
