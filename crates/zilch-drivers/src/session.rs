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

/// The code lane's snapshot stored in a session file.
pub fn load(path: &Path) -> Option<Value> {
    load_field(path, "snapshot")
}

/// The spec lane's snapshot, if the session had one.
pub fn load_spec(path: &Path) -> Option<Value> {
    load_field(path, "spec")
}

fn load_field(path: &Path, field: &str) -> Option<Value> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()?;
    v.get(field).filter(|x| !x.is_null()).cloned()
}

/// Save the code lane's snapshot (see `save_field`).
pub fn save(path: &Path, cwd: &Path, snapshot: &Value) -> std::io::Result<()> {
    save_field(path, cwd, "snapshot", snapshot)
}

/// Update one lane's snapshot in the session file ("snapshot" = code lane, "spec" = spec lane),
/// keeping the other; written atomically (temp file + rename), readable only by the user.
pub fn save_field(path: &Path, cwd: &Path, field: &str, snapshot: &Value) -> std::io::Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let ms = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0);
    let mut body: Value = std::fs::read_to_string(path)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .filter(Value::is_object)
        .unwrap_or_else(|| json!({}));
    body["version"] = json!(1);
    body["cwd"] = json!(cwd);
    body["updated_ms"] = json!(ms);
    body[field] = snapshot.clone();
    let tmp = path.with_extension("json.tmp");
    std::fs::write(&tmp, serde_json::to_vec(&body).unwrap_or_default())?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&tmp, std::fs::Permissions::from_mode(0o600))?;
    }
    std::fs::rename(tmp, path)
}

/// One saved session, for listings.
#[derive(Debug, Clone)]
pub struct Entry {
    pub path: PathBuf,
    pub id: String,
    pub updated_ms: u64,
    /// The orders of the session ("Order #N: …" texts), oldest first.
    pub orders: Vec<String>,
}

/// Every session of a workspace, newest first.
pub fn list(dir: &Path) -> Vec<Entry> {
    let Ok(rd) = std::fs::read_dir(dir) else { return vec![] };
    let mut out: Vec<Entry> = rd
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .filter_map(|path| {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&path).ok()?).ok()?;
            let orders = v["snapshot"]["messages"]
                .as_array()
                .map(|ms| {
                    ms.iter()
                        .filter(|m| m["role"] == "user")
                        .filter_map(|m| m["content"].as_str())
                        .filter(|t| t.starts_with("Order #"))
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let id = path.file_stem()?.to_string_lossy().into_owned();
            Some(Entry { id, updated_ms: v["updated_ms"].as_u64().unwrap_or(0), orders, path })
        })
        .collect();
    out.sort_by_key(|e| std::cmp::Reverse(e.updated_ms));
    out
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
        save_field(&second, Path::new("/work/a"), "spec", &json!({"messages": ["s"]})).unwrap();
        assert_eq!(load(&second).unwrap()["messages"][0], "two", "the code lane is kept");
        assert_eq!(load_spec(&second).unwrap()["messages"][0], "s");
        assert!(load_spec(&first).is_none());
        assert!(latest(&b).is_none(), "another workspace has its own sessions");
        let listed = list(&a);
        assert_eq!(listed.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["2", "1"], "newest first");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(std::fs::metadata(&second).unwrap().permissions().mode() & 0o777, 0o600);
        }
    }
}
