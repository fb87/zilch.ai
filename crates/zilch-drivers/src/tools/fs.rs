use super::{arg_opt_u64, arg_str, sha256, Tool, ToolCtx};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::Path;

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE: usize = 2000;

pub struct ReadFile;

impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> String {
        "Read a text file. Returns numbered lines (`  N\\tline`). Use offset/limit (1-based lines) for large files.".into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string","description":"File path, relative to the working directory or absolute"},
            "offset":{"type":"integer","description":"First line to read (1-based)"},
            "limit":{"type":"integer","description":"Max lines to read (default 2000)"}
        },"required":["path"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("path").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = ctx.resolve_checked(arg_str(&args, "path")?, false)?;
        let bytes = std::fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
        ctx.read_hashes.lock().unwrap().insert(path.clone(), sha256(&bytes));
        if bytes.iter().take(8000).any(|b| *b == 0) {
            bail!("{} looks like a binary file", path.display());
        }
        let text = String::from_utf8_lossy(&bytes);
        let offset = arg_opt_u64(&args, "offset").unwrap_or(1).max(1) as usize;
        let limit = arg_opt_u64(&args, "limit").map(|l| l as usize).unwrap_or(DEFAULT_LIMIT);
        let total = text.lines().count();
        let mut out = String::new();
        for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
            let line = if line.len() > MAX_LINE { &line[..line.floor_char_boundary(MAX_LINE)] } else { line };
            out.push_str(&format!("{:>6}\t{}\n", i + 1, line));
        }
        if total == 0 {
            return Ok("(empty file)".into());
        }
        let shown_end = (offset - 1 + limit).min(total);
        if shown_end < total || offset > 1 {
            out.push_str(&format!("[lines {}-{} of {}]\n", offset, shown_end, total));
        }
        Ok(out)
    }
}

pub struct WriteFile;

impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }
    fn description(&self) -> String {
        "Create a file, or replace one entirely (read it first). To change part of a file use edit_file: it sends \
         only the change."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string"},
            "content":{"type":"string","description":"Full new file content"}
        },"required":["path","content"]})
    }
    fn target_path(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_string)
    }
    fn summarize(&self, args: &Value) -> String {
        let n = args.get("content").and_then(Value::as_str).map(|c| c.lines().count()).unwrap_or(0);
        format!("{} ({n} lines)", args.get("path").and_then(Value::as_str).unwrap_or("?"))
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = ctx.resolve_checked(arg_str(&args, "path")?, true)?;
        let content = arg_str(&args, "content")?;
        let _guard = ctx.locks.lock(&path);
        check_fresh(ctx, &path, true)?;
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(&path, content).with_context(|| format!("cannot write {}", path.display()))?;
        ctx.read_hashes.lock().unwrap().insert(path.clone(), sha256(content.as_bytes()));
        Ok(format!("Wrote {} ({} lines)", ctx.relative(&path), content.lines().count()))
    }
}

/// Refuse to modify a file this agent hasn't seen, or that changed since it last read it
/// (e.g. a parallel worker edited it).
pub(crate) fn check_fresh(ctx: &ToolCtx, path: &Path, allow_new: bool) -> Result<()> {
    match std::fs::read(path) {
        Ok(current) => {
            let known = ctx.read_hashes.lock().unwrap().get(path).cloned();
            match known {
                None => bail!("{} exists but you have not read it yet; call read_file first", ctx.relative(path)),
                Some(h) if h != sha256(&current) => bail!(
                    "{} changed since you last read it (possibly by another worker); call read_file again before modifying it",
                    ctx.relative(path)
                ),
                Some(_) => Ok(()),
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound && allow_new => Ok(()),
        Err(e) => Err(e).with_context(|| format!("cannot read {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::edit::EditFile;

    fn ctx(id: &str, dir: &Path) -> ToolCtx {
        ToolCtx::new(id, dir.to_path_buf())
    }

    #[test]
    fn stale_edits_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "one\ntwo\n").unwrap();
        let (w1, w2) = (ctx("coder#1", dir.path()), ctx("coder#2", dir.path()));
        let edit = |old: &str, new: &str| json!({"path":"f.txt","old_string":old,"new_string":new});

        // Must read before editing or overwriting.
        assert!(EditFile.call(edit("one", "1"), &w1).unwrap_err().to_string().contains("not read it"));
        assert!(WriteFile.call(json!({"path":"f.txt","content":"x"}), &w1).is_err());

        ReadFile.call(json!({"path":"f.txt"}), &w1).unwrap();
        ReadFile.call(json!({"path":"f.txt"}), &w2).unwrap();
        EditFile.call(edit("one", "1"), &w1).unwrap();
        // coder#2's view is now stale.
        let err = EditFile.call(edit("two", "2"), &w2).unwrap_err().to_string();
        assert!(err.contains("changed since you last read"), "{err}");
        // After re-reading it works, and both edits survive.
        ReadFile.call(json!({"path":"f.txt"}), &w2).unwrap();
        EditFile.call(edit("two", "2"), &w2).unwrap();
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).unwrap(), "1\n2\n");
        // Own writes keep the hash fresh: coder#2 can edit again without re-reading.
        EditFile.call(edit("2", "two"), &w2).unwrap();
        // New files can be created without reading.
        WriteFile.call(json!({"path":"sub/new.txt","content":"n"}), &w1).unwrap();
    }
}

#[cfg(test)]
mod sandbox_tests {
    use super::*;
    use crate::tools::search::{GlobTool, GrepTool};
    use crate::tools::{edit::EditFile, ToolPolicy};
    use std::sync::Arc;

    fn ctx(dir: &Path, sandbox: bool) -> ToolCtx {
        ToolCtx::new("t", dir.to_path_buf()).with_policy(Arc::new(ToolPolicy { sandbox, ..Default::default() }))
    }

    #[test]
    fn escapes_are_denied() {
        let outer = tempfile::tempdir().unwrap();
        let ws = outer.path().join("ws");
        std::fs::create_dir_all(ws.join(".git")).unwrap();
        std::fs::write(outer.path().join("secret.txt"), "TOP SECRET").unwrap();
        std::fs::write(ws.join("ok.txt"), "fine").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outer.path(), ws.join("link_out")).unwrap();
        let c = ctx(&ws, true);
        let denied = |r: Result<String>| r.unwrap_err().to_string().contains("access denied");

        assert!(denied(ReadFile.call(json!({"path":"../secret.txt"}), &c)));
        assert!(denied(ReadFile.call(json!({"path": outer.path().join("secret.txt").display().to_string()}), &c)));
        assert!(denied(ReadFile.call(json!({"path":"sub/../../secret.txt"}), &c)));
        #[cfg(unix)]
        {
            assert!(denied(ReadFile.call(json!({"path":"link_out/secret.txt"}), &c)), "symlink escape");
            assert!(denied(WriteFile.call(json!({"path":"link_out/new.txt","content":"x"}), &c)), "write via symlink");
        }
        assert!(denied(WriteFile.call(json!({"path":"../evil.txt","content":"x"}), &c)));
        assert!(denied(WriteFile.call(json!({"path":".git/config","content":"x"}), &c)), ".git writes");
        assert!(denied(EditFile.call(json!({"path":"/etc/hostname","old_string":"a","new_string":"b"}), &c)));
        assert!(denied(GrepTool.call(json!({"pattern":"SECRET","path":".."}), &c)));
        assert!(denied(GlobTool.call(json!({"pattern":"*","path":"/"}), &c)));
        assert!(!outer.path().join("evil.txt").exists());

        // Inside the workspace everything still works, including new nested files.
        assert!(ReadFile.call(json!({"path":"ok.txt"}), &c).unwrap().contains("fine"));
        assert!(ReadFile.call(json!({"path":"./sub/../ok.txt"}), &c).is_ok());
        WriteFile.call(json!({"path":"deep/new/file.txt","content":"x"}), &c).unwrap();
        // Reading .git is allowed (git metadata is useful), writing isn't.
        std::fs::write(ws.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
        assert!(ReadFile.call(json!({"path":".git/HEAD"}), &c).is_ok());

        // Opt-out works.
        let open = ctx(&ws, false);
        assert!(ReadFile.call(json!({"path":"../secret.txt"}), &open).unwrap().contains("TOP SECRET"));
    }
}
