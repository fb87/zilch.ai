use super::{arg_opt_u64, arg_str, sha256, Tool, ToolCtx};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE: usize = 2000;

pub struct ReadFile;

#[async_trait]
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
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = ctx.resolve(arg_str(&args, "path")?);
        let bytes = tokio::fs::read(&path).await.with_context(|| format!("cannot read {}", path.display()))?;
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

#[async_trait]
impl Tool for WriteFile {
    fn name(&self) -> &str {
        "write_file"
    }
    fn description(&self) -> String {
        "Create or overwrite a file with the given content. Existing files must be read first (read_file).".into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string"},
            "content":{"type":"string","description":"Full new file content"}
        },"required":["path","content"]})
    }
    fn mutating(&self) -> bool {
        true
    }
    fn target_path(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_string)
    }
    fn summarize(&self, args: &Value) -> String {
        let n = args.get("content").and_then(Value::as_str).map(|c| c.lines().count()).unwrap_or(0);
        format!("{} ({n} lines)", args.get("path").and_then(Value::as_str).unwrap_or("?"))
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = ctx.resolve(arg_str(&args, "path")?);
        let content = arg_str(&args, "content")?;
        let _guard = ctx.file_locks.lock(&path).await;
        check_fresh(ctx, &path, true).await?;
        if let Some(dir) = path.parent() {
            tokio::fs::create_dir_all(dir).await?;
        }
        tokio::fs::write(&path, content).await.with_context(|| format!("cannot write {}", path.display()))?;
        ctx.read_hashes.lock().unwrap().insert(path.clone(), sha256(content.as_bytes()));
        Ok(format!("Wrote {} ({} lines)", ctx.relative(&path), content.lines().count()))
    }
}

/// Refuse to modify a file this agent hasn't seen, or that changed since it last read it
/// (e.g. a parallel worker edited it).
pub(crate) async fn check_fresh(ctx: &ToolCtx, path: &Path, allow_new: bool) -> Result<()> {
    match tokio::fs::read(path).await {
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
    use crate::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
    use crate::team::file_locks::FileLocks;
    use crate::tools::{edit::EditFile, CancelCell};
    use std::sync::Arc;

    fn ctx(id: &str, dir: &Path) -> ToolCtx {
        let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
        ToolCtx::new(id, dir.to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default())
    }

    #[tokio::test]
    async fn stale_edits_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("f.txt"), "one\ntwo\n").unwrap();
        let (w1, w2) = (ctx("coder#1", dir.path()), ctx("coder#2", dir.path()));
        let edit = |old: &str, new: &str| json!({"path":"f.txt","old_string":old,"new_string":new});

        // Must read before editing or overwriting.
        assert!(EditFile.call(edit("one", "1"), &w1).await.unwrap_err().to_string().contains("not read it"));
        assert!(WriteFile.call(json!({"path":"f.txt","content":"x"}), &w1).await.is_err());

        ReadFile.call(json!({"path":"f.txt"}), &w1).await.unwrap();
        ReadFile.call(json!({"path":"f.txt"}), &w2).await.unwrap();
        EditFile.call(edit("one", "1"), &w1).await.unwrap();
        // coder#2's view is now stale.
        let err = EditFile.call(edit("two", "2"), &w2).await.unwrap_err().to_string();
        assert!(err.contains("changed since you last read"), "{err}");
        // After re-reading it works, and both edits survive.
        ReadFile.call(json!({"path":"f.txt"}), &w2).await.unwrap();
        EditFile.call(edit("two", "2"), &w2).await.unwrap();
        assert_eq!(std::fs::read_to_string(dir.path().join("f.txt")).unwrap(), "1\n2\n");
        // Own writes keep the hash fresh: coder#2 can edit again without re-reading.
        EditFile.call(edit("2", "two"), &w2).await.unwrap();
        // New files can be created without reading.
        WriteFile.call(json!({"path":"sub/new.txt","content":"n"}), &w1).await.unwrap();
    }
}
