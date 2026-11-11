use super::{arg_opt_u64, arg_str, sha256, Tool, ToolCtx};
use anyhow::{bail, Context, Result};
use async_trait::async_trait;
use serde_json::{json, Value};
use std::path::Path;

const DEFAULT_LIMIT: usize = 2000;
const MAX_LINE: usize = 2000;
/// A whole-file read of a longer file returns its outline and first lines instead (an explicit
/// offset/limit reads anything).
const OUTLINE_OVER: usize = 1_500;
const OUTLINE_HEAD: usize = 40;

pub struct ReadFile;

#[async_trait]
impl Tool for ReadFile {
    fn name(&self) -> &str {
        "read_file"
    }
    fn description(&self) -> String {
        "Read a text file as numbered lines; offset/limit (1-based) for a range. Over 1500 lines without a range: its \
         outline (definitions with line numbers) and first lines. Re-reading unchanged lines you have: a short note."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string"},
            "offset":{"type":"integer"},
            "limit":{"type":"integer","description":"Lines (default 2000)"}
        },"required":["path"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("path").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = ctx.resolve_checked(arg_str(&args, "path")?, false)?;
        let bytes = tokio::fs::read(&path).await.with_context(|| format!("cannot read {}", path.display()))?;
        ctx.read_hashes.lock().unwrap().insert(path.clone(), sha256(&bytes));
        if bytes.iter().take(8000).any(|b| *b == 0) {
            bail!("{} looks like a binary file", path.display());
        }
        let text = String::from_utf8_lossy(&bytes);
        let explicit = args.get("offset").is_some_and(|v| !v.is_null()) || args.get("limit").is_some_and(|v| !v.is_null());
        let offset = arg_opt_u64(&args, "offset").unwrap_or(1).max(1) as usize;
        let limit = arg_opt_u64(&args, "limit").map(|l| l as usize).unwrap_or(DEFAULT_LIMIT);
        let total = text.lines().count();
        let hash = sha256(&bytes);
        let shown_path = arg_str(&args, "path")?.to_string();

        // Already in the context and unchanged: don't send it again.
        let end = (offset - 1 + limit).min(total);
        let covered =
            ctx.shown.lock().unwrap().get(&path).is_some_and(|(h, ranges)| {
                *h == hash && total > 0 && offset <= end && ranges.iter().any(|(a, b)| *a <= offset && end <= *b)
            });
        if covered {
            return Ok(format!(
                "[{shown_path} lines {offset}-{end}: unchanged since you read them earlier in this conversation. Use that \
                 output; the text was not sent again.]"
            ));
        }
        let remember = |a: usize, b: usize| {
            let mut s = ctx.shown.lock().unwrap();
            let e = s.entry(path.clone()).or_insert_with(|| (hash.clone(), vec![]));
            if e.0 != hash {
                *e = (hash.clone(), vec![]);
            }
            e.1.push((a, b));
        };

        // A long file read whole: its shape first, so the agent reads only the ranges it needs.
        if !explicit && total > OUTLINE_OVER {
            let mut out = format!("{shown_path}: {total} lines. Definitions (line: signature):\n");
            match crate::repomap::outline(&shown_path, &text) {
                Some(items) if !items.is_empty() => {
                    for it in items.iter().take(150) {
                        out.push_str(&format!("{:>6}: {}\n", it.line, it.sig));
                    }
                    if items.len() > 150 {
                        out.push_str(&format!("  … {} more\n", items.len() - 150));
                    }
                }
                _ => out.push_str("  (no outline for this file type)\n"),
            }
            out.push_str(&format!("First {OUTLINE_HEAD} lines:\n"));
            for (i, line) in text.lines().enumerate().take(OUTLINE_HEAD) {
                let line = if line.len() > MAX_LINE { &line[..line.floor_char_boundary(MAX_LINE)] } else { line };
                out.push_str(&format!("{:>6}\t{}\n", i + 1, line));
            }
            out.push_str(&format!(
                "[lines 1-{OUTLINE_HEAD} of {total}. Read the ranges you need with offset/limit (e.g. offset=120 limit=80); \
                 the whole file: offset=1 limit={total}.]\n"
            ));
            remember(1, OUTLINE_HEAD.min(total));
            return Ok(out);
        }
        let mut out = String::new();
        for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
            let line = if line.len() > MAX_LINE { &line[..line.floor_char_boundary(MAX_LINE)] } else { line };
            out.push_str(&format!("{:>6}\t{}\n", i + 1, line));
        }
        if total == 0 {
            return Ok("(empty file)".into());
        }
        let shown_end = (offset - 1 + limit).min(total);
        if offset <= shown_end {
            remember(offset, shown_end);
        }
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
        let path = ctx.resolve_checked(arg_str(&args, "path")?, true)?;
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

#[cfg(test)]
mod sandbox_tests {
    use super::*;
    use crate::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
    use crate::team::file_locks::FileLocks;
    use crate::tools::search::{GlobTool, GrepTool};
    use crate::tools::{edit::EditFile, CancelCell, ToolPolicy};
    use std::sync::Arc;

    fn ctx(dir: &Path, sandbox: bool) -> ToolCtx {
        let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
        ToolCtx::new("t", dir.to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default())
            .with_policy(Arc::new(ToolPolicy { sandbox, ..Default::default() }), None)
    }

    #[tokio::test]
    async fn escapes_are_denied() {
        let outer = tempfile::tempdir().unwrap();
        let ws = outer.path().join("ws");
        std::fs::create_dir_all(ws.join(".git")).unwrap();
        std::fs::write(outer.path().join("secret.txt"), "TOP SECRET").unwrap();
        std::fs::write(ws.join("ok.txt"), "fine").unwrap();
        #[cfg(unix)]
        std::os::unix::fs::symlink(outer.path(), ws.join("link_out")).unwrap();
        let c = ctx(&ws, true);
        let denied = |r: Result<String>| r.unwrap_err().to_string().contains("access denied");

        assert!(denied(ReadFile.call(json!({"path":"../secret.txt"}), &c).await));
        assert!(denied(ReadFile.call(json!({"path": outer.path().join("secret.txt").display().to_string()}), &c).await));
        assert!(denied(ReadFile.call(json!({"path":"sub/../../secret.txt"}), &c).await));
        #[cfg(unix)]
        {
            assert!(denied(ReadFile.call(json!({"path":"link_out/secret.txt"}), &c).await), "symlink escape");
            assert!(denied(WriteFile.call(json!({"path":"link_out/new.txt","content":"x"}), &c).await), "write via symlink");
        }
        assert!(denied(WriteFile.call(json!({"path":"../evil.txt","content":"x"}), &c).await));
        assert!(denied(WriteFile.call(json!({"path":".git/config","content":"x"}), &c).await), ".git writes");
        assert!(denied(EditFile.call(json!({"path":"/etc/hostname","old_string":"a","new_string":"b"}), &c).await));
        assert!(denied(GrepTool.call(json!({"pattern":"SECRET","path":".."}), &c).await));
        assert!(denied(GlobTool.call(json!({"pattern":"*","path":"/"}), &c).await));
        assert!(!outer.path().join("evil.txt").exists());

        // Inside the workspace everything still works, including new nested files.
        assert!(ReadFile.call(json!({"path":"ok.txt"}), &c).await.unwrap().contains("fine"));
        assert!(ReadFile.call(json!({"path":"./sub/../ok.txt"}), &c).await.is_ok());
        WriteFile.call(json!({"path":"deep/new/file.txt","content":"x"}), &c).await.unwrap();
        // Reading .git is allowed (git metadata is useful), writing isn't.
        std::fs::write(ws.join(".git/HEAD"), "ref: refs/heads/main").unwrap();
        assert!(ReadFile.call(json!({"path":".git/HEAD"}), &c).await.is_ok());

        // Opt-out works.
        let open = ctx(&ws, false);
        assert!(ReadFile.call(json!({"path":"../secret.txt"}), &open).await.unwrap().contains("TOP SECRET"));
    }
}

#[cfg(test)]
mod read_tests {
    use super::*;
    use crate::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
    use crate::team::file_locks::FileLocks;
    use crate::tools::CancelCell;
    use std::sync::Arc;

    fn ctx(dir: &Path) -> ToolCtx {
        let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
        ToolCtx::new("t", dir.to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default())
    }

    #[tokio::test]
    async fn an_unchanged_file_is_not_sent_twice() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "one\ntwo\nthree\n").unwrap();
        let c = ctx(dir.path());
        let read = |args: Value| ReadFile.call(args, &c);
        assert!(read(json!({"path":"a.txt"})).await.unwrap().contains("two"));
        let again = read(json!({"path":"a.txt"})).await.unwrap();
        assert!(again.contains("unchanged since you read them") && !again.contains("two"), "{again}");
        // A range inside what was shown is covered too; a changed file is read for real.
        assert!(read(json!({"path":"a.txt","offset":2,"limit":1})).await.unwrap().contains("unchanged"));
        std::fs::write(dir.path().join("a.txt"), "one\nTWO\nthree\n").unwrap();
        assert!(read(json!({"path":"a.txt"})).await.unwrap().contains("TWO"));
        // Once the earlier outputs may be gone (compaction, reset), it is sent again.
        c.shown.lock().unwrap().clear();
        assert!(read(json!({"path":"a.txt"})).await.unwrap().contains("TWO"));
    }

    #[tokio::test]
    async fn a_long_file_read_whole_returns_its_outline_and_first_lines() {
        let dir = tempfile::tempdir().unwrap();
        let mut src = String::from("//! Big module.\n");
        for i in 0..1600 {
            src.push_str(&format!("pub fn f{i}() {{}}\n"));
        }
        std::fs::write(dir.path().join("big.rs"), &src).unwrap();
        let c = ctx(dir.path());
        let out = ReadFile.call(json!({"path":"big.rs"}), &c).await.unwrap();
        assert!(out.starts_with("big.rs: 1601 lines. Definitions"), "{out}");
        assert!(out.contains("   100: pub fn f98()") && out.contains("… 1450 more"), "outline with line numbers, capped: {out}");
        assert!(out.contains("    40\t") && !out.contains("    41\t"), "first 40 lines only");
        assert!(out.contains("the whole file: offset=1 limit=1601"));
        // Asking explicitly reads anything: a range, or the whole file.
        let part = ReadFile.call(json!({"path":"big.rs","offset":300,"limit":2}), &c).await.unwrap();
        assert!(part.contains("   300\tpub fn f298() {}") && part.contains("[lines 300-301 of 1601]"), "{part}");
        let all = ReadFile.call(json!({"path":"big.rs","offset":1,"limit":1601}), &c).await.unwrap();
        assert!(all.contains("  1601\tpub fn f1599() {}"));
    }
}

#[cfg(test)]
mod repeat_tests {
    use super::*;
    use crate::agent::{ApprovalGate, ApprovalMode, AutoApprove, EventSink};
    use crate::team::file_locks::FileLocks;
    use crate::tools::search::GrepTool;
    use crate::tools::CancelCell;
    use std::sync::Arc;

    #[tokio::test]
    async fn an_identical_grep_with_the_same_result_is_not_sent_twice() {
        let dir = tempfile::tempdir().unwrap();
        let body: String = (0..30).map(|i| format!("fn item_{i}() {{}} // needle\n")).collect();
        std::fs::write(dir.path().join("a.rs"), &body).unwrap();
        let gate = Arc::new(ApprovalGate::new(ApprovalMode::Auto, Arc::new(AutoApprove)));
        let c = ToolCtx::new("t", dir.path().to_path_buf(), gate, EventSink::null(), CancelCell::default(), FileLocks::default());
        let first = GrepTool.call(json!({"pattern":"needle"}), &c).await.unwrap();
        assert!(first.contains("a.rs:1:"));
        let again = GrepTool.call(json!({"pattern":"needle"}), &c).await.unwrap();
        assert!(again.starts_with("[same result as your earlier identical grep"), "{again}");
        // Different arguments, or a changed result, are sent for real.
        assert!(GrepTool.call(json!({"pattern":"needle","case_insensitive":true}), &c).await.unwrap().contains("a.rs:1:"));
        std::fs::write(dir.path().join("a.rs"), format!("{body}fn extra() {{}} // needle\n")).unwrap();
        assert!(GrepTool.call(json!({"pattern":"needle"}), &c).await.unwrap().contains("a.rs:31:"));
        // Short results are always sent (a note would not be shorter).
        assert_eq!(GrepTool.call(json!({"pattern":"nothing_here"}), &c).await.unwrap(), "No matches");
        assert_eq!(GrepTool.call(json!({"pattern":"nothing_here"}), &c).await.unwrap(), "No matches");
    }
}
