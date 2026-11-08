use super::fs::check_fresh;
use super::{arg_bool, arg_str, sha256, Tool, ToolCtx};
use anyhow::{bail, Result};
use async_trait::async_trait;
use serde_json::{json, Value};

pub struct EditFile;

#[async_trait]
impl Tool for EditFile {
    fn name(&self) -> &str {
        "edit_file"
    }
    fn description(&self) -> String {
        "Replace an exact string in a file. `old_string` must match exactly (including whitespace) and be unique \
         unless `replace_all` is true. Read the file first."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string"},
            "old_string":{"type":"string"},
            "new_string":{"type":"string"},
            "replace_all":{"type":"boolean","default":false}
        },"required":["path","old_string","new_string"]})
    }
    fn mutating(&self) -> bool {
        true
    }
    fn target_path(&self, args: &Value) -> Option<String> {
        args.get("path").and_then(Value::as_str).map(str::to_string)
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("path").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    async fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let path = ctx.resolve(arg_str(&args, "path")?);
        let old = arg_str(&args, "old_string")?;
        let new = arg_str(&args, "new_string")?;
        let all = arg_bool(&args, "replace_all");
        if old.is_empty() {
            bail!("old_string must not be empty (use write_file to create files)");
        }
        if old == new {
            bail!("old_string and new_string are identical");
        }
        let _guard = ctx.file_locks.lock(&path).await;
        check_fresh(ctx, &path, false).await?;
        let text = tokio::fs::read_to_string(&path).await?;
        let (updated, n) = apply_edit(&text, old, new, all)?;
        tokio::fs::write(&path, &updated).await?;
        ctx.read_hashes.lock().unwrap().insert(path.clone(), sha256(updated.as_bytes()));
        Ok(format!("Edited {} ({n} replacement{})", ctx.relative(&path), if n == 1 { "" } else { "s" }))
    }
}

pub fn apply_edit(text: &str, old: &str, new: &str, all: bool) -> Result<(String, usize)> {
    let n = text.matches(old).count();
    match (n, all) {
        (0, _) => bail!("old_string not found in file"),
        (1, _) => Ok((text.replacen(old, new, 1), 1)),
        (_, true) => Ok((text.replace(old, new), n)),
        (_, false) => bail!("old_string occurs {n} times; add surrounding context to make it unique or set replace_all"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniqueness_rules() {
        assert!(apply_edit("a b a", "a", "x", false).is_err());
        assert_eq!(apply_edit("a b a", "a", "x", true).unwrap(), ("x b x".into(), 2));
        assert_eq!(apply_edit("a b c", "b", "x", false).unwrap(), ("a x c".into(), 1));
        assert!(apply_edit("abc", "z", "x", false).is_err());
    }
}
