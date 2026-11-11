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
        "Replace an exact, unique string in a file (read it first; whitespace counts; `replace_all` for every \
         occurrence). Several changes to one file: one call with `edits`, applied in order, all or none."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "path":{"type":"string"},
            "old_string":{"type":"string"},
            "new_string":{"type":"string"},
            "replace_all":{"type":"boolean"},
            "edits":{"type":"array","description":"Instead of old_string/new_string",
                "items":{"type":"object","properties":{
                    "old_string":{"type":"string"},
                    "new_string":{"type":"string"},
                    "replace_all":{"type":"boolean"}
                },"required":["old_string","new_string"]}}
        },"required":["path"]})
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
        let path = ctx.resolve_checked(arg_str(&args, "path")?, true)?;
        let edits = edit_list(&args)?;
        let _guard = ctx.file_locks.lock(&path).await;
        check_fresh(ctx, &path, false).await?;
        let text = tokio::fs::read_to_string(&path).await?;
        let (updated, n) = apply_edits(&text, &edits)?;
        tokio::fs::write(&path, &updated).await?;
        ctx.read_hashes.lock().unwrap().insert(path.clone(), sha256(updated.as_bytes()));
        Ok(format!("Edited {} ({n} replacement{})", ctx.relative(&path), if n == 1 { "" } else { "s" }))
    }
}

/// One replacement: (old, new, replace_all).
type Edit<'a> = (&'a str, &'a str, bool);

/// The requested replacements: the `edits` list, or the single old_string/new_string pair.
fn edit_list(args: &Value) -> Result<Vec<Edit<'_>>> {
    let edits: Vec<Edit> = match args.get("edits").and_then(Value::as_array).filter(|e| !e.is_empty()) {
        Some(list) => list
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let s = |k: &str| e.get(k).and_then(Value::as_str).ok_or_else(|| anyhow::anyhow!("edits[{i}]: missing '{k}'"));
                Ok((s("old_string")?, s("new_string")?, arg_bool(e, "replace_all")))
            })
            .collect::<Result<_>>()?,
        None => vec![(arg_str(args, "old_string")?, arg_str(args, "new_string")?, arg_bool(args, "replace_all"))],
    };
    for (i, (old, new, _)) in edits.iter().enumerate() {
        let at = if edits.len() > 1 { format!("edits[{i}]: ") } else { String::new() };
        if old.is_empty() {
            bail!("{at}old_string must not be empty (use write_file to create files)");
        }
        if old == new {
            bail!("{at}old_string and new_string are identical");
        }
    }
    Ok(edits)
}

/// Apply every edit in order to the text; nothing is applied if any of them fails.
pub fn apply_edits(text: &str, edits: &[Edit]) -> Result<(String, usize)> {
    let mut out = text.to_string();
    let mut total = 0;
    for (i, (old, new, all)) in edits.iter().enumerate() {
        let (next, n) = apply_edit(&out, old, new, *all).map_err(|e| {
            if edits.len() > 1 {
                anyhow::anyhow!("edits[{i}]: {e} (no edit was applied)")
            } else {
                e
            }
        })?;
        out = next;
        total += n;
    }
    Ok((out, total))
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

    #[test]
    fn several_edits_apply_in_order_all_or_none() {
        let args = json!({"path":"f","edits":[
            {"old_string":"one","new_string":"1"},
            {"old_string":"1 two","new_string":"1 2"},
            {"old_string":"x","new_string":"y","replace_all":true}
        ]});
        let edits = edit_list(&args).unwrap();
        assert_eq!(apply_edits("one two x x", &edits).unwrap(), ("1 2 y y".into(), 4), "later edits see earlier ones");
        let bad = json!({"path":"f","edits":[{"old_string":"one","new_string":"1"},{"old_string":"missing","new_string":"z"}]});
        let err = apply_edits("one two", &edit_list(&bad).unwrap()).unwrap_err().to_string();
        assert!(err.contains("edits[1]") && err.contains("no edit was applied"), "{err}");
        // The single-edit form still works, and empty or no-op edits are refused up front.
        assert_eq!(edit_list(&json!({"path":"f","old_string":"a","new_string":"b"})).unwrap(), vec![("a", "b", false)]);
        assert!(edit_list(&json!({"path":"f","edits":[{"old_string":"","new_string":"b"}]})).is_err());
        assert!(edit_list(&json!({"path":"f","edits":[{"old_string":"a","new_string":"a"}]})).is_err());
    }
}
