//! The /spec lane's document store: a user-chosen directory holding
//! - `spec.md`   — the requirements specification (working draft),
//! - `design.md` — the design derived from a released spec version,
//! - `plan.md`   — the implementation plan the code lane works from,
//! - `versions/spec-vN.md` — immutable snapshots, one per released spec version.
//!
//! Versioning is enforced here, not left to the prompt: `release_spec` freezes the draft as the
//! next version, and every write of design/plan is stamped with the spec version it is based on,
//! so anyone (the code lane included) can tell when the plan is stale.
//!
//! The tools only ever write inside this directory (paths are fixed by construction, never taken
//! from the model), so they need no approval prompt.

use super::edit::apply_edit;
use super::{arg_bool, arg_str, Tool, ToolCtx};
use anyhow::{bail, Context, Result};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const VERSION_TAG: &str = "<!-- spec-version: ";
const BASED_ON_TAG: &str = "<!-- based-on: spec v";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Doc {
    Spec,
    Design,
    Plan,
}

impl Doc {
    pub fn parse(s: &str) -> Result<Self> {
        match s {
            "spec" => Ok(Doc::Spec),
            "design" => Ok(Doc::Design),
            "plan" => Ok(Doc::Plan),
            _ => bail!("unknown document '{s}' (expected spec | design | plan)"),
        }
    }
    pub fn file(self) -> &'static str {
        match self {
            Doc::Spec => "spec.md",
            Doc::Design => "design.md",
            Doc::Plan => "plan.md",
        }
    }
}

/// Where the spec documents live and what state they are in.
#[derive(Debug, Clone)]
pub struct SpecStore {
    pub dir: PathBuf,
}

/// Snapshot of the documents' versions, for prompts and the UI.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SpecStatus {
    pub spec_exists: bool,
    /// Latest released spec version (0 = nothing released yet).
    pub released: u32,
    /// The draft differs from the latest released version.
    pub unreleased_changes: bool,
    pub design_based_on: Option<u32>,
    pub plan_based_on: Option<u32>,
}

impl SpecStatus {
    pub fn plan_is_stale(&self) -> bool {
        self.released > 0 && self.plan_based_on.is_some_and(|v| v < self.released)
    }
    pub fn design_is_stale(&self) -> bool {
        self.released > 0 && self.design_based_on.is_some_and(|v| v < self.released)
    }

    /// One-paragraph summary used in agent context and `/spec status`.
    pub fn describe(&self, dir: &Path) -> String {
        if !self.spec_exists {
            return format!("No specification yet (spec directory: {}).", dir.display());
        }
        let mut s = match self.released {
            0 => format!("Spec draft at {}/spec.md, not released yet.", dir.display()),
            v => format!(
                "Spec v{v} released ({}/versions/spec-v{v}.md){}.",
                dir.display(),
                if self.unreleased_changes { "; the draft has unreleased changes" } else { "" }
            ),
        };
        for (name, base, stale) in
            [("design.md", self.design_based_on, self.design_is_stale()), ("plan.md", self.plan_based_on, self.plan_is_stale())]
        {
            s.push_str(&match base {
                None => format!(" {name}: not written."),
                Some(v) if stale => format!(" {name}: based on spec v{v} — STALE (spec is v{}).", self.released),
                Some(v) => format!(" {name}: based on spec v{v}."),
            });
        }
        s
    }
}

fn parse_tag(content: &str, tag: &str) -> Option<u32> {
    let line = content.lines().find(|l| l.trim_start().starts_with(tag))?;
    line.trim_start()[tag.len()..].split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

/// Replace (or insert as the first line) a `<!-- … -->` marker line.
fn set_tag(content: &str, tag: &str, line: &str) -> String {
    let mut lines: Vec<&str> = content.lines().filter(|l| !l.trim_start().starts_with(tag)).collect();
    lines.insert(0, line);
    let mut out = lines.join("\n");
    out.push('\n');
    out
}

impl SpecStore {
    pub fn new(dir: impl Into<PathBuf>) -> Self {
        Self { dir: dir.into() }
    }
    pub fn path(&self, d: Doc) -> PathBuf {
        self.dir.join(d.file())
    }
    fn version_path(&self, v: u32) -> PathBuf {
        self.dir.join("versions").join(format!("spec-v{v}.md"))
    }
    pub fn latest_version(&self) -> u32 {
        std::fs::read_dir(self.dir.join("versions"))
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter_map(|e| {
                        let n = e.file_name().to_string_lossy().into_owned();
                        n.strip_prefix("spec-v")?.strip_suffix(".md")?.parse::<u32>().ok()
                    })
                    .max()
                    .unwrap_or(0)
            })
            .unwrap_or(0)
    }

    pub fn status(&self) -> SpecStatus {
        let read = |d: Doc| std::fs::read_to_string(self.path(d)).ok();
        let spec = read(Doc::Spec);
        let released = self.latest_version();
        let unreleased_changes = match (&spec, released) {
            (Some(_), 0) => true,
            (Some(s), v) => std::fs::read_to_string(self.version_path(v)).map(|snap| snap != *s).unwrap_or(true),
            (None, _) => false,
        };
        SpecStatus {
            spec_exists: spec.is_some(),
            released,
            unreleased_changes,
            design_based_on: read(Doc::Design).and_then(|c| parse_tag(&c, BASED_ON_TAG)),
            plan_based_on: read(Doc::Plan).and_then(|c| parse_tag(&c, BASED_ON_TAG)),
        }
    }

    /// Write a document. Design and plan are stamped with the spec version they derive from
    /// (the latest released one) and require that a version exists.
    pub fn write(&self, d: Doc, content: &str) -> Result<String> {
        std::fs::create_dir_all(&self.dir).with_context(|| format!("creating {}", self.dir.display()))?;
        let content = match d {
            Doc::Spec => content.to_string(),
            Doc::Design | Doc::Plan => {
                let v = self.latest_version();
                if v == 0 {
                    bail!("release the spec first (release_spec): design and plan must be based on a released spec version");
                }
                set_tag(content, BASED_ON_TAG, &format!("{BASED_ON_TAG}{v} -->"))
            }
        };
        std::fs::write(self.path(d), &content)?;
        Ok(content)
    }

    /// Freeze the current draft as the next version; returns the new version number.
    pub fn release(&self, summary: &str) -> Result<u32> {
        let draft = std::fs::read_to_string(self.path(Doc::Spec)).context("no spec draft to release; write the spec first")?;
        let st = self.status();
        if st.released > 0 && !st.unreleased_changes {
            bail!("spec v{} is already the current draft; nothing to release", st.released);
        }
        let v = st.released + 1;
        let date = super::today();
        let mut stamped = set_tag(&draft, VERSION_TAG, &format!("{VERSION_TAG}{v} ({date}) -->"));
        // Changelog entry, appended under an existing "## Changelog" heading (or a new one).
        let entry = format!("- v{v} ({date}): {}", summary.trim());
        stamped = if let Some(pos) = stamped.find("## Changelog") {
            let end = stamped[pos..].find('\n').map_or(stamped.len(), |i| pos + i + 1);
            format!("{}{entry}\n{}", &stamped[..end], &stamped[end..])
        } else {
            format!("{}\n## Changelog\n{entry}\n", stamped.trim_end())
        };
        std::fs::create_dir_all(self.dir.join("versions"))?;
        std::fs::write(self.path(Doc::Spec), &stamped)?;
        std::fs::write(self.version_path(v), &stamped)?;
        Ok(v)
    }
}

// ---------------- tools ----------------

pub struct WriteDoc(pub Arc<SpecStore>);
pub struct EditDoc(pub Arc<SpecStore>);
pub struct ReleaseSpec(pub Arc<SpecStore>);

fn doc_arg(args: &Value) -> Result<Doc> {
    Doc::parse(arg_str(args, "doc")?)
}

impl Tool for WriteDoc {
    fn name(&self) -> &str {
        "write_doc"
    }
    fn description(&self) -> String {
        format!(
            "Replace a whole document in the spec directory ({}): doc = spec | design | plan. design and plan are \
             automatically stamped with the released spec version they are based on (release the spec first).",
            self.0.dir.display()
        )
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "doc":{"type":"string","enum":["spec","design","plan"]},
            "content":{"type":"string","description":"Full Markdown content"}
        },"required":["doc","content"]})
    }
    fn summarize(&self, args: &Value) -> String {
        let n = args.get("content").and_then(Value::as_str).map_or(0, |c| c.lines().count());
        format!("{} ({n} lines)", args.get("doc").and_then(Value::as_str).unwrap_or("?"))
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let d = doc_arg(&args)?;
        let _g = ctx.locks.lock(&self.0.path(d));
        let written = self.0.write(d, arg_str(&args, "content")?)?;
        Ok(format!("{} saved ({} lines). {}", d.file(), written.lines().count(), self.0.status().describe(&self.0.dir)))
    }
}

impl Tool for EditDoc {
    fn name(&self) -> &str {
        "edit_doc"
    }
    fn description(&self) -> String {
        "Replace an exact string in a spec-directory document (doc = spec | design | plan). old_string must be unique \
         unless replace_all. Editing design/plan re-stamps them with the current released spec version."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{
            "doc":{"type":"string","enum":["spec","design","plan"]},
            "old_string":{"type":"string"},
            "new_string":{"type":"string"},
            "replace_all":{"type":"boolean"}
        },"required":["doc","old_string","new_string"]})
    }
    fn summarize(&self, args: &Value) -> String {
        args.get("doc").and_then(Value::as_str).unwrap_or("?").to_string()
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let d = doc_arg(&args)?;
        let path = self.0.path(d);
        let _g = ctx.locks.lock(&path);
        let text = std::fs::read_to_string(&path).with_context(|| format!("{} does not exist yet; use write_doc", d.file()))?;
        let (updated, n) = apply_edit(&text, arg_str(&args, "old_string")?, arg_str(&args, "new_string")?, arg_bool(&args, "replace_all"))?;
        self.0.write(d, &updated)?;
        Ok(format!("{} edited ({n} replacement{})", d.file(), if n == 1 { "" } else { "s" }))
    }
}

impl Tool for ReleaseSpec {
    fn name(&self) -> &str {
        "release_spec"
    }
    fn description(&self) -> String {
        "Release the current spec draft as the next version (immutable snapshot + changelog entry). Do this when the \
         user agrees the spec (or a change to it) is ready; then derive or update design and plan from it."
            .into()
    }
    fn schema(&self) -> Value {
        json!({"type":"object","properties":{"summary":{"type":"string","description":"What changed in this version"}},"required":["summary"]})
    }
    fn call(&self, args: Value, ctx: &ToolCtx) -> Result<String> {
        let _g = ctx.locks.lock(&self.0.path(Doc::Spec));
        let v = self.0.release(arg_str(&args, "summary")?)?;
        let st = self.0.status();
        let mut msg = format!("Released spec v{v}.");
        if st.design_based_on.is_some_and(|b| b < v) || st.plan_based_on.is_some_and(|b| b < v) {
            msg.push_str(" design.md / plan.md are now based on an older version: update them to reflect the changes.");
        } else if st.design_based_on.is_none() || st.plan_based_on.is_none() {
            msg.push_str(" Next: write design.md and plan.md from it.");
        }
        Ok(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(dir: &Path) -> ToolCtx {
        ToolCtx::new("spec", dir.into())
    }

    #[test]
    fn versioning_lifecycle() {
        let ws = tempfile::tempdir().unwrap();
        let store = Arc::new(SpecStore::new(ws.path().join("docs/spec")));
        let c = ctx(ws.path());
        let (w, e, r) = (WriteDoc(store.clone()), EditDoc(store.clone()), ReleaseSpec(store.clone()));

        assert!(!store.status().spec_exists);
        assert!(r.call(json!({"summary":"x"}), &c).is_err(), "nothing to release");
        w.call(json!({"doc":"spec","content":"# App — Specification\n## 4. FR-1 login\n## Changelog\n"}), &c).unwrap();
        // Design/plan need a released spec to be based on.
        assert!(w.call(json!({"doc":"plan","content":"# Plan"}), &c).unwrap_err().to_string().contains("release the spec first"));

        r.call(json!({"summary":"initial requirements"}), &c).unwrap();
        let st = store.status();
        assert_eq!((st.released, st.unreleased_changes), (1, false));
        let v1 = std::fs::read_to_string(store.dir.join("versions/spec-v1.md")).unwrap();
        assert!(v1.starts_with("<!-- spec-version: 1 (") && v1.contains("- v1 (") && v1.contains("initial requirements"), "{v1}");
        assert!(r.call(json!({"summary":"again"}), &c).is_err(), "no changes → no new version");

        w.call(json!({"doc":"design","content":"# Design"}), &c).unwrap();
        w.call(json!({"doc":"plan","content":"# Plan\n- M1: FR-1"}), &c).unwrap();
        let st = store.status();
        assert_eq!((st.design_based_on, st.plan_based_on, st.plan_is_stale()), (Some(1), Some(1), false));
        assert!(std::fs::read_to_string(store.path(Doc::Plan)).unwrap().starts_with("<!-- based-on: spec v1 -->"));

        // Spec changes → v2 → plan becomes stale until it is updated.
        e.call(json!({"doc":"spec","old_string":"FR-1 login","new_string":"FR-1 login with SSO"}), &c).unwrap();
        assert!(store.status().unreleased_changes);
        let msg = r.call(json!({"summary":"SSO"}), &c).unwrap();
        assert!(msg.contains("older version"), "{msg}");
        let st = store.status();
        assert!(st.released == 2 && st.plan_is_stale() && st.design_is_stale());
        assert!(st.describe(&store.dir).contains("plan.md: based on spec v1 — STALE (spec is v2)"));
        assert!(std::fs::read_to_string(store.dir.join("versions/spec-v1.md")).unwrap().contains("FR-1 login\n"), "v1 snapshot immutable");

        e.call(json!({"doc":"plan","old_string":"FR-1","new_string":"FR-1 (SSO)"}), &c).unwrap();
        assert_eq!(store.status().plan_based_on, Some(2), "editing re-stamps with the current version");
        // Only the spec directory was ever written.
        let top: Vec<_> = std::fs::read_dir(ws.path()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(top, vec![std::ffi::OsString::from("docs")]);
    }
}
