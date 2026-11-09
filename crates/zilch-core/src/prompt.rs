//! System prompt composition. Every agent gets one at spawn time.

use crate::profile::{AgentRole, Profile, PromptMode};
use std::path::{Path, PathBuf};

pub const MASTER_PROMPT: &str = include_str!("../prompts/master.md");
pub const CODER_PROMPT: &str = include_str!("../prompts/coder.md");
pub const EXPLORER_PROMPT: &str = include_str!("../prompts/explorer.md");
pub const WORKER_PROMPT: &str = include_str!("../prompts/worker.md");
pub const SPEC_PROMPT: &str = include_str!("../prompts/spec.md");

/// Non-removable rules that keep the system working regardless of custom prompts.
const MASTER_CONTRACT: &str = "\
## Rules (always apply)
- You have NO tools that modify files or run commands. All changes go through your workers.
- Any change to the codebase goes through the TODO list: todo_write -> propose_plan -> run_todos.
- `delegate` without a todo_id is read-only exploration: the worker can only read and search, it
  CANNOT run commands (not even tests). To run tests or builds, make it a TODO (or delegate with the
  todo_id of an approved item).
- `delegate` with a todo_id runs a follow-up for that (approved) item.
- Never claim work is done unless a worker report confirms it.";

const SPEC_CONTRACT: &str = "\
## Rules (always apply)
- You can only write the documents in the spec directory (write_doc / edit_doc / release_spec). You cannot change code or run
  commands; the implementation team does that, possibly in parallel with you.
- Record only what the user confirmed; everything else is an assumption or an open question.
- Release a spec version only with the user's agreement, and keep design.md and plan.md based on the
  latest released version. Keep requirement ids stable so the implementation
  team can see what moved.";

const WORKER_CONTRACT: &str = "\
## Rules (always apply)
- Only modify files within your task's scope. If the task needs more, say so in your report.
- You cannot talk to the user; do not ask questions. Make reasonable decisions and note them.
- End EVERY task with a final message in exactly this shape:
  STATUS: done | failed
  SUMMARY: <one or two sentences>
  FILES: <changed files, comma-separated, or 'none'>
  NOTES: <anything the lead must know: open issues, decisions, test results>";

pub struct PromptEnv<'a> {
    pub cwd: &'a Path,
    pub agent_id: &'a str,
    pub tools: &'a [String],
    /// (name, description) of worker profiles — shown to the master.
    pub workers: &'a [(String, String)],
}

pub fn build(profile: &Profile, env: &PromptEnv) -> String {
    let mut parts: Vec<String> = Vec::new();
    match (profile.prompt_mode, &profile.custom_prompt) {
        (PromptMode::Replace, Some(custom)) => parts.push(custom.trim().to_string()),
        (_, custom) => {
            parts.push(profile.default_prompt.trim().to_string());
            if let Some(c) = custom {
                parts.push(c.trim().to_string());
            }
        }
    }
    parts.push(
        match profile.role {
            AgentRole::Master => MASTER_CONTRACT,
            AgentRole::Worker => WORKER_CONTRACT,
            AgentRole::Spec => SPEC_CONTRACT,
        }
        .to_string(),
    );

    let mut envs = format!(
        "## Environment\n- Agent id: {{{{agent_id}}}}\n- Working directory: {{{{cwd}}}}\n- OS: {{{{os}}}}\n- Date: {{{{date}}}}\n- Model: {}\n- Your tools: {{{{tools}}}}",
        profile.binding.model
    );
    if profile.role == AgentRole::Master {
        envs.push_str("\n- Your team (worker profiles for `worker`):\n{{workers}}");
    }
    if env.tools.iter().any(|t| t == "bash") {
        if let Some(line) = cli_tools_line(&installed_cli_tools()) {
            envs.push_str(&format!("\n- {line}"));
        }
    }
    parts.push(envs);

    if let Some((path, text)) = find_agents_md(env.cwd) {
        parts.push(format!("## Project instructions ({})\n{}", path.display(), text.trim()));
    }

    let workers = env.workers.iter().map(|(n, d)| format!("  - {n}: {d}")).collect::<Vec<_>>().join("\n");
    render(
        &parts.join("\n\n"),
        &[
            ("cwd", &env.cwd.display().to_string()),
            ("os", std::env::consts::OS),
            ("date", &today()),
            ("agent_id", env.agent_id),
            ("tools", &env.tools.join(", ")),
            ("workers", &workers),
        ],
    )
}

/// Fast command-line tools worth preferring in `bash` when installed: (binary, alternatives, hint).
const CLI_TOOLS: &[(&str, &[&str], &str)] = &[
    ("rg", &[], "instead of grep -r"),
    ("fd", &["fdfind"], "instead of find"),
    ("jq", &[], "for JSON"),
    ("ast-grep", &["sg"], "structural code search/rewrite"),
];

/// Which of `CLI_TOOLS` are on PATH, as (name as installed, hint). Looked up once per process.
fn installed_cli_tools() -> Vec<(String, &'static str)> {
    static FOUND: std::sync::OnceLock<Vec<(String, &'static str)>> = std::sync::OnceLock::new();
    FOUND.get_or_init(|| find_cli_tools(&std::env::var_os("PATH").unwrap_or_default())).clone()
}

fn find_cli_tools(path: &std::ffi::OsStr) -> Vec<(String, &'static str)> {
    let dirs: Vec<PathBuf> = std::env::split_paths(path).collect();
    let exists = |name: &str| dirs.iter().any(|d| is_executable(&d.join(name)));
    CLI_TOOLS
        .iter()
        .filter_map(|(name, alts, hint)| std::iter::once(name).chain(alts.iter()).find(|n| exists(n)).map(|n| (n.to_string(), *hint)))
        .collect()
}

fn is_executable(p: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        p.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
    }
    #[cfg(not(unix))]
    {
        p.is_file()
    }
}

/// One environment line naming the installed fast CLI tools (None when there are none).
fn cli_tools_line(found: &[(String, &str)]) -> Option<String> {
    if found.is_empty() {
        return None;
    }
    let list = found.iter().map(|(n, h)| format!("{n} ({h})")).collect::<Vec<_>>().join(", ");
    Some(format!("Fast CLI tools installed, prefer them in bash: {list}. To search code, the grep/glob tools are faster still."))
}

/// `{{var}}` substitution. Unknown variables are left untouched.
pub fn render(template: &str, vars: &[(&str, &str)]) -> String {
    let mut out = template.to_string();
    for (k, v) in vars {
        out = out.replace(&format!("{{{{{k}}}}}"), v);
    }
    out
}

/// Nearest AGENTS.md in `cwd` or its ancestors.
pub fn find_agents_md(cwd: &Path) -> Option<(PathBuf, String)> {
    cwd.ancestors().find_map(|d| {
        let p = d.join("AGENTS.md");
        std::fs::read_to_string(&p).ok().map(|t| (p, t))
    })
}

/// Today's date (UTC) as YYYY-MM-DD without a date crate.
pub fn today() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let days = (secs / 86_400) as i64;
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::ModelBinding;

    fn profile(role: AgentRole, custom: Option<&str>, mode: PromptMode) -> Profile {
        Profile {
            name: "p".into(),
            description: "d".into(),
            role,
            binding: ModelBinding { provider: "x".into(), model: "m1".into(), ..Default::default() },
            tools: vec![],
            max_steps: 10,
            context_tokens: 128_000,
            context_explicit: false,
            pricing: Default::default(),
            default_prompt: "DEFAULT PROMPT".into(),
            custom_prompt: custom.map(str::to_string),
            prompt_mode: mode,
        }
    }

    fn env<'a>(tools: &'a [String], workers: &'a [(String, String)]) -> PromptEnv<'a> {
        PromptEnv { cwd: Path::new("/nonexistent/zz"), agent_id: "coder#1", tools, workers }
    }

    #[test]
    fn composition_modes() {
        let tools = vec!["read_file".to_string()];
        let w = vec![("coder".to_string(), "writes code".to_string())];
        let p = build(&profile(AgentRole::Worker, None, PromptMode::Append), &env(&tools, &w));
        assert!(p.contains("DEFAULT PROMPT") && p.contains("STATUS: done") && p.contains("coder#1") && p.contains("read_file"));
        assert!(p.contains("m1"));

        let p = build(&profile(AgentRole::Worker, Some("EXTRA"), PromptMode::Append), &env(&tools, &w));
        assert!(p.contains("DEFAULT PROMPT") && p.contains("EXTRA"));

        let p = build(&profile(AgentRole::Worker, Some("ONLY"), PromptMode::Replace), &env(&tools, &w));
        assert!(!p.contains("DEFAULT PROMPT") && p.contains("ONLY") && p.contains("STATUS: done"));

        let p = build(&profile(AgentRole::Master, None, PromptMode::Append), &env(&tools, &w));
        assert!(p.contains("NO tools that modify files") && p.contains("coder: writes code"));
        assert!(!p.contains("{{"));
    }

    #[test]
    fn installed_fast_cli_tools_are_found_on_path() {
        let a = tempfile::tempdir().unwrap();
        let b = tempfile::tempdir().unwrap();
        let touch = |dir: &Path, name: &str, mode: u32| {
            let p = dir.join(name);
            std::fs::write(&p, "#!/bin/sh\n").unwrap();
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode)).unwrap();
            }
        };
        touch(a.path(), "rg", 0o755);
        touch(b.path(), "fdfind", 0o755); // Debian's name for fd
        touch(b.path(), "jq", 0o644); // not executable
        let path = std::env::join_paths([a.path(), b.path()]).unwrap();
        let found = find_cli_tools(&path);
        let names: Vec<&str> = found.iter().map(|(n, _)| n.as_str()).collect();
        #[cfg(unix)]
        assert_eq!(names, ["rg", "fdfind"]);
        let line = cli_tools_line(&found).unwrap();
        assert!(line.contains("rg (instead of grep -r)") && line.contains("fdfind (instead of find)"), "{line}");
        assert!(cli_tools_line(&[]).is_none());
    }

    #[test]
    fn date_format() {
        let d = today();
        assert_eq!(d.len(), 10);
        assert!(d.starts_with("20"));
    }
}
