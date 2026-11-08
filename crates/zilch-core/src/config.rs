//! Layered configuration: built-in defaults < ~/.config/zilch < ./.zilch < env < CLI flags.

use crate::agent::ApprovalMode;
use crate::llm::openai::OpenAiClient;
use crate::llm::{ModelBinding, Providers};
use crate::profile::{AgentRole, Profile, PromptMode};
use crate::prompt;
use crate::tools::sandbox::BashSandbox;
use crate::tools::COORDINATION_TOOLS;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

pub const MASTER_TOOLS: &[&str] = &["read_file", "glob", "grep", "todo_write", "propose_plan", "run_todos", "delegate"];
pub const WORKER_TOOLS: &[&str] = &["read_file", "write_file", "edit_file", "glob", "grep", "bash"];
pub const READONLY_TOOLS: &[&str] = &["read_file", "glob", "grep"];

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderCfg {
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    /// Discouraged (prefer api_key_env), but handy for local servers that want any token.
    pub api_key: Option<String>,
    pub max_concurrent: Option<usize>,
    /// Extra static HTTP headers for every request.
    pub headers: Option<BTreeMap<String, String>>,
    /// Header carrying each agent's stable conversation id (e.g. "x-opencode-session").
    pub session_header: Option<String>,
    /// Seconds without any data before a request/stream is considered stalled (default 180).
    pub idle_timeout_secs: Option<u64>,
    /// Wire protocol: chat (default) | responses | anthropic.
    pub protocol: Option<crate::llm::protocol::Protocol>,
}

impl ProviderCfg {
    fn merge(&mut self, o: ProviderCfg) {
        merge_opt(&mut self.base_url, o.base_url);
        merge_opt(&mut self.api_key_env, o.api_key_env);
        merge_opt(&mut self.api_key, o.api_key);
        merge_opt(&mut self.max_concurrent, o.max_concurrent);
        merge_opt(&mut self.headers, o.headers);
        merge_opt(&mut self.session_header, o.session_header);
        merge_opt(&mut self.idle_timeout_secs, o.idle_timeout_secs);
        merge_opt(&mut self.protocol, o.protocol);
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoleCfg {
    pub name: Option<String>,
    pub description: Option<String>,
    pub provider: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub extra: Option<Map<String, Value>>,
    pub tools: Option<Vec<String>>,
    pub max_steps: Option<usize>,
    pub context_tokens: Option<usize>,
    pub system_prompt: Option<String>,
    pub system_prompt_file: Option<PathBuf>,
    pub prompt_mode: Option<PromptMode>,
}

impl RoleCfg {
    fn merge(&mut self, o: RoleCfg) {
        merge_opt(&mut self.name, o.name);
        merge_opt(&mut self.description, o.description);
        merge_opt(&mut self.provider, o.provider);
        merge_opt(&mut self.model, o.model);
        merge_opt(&mut self.temperature, o.temperature);
        merge_opt(&mut self.top_p, o.top_p);
        merge_opt(&mut self.max_tokens, o.max_tokens);
        merge_opt(&mut self.reasoning_effort, o.reasoning_effort);
        merge_opt(&mut self.extra, o.extra);
        merge_opt(&mut self.tools, o.tools);
        merge_opt(&mut self.max_steps, o.max_steps);
        merge_opt(&mut self.context_tokens, o.context_tokens);
        if o.system_prompt.is_some() || o.system_prompt_file.is_some() {
            self.system_prompt = o.system_prompt;
            self.system_prompt_file = o.system_prompt_file;
        }
        merge_opt(&mut self.prompt_mode, o.prompt_mode);
    }
    fn absolutize(&mut self, base: &Path) {
        if let Some(p) = &self.system_prompt_file {
            if p.is_relative() {
                self.system_prompt_file = Some(base.join(p));
            }
        }
    }
}

fn merge_opt<T>(dst: &mut Option<T>, src: Option<T>) {
    if src.is_some() {
        *dst = src;
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileConfig {
    pub max_workers: Option<usize>,
    pub approval: Option<ApprovalMode>,
    pub sandbox: Option<bool>,
    pub bash_env_passthrough: Option<Vec<String>>,
    pub bash_sandbox: Option<BashSandbox>,
    pub bash_network: Option<bool>,
    pub bash_hide: Option<Vec<String>>,
    pub bash_writable: Option<Vec<String>>,
    pub bash_overlay: Option<Vec<String>>,
    pub max_session_tokens: Option<u64>,
    #[serde(default)]
    pub providers: BTreeMap<String, ProviderCfg>,
    pub defaults: Option<RoleCfg>,
    #[serde(default)]
    pub roles: BTreeMap<String, RoleCfg>,
}

/// Command-line overrides (highest precedence).
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    pub master_model: Option<String>,
    /// (profile name, model)
    pub worker_models: Vec<(String, String)>,
    pub max_workers: Option<usize>,
    pub approval: Option<ApprovalMode>,
    pub max_session_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub cwd: PathBuf,
    pub max_workers: usize,
    pub approval: ApprovalMode,
    /// Confine file tools to the workspace (default true).
    pub sandbox: bool,
    /// Secret-looking env vars still passed to bash (e.g. GITHUB_TOKEN).
    pub bash_env_passthrough: Vec<String>,
    /// OS sandbox for bash: off | auto (default: bubblewrap when available) | bwrap (required).
    pub bash_sandbox: BashSandbox,
    /// Whether sandboxed commands get network access (default true).
    pub bash_network: bool,
    /// Extra paths hidden from sandboxed commands.
    pub bash_hide: Vec<String>,
    /// Extra writable paths for sandboxed commands (e.g. "~/.cache/pip").
    pub bash_writable: Vec<String>,
    /// Copy-on-write paths for sandboxed commands (None = default tool caches).
    pub bash_overlay: Option<Vec<String>>,
    /// Team-wide token cap per session (prompt + completion), None = unlimited.
    pub max_session_tokens: Option<u64>,
    pub providers: BTreeMap<String, ProviderCfg>,
    /// Master first, then workers.
    pub profiles: Vec<Profile>,
}

impl Config {
    pub fn master(&self) -> &Profile {
        self.profiles.iter().find(|p| p.is_master()).expect("master profile")
    }
    pub fn workers(&self) -> impl Iterator<Item = &Profile> {
        self.profiles.iter().filter(|p| !p.is_master())
    }
    pub fn profile(&self, name: &str) -> Option<&Profile> {
        self.profiles.iter().find(|p| p.name == name)
    }

    /// Load all layers from disk + environment.
    pub fn load(cwd: &Path, ov: &Overrides) -> Result<Self> {
        let mut layers = Vec::new();
        if let Some(d) = dirs::config_dir() {
            layers.push(d.join("zilch"));
        }
        layers.push(cwd.join(".zilch"));
        Self::load_from(cwd, &layers, &std::env::vars().collect(), ov)
    }

    /// Testable loader: explicit layer directories and environment.
    pub fn load_from(cwd: &Path, layer_dirs: &[PathBuf], env: &BTreeMap<String, String>, ov: &Overrides) -> Result<Self> {
        let mut fc = FileConfig::default();
        let mut agent_files: BTreeMap<String, RoleCfg> = BTreeMap::new();
        for dir in layer_dirs {
            let file = dir.join("config.toml");
            if file.exists() {
                let text = std::fs::read_to_string(&file)?;
                let layer: FileConfig = toml::from_str(&text).with_context(|| format!("parsing {}", file.display()))?;
                merge_file(&mut fc, layer, dir);
            }
            let agents = dir.join("agents");
            if let Ok(rd) = std::fs::read_dir(&agents) {
                let mut paths: Vec<_> =
                    rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "toml")).collect();
                paths.sort();
                for p in paths {
                    let mut rc: RoleCfg =
                        toml::from_str(&std::fs::read_to_string(&p)?).with_context(|| format!("parsing {}", p.display()))?;
                    rc.absolutize(dir);
                    let name = rc.name.clone().or_else(|| p.file_stem().map(|s| s.to_string_lossy().into_owned())).unwrap();
                    agent_files.entry(name).or_default().merge(rc);
                }
            }
        }
        // Agent files and [roles.x] describe the same thing; agent files win.
        for (name, rc) in agent_files {
            fc.roles.entry(name).or_default().merge(rc);
        }

        // Environment.
        let openai = fc.providers.entry("openai".into()).or_default();
        if let Some(u) = env.get("OPENAI_BASE_URL") {
            openai.base_url = Some(u.clone());
        }
        openai.base_url.get_or_insert_with(|| "https://api.openai.com/v1".into());
        openai.api_key_env.get_or_insert_with(|| "OPENAI_API_KEY".into());
        let mut defaults = fc.defaults.take().unwrap_or_default();
        if let Some(m) = env.get("ZILCH_MODEL") {
            defaults.model = Some(m.clone());
        }
        if let Some(p) = env.get("ZILCH_PROVIDER") {
            defaults.provider = Some(p.clone());
        }

        // CLI overrides.
        if let Some(m) = &ov.master_model {
            fc.roles.entry("master".into()).or_default().model = Some(m.clone());
        }
        for (name, m) in &ov.worker_models {
            fc.roles.entry(name.clone()).or_default().model = Some(m.clone());
        }

        // Build profiles: built-ins first, then user-defined workers.
        let mut names = vec!["master".to_string(), "coder".to_string(), "explorer".to_string()];
        names.extend(fc.roles.keys().filter(|k| !names.contains(k)).cloned().collect::<Vec<_>>());
        let mut profiles = Vec::new();
        for name in names {
            let rc = fc.roles.get(&name).cloned().unwrap_or_default();
            profiles.push(build_profile(&name, rc, &defaults)?);
        }

        let cfg = Config {
            cwd: cwd.to_path_buf(),
            max_workers: ov.max_workers.or(fc.max_workers).unwrap_or(4).max(1),
            approval: ov.approval.or(fc.approval).unwrap_or_default(),
            sandbox: fc.sandbox.unwrap_or(true),
            bash_env_passthrough: fc.bash_env_passthrough.unwrap_or_default(),
            bash_sandbox: fc.bash_sandbox.unwrap_or_default(),
            bash_network: fc.bash_network.unwrap_or(true),
            bash_hide: fc.bash_hide.unwrap_or_default(),
            bash_writable: fc.bash_writable.unwrap_or_default(),
            bash_overlay: fc.bash_overlay,
            max_session_tokens: ov.max_session_tokens.or(fc.max_session_tokens),
            providers: fc.providers,
            profiles,
        };
        cfg.validate(env)?;
        Ok(cfg)
    }

    /// Fail fast on unknown providers or missing API keys.
    fn validate(&self, env: &BTreeMap<String, String>) -> Result<()> {
        for p in &self.profiles {
            let Some(prov) = self.providers.get(&p.binding.provider) else {
                bail!("profile '{}' uses unknown provider '{}' (define [providers.{}])", p.name, p.binding.provider, p.binding.provider);
            };
            let base = prov.base_url.as_deref().unwrap_or("");
            if base.is_empty() {
                bail!("provider '{}' has no base_url", p.binding.provider);
            }
            let local = ["localhost", "127.0.0.1", "0.0.0.0", "[::1]"].iter().any(|h| base.contains(h));
            if prov.api_key.is_none() && !local {
                if let Some(var) = &prov.api_key_env {
                    if env.get(var).is_none_or(|v| v.is_empty()) {
                        bail!("profile '{}' uses provider '{}' but ${var} is not set", p.name, p.binding.provider);
                    }
                }
            }
        }
        Ok(())
    }

    /// Instantiate one client per provider actually used by some profile.
    pub fn build_providers(&self) -> Result<Providers> {
        let mut out = Providers::default();
        for p in &self.profiles {
            let name = &p.binding.provider;
            if out.contains(name) {
                continue;
            }
            let c = &self.providers[name];
            let key = c.api_key.clone().or_else(|| c.api_key_env.as_ref().and_then(|v| std::env::var(v).ok())).filter(|k| !k.is_empty());
            let headers = c.headers.clone().unwrap_or_default().into_iter().collect();
            let client = OpenAiClient::new(name, c.base_url.clone().unwrap_or_default(), key, c.max_concurrent.unwrap_or(8))
                .with_headers(headers)
                .with_session_header(c.session_header.clone())
                .with_idle_timeout(std::time::Duration::from_secs(c.idle_timeout_secs.unwrap_or(180).max(5)))
                .with_protocol(c.protocol.unwrap_or_default());
            out.insert(name.clone(), Arc::new(client));
        }
        Ok(out)
    }
}

fn merge_file(dst: &mut FileConfig, src: FileConfig, base: &Path) {
    merge_opt(&mut dst.max_workers, src.max_workers);
    merge_opt(&mut dst.approval, src.approval);
    merge_opt(&mut dst.sandbox, src.sandbox);
    merge_opt(&mut dst.bash_env_passthrough, src.bash_env_passthrough);
    merge_opt(&mut dst.bash_sandbox, src.bash_sandbox);
    merge_opt(&mut dst.bash_network, src.bash_network);
    merge_opt(&mut dst.bash_hide, src.bash_hide);
    merge_opt(&mut dst.bash_writable, src.bash_writable);
    merge_opt(&mut dst.bash_overlay, src.bash_overlay);
    merge_opt(&mut dst.max_session_tokens, src.max_session_tokens);
    for (k, v) in src.providers {
        dst.providers.entry(k).or_default().merge(v);
    }
    if let Some(mut d) = src.defaults {
        d.absolutize(base);
        dst.defaults.get_or_insert_with(Default::default).merge(d);
    }
    for (k, mut v) in src.roles {
        v.absolutize(base);
        dst.roles.entry(k).or_default().merge(v);
    }
}

fn build_profile(name: &str, rc: RoleCfg, defaults: &RoleCfg) -> Result<Profile> {
    let (role, default_prompt, default_desc, default_tools): (_, &str, &str, &[&str]) = match name {
        "master" => (AgentRole::Master, prompt::MASTER_PROMPT, "Team lead; plans and coordinates", MASTER_TOOLS),
        "coder" => (
            AgentRole::Worker,
            prompt::CODER_PROMPT,
            "Implements changes: reads, writes and edits files, runs commands and tests",
            WORKER_TOOLS,
        ),
        "explorer" => (
            AgentRole::Worker,
            prompt::EXPLORER_PROMPT,
            "Read-only investigation: finds code, answers questions about the codebase",
            READONLY_TOOLS,
        ),
        _ => (AgentRole::Worker, prompt::WORKER_PROMPT, "General worker", WORKER_TOOLS),
    };

    let tools: Vec<String> = match &rc.tools {
        None => default_tools.iter().map(|s| s.to_string()).collect(),
        Some(list) => {
            for t in list {
                let known = MASTER_TOOLS.contains(&t.as_str()) || WORKER_TOOLS.contains(&t.as_str());
                if !known {
                    bail!("profile '{name}': unknown tool '{t}'");
                }
            }
            list.clone()
        }
    };
    // Structural role enforcement: the master can never get mutating tools, workers never coordinate.
    let tools: Vec<String> = match role {
        AgentRole::Master => tools.into_iter().filter(|t| MASTER_TOOLS.contains(&t.as_str())).collect(),
        AgentRole::Worker => tools.into_iter().filter(|t| !COORDINATION_TOOLS.contains(&t.as_str())).collect(),
    };

    let custom_prompt = match (&rc.system_prompt, &rc.system_prompt_file) {
        (Some(s), _) => Some(s.clone()),
        (None, Some(f)) => Some(std::fs::read_to_string(f).with_context(|| format!("profile '{name}': reading {}", f.display()))?),
        _ => None,
    };

    let binding = ModelBinding {
        provider: rc.provider.clone().or_else(|| defaults.provider.clone()).unwrap_or_else(|| "openai".into()),
        model: rc.model.clone().or_else(|| defaults.model.clone()).unwrap_or_else(|| "gpt-4.1".into()),
        temperature: rc.temperature.or(defaults.temperature),
        top_p: rc.top_p.or(defaults.top_p),
        max_tokens: rc.max_tokens.or(defaults.max_tokens),
        reasoning_effort: rc.reasoning_effort.clone().or_else(|| defaults.reasoning_effort.clone()),
        extra: rc.extra.clone().or_else(|| defaults.extra.clone()).unwrap_or_default(),
    };

    Ok(Profile {
        name: name.to_string(),
        description: rc.description.unwrap_or_else(|| default_desc.into()),
        role,
        binding,
        tools,
        max_steps: rc.max_steps.or(defaults.max_steps).unwrap_or(if role == AgentRole::Master { 60 } else { 80 }),
        context_tokens: rc.context_tokens.or(defaults.context_tokens).unwrap_or(128_000).max(16_000),
        default_prompt: default_prompt.to_string(),
        custom_prompt,
        prompt_mode: rc.prompt_mode.unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect()
    }

    #[test]
    fn layered_models_and_prompts() {
        let dir = tempfile::tempdir().unwrap();
        let user = dir.path().join("user");
        let proj = dir.path().join("proj/.zilch");
        std::fs::create_dir_all(user.join("agents")).unwrap();
        std::fs::create_dir_all(proj.join("agents")).unwrap();
        std::fs::write(
            user.join("config.toml"),
            r#"
max_workers = 2
[providers.local]
base_url = "http://localhost:11434/v1"
[defaults]
model = "base-model"
[roles.master]
model = "user-master"
"#,
        )
        .unwrap();
        std::fs::write(
            proj.join("config.toml"),
            r#"
[roles.coder]
provider = "local"
model = "qwen"
temperature = 0.1
system_prompt_file = "coder_extra.md"
"#,
        )
        .unwrap();
        std::fs::write(proj.join("coder_extra.md"), "Use tabs.").unwrap();
        std::fs::write(
            proj.join("agents/reviewer.toml"),
            r#"
description = "Reviews diffs"
tools = ["read_file", "grep", "bash", "delegate"]
system_prompt = "Be strict."
prompt_mode = "replace"
"#,
        )
        .unwrap();

        let ov = Overrides { worker_models: vec![("explorer".into(), "cli-explorer".into())], ..Default::default() };
        let cfg = Config::load_from(dir.path(), &[user, proj], &env(&[("OPENAI_API_KEY", "k")]), &ov).unwrap();
        assert_eq!(cfg.max_workers, 2);
        assert_eq!(cfg.master().binding.model, "user-master");
        let coder = cfg.profile("coder").unwrap();
        assert_eq!((coder.binding.provider.as_str(), coder.binding.model.as_str()), ("local", "qwen"));
        assert_eq!(coder.custom_prompt.as_deref(), Some("Use tabs."));
        assert_eq!(cfg.profile("explorer").unwrap().binding.model, "cli-explorer");
        let rev = cfg.profile("reviewer").unwrap();
        assert_eq!(rev.binding.model, "base-model");
        assert_eq!(rev.prompt_mode, PromptMode::Replace);
        // Workers never get coordination tools.
        assert_eq!(rev.tools, vec!["read_file", "grep", "bash"]);
        // Master never gets mutating tools.
        assert!(cfg.master().tools.iter().all(|t| !["write_file", "edit_file", "bash"].contains(&t.as_str())));
    }

    #[test]
    fn master_cannot_be_given_write_tools() {
        let dir = tempfile::tempdir().unwrap();
        let proj = dir.path().join(".zilch");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(proj.join("config.toml"), "[roles.master]\ntools = [\"read_file\", \"bash\", \"write_file\"]\n").unwrap();
        let cfg = Config::load_from(dir.path(), &[proj], &env(&[("OPENAI_API_KEY", "k")]), &Overrides::default()).unwrap();
        assert_eq!(cfg.master().tools, vec!["read_file"]);
    }

    #[test]
    fn fails_fast() {
        let dir = tempfile::tempdir().unwrap();
        let err = Config::load_from(dir.path(), &[], &env(&[]), &Overrides::default()).unwrap_err();
        assert!(err.to_string().contains("OPENAI_API_KEY"));

        let proj = dir.path().join(".zilch");
        std::fs::create_dir_all(&proj).unwrap();
        std::fs::write(proj.join("config.toml"), "[roles.coder]\nprovider = \"nope\"\n").unwrap();
        let err = Config::load_from(dir.path(), &[proj], &env(&[("OPENAI_API_KEY", "k")]), &Overrides::default()).unwrap_err();
        assert!(err.to_string().contains("unknown provider"));
    }
}
