//! Configuration: `~/.config/zilch/config.toml`, `./.zilch/config.toml`, environment, CLI flags
//! (later wins). The keys are the previous zilch's, so existing config files keep working, and unknown
//! keys are ignored. Roles map onto groups: `[roles.architect]` (falling back to `[roles.master]`)
//! for the main agent, `[roles.developer]` (falling back to `[roles.coder]`) for the subagents.

use crate::llm::client::HttpClient;
use crate::llm::protocol::Protocol;
use crate::llm::Backend;
use crate::tools::sandbox::{BashSandbox, SandboxSpec};
use crate::tools::ToolPolicy;
use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;
use zilch_ipc::{Group, ModelBinding};

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProviderCfg {
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub api_key: Option<String>,
    pub max_concurrent: Option<usize>,
    pub headers: Option<BTreeMap<String, String>>,
    pub session_header: Option<String>,
    pub idle_timeout_secs: Option<u64>,
    pub protocol: Option<Protocol>,
    /// Fetch prices and context sizes from `{base_url}/models` at startup (default true).
    pub discover: Option<bool>,
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
        merge_opt(&mut self.discover, o.discover);
    }
    fn key(&self, env: &BTreeMap<String, String>) -> Option<String> {
        self.api_key.clone().or_else(|| self.api_key_env.as_ref().and_then(|v| env.get(v).cloned())).filter(|k| !k.is_empty())
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct RoleCfg {
    pub provider: Option<String>,
    pub model: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub max_tokens: Option<u32>,
    pub reasoning_effort: Option<String>,
    pub extra: Option<Map<String, Value>>,
    pub max_steps: Option<usize>,
    pub context_tokens: Option<usize>,
    /// USD per 1M input / output tokens (overrides discovered prices).
    pub price_in: Option<f64>,
    pub price_out: Option<f64>,
    /// Prompt override: inline, or a file (relative to the config directory it was set in);
    /// `prompt_mode` "append" (default: after the built-in prompt) or "replace".
    pub system_prompt: Option<String>,
    pub system_prompt_file: Option<PathBuf>,
    pub prompt_mode: Option<String>,
    /// Custom subagent profiles: their tools and a one-line description for the main agent.
    pub tools: Option<Vec<String>>,
    pub description: Option<String>,
    /// In `agents/*.toml`: the profile's name (default: the file name).
    pub name: Option<String>,
}

impl RoleCfg {
    fn merge(&mut self, o: RoleCfg) {
        merge_opt(&mut self.provider, o.provider);
        merge_opt(&mut self.model, o.model);
        merge_opt(&mut self.temperature, o.temperature);
        merge_opt(&mut self.top_p, o.top_p);
        merge_opt(&mut self.max_tokens, o.max_tokens);
        merge_opt(&mut self.reasoning_effort, o.reasoning_effort);
        merge_opt(&mut self.extra, o.extra);
        merge_opt(&mut self.max_steps, o.max_steps);
        merge_opt(&mut self.context_tokens, o.context_tokens);
        merge_opt(&mut self.price_in, o.price_in);
        merge_opt(&mut self.price_out, o.price_out);
        if o.system_prompt.is_some() || o.system_prompt_file.is_some() {
            self.system_prompt = o.system_prompt;
            self.system_prompt_file = o.system_prompt_file;
        }
        merge_opt(&mut self.prompt_mode, o.prompt_mode);
        merge_opt(&mut self.tools, o.tools);
        merge_opt(&mut self.description, o.description);
        merge_opt(&mut self.name, o.name);
    }

    /// A relative prompt file is relative to the config directory it was set in.
    fn absolutize(&mut self, dir: &Path) {
        if let Some(p) = self.system_prompt_file.as_ref().filter(|p| p.is_relative()) {
            self.system_prompt_file = Some(dir.join(p));
        }
    }

    /// The prompt override this role sets, if any.
    fn prompt_override(&self, role: &str) -> Result<Option<PromptOverride>> {
        let text = match (&self.system_prompt, &self.system_prompt_file) {
            (Some(t), _) => t.clone(),
            (None, Some(p)) => std::fs::read_to_string(p).with_context(|| format!("[roles.{role}] system_prompt_file {}", p.display()))?,
            (None, None) => return Ok(None),
        };
        let replace = match self.prompt_mode.as_deref().unwrap_or("append") {
            "append" => false,
            "replace" => true,
            other => bail!("[roles.{role}] prompt_mode '{other}' (append | replace)"),
        };
        Ok(Some(PromptOverride { text, replace }))
    }
}

/// A role's own prompt: after the built-in one, or instead of it.
#[derive(Debug, Clone, PartialEq)]
pub struct PromptOverride {
    pub text: String,
    pub replace: bool,
}

/// A custom subagent profile (`[roles.<name>]` with any other name, or `agents/<name>.toml`):
/// the main agent can give a plan task to it with the task's `worker` field.
#[derive(Debug, Clone)]
pub struct Profile {
    pub name: String,
    pub description: String,
    pub binding: ModelBinding,
    pub tools: Vec<String>,
    pub prompt: Option<PromptOverride>,
}

impl RoleCfg {
    /// Unset fields taken from `under` (a lower-precedence role).
    fn over(mut self, under: &RoleCfg) -> RoleCfg {
        let mut base = under.clone();
        base.merge(std::mem::take(&mut self));
        base
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
struct FileConfig {
    #[serde(default)]
    providers: BTreeMap<String, ProviderCfg>,
    defaults: Option<RoleCfg>,
    #[serde(default)]
    roles: BTreeMap<String, RoleCfg>,
    sandbox: Option<bool>,
    bash_sandbox: Option<BashSandbox>,
    bash_network: Option<bool>,
    bash_env_passthrough: Option<Vec<String>>,
    bash_hide: Option<Vec<String>>,
    bash_writable: Option<Vec<String>>,
    bash_protect_git: Option<bool>,
    max_workers: Option<usize>,
    loop_limit: Option<usize>,
    max_task_tokens: Option<u64>,
    /// Token quota per group: `[quotas] architect = 200000, developer = 2000000`.
    quotas: Option<BTreeMap<String, u64>>,
    max_session_tokens: Option<u64>,
    max_session_cost: Option<f64>,
    expensive_model_price: Option<f64>,
    expensive_model_minutes: Option<u64>,
    expensive_model_spend: Option<f64>,
    spec_dir: Option<String>,
    approval: Option<String>,
}

fn merge_opt<T>(dst: &mut Option<T>, src: Option<T>) {
    if src.is_some() {
        *dst = src;
    }
}

/// Command-line overrides (highest precedence).
#[derive(Debug, Clone, Default)]
pub struct Overrides {
    /// Model for every group.
    pub model: Option<String>,
    pub architect_model: Option<String>,
    pub developer_model: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Config {
    pub cwd: PathBuf,
    pub sources: Vec<PathBuf>,
    pub providers: BTreeMap<String, ProviderCfg>,
    pub architect: ModelBinding,
    pub developer: ModelBinding,
    pub policy: ToolPolicy,
    /// Parallel LLM calls the I/O pool should allow for (sum of provider limits).
    pub io_threads: usize,
    pub max_steps: u32,
    pub loop_limit: Option<usize>,
    pub max_task_tokens: Option<u64>,
    /// Token quota per group (`[quotas]`).
    pub group_tokens: BTreeMap<Group, u64>,
    /// Context budget of the main agent (`context_tokens` on its role); None = built-in default.
    pub main_context_tokens: Option<usize>,
    /// Session-wide caps, checked before each model call (None = unlimited).
    pub max_session_tokens: Option<u64>,
    pub max_session_cost: Option<f64>,
    /// High-cost model warning: output price (USD/1M) from which a model is high-cost (0 = off),
    /// and after how many minutes of use or USD spent it is flagged (and again every as many).
    pub expensive_model_price: f64,
    pub expensive_model_minutes: u64,
    pub expensive_model_spend: f64,
    /// The spec lane's document directory (`spec_dir`, default `specs`).
    pub spec_dir: PathBuf,
    /// Prompt overrides of the main agent, the subagents and the spec agent.
    pub prompts: BTreeMap<Group, PromptOverride>,
    /// Custom subagent profiles (any other `[roles.<name>]`).
    pub profiles: Vec<Profile>,
    /// `approval`: "plan" (default: approve plans; commands run in the sandbox), "ask" (commands
    /// too), "auto" (nothing asks).
    pub approval: String,
    /// Price overrides from the roles: (model key, input, output) USD per 1M tokens.
    price_overrides: Vec<(String, Option<f64>, Option<f64>)>,
    env: BTreeMap<String, String>,
}

/// `~/.config/zilch` and friends, lowest precedence first (same as zilch).
pub fn user_config_dirs() -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = Vec::new();
    let xdg = std::env::var_os("XDG_CONFIG_HOME")
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|h| h.join(".config")));
    for d in [dirs::config_dir(), xdg].into_iter().flatten() {
        let d = d.join("zilch");
        if !found.contains(&d) {
            found.push(d);
        }
    }
    found
}

impl Config {
    pub fn load(cwd: &Path, ov: &Overrides) -> Result<Self> {
        let mut layers = user_config_dirs();
        layers.push(cwd.join(".zilch"));
        Self::load_from(cwd, &layers, &std::env::vars().collect(), ov)
    }

    pub fn load_from(cwd: &Path, layer_dirs: &[PathBuf], env: &BTreeMap<String, String>, ov: &Overrides) -> Result<Self> {
        let mut fc = FileConfig::default();
        let mut sources = Vec::new();
        for dir in layer_dirs {
            // Roles may also be files of their own: <dir>/agents/<name>.toml.
            if let Ok(rd) = std::fs::read_dir(dir.join("agents")) {
                let mut paths: Vec<PathBuf> =
                    rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|e| e == "toml")).collect();
                paths.sort();
                for p in paths {
                    let mut rc: RoleCfg =
                        toml::from_str(&std::fs::read_to_string(&p)?).with_context(|| format!("parsing {}", p.display()))?;
                    rc.absolutize(dir);
                    let name = rc.name.clone().or_else(|| p.file_stem().map(|s| s.to_string_lossy().into_owned())).unwrap_or_default();
                    fc.roles.entry(name).or_default().merge(rc);
                    sources.push(p);
                }
            }
            let file = dir.join("config.toml");
            if !file.exists() {
                continue;
            }
            let layer: FileConfig =
                toml::from_str(&std::fs::read_to_string(&file)?).with_context(|| format!("parsing {}", file.display()))?;
            for (k, v) in layer.providers {
                fc.providers.entry(k).or_default().merge(v);
            }
            for (k, mut v) in layer.roles {
                v.absolutize(dir);
                fc.roles.entry(k).or_default().merge(v);
            }
            if let Some(d) = layer.defaults {
                fc.defaults.get_or_insert_with(RoleCfg::default).merge(d);
            }
            merge_opt(&mut fc.sandbox, layer.sandbox);
            merge_opt(&mut fc.bash_sandbox, layer.bash_sandbox);
            merge_opt(&mut fc.bash_network, layer.bash_network);
            merge_opt(&mut fc.bash_env_passthrough, layer.bash_env_passthrough);
            merge_opt(&mut fc.bash_hide, layer.bash_hide);
            merge_opt(&mut fc.bash_writable, layer.bash_writable);
            merge_opt(&mut fc.bash_protect_git, layer.bash_protect_git);
            merge_opt(&mut fc.max_workers, layer.max_workers);
            merge_opt(&mut fc.loop_limit, layer.loop_limit);
            merge_opt(&mut fc.max_task_tokens, layer.max_task_tokens);
            merge_opt(&mut fc.max_session_tokens, layer.max_session_tokens);
            merge_opt(&mut fc.max_session_cost, layer.max_session_cost);
            merge_opt(&mut fc.expensive_model_price, layer.expensive_model_price);
            merge_opt(&mut fc.expensive_model_minutes, layer.expensive_model_minutes);
            merge_opt(&mut fc.expensive_model_spend, layer.expensive_model_spend);
            merge_opt(&mut fc.spec_dir, layer.spec_dir);
            merge_opt(&mut fc.approval, layer.approval);
            if let Some(q) = layer.quotas {
                fc.quotas.get_or_insert_with(BTreeMap::new).extend(q);
            }
            sources.push(file);
        }
        // The built-in `openai` provider, adjustable from the environment.
        let openai = fc.providers.entry("openai".into()).or_default();
        if let Some(u) = env.get("OPENAI_BASE_URL") {
            openai.base_url = Some(u.clone());
        }
        openai.base_url.get_or_insert_with(|| "https://api.openai.com/v1".into());
        openai.api_key_env.get_or_insert_with(|| "OPENAI_API_KEY".into());

        let mut defaults = fc.defaults.clone().unwrap_or_default();
        if let Some(m) = env.get("ZILCH_MODEL") {
            defaults.model = Some(m.clone());
        }
        if let Some(p) = env.get("ZILCH_PROVIDER") {
            defaults.provider = Some(p.clone());
        }
        let role = |names: &[&str]| -> RoleCfg {
            let mut r = defaults.clone();
            for n in names.iter().rev() {
                if let Some(rc) = fc.roles.get(*n) {
                    r = rc.clone().over(&r);
                }
            }
            r
        };
        let mut architect = role(&["architect", "master"]);
        let mut developer = role(&["developer", "coder"]);
        for (r, specific) in [(&mut architect, &ov.architect_model), (&mut developer, &ov.developer_model)] {
            if let Some(m) = specific.as_ref().or(ov.model.as_ref()) {
                r.model = Some(m.clone());
            }
        }
        let binding = |r: &RoleCfg, group: &str| -> Result<ModelBinding> {
            let provider = r.provider.clone().unwrap_or_else(|| "openai".into());
            let Some(model) = r.model.clone() else {
                bail!("no model for the {group}: set [defaults] model (or [roles.{group}]) in config.toml, ZILCH_MODEL, or --model")
            };
            Ok(ModelBinding {
                provider,
                model,
                temperature: r.temperature,
                top_p: r.top_p,
                max_tokens: r.max_tokens,
                reasoning_effort: r.reasoning_effort.clone(),
                extra: r.extra.clone().unwrap_or_default(),
            })
        };
        let architect_b = binding(&architect, "architect")?;
        let developer_b = binding(&developer, "developer")?;
        let defaults_policy = ToolPolicy::default();
        let policy = ToolPolicy {
            sandbox: fc.sandbox.unwrap_or(true),
            env_passthrough: fc.bash_env_passthrough.unwrap_or_default(),
            bash_sandbox: fc.bash_sandbox.unwrap_or_default(),
            bash: SandboxSpec {
                network: fc.bash_network.unwrap_or(true),
                hide: fc.bash_hide.unwrap_or_default(),
                writable: fc.bash_writable.unwrap_or_default(),
                overlay: None,
                protect_git: fc.bash_protect_git.unwrap_or(true),
            },
            ..defaults_policy
        };
        let mut policy = policy;
        // Every provider key is a secret for commands, whatever its variable is called.
        policy.secret_env = fc.providers.values().filter_map(|p| p.api_key_env.clone()).collect();
        let io_threads = fc.providers.values().map(|p| p.max_concurrent.unwrap_or(8)).sum::<usize>().clamp(8, 64) + 8;
        let price_overrides: Vec<(String, Option<f64>, Option<f64>)> = [(&architect, &architect_b), (&developer, &developer_b)]
            .iter()
            .filter(|(r, _)| r.price_in.is_some() || r.price_out.is_some())
            .map(|(r, b)| (b.key(), r.price_in, r.price_out))
            .collect();
        // Prompt overrides of the built-in groups, then custom subagent profiles: every other role,
        // inheriting the subagents' model unless it sets its own.
        let mut prompts = BTreeMap::new();
        for (names, g) in [
            (&["architect", "master", "main"][..], Group::Architecture),
            (&["developer", "coder"][..], Group::Development),
            (&["spec"][..], Group::Spec),
        ] {
            for n in names.iter().rev() {
                if let Some(o) = fc.roles.get(*n).map(|r| r.prompt_override(n)).transpose()?.flatten() {
                    prompts.insert(g, o);
                }
            }
        }
        let known = ["architect", "master", "main", "developer", "coder", "spec", "explorer"];
        let mut profiles = Vec::new();
        for (name, rc) in fc.roles.iter().filter(|(n, _)| !known.contains(&n.as_str())) {
            let r = rc.clone().over(&developer);
            profiles.push(Profile {
                name: name.clone(),
                description: rc.description.clone().unwrap_or_else(|| "custom subagent".into()),
                binding: binding(&r, name)?,
                tools: rc.tools.clone().unwrap_or_else(|| crate::DEVELOPER_TOOLS.iter().map(|s| s.to_string()).collect()),
                prompt: rc.prompt_override(name)?,
            });
        }
        let mut group_tokens = BTreeMap::new();
        for (name, q) in fc.quotas.clone().unwrap_or_default() {
            let g = match name.as_str() {
                "main" | "architect" | "architecture" => Group::Architecture,
                "developer" | "development" => Group::Development,
                "reviewer" | "review" => Group::Review,
                other => bail!("unknown group '{other}' in [quotas] (architect, developer, reviewer)"),
            };
            group_tokens.insert(g, q);
        }
        let cfg = Config {
            cwd: cwd.to_path_buf(),
            group_tokens,
            main_context_tokens: architect.context_tokens,
            max_session_tokens: fc.max_session_tokens,
            max_session_cost: fc.max_session_cost,
            expensive_model_price: fc.expensive_model_price.unwrap_or(10.0).max(0.0),
            expensive_model_minutes: fc.expensive_model_minutes.unwrap_or(30).max(1),
            expensive_model_spend: fc.expensive_model_spend.unwrap_or(2.0).max(0.01),
            price_overrides,
            spec_dir: cwd.join(fc.spec_dir.clone().unwrap_or_else(|| "specs".into())),
            prompts,
            profiles,
            approval: match fc.approval.as_deref().unwrap_or("plan") {
                a @ ("ask" | "plan" | "auto") => a.to_string(),
                other => bail!("invalid approval '{other}' (ask | plan | auto)"),
            },
            sources,
            providers: fc.providers,
            architect: architect_b,
            developer: developer_b,
            policy,
            io_threads,
            max_steps: architect.max_steps.or(developer.max_steps).unwrap_or(80) as u32,
            loop_limit: fc.loop_limit,
            max_task_tokens: fc.max_task_tokens,
            env: env.clone(),
        };
        cfg.validate()?;
        Ok(cfg)
    }

    /// Fail fast on unknown providers or missing API keys.
    fn validate(&self) -> Result<()> {
        for (group, b) in [("architect", &self.architect), ("developer", &self.developer)] {
            let Some(p) = self.providers.get(&b.provider) else {
                bail!("the {group}'s provider '{}' is not configured ([providers.{}] in config.toml)", b.provider, b.provider)
            };
            if p.api_key.is_none() && p.api_key_env.is_some() && p.key(&self.env).is_none() {
                bail!("missing API key for provider '{}': set {}", b.provider, p.api_key_env.as_deref().unwrap_or(""));
            }
        }
        Ok(())
    }

    /// One HTTP client per provider used by a group.
    pub fn backends(&self) -> HashMap<String, Arc<dyn Backend>> {
        let mut out: HashMap<String, Arc<dyn Backend>> = HashMap::new();
        // Every usable provider, so /model can switch to any of them.
        for name in self.usable_providers() {
            let p = &self.providers[&name];
            let client = HttpClient::new(&name, p.base_url.clone().unwrap_or_default(), p.key(&self.env), p.max_concurrent.unwrap_or(8))
                .with_headers(p.headers.clone().unwrap_or_default().into_iter().collect())
                .with_session_header(p.session_header.clone())
                .with_protocol(p.protocol.unwrap_or_default())
                .with_idle_timeout(Duration::from_secs(p.idle_timeout_secs.unwrap_or(180)));
            out.insert(name.clone(), Arc::new(client));
        }
        out
    }
}

impl Config {
    /// Prices and context sizes: each configured provider's `/models` list (fetched in parallel,
    /// 5 s timeout, skipped with `discover = false` or without a key), then the role overrides.
    pub fn discover_catalog(&self) -> std::collections::HashMap<String, crate::llm::catalog::ModelInfo> {
        use crate::llm::catalog::{fetch_models, ModelInfo};
        let jobs: Vec<_> = self
            .providers
            .iter()
            .filter(|(_, p)| p.discover != Some(false) && p.protocol != Some(Protocol::Anthropic))
            .filter_map(|(name, p)| {
                let key = p.key(&self.env);
                if p.api_key_env.is_some() && key.is_none() {
                    return None;
                }
                let (name, base) = (name.clone(), p.base_url.clone().unwrap_or_default());
                let headers: Vec<(String, String)> = p.headers.clone().unwrap_or_default().into_iter().collect();
                Some(std::thread::spawn(move || (name, fetch_models(&base, key.as_deref(), &headers))))
            })
            .collect();
        let mut cat = std::collections::HashMap::new();
        for j in jobs {
            if let Ok((provider, models)) = j.join() {
                for (id, info) in models {
                    cat.insert(format!("{provider}/{id}"), info);
                }
            }
        }
        for (key, i, o) in &self.price_overrides {
            let base = cat.get(key).copied().unwrap_or_default();
            cat.insert(key.clone(), base.merged(ModelInfo { context: None, input_per_mtok: *i, output_per_mtok: *o }));
        }
        cat
    }

    /// Providers that can be used (configured, with their key when one is required).
    pub fn usable_providers(&self) -> Vec<String> {
        self.providers.iter().filter(|(_, p)| p.api_key_env.is_none() || p.key(&self.env).is_some()).map(|(n, _)| n.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_overrides_and_custom_profiles_come_from_roles_and_agent_files() {
        let dir = tempfile::tempdir().unwrap();
        let z = dir.path().join(".zilch");
        std::fs::create_dir_all(z.join("agents")).unwrap();
        std::fs::write(z.join("main.md"), "You are terse.").unwrap();
        std::fs::write(
            z.join("config.toml"),
            r#"
[defaults]
model = "base"
[roles.master]
system_prompt_file = "main.md"
prompt_mode = "replace"
[roles.coder]
system_prompt = "Prefer small diffs."
[roles.tester]
description = "runs and writes tests"
tools = ["read_file", "bash"]
model = "fast"
"#,
        )
        .unwrap();
        std::fs::write(z.join("agents").join("reviewer.toml"), "description = \"reviews diffs\"\nsystem_prompt = \"Be strict.\"\n")
            .unwrap();
        let env: BTreeMap<String, String> = [("OPENAI_API_KEY".to_string(), "k".to_string())].into();
        let c = Config::load_from(dir.path(), &[z], &env, &Overrides::default()).unwrap();
        assert_eq!(c.prompts[&Group::Architecture], PromptOverride { text: "You are terse.".into(), replace: true });
        assert_eq!(c.prompts[&Group::Development], PromptOverride { text: "Prefer small diffs.".into(), replace: false });
        let names: Vec<&str> = c.profiles.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["reviewer", "tester"]);
        let tester = &c.profiles[1];
        assert_eq!((tester.binding.model.as_str(), tester.tools.clone()), ("fast", vec!["read_file".to_string(), "bash".to_string()]));
        assert_eq!(c.profiles[0].binding.model, "base", "a profile without a model inherits the subagents' one");
        assert_eq!(c.profiles[0].prompt.as_ref().unwrap().text, "Be strict.");
    }

    #[test]
    fn reads_zilch_config_and_maps_roles_to_groups() {
        let dir = tempfile::tempdir().unwrap();
        let z = dir.path().join(".zilch");
        std::fs::create_dir_all(&z).unwrap();
        std::fs::write(
            z.join("config.toml"),
            r#"
max_workers = 8
spec_dir = "specs"
[providers.opencode]
base_url = "https://example.test/v1"
api_key_env = "OC_KEY"
session_header = "x-opencode-session"
[defaults]
provider = "opencode"
model = "base-free"
[roles.master]
model = "planner-free"
[roles.coder]
temperature = 0.2
[quotas]
developer = 500000
"#,
        )
        .unwrap();
        let env: BTreeMap<String, String> = [("OC_KEY".to_string(), "k".to_string())].into();
        let c = Config::load_from(dir.path(), std::slice::from_ref(&z), &env, &Overrides::default()).unwrap();
        assert_eq!((c.architect.provider.as_str(), c.architect.model.as_str()), ("opencode", "planner-free"));
        assert_eq!((c.developer.model.as_str(), c.developer.temperature), ("base-free", Some(0.2)));
        assert!(c.policy.is_secret_env("OC_KEY"), "provider keys are hidden from commands");
        assert_eq!(c.group_tokens, [(Group::Development, 500_000)].into());

        let ov = Overrides { developer_model: Some("fast-free".into()), ..Default::default() };
        let c = Config::load_from(dir.path(), std::slice::from_ref(&z), &env, &ov).unwrap();
        assert_eq!((c.architect.model.as_str(), c.developer.model.as_str()), ("planner-free", "fast-free"));

        let err = Config::load_from(dir.path(), &[z], &BTreeMap::new(), &Overrides::default()).unwrap_err().to_string();
        assert!(err.contains("missing API key") && err.contains("OC_KEY"), "{err}");
        let err = Config::load_from(dir.path(), &[], &BTreeMap::new(), &Overrides::default()).unwrap_err().to_string();
        assert!(err.contains("no model"), "{err}");
    }
}
