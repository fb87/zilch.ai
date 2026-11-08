//! OS-level sandbox for `bash`, with no external dependencies:
//! - **Linux**: zilch's own namespace sandbox ([`super::sandbox_linux`]), entered through a helper
//!   re-exec of the zilch binary (`zilch __sandbox …`) or the bundled `zilch-sandbox` binary.
//! - **macOS**: the OS's built-in Seatbelt (`sandbox-exec`).
//!
//! Inside: the host filesystem is read-only except the workspace, a private `/tmp` and
//! copy-on-write tool caches; credential stores are hidden; `.git` is read-only; own PID
//! namespace (Linux); optional no-network.
//!
//! The file tools have their own workspace sandbox (see `ToolCtx::resolve_checked`); this module
//! covers arbitrary commands, which those checks can't.

use serde::{Deserialize, Serialize};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BashSandbox {
    /// Run commands directly on the host.
    Off,
    /// Sandbox when the OS supports it; otherwise run unsandboxed with a visible warning.
    #[default]
    Auto,
    /// Refuse to run commands without a working sandbox.
    #[serde(alias = "bwrap")]
    Required,
}

/// Tool caches under `$HOME` mounted copy-on-write, so builds work without letting commands
/// modify (or poison) the host caches.
pub const DEFAULT_OVERLAY: &[&str] =
    &[".cache", ".cargo", ".rustup", ".npm", ".pnpm-store", "go", ".m2", ".gradle", ".nuget", ".local/share/uv"];

/// Credential/secret locations under `$HOME` hidden from sandboxed commands.
pub const DEFAULT_HIDDEN: &[&str] = &[
    ".ssh",
    ".gnupg",
    ".aws",
    ".azure",
    ".kube",
    ".docker",
    ".netrc",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".cargo/credentials",
    ".cargo/credentials.toml",
    ".password-store",
    ".config/gcloud",
    ".config/gh",
    ".config/hub",
    ".config/zilch",
    ".local/share/zilch",
    ".local/share/keyrings",
    ".mozilla",
    ".config/google-chrome",
    ".config/chromium",
    ".bash_history",
    ".zsh_history",
    ".claude",
];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SandboxSpec {
    pub network: bool,
    /// Extra paths to hide (absolute, or `~/…`).
    pub hide: Vec<String>,
    /// Copy-on-write paths: commands see the host contents and may write, but writes go to a
    /// throwaway layer (tool caches like `~/.cargo`, `~/.cache`). `None` = [`DEFAULT_OVERLAY`].
    pub overlay: Option<Vec<String>>,
    /// Make the workspace's `.git/` read-only for commands (default true).
    pub protect_git: bool,
    /// Extra writable paths besides the workspace (absolute, or `~/…`), e.g. `~/.cache/pip`.
    pub writable: Vec<String>,
}

pub fn expand(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), dirs::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(p),
    }
}

// ---------------- Linux helper discovery ----------------

static HELPER: OnceLock<(PathBuf, Vec<String>)> = OnceLock::new();

/// Register how to start the sandbox helper. The `zilch` CLI registers itself
/// (`current_exe`, `["__sandbox"]`) so a single binary is all users need.
pub fn set_helper(program: PathBuf, args: Vec<String>) {
    let _ = HELPER.set((program, args));
}

/// The helper command: the registered one, else a `zilch-sandbox` binary next to the running
/// executable (or one directory up, which covers `target/debug/deps/<test>` in cargo tests).
pub fn helper() -> Option<(PathBuf, Vec<String>)> {
    if let Some(h) = HELPER.get() {
        return Some(h.clone());
    }
    let exe = std::env::current_exe().ok()?;
    let dir = exe.parent()?;
    [dir.join("zilch-sandbox"), dir.parent()?.join("zilch-sandbox")].into_iter().find(|p| p.is_file()).map(|p| (p, vec![]))
}

/// Run `cmd` through the helper with `spec` in `workspace` (used by the probes).
#[cfg(target_os = "linux")]
fn probe(workspace: &Path, spec: &SandboxSpec, cmd: &str) -> bool {
    let (program, args) = wrapper(workspace, spec);
    std::process::Command::new(program)
        .args(args)
        .args(["bash", "-c", cmd])
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Does the native Linux sandbox work here? (User namespaces can be disabled by the kernel,
/// AppArmor, or a container runtime — probed once.)
pub fn native_available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            if helper().is_none() {
                return false;
            }
            let ws = std::env::temp_dir();
            probe(&ws, &SandboxSpec { network: true, overlay: Some(vec![]), ..Default::default() }, "true")
        }
        #[cfg(not(target_os = "linux"))]
        false
    })
}

/// Does copy-on-write (unprivileged overlayfs) work inside the sandbox?
pub fn overlay_available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        #[cfg(target_os = "linux")]
        {
            if !native_available() {
                return false;
            }
            let Some(home) = dirs::home_dir() else { return false };
            let probe_dir = home.join(".cache");
            if !probe_dir.is_dir() {
                return false;
            }
            let marker = probe_dir.join(format!(".zilch-overlay-probe-{}", std::process::id()));
            let spec = SandboxSpec { network: true, overlay: Some(vec![probe_dir.display().to_string()]), ..Default::default() };
            let ok = probe(&std::env::temp_dir(), &spec, &format!("touch '{}'", marker.display()));
            // A real overlay writes to the throwaway layer, never to the host directory.
            let leaked = marker.exists();
            let _ = std::fs::remove_file(&marker);
            ok && !leaked
        }
        #[cfg(not(target_os = "linux"))]
        false
    })
}

// ---------------- macOS (Seatbelt / sandbox-exec) ----------------

/// Does `sandbox-exec` work here (macOS)?
pub fn seatbelt_available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        cfg!(target_os = "macos")
            && std::process::Command::new("sandbox-exec")
                .args(["-p", "(version 1)(allow default)", "/usr/bin/true"])
                .stdin(std::process::Stdio::null())
                .stdout(std::process::Stdio::null())
                .stderr(std::process::Stdio::null())
                .status()
                .is_ok_and(|s| s.success())
    })
}

/// Quote a path for SBPL, resolved to its real location first: Seatbelt matches rules against
/// resolved paths, so a rule written with a symlinked path (macOS: /var → /private/var, /tmp →
/// /private/tmp) would silently never match — leaving "hidden" files readable.
fn sbpl_str(p: &Path) -> String {
    let real = real_path(p);
    let s = real.to_string_lossy();
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

/// Canonicalize the longest existing prefix of `p` and re-append the rest.
pub fn real_path(p: &Path) -> PathBuf {
    let mut existing = p.to_path_buf();
    let mut tail = Vec::new();
    while !existing.exists() {
        match (existing.file_name().map(|s| s.to_os_string()), existing.parent()) {
            (Some(n), Some(parent)) => {
                tail.push(n);
                existing = parent.to_path_buf();
            }
            _ => return p.to_path_buf(),
        }
    }
    let mut r = existing.canonicalize().unwrap_or(existing);
    for t in tail.iter().rev() {
        r.push(t);
    }
    r
}

/// Seatbelt profile with the same policy as the bubblewrap sandbox. Differences forced by the
/// platform: there is no copy-on-write overlay, so tool caches are writable directly, and there
/// is no PID namespace (the process-group kill still applies).
pub fn seatbelt_profile(workspace: &Path, spec: &SandboxSpec) -> String {
    let home = dirs::home_dir();
    let mut p = String::from("(version 1)\n(allow default)\n");
    // Writes: only the workspace, temp dirs, devices, caches and configured paths.
    p.push_str("(deny file-write*)\n(allow file-write*\n");
    let mut writable: Vec<PathBuf> = vec![workspace.to_path_buf(), "/private/tmp".into(), "/private/var/folders".into(), "/dev".into()];
    let overlays: Vec<PathBuf> = match &spec.overlay {
        Some(list) => list.iter().map(|x| expand(x)).collect(),
        None => home.iter().flat_map(|h| DEFAULT_OVERLAY.iter().map(move |x| h.join(x))).collect(),
    };
    writable.extend(overlays);
    if let Some(h) = &home {
        writable.push(h.join("Library/Caches"));
    }
    writable.extend(spec.writable.iter().map(|x| expand(x)));
    for w in &writable {
        p.push_str(&format!("  (subpath {})\n", sbpl_str(w)));
    }
    p.push_str(")\n");
    if spec.protect_git {
        p.push_str(&format!("(deny file-write* (subpath {}))\n", sbpl_str(&workspace.join(".git"))));
    }
    // Secrets: neither readable nor writable (later rules win).
    let mut hidden: Vec<PathBuf> = home.iter().flat_map(|h| DEFAULT_HIDDEN.iter().map(move |x| h.join(x))).collect();
    if let Some(h) = &home {
        hidden.push(h.join("Library/Keychains"));
    }
    hidden.extend(spec.hide.iter().map(|x| expand(x)));
    for h in hidden.iter().filter(|h| !workspace.starts_with(h)) {
        p.push_str(&format!("(deny file-read* file-write* (subpath {}))\n", sbpl_str(h)));
    }
    if !spec.network {
        p.push_str("(deny network*)\n(allow network* (remote unix-socket))\n");
    }
    p
}

// ---------------- platform dispatch ----------------
/// Is an OS sandbox usable on this machine?
pub fn available() -> bool {
    if cfg!(target_os = "linux") {
        native_available()
    } else if cfg!(target_os = "macos") {
        seatbelt_available()
    } else {
        false
    }
}

/// Program + leading arguments that run a command (appended by the caller) inside the sandbox.
pub fn wrapper(workspace: &Path, spec: &SandboxSpec) -> (OsString, Vec<OsString>) {
    if cfg!(target_os = "macos") {
        return ("sandbox-exec".into(), vec!["-p".into(), seatbelt_profile(workspace, spec).into()]);
    }
    let (program, prefix) = helper().unwrap_or_else(|| (PathBuf::from("zilch-sandbox"), vec![]));
    let mut args: Vec<OsString> = prefix.into_iter().map(OsString::from).collect();
    #[cfg(target_os = "linux")]
    {
        let h = super::sandbox_linux::HelperSpec { workspace: workspace.to_path_buf(), spec: spec.clone() };
        args.push(serde_json::to_string(&h).expect("spec serializes").into());
    }
    args.push("--".into());
    (program.into_os_string(), args)
}

/// Human-readable status of the bash sandbox for `mode`, and whether commands are confined.
pub fn status(mode: BashSandbox, network: bool) -> (bool, String) {
    let net = if network { "network on" } else { "no network" };
    let why = if cfg!(target_os = "macos") {
        "sandbox-exec unavailable"
    } else if cfg!(target_os = "linux") {
        "user namespaces unavailable or helper missing"
    } else {
        "no sandbox for this OS"
    };
    match mode {
        BashSandbox::Off => (false, "OFF: commands run directly on the host".into()),
        _ if available() && cfg!(target_os = "macos") => {
            (true, format!("sandbox-exec: writes limited to workspace/tmp/caches, secrets hidden, .git protected, {net}"))
        }
        _ if available() => {
            let ovl = if overlay_available() { "copy-on-write caches" } else { "private ~/.cache" };
            (true, format!("native namespaces: read-only host, secrets hidden, {ovl}, .git protected, own PID ns, {net}"))
        }
        BashSandbox::Auto => (false, format!("UNAVAILABLE ({why}): commands run directly on the host")),
        BashSandbox::Required => (false, format!("REQUIRED but unavailable ({why}): bash will refuse to run")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seatbelt_profile_encodes_the_policy() {
        let spec = SandboxSpec {
            network: false,
            hide: vec!["/srv/secret \"x\"".into()],
            writable: vec![],
            overlay: Some(vec![]),
            protect_git: true,
        };
        let p = seatbelt_profile(Path::new("/Users/u/proj"), &spec);
        assert!(p.starts_with("(version 1)\n(allow default)\n(deny file-write*)"));
        assert!(p.contains("(subpath \"/Users/u/proj\")"), "workspace writable");
        assert!(p.contains("(deny file-write* (subpath \"/Users/u/proj/.git\"))"), ".git protected");
        assert!(p.contains("(deny file-read* file-write* (subpath \"/srv/secret \\\"x\\\"\"))"), "hidden + quoting");
        assert!(p.contains("(deny network*)"));
        // Deny rules for .git / secrets come after the workspace allow (SBPL: last match wins).
        assert!(p.find(".git\"))").unwrap() > p.find("(subpath \"/Users/u/proj\")").unwrap());
        let open = seatbelt_profile(Path::new("/w"), &SandboxSpec { network: true, protect_git: false, ..Default::default() });
        assert!(!open.contains("network") && !open.contains("/w/.git\""), "{open}");
    }

    #[cfg(unix)]
    #[test]
    fn seatbelt_rules_use_real_paths() {
        // Regression (found by macOS CI): Seatbelt matches resolved paths, so symlinked paths in
        // rules never matched — e.g. a hidden dir under /var/folders (→ /private/var/folders).
        let base = tempfile::tempdir().unwrap();
        let real = base.path().canonicalize().unwrap().join("real");
        std::fs::create_dir_all(real.join("ws")).unwrap();
        std::fs::create_dir_all(real.join("vault")).unwrap();
        let link = base.path().join("link");
        std::os::unix::fs::symlink(&real, &link).unwrap();
        let spec = SandboxSpec {
            hide: vec![link.join("vault").display().to_string()],
            overlay: Some(vec![]),
            protect_git: true,
            network: true,
            ..Default::default()
        };
        let p = seatbelt_profile(&link.join("ws"), &spec);
        assert!(p.contains(&format!("(subpath \"{}\")", real.join("ws").display())), "workspace resolved: {p}");
        assert!(p.contains(&format!("(subpath \"{}\"))", real.join("vault").display())), "hidden dir resolved: {p}");
        assert!(p.contains(&format!("{}\"))", real.join("ws/.git").display())), "non-existent .git resolved via parent: {p}");
        assert!(!p.contains(&link.display().to_string()), "no symlinked path may remain: {p}");
    }

    #[cfg(target_os = "macos")]
    #[test]
    fn seatbelt_confines_commands_on_macos() {
        assert!(seatbelt_available(), "sandbox-exec should work on macOS");
        let outer = std::env::temp_dir().join(format!("zilch-sbx-{}", std::process::id()));
        let ws = outer.join("ws");
        std::fs::create_dir_all(ws.join(".git")).unwrap();
        let vault = outer.join("vault");
        std::fs::create_dir_all(&vault).unwrap();
        std::fs::write(vault.join("key"), "SECRET").unwrap();
        let spec = SandboxSpec {
            network: false,
            hide: vec![vault.display().to_string()],
            writable: vec![],
            overlay: Some(vec![]),
            protect_git: true,
        };
        let (prog, args) = wrapper(&ws, &spec);
        let run = |cmd: &str| {
            let o = std::process::Command::new(&prog)
                .args(&args)
                .args(["bash", "-c", &format!("exec 2>&1\n{cmd}")])
                .current_dir(&ws)
                .output()
                .unwrap();
            String::from_utf8_lossy(&o.stdout).into_owned()
        };
        assert!(!run(&format!("cat {}/key", vault.display())).contains("SECRET"));
        let home_probe = dirs::home_dir().unwrap().join(".zilch-sbx-probe");
        run(&format!("touch {}", home_probe.display()));
        assert!(!home_probe.exists(), "write outside workspace escaped");
        run("touch .git/evil");
        assert!(!ws.join(".git/evil").exists(), ".git must be read-only");
        run("echo ok > inside.txt");
        assert!(ws.join("inside.txt").exists());
        let _ = std::fs::remove_dir_all(&outer);
    }
}
