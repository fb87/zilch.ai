//! OS-level sandbox for `bash` using bubblewrap (Linux).
//!
//! Inside the sandbox:
//! - the whole filesystem is **read-only**, except the workspace and a private `/tmp`
//!   (plus any explicitly configured writable paths);
//! - credential stores in `$HOME` (`~/.ssh`, `~/.aws`, `~/.gnupg`, zilch's own config/logs, …) are
//!   **hidden** (empty tmpfs / `/dev/null`);
//! - it gets its own PID namespace (killing bwrap kills everything inside) and, optionally,
//!   **no network**.
//!
//! The file tools have their own workspace sandbox (see `ToolCtx::resolve_checked`); this module
//! covers arbitrary commands, which those checks can't.

use serde::Deserialize;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BashSandbox {
    /// Run commands directly on the host.
    Off,
    /// Use bubblewrap when it works on this machine; otherwise run unsandboxed with a warning.
    #[default]
    Auto,
    /// Require bubblewrap; refuse to run commands without it.
    Bwrap,
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

#[derive(Debug, Clone, Default)]
pub struct SandboxSpec {
    pub network: bool,
    /// Extra paths to hide (absolute, or `~/…`).
    pub hide: Vec<String>,
    /// Copy-on-write paths: commands see the host contents and may write, but writes go to a
    /// throwaway layer (tool caches like `~/.cargo`, `~/.cache`). `None` = [`DEFAULT_OVERLAY`].
    pub overlay: Option<Vec<String>>,
    /// Extra writable paths besides the workspace (absolute, or `~/…`), e.g. `~/.cache/pip`.
    pub writable: Vec<String>,
}

pub fn expand(p: &str) -> PathBuf {
    match (p.strip_prefix("~/"), dirs::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(p),
    }
}

/// Does bubblewrap work here? (Probed once: it can be installed but blocked, e.g. inside
/// containers without unprivileged user namespaces.)
pub fn bwrap_available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        std::process::Command::new("bwrap")
            .args(["--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--unshare-pid", "--die-with-parent", "--", "true"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success())
    })
}

/// Does bubblewrap's copy-on-write overlay work here (needs unprivileged overlayfs)?
pub fn overlay_available() -> bool {
    static OK: OnceLock<bool> = OnceLock::new();
    *OK.get_or_init(|| {
        let Ok(dir) = tempfile_dir() else { return false };
        let ok = std::process::Command::new("bwrap")
            .args(["--ro-bind", "/", "/", "--dev", "/dev", "--overlay-src"])
            .arg(&dir)
            .arg("--tmp-overlay")
            .arg(&dir)
            .args(["--unshare-pid", "--die-with-parent", "--", "true"])
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .is_ok_and(|s| s.success());
        let _ = std::fs::remove_dir(&dir);
        ok
    })
}

fn tempfile_dir() -> std::io::Result<PathBuf> {
    let d = std::env::temp_dir().join(format!("zilch-ovl-probe-{}", std::process::id()));
    std::fs::create_dir_all(&d)?;
    Ok(d)
}

/// bubblewrap arguments (everything before `--`) for running a command in `workspace`.
pub fn bwrap_args(workspace: &Path, spec: &SandboxSpec) -> Vec<OsString> {
    let mut a: Vec<OsString> = Vec::new();
    let mut push = |xs: &[&std::ffi::OsStr]| a.extend(xs.iter().map(|x| x.to_os_string()));
    push(&["--ro-bind".as_ref(), "/".as_ref(), "/".as_ref()]);
    push(&["--dev".as_ref(), "/dev".as_ref(), "--proc".as_ref(), "/proc".as_ref()]);
    push(&["--tmpfs".as_ref(), "/tmp".as_ref()]);

    // Tool caches: copy-on-write (host contents visible, writes discarded). Mounted before the
    // hides so credentials inside them (e.g. ~/.cargo/credentials) are still hidden.
    let home = dirs::home_dir();
    let overlays: Vec<PathBuf> = match &spec.overlay {
        Some(list) => list.iter().map(|p| expand(p)).collect(),
        None => home.iter().flat_map(|h| DEFAULT_OVERLAY.iter().map(move |p| h.join(p))).collect(),
    };
    let overlay_ok = overlay_available();
    for o in overlays.iter().filter(|p| p.is_dir() && !workspace.starts_with(p)) {
        if overlay_ok {
            push(&["--overlay-src".as_ref(), o.as_os_str(), "--tmp-overlay".as_ref(), o.as_os_str()]);
        } else if home.as_ref().is_some_and(|h| *o == h.join(".cache")) {
            // No overlay support: at least give tools a private, empty, writable cache dir.
            push(&["--tmpfs".as_ref(), o.as_os_str()]);
        }
    }

    // Hide secrets: directories become empty tmpfs, files become /dev/null.
    let mut hidden: Vec<PathBuf> = dirs::home_dir().map(|h| DEFAULT_HIDDEN.iter().map(|p| h.join(p)).collect()).unwrap_or_default();
    hidden.extend(spec.hide.iter().map(|p| expand(p)));
    for h in hidden {
        // Never hide the workspace itself (or a parent of it).
        if workspace.starts_with(&h) {
            continue;
        }
        if h.is_dir() {
            push(&["--tmpfs".as_ref(), h.as_os_str()]);
        } else if h.exists() {
            push(&["--ro-bind".as_ref(), "/dev/null".as_ref(), h.as_os_str()]);
        }
    }

    for w in spec.writable.iter().map(|p| expand(p)).filter(|p| p.exists()) {
        push(&["--bind".as_ref(), w.as_os_str(), w.as_os_str()]);
    }
    // The workspace is writable (bound last so it wins over /tmp or hidden parents).
    push(&["--bind".as_ref(), workspace.as_os_str(), workspace.as_os_str()]);
    push(&["--chdir".as_ref(), workspace.as_os_str()]);
    push(&["--setenv".as_ref(), "TMPDIR".as_ref(), "/tmp".as_ref()]);
    push(&["--unshare-pid".as_ref(), "--unshare-ipc".as_ref(), "--unshare-uts".as_ref(), "--die-with-parent".as_ref()]);
    if !spec.network {
        push(&["--unshare-net".as_ref()]);
    }
    a
}

/// Human-readable status of the bash sandbox for `mode`, and whether commands are confined.
pub fn status(mode: BashSandbox, network: bool) -> (bool, String) {
    let net = if network { "network on" } else { "no network" };
    match mode {
        BashSandbox::Off => (false, "OFF: commands run directly on the host".into()),
        _ if bwrap_available() => {
            let ovl = if overlay_available() { "copy-on-write caches" } else { "private ~/.cache" };
            (true, format!("bubblewrap: read-only host, secrets hidden, {ovl}, {net}"))
        }
        BashSandbox::Auto => (false, "UNAVAILABLE (bubblewrap missing or blocked): commands run directly on the host".into()),
        BashSandbox::Bwrap => (false, "REQUIRED but bubblewrap is unavailable: bash will refuse to run".into()),
    }
}
