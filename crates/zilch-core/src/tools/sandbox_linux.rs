//! Native Linux sandbox (no external bubblewrap): user + mount + PID + IPC + UTS (+ network)
//! namespaces set up directly with libc.
//!
//! Must run in a **single-threaded** process (unshare(CLONE_NEWUSER) refuses multithreaded
//! callers), so it is entered through a helper re-exec: `zilch __sandbox <spec-json> -- cmd…`
//! (or the `zilch-sandbox` binary). Layout inside:
//! - every mount read-only, except: the workspace, a private `/tmp` and `/dev/shm`, configured
//!   writable paths, and copy-on-write overlays over tool caches;
//! - secret paths covered by an empty tmpfs (dirs) or `/dev/null` (files);
//! - `<workspace>/.git` read-only (optional);
//! - own PID namespace with a fresh `/proc` (the command is PID 1; when it exits, every process
//!   inside dies), dies with its parent, `no_new_privs`;
//! - optional private network namespace with only loopback (brought up).

use super::sandbox::{expand, SandboxSpec, DEFAULT_HIDDEN, DEFAULT_OVERLAY};
use serde::{Deserialize, Serialize};
use std::ffi::{CString, OsStr};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// Everything the helper needs, passed as one JSON argument.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HelperSpec {
    pub workspace: PathBuf,
    pub spec: SandboxSpec,
}

fn cstr(p: &OsStr) -> CString {
    CString::new(p.as_bytes()).unwrap_or_else(|_| CString::new("/nonexistent").unwrap())
}

fn err(what: &str) -> String {
    format!("{what}: {}", std::io::Error::last_os_error())
}

fn write_file(path: &str, data: &str) -> Result<(), String> {
    std::fs::write(path, data).map_err(|e| format!("writing {path}: {e}"))
}

unsafe fn mount(src: Option<&OsStr>, target: &Path, fstype: Option<&str>, flags: libc::c_ulong, data: Option<&str>) -> Result<(), String> {
    let src_c = src.map(cstr);
    let tgt = cstr(target.as_os_str());
    let fs_c = fstype.map(|f| CString::new(f).unwrap());
    let data_c = data.map(|d| CString::new(d).unwrap());
    let r = libc::mount(
        src_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        tgt.as_ptr(),
        fs_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr()),
        flags,
        data_c.as_ref().map_or(std::ptr::null(), |c| c.as_ptr() as *const libc::c_void),
    );
    if r == 0 {
        Ok(())
    } else {
        Err(err(&format!("mount {}", target.display())))
    }
}

unsafe fn bind(src: &Path, dst: &Path) -> Result<(), String> {
    mount(Some(src.as_os_str()), dst, None, libc::MS_BIND | libc::MS_REC, None)
}

/// Mount points from /proc/self/mountinfo (field 5, octal escapes decoded).
fn mount_points() -> Vec<PathBuf> {
    let info = std::fs::read_to_string("/proc/self/mountinfo").unwrap_or_default();
    info.lines()
        .filter_map(|l| l.split(' ').nth(4))
        .map(|raw| {
            let mut out = Vec::new();
            let b = raw.as_bytes();
            let mut i = 0;
            while i < b.len() {
                if b[i] == b'\\' && i + 3 < b.len() {
                    if let Ok(v) = u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 4]).unwrap_or("x"), 8) {
                        out.push(v);
                        i += 4;
                        continue;
                    }
                }
                out.push(b[i]);
                i += 1;
            }
            PathBuf::from(OsStr::from_bytes(&out))
        })
        .collect()
}

/// Flags of an existing mount that must be preserved on a bind-remount inside a user namespace
/// (the kernel locks them; dropping one fails with EPERM).
unsafe fn locked_flags(p: &Path) -> libc::c_ulong {
    let c = cstr(p.as_os_str());
    let mut st: libc::statvfs = std::mem::zeroed();
    if libc::statvfs(c.as_ptr(), &mut st) != 0 {
        return 0;
    }
    let f = st.f_flag;
    let mut out = 0;
    for (st_flag, ms_flag) in [
        (libc::ST_NOSUID, libc::MS_NOSUID),
        (libc::ST_NODEV, libc::MS_NODEV),
        (libc::ST_NOEXEC, libc::MS_NOEXEC),
        (libc::ST_NOATIME, libc::MS_NOATIME),
        (libc::ST_NODIRATIME, libc::MS_NODIRATIME),
        (libc::ST_RELATIME, libc::MS_RELATIME),
    ] {
        if f & st_flag != 0 {
            out |= ms_flag;
        }
    }
    out
}

unsafe fn loopback_up() {
    let fd = libc::socket(libc::AF_INET, libc::SOCK_DGRAM | libc::SOCK_CLOEXEC, 0);
    if fd < 0 {
        return;
    }
    let mut ifr: libc::ifreq = std::mem::zeroed();
    for (i, b) in b"lo".iter().enumerate() {
        ifr.ifr_name[i] = *b as libc::c_char;
    }
    if libc::ioctl(fd, libc::SIOCGIFFLAGS, &mut ifr) == 0 {
        ifr.ifr_ifru.ifru_flags |= libc::IFF_UP as libc::c_short;
        libc::ioctl(fd, libc::SIOCSIFFLAGS, &ifr);
    }
    libc::close(fd);
}

/// Build the filesystem view (runs as PID 1 inside the new namespaces).
unsafe fn setup_mounts(h: &HelperSpec) -> Result<(), String> {
    let ws = &h.workspace;
    let spec = &h.spec;
    let home = dirs::home_dir();
    // Nothing we do may propagate back to the host.
    mount(None, Path::new("/"), None, libc::MS_REC | libc::MS_PRIVATE, None)?;

    let mut keep_writable: Vec<PathBuf> = Vec::new();

    // Writable dirs: the workspace and configured paths. Open handles FIRST: the private /tmp
    // mounted next would hide anything that lives under /tmp (temp-dir workspaces), so they are
    // bound back through /proc/self/fd afterwards.
    let mut writable: Vec<PathBuf> = vec![ws.clone()];
    writable.extend(spec.writable.iter().map(|p| expand(p)).filter(|p| p.exists()));
    let mut handles = Vec::new();
    for w in &writable {
        let c = cstr(w.as_os_str());
        let fd = libc::open(c.as_ptr(), libc::O_PATH | libc::O_CLOEXEC);
        if fd < 0 {
            return Err(err(&format!("open {}", w.display())));
        }
        handles.push((w.clone(), fd));
    }

    // Private scratch space.
    mount(Some(OsStr::new("tmpfs")), Path::new("/tmp"), Some("tmpfs"), libc::MS_NOSUID | libc::MS_NODEV, Some("mode=1777"))?;
    keep_writable.push("/tmp".into());
    if Path::new("/dev/shm").is_dir() {
        mount(Some(OsStr::new("tmpfs")), Path::new("/dev/shm"), Some("tmpfs"), libc::MS_NOSUID | libc::MS_NODEV, Some("mode=1777"))?;
        keep_writable.push("/dev/shm".into());
    }

    // Writable binds (new mounts, so the read-only pass below skips them).
    for (w, fd) in handles {
        if !w.exists() {
            // Hidden under the new /tmp: recreate the mount point inside it.
            if w.is_file() {
                let _ = std::fs::File::create(&w);
            } else {
                std::fs::create_dir_all(&w).map_err(|e| format!("mkdir {}: {e}", w.display()))?;
            }
        }
        let src = PathBuf::from(format!("/proc/self/fd/{fd}"));
        bind(&src, &w)?;
        libc::close(fd);
        keep_writable.push(w);
    }

    // Copy-on-write tool caches: overlayfs with the upper layer in the private /tmp.
    let overlays: Vec<PathBuf> = match &spec.overlay {
        Some(list) => list.iter().map(|p| expand(p)).collect(),
        None => home.iter().flat_map(|h| DEFAULT_OVERLAY.iter().map(move |p| h.join(p))).collect(),
    };
    for (i, o) in overlays.iter().filter(|p| p.is_dir() && !ws.starts_with(p)).enumerate() {
        let base = PathBuf::from(format!("/tmp/.zilch-overlay/{i}"));
        let (upper, work) = (base.join("upper"), base.join("work"));
        let _ = std::fs::create_dir_all(&upper);
        let _ = std::fs::create_dir_all(&work);
        let opts = format!("lowerdir={},upperdir={},workdir={}", o.display(), upper.display(), work.display());
        if mount(Some(OsStr::new("overlay")), o, Some("overlay"), 0, Some(&opts)).is_ok() {
            keep_writable.push(o.clone());
        } else if home.as_ref().is_some_and(|h| *o == h.join(".cache")) {
            // No unprivileged overlayfs: give tools a private, empty cache instead.
            mount(Some(OsStr::new("tmpfs")), o, Some("tmpfs"), libc::MS_NOSUID | libc::MS_NODEV, None)?;
            keep_writable.push(o.clone());
        }
    }

    // Hide secrets (these mounts become read-only in the pass below).
    let mut hidden: Vec<PathBuf> = home.iter().flat_map(|h| DEFAULT_HIDDEN.iter().map(move |p| h.join(p))).collect();
    hidden.extend(spec.hide.iter().map(|p| expand(p)));
    for p in hidden.iter().filter(|p| !ws.starts_with(p)) {
        if p.is_dir() {
            mount(Some(OsStr::new("tmpfs")), p, Some("tmpfs"), libc::MS_NOSUID | libc::MS_NODEV, Some("size=4k,mode=0700"))?;
        } else if p.exists() {
            bind(Path::new("/dev/null"), p)?;
        }
    }

    // Git metadata: a separate mount that the read-only pass will cover.
    let git = ws.join(".git");
    if spec.protect_git && git.exists() {
        bind(&git, &git)?;
    }

    // Fresh /proc for the new PID namespace (only sandbox processes visible).
    mount(Some(OsStr::new("proc")), Path::new("/proc"), Some("proc"), libc::MS_NOSUID | libc::MS_NODEV | libc::MS_NOEXEC, None)?;

    // Read-only pass: every other mount point becomes read-only, keeping its locked flags.
    for mp in mount_points() {
        if keep_writable.contains(&mp) || mp.starts_with("/proc") {
            continue;
        }
        let flags = libc::MS_REMOUNT | libc::MS_BIND | libc::MS_RDONLY | locked_flags(&mp);
        if let Err(e) = mount(None, &mp, None, flags, None) {
            // Unreachable mount points (permission-denied paths, stale entries) can't be written
            // through either; anything else means a host mount could stay writable → refuse.
            let errno = std::io::Error::last_os_error().raw_os_error().unwrap_or(0);
            if errno != libc::EACCES && errno != libc::ENOENT {
                return Err(e);
            }
        }
    }
    Ok(())
}

/// Entry point of the helper process. Never returns: execs the command or exits with an error.
pub fn helper_main(mut args: impl Iterator<Item = String>) -> ! {
    let fail = |msg: String| -> ! {
        eprintln!("zilch sandbox: {msg}");
        std::process::exit(126);
    };
    let spec_json = args.next().unwrap_or_else(|| fail("missing spec".into()));
    if args.next().as_deref() != Some("--") {
        fail("expected `--` before the command".into());
    }
    let argv: Vec<String> = args.collect();
    if argv.is_empty() {
        fail("missing command".into());
    }
    let h: HelperSpec = serde_json::from_str(&spec_json).unwrap_or_else(|e| fail(format!("bad spec: {e}")));

    unsafe {
        // Die if zilch dies.
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
        let (uid, gid) = (libc::getuid(), libc::getgid());
        let mut flags = libc::CLONE_NEWUSER | libc::CLONE_NEWNS | libc::CLONE_NEWPID | libc::CLONE_NEWIPC | libc::CLONE_NEWUTS;
        if !h.spec.network {
            flags |= libc::CLONE_NEWNET;
        }
        if libc::unshare(flags) != 0 {
            fail(err("unshare (are unprivileged user namespaces disabled?)"));
        }
        // Keep the same identity inside, so files stay owned by the user.
        let _ = write_file("/proc/self/setgroups", "deny");
        if let Err(e) = write_file("/proc/self/uid_map", &format!("{uid} {uid} 1")) {
            fail(e);
        }
        if let Err(e) = write_file("/proc/self/gid_map", &format!("{gid} {gid} 1")) {
            fail(e);
        }

        // The new PID namespace applies to children: fork so the command becomes PID 1.
        let pid = libc::fork();
        if pid < 0 {
            fail(err("fork"));
        }
        if pid > 0 {
            // Outer helper: wait and propagate the exit status.
            let mut status = 0;
            loop {
                let r = libc::waitpid(pid, &mut status, 0);
                if r == pid {
                    break;
                }
                if r < 0 && std::io::Error::last_os_error().raw_os_error() != Some(libc::EINTR) {
                    fail(err("waitpid"));
                }
            }
            if libc::WIFEXITED(status) {
                std::process::exit(libc::WEXITSTATUS(status));
            }
            std::process::exit(128 + libc::WTERMSIG(status));
        }

        // PID 1 inside the sandbox.
        libc::prctl(libc::PR_SET_PDEATHSIG, libc::SIGKILL);
        if let Err(e) = setup_mounts(&h) {
            fail(e);
        }
        if !h.spec.network {
            loopback_up();
        }
        libc::prctl(libc::PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0);
        let ws = cstr(h.workspace.as_os_str());
        if libc::chdir(ws.as_ptr()) != 0 {
            fail(err("chdir"));
        }
        std::env::set_var("TMPDIR", "/tmp");
        let c_argv: Vec<CString> = argv.iter().map(|a| CString::new(a.as_bytes()).unwrap()).collect();
        let mut ptrs: Vec<*const libc::c_char> = c_argv.iter().map(|c| c.as_ptr()).collect();
        ptrs.push(std::ptr::null());
        libc::execvp(ptrs[0], ptrs.as_ptr());
        fail(err(&format!("exec {}", argv[0])));
    }
}
