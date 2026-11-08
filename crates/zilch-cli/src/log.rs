//! Append-only JSONL session log of every agent event, for debugging and auditing.
//! Streamed text deltas are coalesced into one `text` record per assistant message.

use serde_json::json;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};
use zilch_core::{AgentEvent, EventKind};

pub struct EventLog {
    pub path: PathBuf,
    out: BufWriter<File>,
    text: HashMap<String, String>,
}

fn now_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

impl EventLog {
    /// Default location: `<data dir>/zilch/logs/<unix-ms>-<pid>.jsonl`.
    pub fn default_path() -> Option<PathBuf> {
        dirs::data_dir().map(|d| d.join("zilch").join("logs").join(format!("{}-{}.jsonl", now_ms(), std::process::id())))
    }

    pub fn open(path: &Path) -> std::io::Result<Self> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        #[cfg(unix)]
        {
            // Logs contain code and prompts: owner-only.
            use std::os::unix::fs::PermissionsExt;
            let _ = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600));
        }
        Ok(Self { path: path.to_path_buf(), out: BufWriter::new(file), text: HashMap::new() })
    }

    pub fn record(&mut self, kind: &str, data: serde_json::Value) {
        let _ = writeln!(self.out, "{}", json!({"ts": now_ms() as u64, "agent": "zilch", "type": kind, "data": data}));
        let _ = self.out.flush();
    }

    pub fn handle(&mut self, ev: &AgentEvent) {
        let line = match &ev.kind {
            EventKind::TextDelta(t) => {
                self.text.entry(ev.agent.clone()).or_default().push_str(t);
                return;
            }
            EventKind::TextDone => {
                let text = self.text.remove(&ev.agent).unwrap_or_default();
                json!({"ts": now_ms() as u64, "agent": ev.agent, "type": "text", "data": text})
            }
            _ => {
                let mut v = serde_json::to_value(ev).unwrap_or_default();
                if let Some(o) = v.as_object_mut() {
                    o.insert("ts".into(), json!(now_ms() as u64));
                    // Flatten {"kind": {"type", "data"}} into the record.
                    if let Some(serde_json::Value::Object(k)) = o.remove("kind") {
                        o.extend(k);
                    }
                }
                v
            }
        };
        let _ = writeln!(self.out, "{line}");
        let _ = self.out.flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coalesces_text_and_serializes_events() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("logs/s.jsonl");
        let mut log = EventLog::open(&p).unwrap();
        let ev = |agent: &str, kind| AgentEvent { agent: agent.into(), kind };
        log.handle(&ev("master", EventKind::TextDelta("Hel".into())));
        log.handle(&ev("coder#1", EventKind::TextDelta("other".into())));
        log.handle(&ev("master", EventKind::TextDelta("lo".into())));
        log.handle(&ev("master", EventKind::TextDone));
        log.handle(&ev("coder#1", EventKind::ToolEnd { call_id: "c".into(), name: "bash".into(), ok: false, summary: "Error: x".into() }));
        log.handle(&ev("master", EventKind::Notice("n".into())));
        let lines: Vec<serde_json::Value> =
            std::fs::read_to_string(&p).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!((lines[0]["type"].as_str(), lines[0]["data"].as_str()), (Some("text"), Some("Hello")));
        assert_eq!(lines[1]["type"], "tool_end");
        assert_eq!(lines[1]["data"]["ok"], false);
        assert_eq!(lines[1]["agent"], "coder#1");
        assert_eq!(lines[2]["data"], "n");
        assert!(lines.iter().all(|l| l["ts"].as_u64().is_some()));
    }
}
