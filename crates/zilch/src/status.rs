//! Live status bar for the REPL: one row above the input line, redrawn in place.
//!
//! rustyline owns the bottom of the screen while the user types, so we cannot draw there
//! ourselves. Instead the prompt is two rows, `"\n[code] › "`, and the line editor's highlighter
//! fills the first (empty) row with the current status each time rustyline redraws. Its cursor
//! math uses the plain prompt, which has the same rows, so nothing shifts. A small thread asks for
//! a redraw a few times a second while something runs, which animates the spinner and timers.
//!
//! On Enter, rustyline does a final forced refresh; the status row is blanked for it, so
//! scrollback keeps only an empty spacer row above each typed line, never a stale status.

use rustyline::completion::Completer;
use rustyline::highlight::{CmdKind, Highlighter};
use rustyline::hint::Hinter;
use rustyline::validate::Validator;
use rustyline::{ExternalPrinter, Helper};
use std::borrow::Cow;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Shared between the renderer (writes the text), the line editor (draws it) and the refresher.
#[derive(Default)]
pub struct StatusBar {
    text: Mutex<String>,
    /// Something is running: keep redrawing so spinners and elapsed times move.
    busy: AtomicBool,
    /// The text changed since the last redraw.
    dirty: AtomicBool,
    /// The line editor is waiting for input (redraw requests are only sent then).
    pub reading: AtomicBool,
    /// Set during rustyline's final refresh on Enter: draw an empty status row.
    accepting: AtomicBool,
}

impl StatusBar {
    pub fn set(&self, text: String, busy: bool) {
        let mut t = self.text.lock().unwrap();
        if *t != text {
            *t = text;
            self.dirty.store(true, Ordering::SeqCst);
        }
        self.busy.store(busy, Ordering::SeqCst);
    }

    pub fn text(&self) -> String {
        self.text.lock().unwrap().clone()
    }
}

/// Completion: (line, cursor) → (where the word starts, candidates).
pub type CompleteFn = dyn Fn(&str, usize) -> (usize, Vec<String>) + Send + Sync;

/// The rustyline helper: the prompt highlighter draws the status row; the completer offers commands,
/// arguments and @ mentions.
pub struct StatusHelper {
    pub bar: Arc<StatusBar>,
    pub complete: Option<Box<CompleteFn>>,
}

impl Highlighter for StatusHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, default: bool) -> Cow<'b, str> {
        // Only our two-row prompt carries a status row (approval questions etc. are left alone).
        match prompt.strip_prefix('\n') {
            Some(rest) if default && !self.bar.accepting.load(Ordering::SeqCst) => {
                // Autowrap (DECAWM) is off while the row is drawn: a terminal that draws some glyph wider
                // than counted cuts the row instead of wrapping it (a wrapped row stays behind as a
                // stale copy on every redraw).
                Cow::Owned(format!("\r\x1b[2K\x1b[?7l{}\x1b[?7h\n{rest}", self.bar.text()))
            }
            _ => Cow::Borrowed(prompt),
        }
    }

    fn highlight_char(&self, _line: &str, _pos: usize, kind: CmdKind) -> bool {
        // Returning true makes rustyline do a forced refresh when the line is accepted, which is
        // when we blank the status row (see module docs).
        self.bar.accepting.store(kind == CmdKind::ForcedRefresh, Ordering::SeqCst);
        true
    }
}

impl Completer for StatusHelper {
    type Candidate = String;
    fn complete(&self, line: &str, pos: usize, _ctx: &rustyline::Context<'_>) -> rustyline::Result<(usize, Vec<String>)> {
        Ok(self.complete.as_ref().map_or((pos, vec![]), |f| f(line, pos)))
    }
}
impl Hinter for StatusHelper {
    type Hint = String;
}
impl Validator for StatusHelper {}
impl Helper for StatusHelper {}

/// Ask rustyline to redraw the prompt (and so the status row) while it waits for input: an
/// external print of "cursor up, newline" moves nowhere and triggers a full prompt refresh.
pub fn spawn_refresher(bar: Arc<StatusBar>, mut printer: Box<dyn ExternalPrinter + Send>) {
    std::thread::Builder::new()
        .name("zilch-status".into())
        .spawn(move || loop {
            std::thread::sleep(Duration::from_millis(250));
            if !bar.reading.load(Ordering::SeqCst) {
                continue;
            }
            let busy = bar.busy.load(Ordering::SeqCst);
            if (busy || bar.dirty.swap(false, Ordering::SeqCst)) && printer.print("\x1b[1A\n".into()).is_err() {
                break;
            }
        })
        .expect("spawn status thread");
}

/// Clip a line so it occupies at most `width - 1` terminal columns (ANSI escapes take none, wide
/// characters take two). The status row must never wrap: a wrapped row would leave a stale copy
/// behind on every redraw. Line breaks become spaces, other control characters are dropped.
pub fn clip(s: &str, width: usize) -> String {
    use unicode_width::UnicodeWidthChar;
    let max = width.saturating_sub(1).max(10);
    let mut out = String::new();
    let mut cols = 0;
    let mut in_esc = false;
    for ch in s.chars() {
        if in_esc {
            out.push(ch);
            if ch.is_ascii_alphabetic() {
                in_esc = false;
            }
            continue;
        }
        if ch == '\x1b' {
            in_esc = true;
            out.push(ch);
            continue;
        }
        let ch = match ch {
            '\n' | '\r' | '\t' if out.ends_with(' ') => continue,
            '\n' | '\r' | '\t' => ' ',
            c if c.is_control() => continue,
            c => c,
        };
        let w = ch.width().unwrap_or(0);
        if cols + w > max - 1 {
            out.push('…');
            out.push_str("\x1b[0m");
            break;
        }
        out.push(ch);
        cols += w;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_status_row_stays_one_row() {
        assert_eq!(clip("⠸ 2 bash(set -u\nD=/x\r\n\tls)  │  idle", 80), "⠸ 2 bash(set -u D=/x ls)  │  idle");
        let long = clip(&"x".repeat(200), 40);
        let visible = long.replace("\x1b[0m", "");
        assert!(visible.chars().count() <= 40 && visible.ends_with('…'), "{visible}");
    }
}
