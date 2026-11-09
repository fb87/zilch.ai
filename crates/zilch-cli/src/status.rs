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

/// The rustyline helper: only the prompt highlighter does anything.
pub struct StatusHelper(pub Arc<StatusBar>);

impl Highlighter for StatusHelper {
    fn highlight_prompt<'b, 's: 'b, 'p: 'b>(&'s self, prompt: &'p str, default: bool) -> Cow<'b, str> {
        // Only our two-row prompt carries a status row (approval questions etc. are left alone).
        match prompt.strip_prefix('\n') {
            Some(rest) if default && !self.0.accepting.load(Ordering::SeqCst) => Cow::Owned(format!("\r\x1b[2K{}\n{rest}", self.0.text())),
            _ => Cow::Borrowed(prompt),
        }
    }

    fn highlight_char(&self, _line: &str, _pos: usize, kind: CmdKind) -> bool {
        // Returning true makes rustyline do a forced refresh when the line is accepted, which is
        // when we blank the status row (see module docs).
        self.0.accepting.store(kind == CmdKind::ForcedRefresh, Ordering::SeqCst);
        true
    }
}

impl Completer for StatusHelper {
    type Candidate = String;
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
