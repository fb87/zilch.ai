//! Word wrapping with a hanging indent, for terminal output.
//!
//! Terminals wrap long lines at column 0, so a wrapped bullet or tool line loses its indentation
//! and the output turns ragged. We wrap ourselves instead: continuation rows line up under the
//! text after the line's indent and marker (`- `, `1. `, `⎿ `, `✓ `, `│ ` …).
//!
//! ANSI escapes take no columns; the active style is closed before each break and re-opened after
//! the indent, so a dim line stays dim on every row (rustyline prints each row separately).
//! Code blocks (``` fences) and tables are left alone: re-indenting them would corrupt copy/paste.

use unicode_width::UnicodeWidthChar;

const RESET: &str = "\x1b[0m";

/// Wraps a stream of lines, remembering whether we are inside a code fence.
#[derive(Default)]
pub struct Wrapper {
    in_fence: bool,
}

impl Wrapper {
    /// Wrap one logical line (no `\n`) to `width` columns. Returns the rows joined by `\n`.
    pub fn line(&mut self, line: &str, width: usize) -> String {
        let fence = visible(line).trim_start().starts_with("```");
        if fence {
            self.in_fence = !self.in_fence;
            return line.to_string();
        }
        if self.in_fence || visible(line).trim_start().starts_with('|') {
            return line.to_string();
        }
        wrap_line(line, width)
    }

    /// Wrap a block of text (several lines).
    pub fn text(&mut self, text: &str, width: usize) -> String {
        text.split('\n').map(|l| self.line(l, width)).collect::<Vec<_>>().join("\n")
    }
}

/// The text without ANSI escape sequences.
fn visible(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut esc = false;
    for ch in s.chars() {
        if esc {
            esc = !ch.is_ascii_alphabetic();
        } else if ch == '\x1b' {
            esc = true;
        } else {
            out.push(ch);
        }
    }
    out
}

fn width_of(s: &str) -> usize {
    visible(s).chars().map(|c| c.width().unwrap_or(0)).sum()
}

/// A list/status marker: bullets, `12.`/`3)`, quote `>`, or a short run of symbols (`⎿`, `✓`, `●`).
fn is_marker(tok: &str) -> bool {
    let v = visible(tok);
    if v.is_empty() || v.chars().count() > 4 {
        return false;
    }
    let digits = v.trim_end_matches(['.', ')']);
    if digits.len() < v.len() && !digits.is_empty() && digits.chars().all(|c| c.is_ascii_digit()) {
        return true;
    }
    v.chars().all(|c| !c.is_alphanumeric() && !c.is_whitespace() && !"([{\"'`".contains(c))
}

/// Hanging indent (in columns) for `line`: leading spaces plus up to two marker tokens.
fn hanging_indent(line: &str) -> usize {
    let v = visible(line);
    let lead = v.len() - v.trim_start().len();
    let mut cols = lead;
    let mut rest = &v[lead..];
    for _ in 0..2 {
        let Some((tok, after)) = rest.split_once(' ') else { break };
        if !is_marker(tok) {
            break;
        }
        cols += tok.chars().map(|c| c.width().unwrap_or(0)).sum::<usize>() + 1;
        rest = after.trim_start_matches(' ');
        cols += after.len() - rest.len();
    }
    cols
}

/// Wrap one line to `width` columns (0 = don't wrap).
pub fn wrap_line(line: &str, width: usize) -> String {
    // Stay one column short of the edge: writing the last column leaves some terminals in a
    // pending-wrap state that turns the following newline into a blank row.
    let max = width.saturating_sub(1);
    if max < 20 || width_of(line) <= max {
        return line.to_string();
    }
    let mut indent = hanging_indent(line);
    if indent > max / 2 {
        indent = 0;
    }
    let pad = " ".repeat(indent);

    // Split into words, each with its trailing spaces; escapes stay attached to their word.
    let mut words: Vec<(String, usize, String)> = Vec::new(); // (word, cols, spaces)
    let (mut word, mut cols, mut spaces) = (String::new(), 0usize, String::new());
    let mut esc = false;
    for ch in line.chars() {
        if esc {
            word.push(ch);
            esc = !ch.is_ascii_alphabetic();
            continue;
        }
        if ch == '\x1b' {
            if !spaces.is_empty() {
                words.push((std::mem::take(&mut word), cols, std::mem::take(&mut spaces)));
                cols = 0;
            }
            esc = true;
            word.push(ch);
        } else if ch == ' ' {
            spaces.push(ch);
        } else {
            if !spaces.is_empty() {
                words.push((std::mem::take(&mut word), cols, std::mem::take(&mut spaces)));
                cols = 0;
            }
            word.push(ch);
            cols += ch.width().unwrap_or(0);
        }
    }
    words.push((word, cols, spaces));

    let mut out = String::with_capacity(line.len() + 16);
    let mut style = String::new(); // SGR sequences active right now
    let mut col = 0usize;
    let mut row_has_word = false; // a visible word is on the current row
    let mut just_broke = false; // at the start of a continuation row (spaces there are dropped)
    let mut pending_spaces = String::new();
    for (word, wcols, trailing) in words {
        if row_has_word && wcols > 0 && col + pending_spaces.len() + wcols > max {
            break_row(&mut out, &style, &pad);
            col = indent;
            row_has_word = false;
            just_broke = true;
        }
        if !just_broke {
            out.push_str(&pending_spaces);
            col += pending_spaces.len();
        }
        pending_spaces.clear();
        // A word longer than a whole row is cut by character.
        let mut esc = false;
        let mut seq = String::new();
        for ch in word.chars() {
            if esc || ch == '\x1b' {
                out.push(ch);
                seq.push(ch);
                esc = ch == '\x1b' || !ch.is_ascii_alphabetic();
                if !esc {
                    track_style(&mut style, &seq);
                    seq.clear();
                }
                continue;
            }
            let w = ch.width().unwrap_or(0);
            if col + w > max && col > indent {
                break_row(&mut out, &style, &pad);
                col = indent;
            }
            out.push(ch);
            col += w;
        }
        if wcols > 0 {
            row_has_word = true;
            just_broke = false;
        }
        pending_spaces = trailing;
    }
    out.push_str(&pending_spaces);
    out
}

fn track_style(style: &mut String, seq: &str) {
    if !seq.ends_with('m') {
        return;
    }
    if seq == RESET || seq == "\x1b[m" {
        style.clear();
    } else {
        style.push_str(seq);
    }
}

fn break_row(out: &mut String, style: &str, pad: &str) {
    if !style.is_empty() {
        out.push_str(RESET);
    }
    out.push('\n');
    out.push_str(pad);
    out.push_str(style);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(s: &str) -> Vec<String> {
        s.split('\n').map(visible).collect()
    }

    #[test]
    fn short_lines_are_untouched() {
        assert_eq!(wrap_line("hello world", 80), "hello world");
        assert_eq!(wrap_line("anything at all", 0), "anything at all");
    }

    #[test]
    fn plain_text_wraps_at_word_boundaries() {
        let r = rows(&wrap_line("the quick brown fox jumps over the lazy dog again and again", 31));
        assert!(r.iter().all(|l| width_of(l) <= 30), "{r:?}");
        assert_eq!(r.join(" ").split_whitespace().count(), 12);
        assert!(r.iter().all(|l| !l.starts_with(' ')));
    }

    #[test]
    fn bullets_numbers_and_glyphs_get_a_hanging_indent() {
        for (line, indent) in [
            ("- a bullet that is long enough to need wrapping onto another row", 2),
            ("  12. a numbered item that is long enough to need wrapping onto rows", 6),
            ("  ⎿ ✗ a failing tool summary that is long enough to need wrapping", 6),
            ("  ✓ [a1] Fix currency rounding (coder#1) changed ledger/currency.py and tests", 4),
            ("> a quoted line that is long enough to need wrapping onto another row", 2),
        ] {
            let r = rows(&wrap_line(line, 40));
            assert!(r.len() > 1, "{line}");
            for cont in &r[1..] {
                assert_eq!(cont.len() - cont.trim_start().len(), indent, "{line} → {r:?}");
                assert!(width_of(cont) <= 39);
            }
        }
    }

    #[test]
    fn leading_style_keeps_the_indent() {
        let s = format!("\x1b[2m  │ coder#1: {}{RESET}", "word ".repeat(15));
        let r = rows(&wrap_line(&s, 40));
        assert!(r[0].starts_with("  │ coder#1:"), "{r:?}");
        assert!(r[1..].iter().all(|l| l.starts_with("    ") && !l.starts_with("     ")), "{r:?}");
    }

    #[test]
    fn prose_starting_with_a_word_is_not_a_marker() {
        let r = rows(&wrap_line("Note: this sentence is long enough that it has to be wrapped", 30));
        assert!(r[1..].iter().all(|l| !l.starts_with(' ')), "{r:?}");
    }

    #[test]
    fn style_is_reopened_on_every_row() {
        let dim = "\x1b[2m";
        let s = format!("  ⎿ {dim}one two three four five six seven eight nine ten eleven{RESET}");
        let out = wrap_line(&s, 30);
        let lines: Vec<&str> = out.split('\n').collect();
        assert!(lines.len() > 1);
        assert!(lines[0].ends_with(RESET));
        for l in &lines[1..] {
            assert!(l.trim_start().starts_with(dim), "{l:?}");
        }
    }

    #[test]
    fn overlong_words_are_cut() {
        let url = "x".repeat(70);
        let r = rows(&wrap_line(&format!("- see {url}"), 30));
        assert!(r.iter().all(|l| width_of(l) <= 29), "{r:?}");
        assert_eq!(r.concat().matches('x').count(), 70);
    }

    #[test]
    fn wide_characters_count_double() {
        let r = rows(&wrap_line(&"日本語 ".repeat(10), 21));
        assert!(r.iter().all(|l| width_of(l) <= 20), "{r:?}");
    }

    #[test]
    fn code_fences_and_tables_are_left_alone() {
        let long = format!("    let x = {};", "a + ".repeat(30));
        let mut w = Wrapper::default();
        let text = format!("```rust\n{long}\n```\n| a | {} |", "b ".repeat(40));
        assert_eq!(w.text(&text, 40), text);
        // After the fence closes, prose wraps again.
        assert!(w.line(&"word ".repeat(20), 40).contains('\n'));
    }
}
