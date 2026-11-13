//! Terminal rendering of the markdown that models write, line by line (so it streams).
//!
//! Only VT100/VT220 features are used, so it looks right on any terminal that emulates them:
//! - attributes: bold (SGR 1), underline (4), reverse video (7), switched off individually with
//!   the VT220 codes 22/24/27 so they nest; no colors, italics, dim or strike-through;
//! - lines, borders and bullets from the VT100 line-drawing set (DEC Special Graphics, selected
//!   with `ESC ( 0` and back to ASCII with `ESC ( B`), never left on past the end of a line.
//!
//! Block level: headings, bullet/numbered/task lists, block quotes, horizontal rules, fenced code
//! blocks and pipe tables. Inline: `code` (reverse), **bold**, *emphasis* (underline) and
//! [links](url). Code blocks keep their text verbatim (no gutter, no re-indenting) so copying from
//! the terminal still yields working code. Tables are buffered until they end, then aligned.
//! Without color/escapes (pipes, NO_COLOR) the markdown is left as is: it reads fine raw.

use crate::wrap::{visible, wrap_cell, wrap_line, Wrapper};
use unicode_width::UnicodeWidthStr;

const BOLD: &str = "\x1b[1m";
const BOLD_OFF: &str = "\x1b[22m";
const UNDERLINE: &str = "\x1b[4m";
const UNDERLINE_OFF: &str = "\x1b[24m";
const REVERSE: &str = "\x1b[7m";
const REVERSE_OFF: &str = "\x1b[27m";

/// Characters drawn from the DEC Special Graphics set (ASCII letters select line pieces).
fn gfx(s: &str) -> String {
    format!("\x1b(0{s}\x1b(B")
}
const H: char = 'q'; // ─
const V: char = 'x'; // │
const BULLET: char = '~'; // · centered dot

fn rule(n: usize) -> String {
    gfx(&H.to_string().repeat(n))
}

pub struct Markdown {
    styled: bool,
    plain: Wrapper,
    /// Inside a fenced code block: the fence marker (``` or ~~~).
    fence: Option<String>,
    /// Pipe-table rows waiting for the table to end.
    table: Vec<String>,
    /// Columns the whole answer is shifted right by, so it stands apart from zilch's own lines.
    indent: usize,
}

impl Markdown {
    /// `styled`: the output is a terminal that takes escape sequences.
    pub fn new(styled: bool) -> Self {
        Self { styled, plain: Wrapper::default(), fence: None, table: Vec::new(), indent: 0 }
    }

    /// Shift every rendered line right by `n` columns (code blocks included; 0 = off).
    pub fn with_indent(mut self, n: usize) -> Self {
        self.indent = n;
        self
    }

    /// Width left for the content once the indent is taken (0 = don't wrap stays 0).
    fn inner(&self, width: usize) -> usize {
        if width == 0 {
            0
        } else {
            width.saturating_sub(self.indent).max(1)
        }
    }

    /// Prefix every non-empty display line with the indent.
    fn pad(&self, s: String) -> String {
        if self.indent == 0 {
            return s;
        }
        let pad = " ".repeat(self.indent);
        s.split('\n').map(|l| if l.is_empty() { String::new() } else { format!("{pad}{l}") }).collect::<Vec<_>>().join("\n")
    }

    /// Render one source line (no `\n`) into display lines joined by `\n`. Returns None while a
    /// table is being collected (nothing to print yet).
    pub fn line(&mut self, src: &str, width: usize) -> Option<String> {
        let w = self.inner(width);
        self.render_line(src, w).map(|s| self.pad(s))
    }

    /// Whatever is still buffered (a table ending the answer); resets the state.
    pub fn finish(&mut self, width: usize) -> Option<String> {
        self.fence = None;
        self.plain = Wrapper::default();
        let w = self.inner(width);
        (!self.table.is_empty()).then(|| self.flush_table(w)).map(|s| self.pad(s))
    }

    /// Render a whole text (e.g. a buffered answer).
    pub fn text(&mut self, text: &str, width: usize) -> String {
        let mut out: Vec<String> = text.split('\n').filter_map(|l| self.line(l, width)).collect();
        out.extend(self.finish(width));
        out.join("\n")
    }

    fn render_line(&mut self, src: &str, width: usize) -> Option<String> {
        if !self.styled {
            return Some(self.plain.line(src, width));
        }
        let trimmed = src.trim_start();
        if let Some(fence) = &self.fence {
            if trimmed.starts_with(fence.as_str()) && trimmed.trim_start_matches(fence.as_str()).trim().is_empty() {
                self.fence = None;
                return Some(rule(8));
            }
            return Some(src.to_string());
        }
        // A table ends at the first line that is not a row.
        let is_row = trimmed.starts_with('|') && trimmed.len() > 1;
        let mut out = Vec::new();
        if !is_row && !self.table.is_empty() {
            out.push(self.flush_table(width));
        }
        if is_row {
            self.table.push(src.to_string());
            return if out.is_empty() { None } else { Some(out.join("\n")) };
        }
        out.push(self.block(src, trimmed, width));
        Some(out.join("\n"))
    }

    fn block(&mut self, src: &str, trimmed: &str, width: usize) -> String {
        let indent = &src[..src.len() - trimmed.len()];
        // Opening fence: a short rule with the language.
        for marker in ["```", "~~~"] {
            if let Some(lang) = trimmed.strip_prefix(marker) {
                self.fence = Some(marker.to_string());
                let lang = lang.trim();
                return if lang.is_empty() { rule(8) } else { format!("{} {lang}", rule(3)) };
            }
        }
        // Horizontal rule.
        let compact: String = trimmed.chars().filter(|c| !c.is_whitespace()).collect();
        if compact.len() >= 3 && ['-', '*', '_'].iter().any(|m| compact.chars().all(|c| c == *m)) {
            return rule(width.saturating_sub(1).clamp(3, 60));
        }
        // Heading: #, ## bold (# also underlined), ### and deeper underlined.
        let hashes = trimmed.chars().take_while(|c| *c == '#').count();
        if (1..=6).contains(&hashes) && trimmed[hashes..].starts_with(' ') {
            let title = inline(trimmed[hashes..].trim().trim_end_matches('#').trim_end());
            let styled = match hashes {
                1 => format!("{BOLD}{UNDERLINE}{title}{UNDERLINE_OFF}{BOLD_OFF}"),
                2 => format!("{BOLD}{title}{BOLD_OFF}"),
                _ => format!("{UNDERLINE}{title}{UNDERLINE_OFF}"),
            };
            return wrap_line(&styled, width);
        }
        // Block quote (possibly nested: "> > x").
        if trimmed.starts_with('>') {
            let mut rest = trimmed;
            let mut depth = 0;
            while let Some(r) = rest.strip_prefix('>') {
                depth += 1;
                rest = r.strip_prefix(' ').unwrap_or(r);
            }
            // The bar is drawn on every row, so a wrapped quote stays marked.
            let bars = format!("{indent}{}", format!("{} ", gfx(&V.to_string())).repeat(depth));
            let used = indent.len() + 2 * depth;
            let text = inline(rest);
            if width == 0 || width.saturating_sub(1) < used + 20 {
                return format!("{bars}{text}");
            }
            return wrap_cell(&text, width - 1 - used).iter().map(|r| format!("{bars}{r}")).collect::<Vec<_>>().join("\n");
        }
        // Lists: a centered dot (nested: a dash); task boxes stay ASCII.
        let level = indent.chars().map(|c| if c == '\t' { 4 } else { 1 }).sum::<usize>() / 2;
        for bullet in ["- ", "* ", "+ "] {
            if let Some(item) = trimmed.strip_prefix(bullet) {
                let mark = if level % 2 == 0 { gfx(&BULLET.to_string()) } else { "-".into() };
                let item = if let Some(i) = item.strip_prefix("[ ] ") {
                    format!("[ ] {}", inline(i))
                } else if let Some(i) = item.strip_prefix("[x] ").or_else(|| item.strip_prefix("[X] ")) {
                    format!("[x] {}", inline(i))
                } else {
                    inline(item)
                };
                return wrap_line(&format!("{indent}{mark} {item}"), width);
            }
        }
        // Paragraphs, numbered items, blank lines: inline markup only.
        wrap_line(&format!("{indent}{}", inline(trimmed)), width)
    }

    fn flush_table(&mut self, width: usize) -> String {
        let rows = std::mem::take(&mut self.table);
        render_table(&rows, width).unwrap_or_else(|| rows.iter().map(|r| wrap_line(&inline(r), width)).collect::<Vec<_>>().join("\n"))
    }
}

fn visible_width(s: &str) -> usize {
    UnicodeWidthStr::width(visible(s).as_str())
}

fn split_row(row: &str) -> Vec<String> {
    let t = row.trim();
    let t = t.strip_prefix('|').unwrap_or(t);
    let t = t.strip_suffix('|').unwrap_or(t);
    // Split on pipes that are neither escaped nor inside inline code.
    let mut cells = vec![String::new()];
    let (mut code, mut prev) = (false, '\0');
    for ch in t.chars() {
        if ch == '`' {
            code = !code;
        }
        if ch == '|' && !code && prev != '\\' {
            cells.push(String::new());
        } else {
            cells.last_mut().unwrap().push(ch);
        }
        prev = ch;
    }
    cells.into_iter().map(|c| c.trim().replace("\\|", "|")).collect()
}

#[derive(Clone, Copy)]
enum Align {
    Left,
    Center,
    Right,
}

/// Aligned table with line-drawing borders, or None if it is not a table (no separator row).
/// A table wider than the terminal gets narrower columns whose cells wrap inside the borders;
/// when even that cannot fit, each row is listed as `header: value` lines instead.
fn render_table(rows: &[String], width: usize) -> Option<String> {
    if rows.len() < 2 {
        return None;
    }
    let sep = split_row(&rows[1]);
    if !sep.iter().all(|c| c.contains('-') && c.trim_matches(':').chars().all(|ch| ch == '-')) {
        return None;
    }
    let aligns: Vec<Align> = sep
        .iter()
        .map(|c| match (c.starts_with(':'), c.ends_with(':')) {
            (true, true) => Align::Center,
            (false, true) => Align::Right,
            _ => Align::Left,
        })
        .collect();
    let body: Vec<Vec<String>> =
        std::iter::once(&rows[0]).chain(rows.iter().skip(2)).map(|r| split_row(r).iter().map(|c| inline(c)).collect()).collect();
    let cols = body.iter().map(Vec::len).max().unwrap_or(0).max(aligns.len());
    let mut widths = vec![0usize; cols];
    for r in &body {
        for (i, c) in r.iter().enumerate() {
            widths[i] = widths[i].max(visible_width(c));
        }
    }
    if width > 0 {
        // Columns for cell text once the borders (`│ ` … ` │ ` … ` │`) and the right margin are taken.
        let avail = width.saturating_sub(1).saturating_sub(3 * cols + 1);
        if avail < 4 * cols {
            return Some(table_as_records(&body, width));
        }
        // Narrow the widest column one step at a time until the table fits.
        while widths.iter().sum::<usize>() > avail {
            let (i, _) = widths.iter().enumerate().max_by_key(|(_, w)| **w).unwrap();
            widths[i] -= 1;
        }
    }
    // Corner/tee pieces of the line-drawing set: l k m j (corners), t u (side tees),
    // w v (top/bottom tees), n (cross).
    let border = |l: char, m: char, r: char| {
        let segs: Vec<String> = widths.iter().map(|w| H.to_string().repeat(w + 2)).collect();
        gfx(&format!("{l}{}{r}", segs.join(&m.to_string())))
    };
    let bar = gfx(&V.to_string());
    let mut out = vec![border('l', 'w', 'k')];
    for (ri, r) in body.iter().enumerate() {
        let cells: Vec<Vec<String>> =
            widths.iter().enumerate().map(|(i, w)| wrap_cell(r.get(i).map(String::as_str).unwrap_or(""), *w)).collect();
        let height = cells.iter().map(Vec::len).max().unwrap_or(1);
        for k in 0..height {
            let mut s = bar.clone();
            for (i, w) in widths.iter().enumerate() {
                let cell = cells[i].get(k).map(String::as_str).unwrap_or("");
                let pad = w.saturating_sub(visible_width(cell));
                let (lp, rp) = match aligns.get(i).copied().unwrap_or(Align::Left) {
                    Align::Left => (0, pad),
                    Align::Right => (pad, 0),
                    Align::Center => (pad / 2, pad - pad / 2),
                };
                let cell = if ri == 0 && !cell.is_empty() { format!("{BOLD}{cell}{BOLD_OFF}") } else { cell.to_string() };
                s.push_str(&format!(" {}{cell}{} {bar}", " ".repeat(lp), " ".repeat(rp)));
            }
            out.push(s);
        }
        if ri == 0 {
            out.push(border('t', 'n', 'u'));
        }
    }
    out.push(border('m', 'v', 'j'));
    Some(out.join("\n"))
}

/// A table too wide for any column layout: one block per row, `header: value` per cell.
fn table_as_records(body: &[Vec<String>], width: usize) -> String {
    let header = &body[0];
    let mut out = Vec::new();
    for (ri, r) in body.iter().enumerate().skip(1) {
        if ri > 1 {
            out.push(String::new());
        }
        for (i, cell) in r.iter().enumerate() {
            let name = header.get(i).map(String::as_str).unwrap_or("");
            let mark = if i == 0 { gfx(&BULLET.to_string()) } else { " ".into() };
            out.push(wrap_line(&format!("{mark} {BOLD}{name}{BOLD_OFF}: {cell}"), width));
        }
    }
    out.join("\n")
}

/// Inline markup → VT100 attributes. Code spans are literal (reverse video); `**`/`__` are bold,
/// `*`/`_` underline (VT100 has no italics). Emphasis needs a word boundary outside and
/// non-space inside, so `snake_case`, `a * b` and `2*3*4` stay as they are. `~~strike~~` is left
/// as written (no VT100 equivalent).
pub fn inline(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let mut out = String::with_capacity(s.len() + 16);
    let mut i = 0;
    'scan: while i < chars.len() {
        let c = chars[i];
        // Escaped punctuation.
        if c == '\\' && chars.get(i + 1).is_some_and(|n| n.is_ascii_punctuation()) {
            out.push(chars[i + 1]);
            i += 2;
            continue;
        }
        // `code`, also ``code with ` inside``.
        if c == '`' {
            let ticks = chars[i..].iter().take_while(|c| **c == '`').count();
            if let Some(close) = find_run(&chars, i + ticks, '`', ticks) {
                let code: String = chars[i + ticks..close].iter().collect();
                out.push_str(&format!("{REVERSE}{}{REVERSE_OFF}", code.trim()));
                i = close + ticks;
                continue;
            }
            // Unclosed: the backticks are literal.
            out.extend(&chars[i..i + ticks]);
            i += ticks;
            continue;
        }
        // [text](url)
        if c == '[' {
            if let Some((text, url, end)) = link(&chars, i) {
                let shown = format!("{UNDERLINE}{}{UNDERLINE_OFF}", inline(&text));
                if text == url {
                    out.push_str(&shown);
                } else {
                    out.push_str(&format!("{shown} ({url})"));
                }
                i = end;
                continue;
            }
        }
        // Emphasis.
        let open_ok = i == 0 || !chars[i - 1].is_alphanumeric();
        if open_ok {
            for (marker, on, off) in
                [("**", BOLD, BOLD_OFF), ("__", BOLD, BOLD_OFF), ("*", UNDERLINE, UNDERLINE_OFF), ("_", UNDERLINE, UNDERLINE_OFF)]
            {
                let m: Vec<char> = marker.chars().collect();
                let start = i + m.len();
                if chars[i..].starts_with(&m) && chars.get(start).is_some_and(|c| !c.is_whitespace() && *c != m[0]) {
                    if let Some(end) = find_closing(&chars, start, &m) {
                        let inner: String = chars[start..end].iter().collect();
                        out.push_str(&format!("{on}{}{off}", inline(&inner)));
                        i = end + m.len();
                        continue 'scan;
                    }
                }
            }
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Index of the next run of exactly `n` `ch` characters at or after `from`.
fn find_run(chars: &[char], from: usize, ch: char, n: usize) -> Option<usize> {
    let mut i = from;
    while i < chars.len() {
        if chars[i] == ch {
            let run = chars[i..].iter().take_while(|c| **c == ch).count();
            if run == n {
                return Some(i);
            }
            i += run;
        } else {
            i += 1;
        }
    }
    None
}

/// Closing emphasis marker: preceded by non-space, followed by a word boundary, and not part of
/// a longer run of the same character.
fn find_closing(chars: &[char], from: usize, m: &[char]) -> Option<usize> {
    let mut i = from + 1;
    while i + m.len() <= chars.len() {
        if chars[i] == '`' {
            // Skip code spans inside emphasis.
            let ticks = chars[i..].iter().take_while(|c| **c == '`').count();
            i = find_run(chars, i + ticks, '`', ticks)? + ticks;
            continue;
        }
        if chars[i..].starts_with(m)
            && !chars[i - 1].is_whitespace()
            && chars.get(i + m.len()).is_none_or(|c| !c.is_alphanumeric() && *c != m[0])
        {
            return Some(i);
        }
        i += 1;
    }
    None
}

/// `[text](url)` starting at `i`: (text, url, index after the closing paren).
fn link(chars: &[char], i: usize) -> Option<(String, String, usize)> {
    let close = (i + 1..chars.len()).find(|&j| chars[j] == ']')?;
    if chars.get(close + 1) != Some(&'(') {
        return None;
    }
    let end = (close + 2..chars.len()).find(|&j| chars[j] == ')')?;
    let text: String = chars[i + 1..close].iter().collect();
    let url: String = chars[close + 2..end].iter().collect();
    if text.is_empty() || url.is_empty() || url.contains(' ') {
        return None;
    }
    Some((text, url, end + 1))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What a VT100 shows: escapes removed, line-drawing letters mapped to their glyphs.
    fn screen(s: &str) -> String {
        let mut out = String::new();
        let mut graphics = false;
        for p in crate::wrap::pieces(s) {
            match p {
                crate::wrap::Piece::Esc("\x1b(0") => graphics = true,
                crate::wrap::Piece::Esc("\x1b(B") => graphics = false,
                crate::wrap::Piece::Esc(_) => {}
                crate::wrap::Piece::Char(c) if graphics => out.push(match c {
                    'q' => '─',
                    'x' => '│',
                    'l' => '┌',
                    'k' => '┐',
                    'm' => '└',
                    'j' => '┘',
                    't' => '├',
                    'u' => '┤',
                    'w' => '┬',
                    'v' => '┴',
                    'n' => '┼',
                    '~' => '·',
                    other => other,
                }),
                crate::wrap::Piece::Char(c) => out.push(c),
            }
        }
        out
    }

    fn render(src: &str) -> String {
        Markdown::new(true).text(src, 80)
    }

    /// Every escape sequence used must be a VT100/VT220 one.
    fn assert_vt220_only(s: &str) {
        for p in crate::wrap::pieces(s) {
            if let crate::wrap::Piece::Esc(e) = p {
                let ok = ["\x1b[1m", "\x1b[22m", "\x1b[4m", "\x1b[24m", "\x1b[7m", "\x1b[27m", "\x1b[0m", "\x1b(0", "\x1b(B"].contains(&e);
                assert!(ok, "non-VT220 sequence {e:?} in {s:?}");
            }
        }
        // Line drawing is always switched back to ASCII on the same line.
        for l in s.lines() {
            assert_eq!(l.matches("\x1b(0").count(), l.matches("\x1b(B").count(), "{l:?}");
        }
    }

    #[test]
    fn inline_styles() {
        assert_eq!(inline("a **b** c"), format!("a {BOLD}b{BOLD_OFF} c"));
        assert_eq!(inline("an *em* word"), format!("an {UNDERLINE}em{UNDERLINE_OFF} word"));
        assert_eq!(inline("run `cargo test` now"), format!("run {REVERSE}cargo test{REVERSE_OFF} now"));
        assert_eq!(screen(&inline("**bold with `code` inside**")), "bold with code inside");
        assert_eq!(screen(&inline("see [docs](https://x.io)")), "see docs (https://x.io)");
        assert_eq!(screen(&inline("~~old~~")), "~~old~~", "no VT100 strike-through: left as is");
    }

    #[test]
    fn things_that_are_not_emphasis_stay_literal() {
        for s in ["snake_case_name", "a * b * c", "2*3*4", "file_*.rs", "**", "x ** y", "`unclosed", "[a] (b)", "\\*not em\\*", "***"] {
            let want = s.replace('\\', "");
            assert_eq!(screen(&inline(s)), want, "{s}");
        }
        assert_eq!(screen(&inline("`a_b*c*`")), "a_b*c*", "code is literal");
    }

    #[test]
    fn blocks() {
        let out = render("# Title\n## Sub\n- one\n  - nested\n- [ ] todo\n- [x] done\n1. first\n> quoted\n---\nend");
        assert_vt220_only(&out);
        let out = screen(&out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "Title");
        assert_eq!(lines[1], "Sub");
        assert_eq!(lines[2], "· one");
        assert_eq!(lines[3], "  - nested");
        assert_eq!(lines[4], "· [ ] todo");
        assert_eq!(lines[5], "· [x] done");
        assert_eq!(lines[6], "1. first");
        assert_eq!(lines[7], "│ quoted");
        assert!(lines[8].chars().all(|c| c == '─'));
        assert_eq!(lines[9], "end");
    }

    #[test]
    fn code_blocks_are_verbatim() {
        let src = "```rust\nfn main() { let x = a * b; } // **not bold** and a very long line that goes on and on past the edge\n```\nafter **b**";
        let out = render(src);
        assert_vt220_only(&out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(screen(lines[0]), "─── rust");
        assert_eq!(lines[1], "fn main() { let x = a * b; } // **not bold** and a very long line that goes on and on past the edge");
        assert_eq!(screen(lines[2]), "────────");
        assert_eq!(screen(lines[3]), "after b");
    }

    #[test]
    fn tables_are_aligned() {
        let out = render("| Name | Qty |\n|:-----|----:|\n| apple | 3 |\n| `kiwi` | 12 |\ndone");
        assert_vt220_only(&out);
        let out = screen(&out);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines[0], "┌───────┬─────┐");
        assert_eq!(lines[1], "│ Name  │ Qty │");
        assert_eq!(lines[2], "├───────┼─────┤");
        assert_eq!(lines[3], "│ apple │   3 │");
        assert_eq!(lines[4], "│ kiwi  │  12 │");
        assert_eq!(lines[5], "└───────┴─────┘");
        assert_eq!(lines[6], "done");
    }

    #[test]
    fn table_at_end_and_too_wide_tables() {
        let mut md = Markdown::new(true);
        assert!(md.line("| a | b |", 80).is_none(), "rows are buffered");
        assert!(md.line("|---|---|", 80).is_none());
        assert!(md.line("| 1 | 2 |", 80).is_none());
        assert!(screen(&md.finish(80).unwrap()).contains("│ 1 │ 2 │"));
    }

    fn assert_fits(out: &str, width: usize) {
        for l in out.split('\n') {
            assert!(visible_width(l) < width, "wider than {width}: {:?}", screen(l));
        }
    }

    #[test]
    fn wide_tables_wrap_their_cells_inside_the_borders() {
        let src = "| Segment | Flags | Direction | Key Fields |\n|---|---|---|---|\n\
                   | 1 | SYN | Client → Server | seq = x, options (MSS, etc.) |\n\
                   | 3 | ACK | Client → Server | ack = y + 1 (may carry data with TFO) |";
        let out = Markdown::new(true).with_indent(2).text(src, 60);
        assert_vt220_only(&out);
        assert_fits(&out, 60);
        let s = screen(&out);
        assert!(s.lines().all(|l| ["  ┌", "  │", "  ├", "  └"].iter().any(|p| l.starts_with(p))), "{s}");
        let text: String = s.split_whitespace().collect::<Vec<_>>().join(" ");
        for word in ["Segment", "Direction", "options", "(MSS,", "TFO)"] {
            assert!(text.contains(word), "{word} kept: {s}");
        }
        // Far too many columns for the width: one block per row instead.
        let many = format!("{}\n{}|\n{}", "| a | b | c | d | e | f | g | h |", "|---".repeat(8), "| 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |");
        let out = Markdown::new(true).text(&many, 30);
        assert_fits(&out, 30);
        let s = screen(&out);
        assert!(s.contains("· a: 1") && s.contains("  h: 8"), "{s}");
    }

    #[test]
    fn wrapped_quotes_keep_their_bar() {
        let out = Markdown::new(true).with_indent(2).text(&format!("> {}", "a long quoted sentence ".repeat(6)), 40);
        assert_fits(&out, 40);
        let s = screen(&out);
        assert!(s.lines().count() > 2);
        assert!(s.lines().all(|l| l.starts_with("  │ ")), "{s}");
    }

    #[test]
    fn without_escapes_markdown_is_left_alone() {
        let src = "# T\n- **a** `b`\n| x |\n|---|";
        assert_eq!(Markdown::new(false).text(src, 80), src);
    }

    #[test]
    fn long_list_items_wrap_under_the_text() {
        let out = Markdown::new(true).text(&format!("- {}", "word ".repeat(20)), 40);
        assert_vt220_only(&out);
        let out = screen(&out);
        let lines: Vec<&str> = out.lines().collect();
        assert!(lines.len() > 1);
        assert!(lines[1..].iter().all(|l| l.starts_with("  ") && !l.starts_with("   ")), "{lines:?}");
    }

    #[test]
    fn indent_shifts_every_block_and_still_fits() {
        let src = format!("## Head\n\n{}\n- item\n> quote\n| a | b |\n|---|---|\n| 1 | 2 |\n```py\nx = 1\n```\n---", "word ".repeat(30));
        let out = Markdown::new(true).with_indent(2).text(&src, 40);
        assert_vt220_only(&out);
        let out = screen(&out);
        for l in out.lines() {
            if l.is_empty() {
                continue; // blank lines carry no trailing spaces
            }
            assert!(l.starts_with("  ") && !l.starts_with("   ") || l.starts_with("    "), "indented: {l:?}");
            assert!(UnicodeWidthStr::width(l) <= 39, "fits the terminal: {l:?}");
        }
        assert!(out.contains("\n\n"), "blank line kept, without padding");
        assert!(out.contains("  x = 1"), "code is indented with the rest");
        assert!(out.contains("  ┌───┬───┐"));
        // A table that would fit unindented but not indented gets narrower columns instead.
        let t = Markdown::new(true).with_indent(4).text("| aaaaaaaaaaaaaaaa | b |\n|---|---|", 28);
        assert_fits(&t, 28);
        assert!(screen(&t).starts_with("    ┌"), "{t:?}");
    }

    #[test]
    fn zero_indent_is_unchanged() {
        let src = "# T\n- a\n```\ncode\n```";
        assert_eq!(Markdown::new(true).with_indent(0).text(src, 80), Markdown::new(true).text(src, 80));
    }
}
