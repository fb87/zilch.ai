//! The terminal front end: the REPL (prompt, status row, output) and `-p` (one order), driving a
//! session through [`Link`], in this process or in the daemon.

use crate::session::{Link, Out};
use crate::{completion, status, terminal_width, DIM, RESET};
use std::io::{BufRead, IsTerminal, Write};
use std::sync::Arc;
use std::time::Duration;
use zilch_ipc::TicketId;

/// Why the REPL ended.
#[derive(Debug, PartialEq)]
pub enum Leave {
    /// Ctrl-D, /exit or /quit.
    Exit,
    /// /detach (daemon sessions keep running).
    Detach,
    /// /stop: end the session (daemon sessions).
    Stop,
}

/// Remove ANSI escape sequences (output for a pipe or file).
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                for c in chars.by_ref() {
                    if c.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
            continue;
        }
        out.push(c);
    }
    out
}

/// The REPL. `remote`: the session lives in the daemon (/detach and /stop are offered).
pub fn repl(link: Arc<dyn Link>, banner: &str, replay: usize, remote: bool) -> anyhow::Result<Leave> {
    let color = std::io::stdout().is_terminal();
    let (d, r) = if color { (DIM, RESET) } else { ("", "") };
    let fix = move |s: String| if color { s } else { strip_ansi(&s) };
    println!("{}", fix(banner.to_string()));
    let hint = if remote { "Type an order, or /help. /detach leaves it running." } else { "Type an order, or /help." };
    println!("{d}{hint}{r}");
    // Tab lists the choices (commands, their arguments, @ mentions) instead of cycling through them.
    let editor_cfg = rustyline::Config::builder().completion_type(rustyline::CompletionType::List).build();
    let mut rl: rustyline::Editor<status::StatusHelper, rustyline::history::DefaultHistory> = rustyline::Editor::with_config(editor_cfg)?;
    let mut printer = rl.create_external_printer()?;
    // Completion from what is live now: process ids, and the models the providers list.
    let l_complete = link.clone();
    let complete = move |line: &str, pos: usize| {
        let st = l_complete.state();
        completion::complete(line, pos, &completion::Live { pids: st.pids, models: st.models })
    };
    // The live status row above the prompt (terminals only), redrawn while something runs.
    let bar = color.then(|| Arc::new(status::StatusBar::default()));
    rl.set_helper(Some(status::StatusHelper { bar: bar.clone().unwrap_or_default(), complete: Some(Box::new(complete)) }));
    if let Some(bar) = &bar {
        status::spawn_refresher(bar.clone(), Box::new(rl.create_external_printer()?));
        let (l, bar) = (link.clone(), bar.clone());
        std::thread::Builder::new().name("zilch-status".into()).spawn(move || loop {
            let st = l.state();
            bar.set(status::clip(&st.status, terminal_width()), st.busy);
            std::thread::sleep(Duration::from_millis(100));
        })?;
    }
    let output = link.subscribe(replay);
    std::thread::Builder::new().name("zilch-output".into()).spawn(move || {
        for out in output.iter() {
            let text = match out {
                Out::Line { text } => text,
                Out::Ended { reason } => format!("── session ended: {reason} (Enter leaves) ──"),
                _ => continue,
            };
            if rustyline::ExternalPrinter::print(&mut printer, format!("{}\n", fix(text))).is_err() {
                break;
            }
        }
    })?;
    let leave = loop {
        // A line answers the question only if its prompt was showing when it was typed.
        let st = link.state();
        let asked = st.question.map(|(q, t, _)| (q, t));
        // With a status row the prompt is two rows; the first is filled with the status (status.rs).
        let row = if bar.is_some() { "\n" } else { "" };
        let prompt = match st.question {
            Some((_, _, true)) => format!("{row}allow? [y/n/a] › "),
            Some((_, t, false)) => format!("{row}plan #{t}? [y/n/feedback] › "),
            None => match (bar.is_some(), st.spec) {
                (true, false) => format!("{row}⧉ › "),
                (true, true) => format!("{row}⌬ › "),
                (false, false) => "[code] › ".to_string(),
                (false, true) => "[spec] › ".to_string(),
            },
        };
        if let Some(b) = &bar {
            b.reading.store(true, std::sync::atomic::Ordering::SeqCst);
        }
        let read = rl.readline(&prompt);
        if let Some(b) = &bar {
            b.reading.store(false, std::sync::atomic::Ordering::SeqCst);
        }
        let line = match read {
            Ok(l) => l,
            Err(rustyline::error::ReadlineError::Interrupted) => {
                println!("{}", link.interrupt());
                continue;
            }
            Err(rustyline::error::ReadlineError::Eof) => break if remote { Leave::Detach } else { Leave::Exit },
            Err(e) => return Err(e.into()),
        };
        if link.closed() {
            break Leave::Detach;
        }
        let input = line.trim();
        match input {
            "/exit" | "/quit" => break if remote { Leave::Detach } else { Leave::Exit },
            "/detach" if remote => break Leave::Detach,
            "/stop" if remote => break Leave::Stop,
            _ => {}
        }
        if !input.is_empty() && asked.is_none() {
            let _ = rl.add_history_entry(input);
        }
        let out = link.input(input, asked, bar.is_some());
        if !out.is_empty() {
            print!("{}", if color { out } else { strip_ansi(&out) });
            let _ = std::io::stdout().flush();
        }
    };
    Ok(leave)
}

/// `-p`: place one order, show its progress, answer questions from stdin, and return the exit
/// code once it is served (0 = served).
pub fn one_order(link: &dyn Link, banner: &str, order: &str) -> anyhow::Result<i32> {
    let color = std::io::stdout().is_terminal();
    let fix = |s: String| if color { s } else { strip_ansi(&s) };
    eprintln!("{}", if std::io::stderr().is_terminal() { banner.to_string() } else { strip_ansi(banner) });
    let output = link.subscribe(0);
    link.order(order);
    let stdin = std::io::stdin();
    let mut mine: Option<TicketId> = None;
    let mut matched = false;
    for out in output.iter() {
        match out {
            Out::Line { text } => {
                println!("{}", fix(text));
                let _ = std::io::stdout().flush();
            }
            // The first order opened after ours was placed, preferring one with our text.
            Out::Opened { ticket, text } => {
                if mine.is_none() || (text == order && !matched) {
                    mine = Some(ticket);
                    matched = text == order;
                }
            }
            Out::Question { qid, ticket, command, ask: true } if mine.is_none_or(|m| m == ticket) => {
                // Approval must be explicit; an empty line asks again, end of input rejects.
                let answer = loop {
                    eprint!("{}", if command { "allow? [y/n/a] › " } else { "plan? [y/n/feedback] › " });
                    let mut line = String::new();
                    if stdin.lock().read_line(&mut line)? == 0 {
                        break "n".to_string();
                    }
                    if !line.trim().is_empty() {
                        break line.trim().to_string();
                    }
                };
                link.answer(qid, &answer);
            }
            Out::Ended { reason } => {
                eprintln!("session ended: {reason}");
                return Ok(1);
            }
            Out::Served { ticket, ok } if mine.is_none_or(|m| m == ticket) => {
                for line in link.totals() {
                    eprintln!("{}", fix(line));
                }
                return Ok(if ok { 0 } else { 1 });
            }
            _ => {}
        }
    }
    Ok(1)
}

#[cfg(test)]
mod tests {
    #[test]
    fn colour_codes_are_stripped() {
        assert_eq!(super::strip_ansi("\x1b[1m\x1b[32m✓\x1b[0m done \x1b[2mdim\x1b[0m"), "✓ done dim");
        assert_eq!(super::strip_ansi("plain"), "plain");
    }
}
