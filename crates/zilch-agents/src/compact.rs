//! History compaction (ported from zilch): keeps a long-lived conversation — above all the
//! resident main agent, which hears every order, report and note — inside its context budget, so
//! the session does not get slower and slower as it grows.
//!
//! Applied only when the estimated prompt reaches `COMPACT_AT` of the budget:
//! 1. Elide large tool outputs outside the recent window (no model call).
//! 2. If still too big, summarize the older turns — as an ordinary `Complete` syscall, since a
//!    step never does I/O — and keep the recent turns verbatim. A cut never separates an assistant
//!    tool call from its results.
//! 3. If summarizing fails, keep the result of step 1.

use zilch_ipc::llm::Role;
use zilch_ipc::{ChatRequest, Message, ModelBinding, ToolSpec};

/// Compact when the prompt reaches this fraction of the context budget…
pub const COMPACT_AT: f64 = 0.75;
/// …and keep about this fraction of the budget as verbatim recent history.
const KEEP_RECENT: f64 = 0.35;
/// Tool outputs larger than this (chars) are elided outside the recent window.
const ELIDE_TOOL_OVER: usize = 1_500;
const SUMMARY_TAG: &str = "<conversation-summary>";

const SUMMARY_PROMPT: &str = "You compress an AI agent's conversation history so it can continue working \
with a smaller context. Write a dense summary that preserves: the user's goals and constraints, decisions made, \
work completed (files changed, results, test outcomes), current state of any plan, open problems, and facts about \
the codebase the agent will need again (paths, names, conventions). Omit pleasantries and raw tool output. Use \
terse bullet points.";

/// Rough token estimate (≈4 chars/token), deliberately conservative.
pub fn estimate_tokens(system: &str, history: &[Message], tools: &[ToolSpec]) -> usize {
    let mut chars = system.len() + serde_json::to_string(tools).map(|s| s.len()).unwrap_or(0);
    for m in history {
        chars += 16 + m.content.as_ref().map_or(0, String::len) + m.reasoning_content.as_ref().map_or(0, String::len);
        for c in &m.tool_calls {
            chars += c.function.name.len() + c.function.arguments.len() + 16;
        }
    }
    chars / 4 + 1
}

fn msg_tokens(m: &Message) -> usize {
    estimate_tokens("", std::slice::from_ref(m), &[])
}

/// A verbatim tail may start at a user message or an assistant message (the start of a tool-call
/// group) — never at a tool result.
fn valid_cut(history: &[Message], i: usize) -> bool {
    i > 0 && history[i].role != Role::Tool && history[i].role != Role::System
}

/// Where the verbatim tail starts: the earliest valid cut whose tail fits `keep`, always keeping
/// at least the last message group. 0 = nothing to compact.
pub fn cut_point(history: &[Message], keep: usize) -> usize {
    let mut acc = 0;
    let mut best = 0;
    for i in (1..history.len()).rev() {
        acc += msg_tokens(&history[i]);
        if valid_cut(history, i) {
            if acc > keep && best != 0 {
                break;
            }
            best = i;
        }
    }
    let starts_with_summary = history.first().is_some_and(|m| m.text().starts_with(SUMMARY_TAG));
    if starts_with_summary && best <= 2 {
        return 0;
    }
    best
}

/// First index of the most recent messages that together fit in `keep` tokens.
pub fn recent_window_start(history: &[Message], keep: usize) -> usize {
    let mut acc = 0;
    for i in (0..history.len()).rev() {
        acc += msg_tokens(&history[i]);
        if acc > keep {
            return i + 1;
        }
    }
    0
}

/// Step 1: shorten big tool results before `upto`. Returns true if anything changed.
pub fn elide_old_tool_outputs(history: &mut [Message], upto: usize) -> bool {
    let mut changed = false;
    for m in history.iter_mut().take(upto) {
        if m.role == Role::Tool {
            if let Some(c) = &m.content {
                if c.len() > ELIDE_TOOL_OVER {
                    let head: String = c.chars().take(400).collect();
                    m.content = Some(format!(
                        "{head}\n… [older tool output elided to save context ({} chars). Rely on what you already concluded from it; re-read only if you need exact text you have not noted.]",
                        c.len()
                    ));
                    changed = true;
                }
            }
        }
        if m.reasoning_content.is_some() {
            m.reasoning_content = None;
            changed = true;
        }
    }
    changed
}

/// Old turns as plain text for the summarizer (tool output clipped).
fn transcript(msgs: &[Message]) -> String {
    let mut s = String::new();
    for m in msgs {
        let who = match m.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
            Role::Tool => "tool result",
        };
        let text = m.text();
        let text: String = if m.role == Role::Tool && text.len() > 2_000 {
            format!("{}… [clipped]", &text[..text.floor_char_boundary(2_000)])
        } else {
            text.to_string()
        };
        if !text.is_empty() {
            s.push_str(&format!("[{who}] {text}\n"));
        }
        for c in &m.tool_calls {
            let args: String = c.function.arguments.chars().take(500).collect();
            s.push_str(&format!("[assistant called {}] {args}\n", c.function.name));
        }
    }
    s
}

/// What needs doing before the next request, if anything.
pub enum Plan {
    /// Fits (possibly after eliding old tool outputs in place).
    Fits,
    /// Summarize `history[..cut]` with this request, then splice.
    Summarize { cut: usize, request: Box<ChatRequest> },
}

/// Step 1 in place, and decide whether step 2 is needed. `budget` is the context size in tokens.
pub fn prepare(
    binding: &ModelBinding,
    session: Option<String>,
    system: &str,
    tools: &[ToolSpec],
    history: &mut [Message],
    budget: usize,
) -> Plan {
    if (estimate_tokens(system, history, tools) as f64) < budget as f64 * COMPACT_AT {
        return Plan::Fits;
    }
    let fixed = estimate_tokens(system, &[], tools);
    let keep = ((budget as f64 * KEEP_RECENT) as usize).saturating_sub(fixed).max(256);
    let window = recent_window_start(history, keep);
    elide_old_tool_outputs(history, window);
    let cut = cut_point(history, keep);
    if (estimate_tokens(system, history, tools) as f64) < budget as f64 * COMPACT_AT || cut == 0 {
        return Plan::Fits;
    }
    let request = ChatRequest {
        binding: ModelBinding { max_tokens: Some(binding.max_tokens.unwrap_or(4_000).min(4_000)), ..binding.clone() },
        messages: vec![
            Message::system(SUMMARY_PROMPT),
            Message::user(format!("Conversation to summarize:\n\n{}", transcript(&history[..cut]))),
        ],
        tools: vec![],
        session,
    };
    Plan::Summarize { cut, request: Box::new(request) }
}

/// Step 2: replace `history[..cut]` by the summary (keeping the task being worked on verbatim
/// when the cut falls inside a turn).
pub fn splice(history: &[Message], cut: usize, summary: &str) -> Vec<Message> {
    let current_task = if history[cut].role == Role::Assistant {
        history[..cut]
            .iter()
            .rposition(|m| m.role == Role::User && !m.text().starts_with(SUMMARY_TAG))
            .map(|i| history[i].text().to_string())
    } else {
        None
    };
    let mut text = format!("{SUMMARY_TAG}\nEarlier conversation, compacted:\n{}\n", summary.trim());
    if let Some(task) = current_task {
        text.push_str(&format!("\nThe request you are currently working on (verbatim):\n{task}\n"));
    }
    text.push_str("</conversation-summary>");
    let mut out = Vec::with_capacity(history.len() - cut + 2);
    out.push(Message::user(text));
    // Keep user/assistant alternation: a tail starting with a user message needs an ack first.
    if history[cut].role == Role::User {
        out.push(Message::assistant("Understood. Continuing from that summary."));
    }
    out.extend_from_slice(&history[cut..]);
    out
}

/// Between orders: a resident agent whose history passed this share of its budget is digested
/// before the next order goes to the model (see `digest_request`).
pub const DIGEST_AT: f64 = 0.40;
/// ...and, idle long enough that the provider has likely dropped its prompt cache, past this share.
pub const DIGEST_WHEN_COLD: f64 = 0.20;

const DIGEST_INSTRUCTION: &str = "[compaction] Do not call any tool and do not continue the work. Write a digest of \
this whole conversation so far, to replace it: you will continue from the digest alone. One short block per \
request the user made: what was asked, what was done (files changed, results, test outcomes), decisions and the \
user's preferences, and anything left open. Then the facts about the codebase you will need again (paths, names, \
conventions). Terse bullet points; no raw tool output.";

/// The digest request: the agent's own prefix (its system prompt, history and tools, exactly as it
/// last sent them) plus one instruction, so a caching provider serves almost all of the input from
/// its cache.
pub fn digest_request(
    binding: &ModelBinding,
    session: Option<String>,
    system: &str,
    tools: &[ToolSpec],
    history: &[Message],
) -> ChatRequest {
    let mut messages = Vec::with_capacity(history.len() + 2);
    messages.push(Message::system(system));
    messages.extend(history.iter().cloned());
    messages.push(Message::user(DIGEST_INSTRUCTION));
    ChatRequest {
        binding: ModelBinding { max_tokens: Some(binding.max_tokens.unwrap_or(4_000).min(4_000)), ..binding.clone() },
        messages,
        tools: tools.to_vec(),
        session,
    }
}

/// The history after a digest: the digest, then the last request and reply verbatim (so a
/// follow-up like "and the other one?" keeps its referent).
pub fn digest_splice(history: &[Message], digest: &str) -> Vec<Message> {
    let last_user = history.iter().rposition(|m| m.role == Role::User && !m.text().starts_with("<conversation-summary>"));
    let last_reply = history.iter().rev().find(|m| m.role == Role::Assistant && m.tool_calls.is_empty() && !m.text().is_empty());
    let mut text = format!("<conversation-summary>\nEarlier conversation, compacted:\n{}\n", digest.trim());
    if let (Some(u), Some(r)) = (last_user, last_reply) {
        text.push_str(&format!("\nThe last request (verbatim):\n{}\n\nYour reply to it (verbatim):\n{}\n", history[u].text(), r.text()));
    }
    text.push_str("</conversation-summary>");
    vec![Message::user(text), Message::assistant("Understood. Continuing from that summary.")]
}

#[cfg(test)]
mod tests {
    use super::*;
    use zilch_ipc::llm::FunctionCall;
    use zilch_ipc::ToolCall;

    fn tool_turn(i: usize, out: usize) -> Vec<Message> {
        let mut a = Message::assistant("");
        a.tool_calls = vec![ToolCall {
            id: format!("c{i}"),
            kind: "function".into(),
            function: FunctionCall { name: "read_file".into(), arguments: "{}".into() },
        }];
        vec![a, Message::tool(format!("c{i}"), "x".repeat(out))]
    }

    #[test]
    fn small_histories_are_left_alone_and_big_ones_cut_at_a_safe_point() {
        let b = ModelBinding::default();
        let mut h = vec![Message::user("Order #1: hi"), Message::assistant("hello")];
        assert!(matches!(prepare(&b, None, "s", &[], &mut h, 10_000), Plan::Fits));

        let mut h = vec![Message::user("Order #1: big")];
        for i in 0..20 {
            h.extend(tool_turn(i, 400));
        }
        h.push(Message::assistant("done"));
        h.push(Message::user("Order #2: next"));
        let Plan::Summarize { cut, request } = prepare(&b, Some("s1".into()), "s", &[], &mut h, 2_000) else { panic!("should summarize") };
        assert_ne!(h[cut].role, Role::Tool, "never cut between a call and its result");
        assert!(request.messages[1].text().contains("Order #1: big") && request.tools.is_empty());
        let out = splice(&h, cut, "- read 20 files");
        assert!(out[0].text().starts_with(SUMMARY_TAG) && out[0].text().contains("read 20 files"));
        assert_eq!(out.last().unwrap().text(), "Order #2: next", "the recent tail is kept verbatim");
        assert!(estimate_tokens("s", &out, &[]) < estimate_tokens("s", &h, &[]));
    }

    #[test]
    fn old_big_tool_outputs_are_elided_first() {
        let mut h = vec![Message::user("go")];
        h.extend(tool_turn(0, 10_000));
        h.extend(tool_turn(1, 10));
        assert!(elide_old_tool_outputs(&mut h, 2 + 1));
        assert!(h[2].text().contains("elided") && h[2].text().len() < 1_000);
        assert_eq!(h[4].text().len(), 10, "recent ones untouched");
    }
}
