//! History compaction: keeps long-running agents (especially the master, which accumulates
//! every worker report) inside their context window.
//!
//! Strategy, applied only when the estimated prompt exceeds `COMPACT_AT` of the budget:
//! 1. Elide large tool outputs outside the recent window (cheap, lossless for recent work).
//! 2. If still too big, summarize the older turns with the agent's own model and keep the
//!    recent turns verbatim. A cut never separates an assistant tool call from its results.
//! 3. If summarization fails, keep the result of step 1 (never fail the turn because of this).

use crate::llm::{complete, ChatRequest, LlmClient, Message, ModelBinding, Role, ToolSpec};

/// Compact when the prompt reaches this fraction of the context budget...
pub const COMPACT_AT: f64 = 0.75;
/// ...and aim to keep this fraction of the budget as verbatim recent history.
const KEEP_RECENT: f64 = 0.35;
/// Tool outputs larger than this (chars) are elided outside the recent window.
const ELIDE_TOOL_OVER: usize = 1_500;

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

/// A verbatim tail may start at a user message or at an assistant message (the start of a
/// tool-call group) — never at a tool result, so no call is separated from its results.
fn valid_cut(history: &[Message], i: usize) -> bool {
    i > 0 && history[i].role != Role::Tool && history[i].role != Role::System
}

/// Index where the verbatim tail starts (the earliest valid cut whose tail fits `keep_budget`,
/// but always keeping at least the last message group). Returns 0 if there is nothing to compact.
pub fn cut_point(history: &[Message], keep_budget: usize) -> usize {
    let mut acc = 0;
    let mut best = 0;
    for i in (1..history.len()).rev() {
        acc += msg_tokens(&history[i]);
        if valid_cut(history, i) {
            if acc > keep_budget && best != 0 {
                break;
            }
            best = i;
        }
    }
    // Don't re-summarize a history that is just [summary, ack?, small tail].
    let starts_with_summary = history.first().is_some_and(|m| m.text().starts_with("<conversation-summary>"));
    if starts_with_summary && best <= 2 {
        return 0;
    }
    best
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

/// Render old turns as plain text for the summarizer (tool output clipped).
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
        let text: String = if m.role == Role::Tool && text.len() > 2_000 { format!("{}… [clipped]", &text[..text.floor_char_boundary(2_000)]) } else { text.to_string() };
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

const SUMMARY_PROMPT: &str = "You compress an AI agent's conversation history so it can continue working \
with a smaller context. Write a dense summary that preserves: the user's goals and constraints, decisions made, \
work completed (files changed, results, test outcomes), current state of any plan/TODOs, open problems, and \
facts about the codebase the agent will need again (paths, names, conventions). Omit pleasantries and raw tool \
output. Use terse bullet points.";

/// Step 2: summarize `history[..cut]` and splice the summary in. Returns the new history.
pub async fn summarize(llm: &dyn LlmClient, binding: &ModelBinding, session: Option<String>, history: &[Message], cut: usize) -> anyhow::Result<Vec<Message>> {
    let req = ChatRequest {
        binding: ModelBinding { max_tokens: Some(binding.max_tokens.unwrap_or(4_000).min(4_000)), ..binding.clone() },
        messages: vec![Message::system(SUMMARY_PROMPT), Message::user(format!("Conversation to summarize:\n\n{}", transcript(&history[..cut])))],
        tools: vec![],
        session,
    };
    let (msg, _) = complete(llm, &req, |_| {}).await?;
    let summary = msg.text().trim().to_string();
    if summary.is_empty() {
        anyhow::bail!("empty summary");
    }
    // If the cut falls inside a turn, the instruction being worked on is in the summarized part:
    // quote it verbatim so the agent never loses its task.
    let current_task = if history[cut].role == Role::Assistant {
        history[..cut].iter().rposition(|m| m.role == Role::User && !m.text().starts_with("<conversation-summary>")).map(|i| history[i].text().to_string())
    } else {
        None
    };
    let mut text = format!("<conversation-summary>\nEarlier conversation, compacted:\n{summary}\n");
    if let Some(task) = current_task {
        text.push_str(&format!("\nThe request you are currently working on (verbatim):\n{task}\n"));
    }
    text.push_str("</conversation-summary>");

    let mut out = Vec::with_capacity(history.len() - cut + 2);
    out.push(Message::user(text));
    // Keep strict user/assistant alternation: the tail continues with either a user message
    // (needs an assistant ack in between) or an assistant tool-call group (follows directly).
    if history[cut].role == Role::User {
        out.push(Message::assistant("Understood. Continuing from that summary."));
    }
    out.extend_from_slice(&history[cut..]);
    Ok(out)
}

/// What a compaction pass did, for logging.
#[derive(Debug, Clone, PartialEq)]
pub struct CompactReport {
    pub before: usize,
    pub after: usize,
    pub summarized: bool,
}

/// Compact `history` in place if needed. `budget` is the agent's context size in tokens.
pub async fn maybe_compact(
    llm: &dyn LlmClient,
    binding: &ModelBinding,
    session: Option<String>,
    system: &str,
    tools: &[ToolSpec],
    history: &mut Vec<Message>,
    budget: usize,
) -> Option<CompactReport> {
    let before = estimate_tokens(system, history, tools);
    if (before as f64) < budget as f64 * COMPACT_AT {
        return None;
    }
    let fixed = estimate_tokens(system, &[], tools);
    let keep = ((budget as f64 * KEEP_RECENT) as usize).saturating_sub(fixed).max(256);

    // Eliding content never breaks call/result pairing, so it may reach inside the current
    // turn (a worker's single task can span dozens of tool steps).
    let window = recent_window_start(history, keep);
    elide_old_tool_outputs(history, window);
    let mid = estimate_tokens(system, history, tools);
    let report = |after, summarized| if after < before { Some(CompactReport { before, after, summarized }) } else { None };
    // Summaries must cut at a user turn so no tool call is separated from its results.
    let cut = cut_point(history, keep);
    if (mid as f64) < budget as f64 * COMPACT_AT || cut == 0 {
        return report(mid, false);
    }
    match summarize(llm, binding, session, history, cut).await {
        Ok(new) => {
            *history = new;
            report(estimate_tokens(system, history, tools), true)
        }
        Err(_) => report(mid, false),
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::{FunctionCall, ToolCall};

    fn call_msg(id: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: None,
            tool_calls: vec![ToolCall { id: id.into(), kind: "function".into(), function: FunctionCall { name: "read_file".into(), arguments: "{}".into() } }],
            tool_call_id: None,
            reasoning_content: None,
        }
    }

    fn history(turns: usize) -> Vec<Message> {
        let mut h = vec![];
        for i in 0..turns {
            h.push(Message::user(format!("task {i}")));
            h.push(call_msg(&format!("c{i}")));
            h.push(Message::tool(format!("c{i}"), "x".repeat(4_000)));
            h.push(Message::assistant(format!("done {i}")));
        }
        h
    }

    #[test]
    fn cut_lands_on_user_turns() {
        let h = history(10);
        let cut = cut_point(&h, 2_500);
        assert!(cut > 0 && cut < h.len());
        assert_ne!(h[cut].role, Role::Tool);
        // The verbatim tail never starts with an orphan tool result.
        assert_ne!(h[cut].role, Role::Tool);
    }

    struct FakeSummarizer {
        fail: bool,
        seen: std::sync::Mutex<Vec<ChatRequest>>,
    }

    #[async_trait::async_trait]
    impl LlmClient for FakeSummarizer {
        async fn stream(&self, req: &ChatRequest) -> anyhow::Result<crate::llm::EventStream> {
            self.seen.lock().unwrap().push(req.clone());
            if self.fail {
                anyhow::bail!("provider down");
            }
            Ok(Box::pin(futures::stream::iter(vec![Ok(crate::llm::LlmEvent::TextDelta("- user wants tasks 0-9 done\n- tasks 0-5 done".into()))])))
        }
    }

    fn big_history(turns: usize) -> Vec<Message> {
        // Big assistant texts (not tool output) so eliding alone can't get under budget.
        let mut h = vec![];
        for i in 0..turns {
            h.push(Message::user(format!("task {i}")));
            h.push(call_msg(&format!("c{i}")));
            h.push(Message::tool(format!("c{i}"), "ok"));
            h.push(Message::assistant(format!("report {i}: {}", "y".repeat(3_000))));
        }
        h
    }

    #[tokio::test]
    async fn summarizes_when_over_budget() {
        let llm = FakeSummarizer { fail: false, seen: Default::default() };
        let mut h = big_history(12);
        let len_before = h.len();
        let r = maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 8_000).await.unwrap();
        assert!(r.summarized && r.after < r.before, "{r:?}");
        assert!((r.after as f64) < 8_000.0 * COMPACT_AT);
        assert!(h[0].text().contains("<conversation-summary>") && h[0].text().contains("tasks 0-5 done"));
        assert_eq!(h[1].role, Role::Assistant);
        assert_well_formed(&h);
        assert!(h.len() < len_before);
        // The most recent turn survives verbatim.
        assert_eq!(h.last().unwrap().text(), big_history(12).last().unwrap().text());
        // The summarizer saw the old turns, not the tail.
        let seen = llm.seen.lock().unwrap();
        assert!(seen[0].messages[1].text().contains("task 0") && seen[0].tools.is_empty());
    }

    #[tokio::test]
    async fn falls_back_to_eliding_when_summary_fails() {
        let llm = FakeSummarizer { fail: true, seen: Default::default() };
        let mut h = history(12); // big tool outputs: eliding helps
        let r = maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 6_000).await.unwrap();
        assert!(!r.summarized && r.after < r.before);
        assert!(h[2].text().contains("elided"));
        assert_eq!(h.len(), 48, "no messages dropped without a summary");
    }

    #[tokio::test]
    async fn compacts_inside_a_single_long_turn() {
        // One task, many tool steps (the shape of every worker run): no earlier user turn to cut at.
        let llm = FakeSummarizer { fail: false, seen: Default::default() };
        let mut h = vec![Message::user("fix everything")];
        for i in 0..12 {
            h.push(call_msg(&format!("c{i}")));
            h.push(Message::tool(format!("c{i}"), "z".repeat(4_000)));
        }
        let r = maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 8_000).await.unwrap();
        assert!(r.after < r.before && !r.summarized, "{r:?}");
        assert!(h[2].text().contains("elided"), "old output in the same turn is elided");
        assert_eq!(h.last().unwrap().text().len(), 4_000, "latest output kept verbatim");
        assert_eq!(h.len(), 25);
        // Nothing left to gain: a second pass reports nothing (no notice spam).
        let again = maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 8_000).await;
        assert!(again.is_none_or(|r| r.after < r.before));
    }

    /// Every tool result follows an assistant call with its id; no two user messages in a row.
    fn assert_well_formed(h: &[Message]) {
        let mut open: Vec<String> = vec![];
        for (i, m) in h.iter().enumerate() {
            match m.role {
                Role::Assistant => open = m.tool_calls.iter().map(|c| c.id.clone()).collect(),
                Role::Tool => assert!(open.contains(m.tool_call_id.as_ref().unwrap()), "orphan tool result at {i}"),
                Role::User => assert!(i == 0 || h[i - 1].role != Role::User, "two user messages in a row at {i}"),
                Role::System => {}
            }
        }
    }

    #[tokio::test]
    async fn summarizes_mid_turn_and_keeps_task_verbatim() {
        let llm = FakeSummarizer { fail: false, seen: Default::default() };
        let mut h = vec![Message::user("TASK: fix the parser in src/parse.rs")];
        for i in 0..12 {
            let mut m = call_msg(&format!("c{i}"));
            m.content = Some(format!("thinking about step {i}: {}", "w".repeat(3_000)));
            h.push(m);
            h.push(Message::tool(format!("c{i}"), "ok"));
        }
        let r = maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 8_000).await.unwrap();
        assert!(r.summarized && (r.after as f64) < 8_000.0 * COMPACT_AT, "{r:?}");
        assert!(h[0].text().contains("TASK: fix the parser in src/parse.rs"), "task quoted verbatim");
        assert_eq!(h[1].role, Role::Assistant, "tail resumes at a tool-call group, no ack needed");
        assert!(!h[1].tool_calls.is_empty());
        assert_well_formed(&h);
        // A second pass right after doesn't summarize the summary again.
        let before = llm.seen.lock().unwrap().len();
        let _ = maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 8_000).await;
        assert_eq!(llm.seen.lock().unwrap().len(), before);
    }

    #[tokio::test]
    async fn no_op_under_threshold() {
        let llm = FakeSummarizer { fail: false, seen: Default::default() };
        let mut h = history(2);
        assert!(maybe_compact(&llm, &ModelBinding::default(), None, "sys", &[], &mut h, 128_000).await.is_none());
        assert!(llm.seen.lock().unwrap().is_empty());
    }

    #[test]
    fn eliding_shrinks_old_outputs_only() {
        let mut h = history(4);
        let cut = 8;
        assert!(elide_old_tool_outputs(&mut h, cut));
        assert!(h[2].text().contains("elided"));
        assert_eq!(h[10].text().len(), 4_000); // recent tool output untouched
    }
}
