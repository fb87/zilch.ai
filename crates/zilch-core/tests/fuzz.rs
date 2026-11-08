//! Property-based fuzzing (proptest) of everything that handles untrusted input:
//! provider streams, model-generated tool arguments, TODO graphs, file scopes, history.
//! More cases: PROPTEST_CASES=10000 cargo test -p zilch-core --test fuzz

use proptest::prelude::*;
use std::collections::HashSet;
use zilch_core::agent::compact::{cut_point, elide_old_tool_outputs, estimate_tokens, recent_window_start};
use zilch_core::llm::accumulate::Accumulator;
use zilch_core::llm::sse::{parse_chunk, SseDecoder};
use zilch_core::llm::{FunctionCall, LlmEvent, Message, Role, ToolCall};
use zilch_core::team::todo::{scopes_overlap, TodoInput, TodoList, TodoStatus};
use zilch_core::tools::edit::apply_edit;
use zilch_core::tools::{normalize, truncate_output, Scope};

// ---------- provider stream ----------

/// Feed `raw` to a decoder in the given chunk sizes; return all decoded events.
fn decode(raw: &[u8], cuts: &[usize]) -> Vec<LlmEvent> {
    let mut d = SseDecoder::default();
    let mut out = vec![];
    let mut pos = 0;
    let mut i = 0;
    let mut payloads = vec![];
    while pos < raw.len() {
        let n = cuts.get(i).copied().unwrap_or(raw.len()).max(1).min(raw.len() - pos);
        payloads.extend(d.push(&raw[pos..pos + n]));
        pos += n;
        i += 1;
    }
    payloads.extend(d.finish());
    for p in payloads {
        match parse_chunk(&p) {
            Ok(Some(evs)) => out.extend(evs),
            Ok(None) => break,
            Err(_) => {}
        }
    }
    out
}

fn text_chunk(t: &str) -> String {
    format!("data: {}\n\n", serde_json::json!({"choices":[{"delta":{"content": t}}]}))
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

    /// Arbitrary bytes never panic the decoder or parser.
    #[test]
    fn sse_never_panics(bytes in proptest::collection::vec(any::<u8>(), 0..2048), cuts in proptest::collection::vec(1usize..64, 0..64)) {
        let _ = decode(&bytes, &cuts);
    }

    /// Arbitrary strings never panic the chunk parser (including valid-JSON-but-wrong-shape).
    #[test]
    fn parse_chunk_never_panics(s in ".{0,400}") {
        let _ = parse_chunk(&s);
        let _ = parse_chunk(&format!("{{\"choices\":[{{\"delta\":{s}}}]}}"));
        let _ = parse_chunk(&format!("{{\"choices\":[{{\"delta\":{{\"tool_calls\":[{s}]}}}}]}}"));
    }

    /// However a valid stream is chunked (and whatever line endings), the text is identical.
    #[test]
    fn sse_chunking_is_transparent(parts in proptest::collection::vec("\\PC{0,30}", 1..20), cuts in proptest::collection::vec(1usize..40, 0..200), crlf in any::<bool>()) {
        let mut raw: String = parts.iter().map(|p| text_chunk(p)).collect();
        raw.push_str("data: [DONE]\n\n");
        if crlf { raw = raw.replace('\n', "\r\n"); }
        let evs = decode(raw.as_bytes(), &cuts);
        let text: String = evs.iter().filter_map(|e| if let LlmEvent::TextDelta(t) = e { Some(t.as_str()) } else { None }).collect();
        prop_assert_eq!(text, parts.concat());
    }

    /// Tool-call arguments split at arbitrary points (and interleaved across parallel calls)
    /// reassemble exactly.
    #[test]
    fn accumulator_reassembles_split_args(
        calls in proptest::collection::vec(("[a-z_]{1,12}", "\\PC{0,80}"), 1..5),
        splits in proptest::collection::vec(1usize..10, 1..40),
        seed in any::<u64>(),
    ) {
        // Build per-call delta lists: name+id on first piece, args in random pieces.
        let mut streams: Vec<Vec<LlmEvent>> = vec![];
        for (idx, (name, args)) in calls.iter().enumerate() {
            let chars: Vec<char> = args.chars().collect();
            let mut evs = vec![LlmEvent::ToolCallDelta { index: idx, id: Some(format!("id{idx}")), name: Some(name.clone()), args: String::new() }];
            let mut pos = 0;
            let mut k = 0;
            while pos < chars.len() {
                let n = splits[k % splits.len()].min(chars.len() - pos);
                evs.push(LlmEvent::ToolCallDelta { index: idx, id: None, name: None, args: chars[pos..pos + n].iter().collect() });
                pos += n;
                k += 1;
            }
            streams.push(evs);
        }
        // Interleave streams pseudo-randomly, preserving per-call order.
        let mut acc = Accumulator::default();
        let mut cursors = vec![0usize; streams.len()];
        let mut rng = seed;
        loop {
            let live: Vec<usize> = (0..streams.len()).filter(|&s| cursors[s] < streams[s].len()).collect();
            if live.is_empty() { break; }
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let s = live[(rng >> 33) as usize % live.len()];
            acc.push(&streams[s][cursors[s]]);
            cursors[s] += 1;
        }
        let msg = acc.finish();
        prop_assert_eq!(msg.tool_calls.len(), calls.len());
        for (i, (name, args)) in calls.iter().enumerate() {
            prop_assert_eq!(&msg.tool_calls[i].function.name, name);
            let expected = if args.trim().is_empty() { "{}".to_string() } else { args.clone() };
            prop_assert_eq!(&msg.tool_calls[i].function.arguments, &expected);
            prop_assert_eq!(&msg.tool_calls[i].id, &format!("id{i}"));
        }
    }

    /// Arbitrary event sequences never panic the accumulator.
    #[test]
    fn accumulator_never_panics(evs in proptest::collection::vec(
        prop_oneof![
            "\\PC{0,20}".prop_map(LlmEvent::TextDelta),
            "\\PC{0,20}".prop_map(LlmEvent::ReasoningDelta),
            (0usize..1000, proptest::option::of("[a-z]{0,5}"), proptest::option::of("[a-z_]{0,8}"), "\\PC{0,20}")
                .prop_map(|(index, id, name, args)| LlmEvent::ToolCallDelta { index, id, name, args }),
            "[a-z_]{0,10}".prop_map(LlmEvent::Finish),
        ], 0..60)) {
        let mut acc = Accumulator::default();
        for e in &evs { acc.push(e); }
        let m = acc.finish();
        for c in &m.tool_calls { prop_assert!(!c.id.is_empty() && !c.function.arguments.is_empty()); }
    }

    // ---------- edit_file ----------

    /// Model-provided edit strings: never panic; results obey the uniqueness contract.
    #[test]
    fn apply_edit_contract(text in "\\PC{0,200}", old in "\\PC{0,8}", new in "\\PC{0,8}", all in any::<bool>()) {
        prop_assume!(!old.is_empty());
        let n = text.matches(old.as_str()).count();
        match apply_edit(&text, &old, &new, all) {
            Ok((out, k)) => {
                prop_assert!(n == 1 || all);
                prop_assert_eq!(k, n);
                prop_assert_eq!(out, text.replace(old.as_str(), new.as_str()));
            }
            Err(_) => prop_assert!(n == 0 || (n > 1 && !all)),
        }
    }

    // ---------- output truncation ----------

    #[test]
    fn truncate_is_char_safe_and_bounded(s in "\\PC{0,3000}", max in 1usize..500) {
        let t = truncate_output(&s, max);
        if s.len() <= max {
            prop_assert_eq!(&t, &s);
        } else {
            prop_assert!(t.contains("truncated"));
            prop_assert!(t.len() <= max + 64, "len {} for max {max}", t.len());
        }
    }

    // ---------- paths & scopes ----------

    #[test]
    fn normalize_never_escapes_root_lexically(parts in proptest::collection::vec(prop_oneof![Just("..".to_string()), Just(".".to_string()), "[a-z]{1,5}"], 0..12)) {
        let p = std::path::PathBuf::from("/root").join(parts.join("/"));
        let n = normalize(&p);
        prop_assert!(n.components().all(|c| !matches!(c, std::path::Component::ParentDir | std::path::Component::CurDir)));
    }

    #[test]
    fn scope_patterns_never_panic(pats in proptest::collection::vec("\\PC{0,12}", 0..5), path in "\\PC{0,30}") {
        if let Ok(s) = Scope::new(&pats) { let _ = s.contains(&path); }
        let _ = scopes_overlap(&pats, std::slice::from_ref(&path));
    }

    #[test]
    fn scope_overlap_is_symmetric(a in proptest::collection::vec("[a-c*/.]{1,6}", 0..4), b in proptest::collection::vec("[a-c*/.]{1,6}", 0..4)) {
        prop_assert_eq!(scopes_overlap(&a, &b), scopes_overlap(&b, &a));
        // Identical non-empty scopes always overlap; empty means "touches anything".
        prop_assert!(scopes_overlap(&a, &a));
    }
}

// ---------- TODO graphs ----------

fn todo_inputs() -> impl Strategy<Value = Vec<TodoInput>> {
    proptest::collection::vec((0u8..12, proptest::collection::vec(0u8..12, 0..4), proptest::collection::vec("[a-c]\\.rs|src/\\*\\*", 0..3)), 0..12)
        .prop_map(|items| {
            items
                .into_iter()
                .map(|(id, deps, files)| TodoInput {
                    id: format!("t{id}"),
                    title: None,
                    task: Some(format!("do t{id}")),
                    worker: None,
                    files: Some(files),
                    depends_on: Some(deps.into_iter().map(|d| format!("t{d}")).collect()),
                    model: None,
                    provider: None,
                    status: None,
                    remove: false,
                })
                .collect()
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, .. ProptestConfig::default() })]

    /// Random (possibly cyclic, dangling, duplicate) TODO batches: apply never panics; when it
    /// succeeds the graph is valid and the waves are a correct topological layering.
    #[test]
    fn todo_graphs_are_valid_or_rejected(batch in todo_inputs(), batch2 in todo_inputs()) {
        let mut list = TodoList::default();
        for b in [batch, batch2] {
            let snapshot = list.clone();
            match list.apply(b, "coder") {
                Ok(_) => {
                    prop_assert!(list.validate().is_ok());
                    let waves = list.waves().unwrap();
                    let mut seen: HashSet<String> = list.items.iter().filter(|t| t.status.is_terminal()).map(|t| t.id.clone()).collect();
                    let mut count = 0;
                    for w in &waves {
                        for id in w {
                            let t = list.get(id).unwrap();
                            prop_assert!(t.depends_on.iter().all(|d| seen.contains(d)), "{id} scheduled before its deps");
                        }
                        seen.extend(w.iter().cloned());
                        count += w.len();
                    }
                    prop_assert_eq!(count, list.items.iter().filter(|t| !t.status.is_terminal()).count());
                    let ids: HashSet<&str> = list.items.iter().map(|t| t.id.as_str()).collect();
                    prop_assert_eq!(ids.len(), list.items.len(), "duplicate ids");
                }
                Err(_) => {
                    // A rejected batch must not leave a half-applied invalid graph behind in callers
                    // that validate on a copy (todo_write does) — restore and continue.
                    list = snapshot;
                }
            }
        }
        // Approval bookkeeping: after approving, nothing open is unapproved.
        list.approve_all_open();
        prop_assert!(list.unapproved_open().is_empty());
        prop_assert!(list.items.iter().all(|t| t.status == TodoStatus::Pending));
    }
}

// ---------- history compaction ----------

fn history_strategy() -> impl Strategy<Value = Vec<Message>> {
    // Generate well-formed histories: user turns containing assistant tool-call groups.
    proptest::collection::vec((
        "\\PC{0,40}",
        proptest::collection::vec((0usize..3, 0usize..6000, 0usize..2000), 0..5),
    ), 1..12)
    .prop_map(|turns| {
        let mut h = vec![];
        let mut id = 0;
        for (utext, groups) in turns {
            h.push(Message::user(utext));
            for (ncalls, out_len, text_len) in groups {
                let calls: Vec<ToolCall> = (0..ncalls.max(1)).map(|_| {
                    id += 1;
                    ToolCall { id: format!("c{id}"), kind: "function".into(), function: FunctionCall { name: "read_file".into(), arguments: "{}".into() } }
                }).collect();
                h.push(Message { role: Role::Assistant, content: Some("t".repeat(text_len)), tool_calls: calls.clone(), tool_call_id: None, reasoning_content: None });
                for c in calls { h.push(Message::tool(c.id, "o".repeat(out_len))); }
            }
            h.push(Message::assistant("done"));
        }
        h
    })
}

fn well_formed(h: &[Message]) -> Result<(), String> {
    let mut open: Vec<String> = vec![];
    for (i, m) in h.iter().enumerate() {
        match m.role {
            Role::Assistant => open = m.tool_calls.iter().map(|c| c.id.clone()).collect(),
            Role::Tool => {
                if !open.contains(m.tool_call_id.as_ref().unwrap()) {
                    return Err(format!("orphan tool result at {i}"));
                }
            }
            Role::User => {
                if i > 0 && h[i - 1].role == Role::User {
                    return Err(format!("two user messages at {i}"));
                }
            }
            Role::System => {}
        }
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, .. ProptestConfig::default() })]

    /// Cutting the history at `cut_point` never orphans a tool result, for any budget.
    #[test]
    fn cut_point_never_splits_tool_groups(h in history_strategy(), keep in 0usize..20_000) {
        let cut = cut_point(&h, keep);
        prop_assert!(cut < h.len().max(1));
        if cut > 0 {
            prop_assert!(h[cut].role != Role::Tool, "tail starts with a tool result");
            prop_assert!(well_formed(&h[cut..]).is_ok() || h[cut].role == Role::Assistant);
            // Splicing a summary in front (as summarize() does) yields a well-formed history.
            let mut spliced = vec![Message::user("<conversation-summary>s</conversation-summary>")];
            if h[cut].role == Role::User { spliced.push(Message::assistant("ok")); }
            spliced.extend_from_slice(&h[cut..]);
            prop_assert!(well_formed(&spliced).is_ok(), "{:?}", well_formed(&spliced));
        }
    }

    /// Eliding never changes structure and never grows the history.
    #[test]
    fn elision_preserves_structure(mut h in history_strategy(), keep in 0usize..20_000) {
        let before = estimate_tokens("", &h, &[]);
        let roles: Vec<Role> = h.iter().map(|m| m.role).collect();
        let w = recent_window_start(&h, keep);
        elide_old_tool_outputs(&mut h, w);
        prop_assert_eq!(roles, h.iter().map(|m| m.role).collect::<Vec<_>>());
        prop_assert!(estimate_tokens("", &h, &[]) <= before);
        prop_assert!(well_formed(&h).is_ok());
    }
}
