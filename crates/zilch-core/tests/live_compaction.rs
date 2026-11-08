//! LIVE test of context management against a real (free) model. Opt-in:
//!   OPENCODE_API_KEY=... cargo test -p zilch-core --test live_compaction -- --ignored --nocapture
//! Optional: ZILCH_LIVE_MODEL (default "beam-free"), ZILCH_LIVE_BASE_URL.
//!
//! Builds a long worker-style history with specific facts buried in early turns, compacts it with
//! the real model, then asks the same model questions answerable ONLY from the compacted part.

use zilch_core::agent::compact::{estimate_tokens, maybe_compact, COMPACT_AT};
use zilch_core::llm::openai::OpenAiClient;
use zilch_core::llm::{complete, ChatRequest, FunctionCall, Message, ModelBinding, Role, ToolCall};

fn call(id: &str, path: &str) -> Message {
    Message {
        role: Role::Assistant,
        content: None,
        tool_calls: vec![ToolCall { id: id.into(), kind: "function".into(), function: FunctionCall { name: "read_file".into(), arguments: format!("{{\"path\":\"{path}\"}}") } }],
        tool_call_id: None,
        reasoning_content: None,
    }
}

/// Facts the agent must still know after compaction: (question, expected substring).
const FACTS: &[(&str, &str)] = &[
    ("Which file contained the bug you fixed first? Answer with just the path.", "src/billing/tax.rs"),
    ("What exact value did you change the VAT rate constant to? Answer with just the number.", "0.19"),
    ("What is the name of the integration test that was failing? Answer with just the name.", "test_invoice_rounding_eu"),
];

fn long_history() -> Vec<Message> {
    let filler = |n: usize| (0..n).map(|i| format!("    let value_{i} = compute_{i}(input) * FACTOR; // routine line {i}\n")).collect::<String>();
    let mut h = vec![Message::user(
        "TASK [t1]: Fix the EU invoice rounding bug. The failing integration test is test_invoice_rounding_eu. \
         Find the cause, fix it, and report.",
    )];
    // Early turns carry the facts.
    h.push(call("c1", "src/billing/mod.rs"));
    h.push(Message::tool("c1", format!("pub mod tax;\npub mod invoice;\n{}", filler(120))));
    h.push(Message::assistant(
        "The rounding happens in the tax module. Reading src/billing/tax.rs next — I suspect the VAT constant.",
    ));
    h.push(call("c2", "src/billing/tax.rs"));
    h.push(Message::tool("c2", format!("pub const VAT_RATE: f64 = 0.16; // outdated\n{}", filler(150))));
    h.push(Message::assistant(
        "Found it: the bug is in src/billing/tax.rs — VAT_RATE is 0.16 but Germany's rate is 19%. I changed VAT_RATE \
         from 0.16 to 0.19 and switched rounding to half-up on cents.",
    ));
    // Lots of later, unrelated work to push the facts out of the recent window.
    for i in 3..40 {
        h.push(call(&format!("c{i}"), &format!("src/other/module_{i}.rs")));
        h.push(Message::tool(format!("c{i}"), filler(60)));
        h.push(Message::assistant(format!("module_{i}.rs looks fine; nothing related to invoices there. {}", "Checked every function. ".repeat(20))));
    }
    h
}

#[tokio::test]
#[ignore = "live: needs OPENCODE_API_KEY and network"]
async fn compaction_preserves_what_matters_with_a_real_model() {
    let key = std::env::var("OPENCODE_API_KEY").expect("set OPENCODE_API_KEY");
    let base = std::env::var("ZILCH_LIVE_BASE_URL").unwrap_or_else(|_| "https://opencode.ai/inference/go/openai/v1".into());
    let model = std::env::var("ZILCH_LIVE_MODEL").unwrap_or_else(|_| "beam-free".into());
    let client = OpenAiClient::new("live", base, Some(key), 2).with_session_header(Some("x-opencode-session".into()));
    let binding = ModelBinding { provider: "live".into(), model, ..Default::default() };
    let system = "You are a CODER on a software team. Report precisely.";
    let session = Some("zilch-live-compaction-test".to_string());

    let mut history = long_history();
    let budget = 16_000;
    let before = estimate_tokens(system, &history, &[]);
    println!("history: {} messages, ~{before} tokens, budget {budget} (compacts at {:.0})", history.len(), budget as f64 * COMPACT_AT);
    assert!(before as f64 > budget as f64 * COMPACT_AT, "fixture must exceed the threshold");

    let report = maybe_compact(&client, &binding, session.clone(), system, &[], &mut history, budget).await.expect("should compact");
    println!("compaction: {report:?}\n--- summary message ---\n{}\n-----------------------", history[0].text());
    assert!(report.summarized, "expected a real summary, got {report:?}");
    assert!((report.after as f64) < budget as f64 * COMPACT_AT, "still over threshold: {report:?}");
    assert!(history[0].text().contains("test_invoice_rounding_eu"), "current task must be quoted verbatim");

    // Ask the real model questions only the compacted part can answer.
    let mut correct = 0;
    for (q, expected) in FACTS {
        let mut msgs = vec![Message::system(system)];
        msgs.extend(history.iter().cloned());
        msgs.push(Message::user(format!("Quick check before you continue: {q}")));
        let req = ChatRequest { binding: binding.clone(), messages: msgs, tools: vec![], session: session.clone() };
        let (ans, _) = complete(&client, &req, |_| {}).await.expect("answer");
        let ok = ans.text().contains(expected);
        println!("Q: {q}\nA: {}\n=> {}\n", ans.text().trim(), if ok { "✓" } else { "✗" });
        correct += ok as usize;
    }
    println!("recall: {correct}/{}", FACTS.len());
    assert_eq!(correct, FACTS.len(), "the agent forgot facts that were compacted away");
}
