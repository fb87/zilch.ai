//! The real HTTP client against wiremock: streaming, tool-call assembly, per-provider routing.

use std::collections::BTreeMap;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};
use zilch_core::llm::{complete, ChatRequest, Message};
use zilch_core::{Config, Overrides};

fn sse(chunks: &[&str]) -> String {
    chunks.iter().map(|c| format!("data: {c}\n\n")).collect::<String>() + "data: [DONE]\n\n"
}

#[tokio::test]
async fn each_agent_hits_its_own_provider_and_model() {
    let a = MockServer::start().await;
    let b = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(header("authorization", "Bearer key-a"))
        .and(body_partial_json(serde_json::json!({"model":"big-planner","stream":true})))
        .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(sse(&[
            r#"{"choices":[{"delta":{"content":"plan"}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"t1","function":{"name":"glob","arguments":"{\"pat"}}]}}]}"#,
            r#"{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"tern\":\"*\"}"}}]},"finish_reason":"tool_calls"}]}"#,
            r#"{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":3}}"#,
        ])))
        .expect(1)
        .mount(&a)
        .await;
    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .and(body_partial_json(serde_json::json!({"model":"small-coder","temperature":0.2})))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[r#"{"choices":[{"delta":{"content":"code"}}]}"#])))
        .expect(1)
        .mount(&b)
        .await;

    let dir = tempfile::tempdir().unwrap();
    let z = dir.path().join(".zilch");
    std::fs::create_dir_all(&z).unwrap();
    std::fs::write(
        z.join("config.toml"),
        format!(
            r#"
[providers.alpha]
base_url = "{}/v1"
api_key_env = "ALPHA_KEY"
[providers.beta]
base_url = "{}/v1"
[roles.master]
provider = "alpha"
model = "big-planner"
[roles.coder]
provider = "beta"
model = "small-coder"
temperature = 0.2
"#,
            a.uri(),
            b.uri()
        ),
    )
    .unwrap();
    std::env::set_var("ALPHA_KEY", "key-a");
    let env: BTreeMap<String, String> = [("ALPHA_KEY".into(), "key-a".into()), ("OPENAI_API_KEY".into(), "x".into())].into();
    let cfg = Config::load_from(dir.path(), &[z], &env, &Overrides::default()).unwrap();
    let providers = cfg.build_providers().unwrap();

    let m = cfg.master();
    let req = ChatRequest { binding: m.binding.clone(), messages: vec![Message::user("hi")], tools: vec![], session: None };
    let mut streamed = String::new();
    let (msg, usage) = complete(providers.get(&m.binding.provider).unwrap().as_ref(), &req, |t| streamed.push_str(t)).await.unwrap();
    assert_eq!(streamed, "plan");
    assert_eq!(msg.tool_calls[0].function.arguments, r#"{"pattern":"*"}"#);
    assert_eq!(usage.prompt_tokens, 7);

    let c = cfg.profile("coder").unwrap();
    let req = ChatRequest { binding: c.binding.clone(), messages: vec![Message::user("hi")], tools: vec![], session: None };
    let (msg, _) = complete(providers.get(&c.binding.provider).unwrap().as_ref(), &req, |_| {}).await.unwrap();
    assert_eq!(msg.text(), "code");
}

#[tokio::test]
async fn provider_errors_surface() {
    let s = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(400).set_body_string("bad model")).mount(&s).await;
    let client = zilch_core::llm::openai::OpenAiClient::new("t", format!("{}/v1", s.uri()), None, 2);
    let req = ChatRequest { binding: Default::default(), messages: vec![Message::user("x")], tools: vec![], session: None };
    let err = complete(&client, &req, |_| {}).await.unwrap_err().to_string();
    assert!(err.contains("400") && err.contains("bad model"), "{err}");
}
