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

#[tokio::test]
async fn stalled_stream_times_out_instead_of_hanging() {
    let s = MockServer::start().await;
    // Headers arrive, then the body is delayed far beyond the idle timeout.
    Mock::given(method("POST"))
        .respond_with(
            ResponseTemplate::new(200)
                .set_body_string(sse(&[r#"{"choices":[{"delta":{"content":"late"}}]}"#]))
                .set_delay(std::time::Duration::from_secs(30)),
        )
        .mount(&s)
        .await;
    let client = zilch_core::llm::openai::OpenAiClient::new("t", format!("{}/v1", s.uri()), None, 2)
        .with_idle_timeout(std::time::Duration::from_millis(300));
    let req = ChatRequest { binding: Default::default(), messages: vec![Message::user("x")], tools: vec![], session: None };
    let t = std::time::Instant::now();
    let err = complete(&client, &req, |_| {}).await.unwrap_err().to_string();
    assert!(t.elapsed() < std::time::Duration::from_secs(20), "hung for {:?}", t.elapsed());
    assert!(err.contains("no response headers") || err.contains("stalled") || err.contains("failed"), "{err}");
}

#[tokio::test]
async fn retry_after_is_honoured_then_succeeds() {
    let s = MockServer::start().await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(429).insert_header("retry-after", "1").set_body_string("slow down"))
        .up_to_n_times(1)
        .mount(&s)
        .await;
    Mock::given(method("POST"))
        .respond_with(ResponseTemplate::new(200).set_body_string(sse(&[r#"{"choices":[{"delta":{"content":"ok"}}]}"#])))
        .mount(&s)
        .await;
    let client = zilch_core::llm::openai::OpenAiClient::new("t", format!("{}/v1", s.uri()), None, 2);
    let req = ChatRequest { binding: Default::default(), messages: vec![Message::user("x")], tools: vec![], session: None };
    let t = std::time::Instant::now();
    let (msg, _) = complete(&client, &req, |_| {}).await.unwrap();
    assert_eq!(msg.text(), "ok");
    assert!(t.elapsed() >= std::time::Duration::from_millis(900), "Retry-After ignored: {:?}", t.elapsed());
}

#[tokio::test]
async fn client_errors_are_not_retried() {
    let s = MockServer::start().await;
    Mock::given(method("POST")).respond_with(ResponseTemplate::new(401).set_body_string("bad key")).expect(1).mount(&s).await;
    let client = zilch_core::llm::openai::OpenAiClient::new("t", format!("{}/v1", s.uri()), None, 2);
    let req = ChatRequest { binding: Default::default(), messages: vec![Message::user("x")], tools: vec![], session: None };
    assert!(complete(&client, &req, |_| {}).await.unwrap_err().to_string().contains("401"));
}

#[tokio::test]
async fn anthropic_protocol_over_http() {
    use wiremock::matchers::header_exists;
    let s = MockServer::start().await;
    let body = [
        r#"{"type":"message_start","message":{"usage":{"input_tokens":11}}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_9","name":"glob"}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\"pattern\":\"*\"}"}}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":5}}"#,
        r#"{"type":"message_stop"}"#,
    ]
    .iter()
    .map(|c| format!("event: x\ndata: {c}\n\n"))
    .collect::<String>();
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("x-api-key", "ak"))
        .and(header("anthropic-version", "2023-06-01"))
        .and(body_partial_json(serde_json::json!({"model":"claude-x","stream":true,"system":[{"type":"text","text":"S","cache_control":{"type":"ephemeral"}}],"max_tokens":8192})))
        .respond_with(ResponseTemplate::new(200).set_body_string(body))
        .expect(1)
        .mount(&s)
        .await;
    // A bearer token must NOT be sent to an Anthropic endpoint.
    Mock::given(method("POST")).and(header_exists("authorization")).respond_with(ResponseTemplate::new(400)).mount(&s).await;
    let client = zilch_core::llm::openai::OpenAiClient::new("a", format!("{}/v1", s.uri()), Some("ak".into()), 2)
        .with_protocol(zilch_core::llm::protocol::Protocol::Anthropic);
    let req = ChatRequest {
        binding: zilch_core::llm::ModelBinding { model: "claude-x".into(), ..Default::default() },
        messages: vec![Message::system("S"), Message::user("go")],
        tools: vec![],
        session: None,
    };
    let (msg, usage) = complete(&client, &req, |_| {}).await.unwrap();
    assert_eq!(msg.tool_calls[0].id, "toolu_9");
    assert_eq!(msg.tool_calls[0].function.arguments, r#"{"pattern":"*"}"#);
    assert_eq!((usage.prompt_tokens, usage.completion_tokens), (11, 5));
}

/// A model the provider does not list is reported at startup, naming the agent and what exists.
#[tokio::test]
async fn unlisted_models_are_reported_per_agent() {
    let s = MockServer::start().await;
    Mock::given(method("GET"))
        .and(path("/v1/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"data": [{"id": "our-model"}, {"id": "our-coder"}]})))
        .mount(&s)
        .await;
    let dir = tempfile::tempdir().unwrap();
    let proj = dir.path().join(".zilch");
    std::fs::create_dir_all(&proj).unwrap();
    std::fs::write(
        proj.join("config.toml"),
        format!(
            r#"
[providers.xyz]
base_url = "{}/v1"
api_key_env = "XYZ_KEY"
[defaults]
provider = "xyz"
model = "beam-free"
[roles.master]
model = "our-model"
[roles.coder]
model = "our-coder"
"#,
            s.uri()
        ),
    )
    .unwrap();
    let env: BTreeMap<String, String> = [("XYZ_KEY".to_string(), "k".to_string())].into();
    let cfg = Config::load_from(dir.path(), &[proj], &env, &Overrides::default()).unwrap();
    let (_, warnings) = cfg.discover_checked().await;
    // master, coder: listed. spec: inherits the master's model. explorer: stale [defaults] model.
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("agent 'explorer' uses model 'beam-free'"), "{}", warnings[0]);
    assert!(warnings[0].contains("our-coder, our-model"), "{}", warnings[0]);
}
