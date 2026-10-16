mod common;

use common::{MockServer, Resp};
use freshrss_tts::director::{build_messages, director_schema, plan_story, Completer, Mood, PlanOptions};
use freshrss_tts::llm::ChatClient;
use freshrss_tts::textprep::split_sentences;
use serde_json::json;

#[test]
fn sends_a_schema_constrained_request_and_reads_the_content() {
    let server =
        MockServer::start(|_| Resp::json(json!({"choices": [{"message": {"content": "{\"ok\": 1}"}}]})));
    let mut client = ChatClient::new(&format!("{}/", server.url)).with_api_key(Some("k".into()));
    client.max_tokens = 123;
    let schema = director_schema();
    let msgs = build_messages(&split_sentences("Hello there."), None, None, None);
    assert_eq!(client.chat_json(&msgs, &schema).unwrap(), "{\"ok\": 1}");

    let req = &server.requests()[0];
    assert_eq!(
        (req.method.as_str(), req.path()),
        ("POST", "/v1/chat/completions")
    );
    assert_eq!(req.headers["authorization"], "Bearer k");
    let body = req.json();
    assert_eq!(body["response_format"]["schema"], schema);
    assert_eq!(body["response_format"]["json_schema"]["schema"], schema);
    assert_eq!(body["max_tokens"], 123);
    assert_eq!(body["chat_template_kwargs"], json!({"enable_thinking": false}));
    assert_eq!(body["messages"][0]["role"], "system");
    assert!(body["messages"][1]["content"]
        .as_str()
        .unwrap()
        .contains("[0] (new paragraph) Hello there."));
}

#[test]
fn null_content_becomes_empty_and_http_errors_surface() {
    let ok = MockServer::start(|_| Resp::json(json!({"choices": [{"message": {"content": null}}]})));
    assert_eq!(ChatClient::new(&ok.url).chat_json(&[], &json!({})).unwrap(), "");

    let bad = MockServer::start(|_| Resp::status(500, "boom"));
    assert!(ChatClient::new(&bad.url).chat_json(&[], &json!({})).is_err());
}

#[test]
fn director_survives_a_dead_llm_and_uses_a_live_one() {
    let sentences = split_sentences("It was dark. Something moved.");

    let dead = ChatClient::new("http://127.0.0.1:1");
    let plan = plan_story(&sentences, Some(&dead), &PlanOptions::default());
    assert!(plan.segments.iter().all(|s| s.directive.mood == Mood::Neutral));

    let reply = json!({"narrator_voice": "Old man", "events": [{"i": 1, "mood": "dread"}]}).to_string();
    let live = MockServer::start(move |_| Resp::json(json!({"choices": [{"message": {"content": reply}}]})));
    let plan = plan_story(
        &sentences,
        Some(&ChatClient::new(&live.url)),
        &PlanOptions::default(),
    );
    assert_eq!(plan.narrator_voice, "Old man");
    assert_eq!(plan.segments[1].directive.mood, Mood::Dread);
}
