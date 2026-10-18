mod common;

use anyhow::Result;
use common::{llama_mock, speech_tokens, MockServer, Resp};
use freshrss_tts::director::{Completer, Message, PlanOptions};
use freshrss_tts::engines::maya1::{self, CODE_TOKEN_OFFSET};
use freshrss_tts::engines::maya1_llamacpp::{Decoder, Maya1LlamaCpp};
use freshrss_tts::engines::{Chunk, Engine};
use freshrss_tts::pipeline;
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};

/// Records the code levels it is given and returns 2048 samples per frame of a fixed byte pattern.
struct FakeDecoder(Arc<Mutex<Vec<[Vec<u32>; 3]>>>);
impl Decoder for FakeDecoder {
    fn decode(&mut self, levels: &[Vec<u32>; 3]) -> Result<Vec<u8>> {
        self.0.lock().unwrap().push(levels.clone());
        Ok([1u8, 0].repeat(levels[0].len() * maya1::SAMPLES_PER_FRAME))
    }
}

#[test]
fn tokenize_posts_special_token_flags() {
    let server = llama_mock(vec![]);
    let engine = Maya1LlamaCpp::new(&server.url, Box::new(FakeDecoder(Default::default())));
    assert_eq!(engine.tokenize("one two three").unwrap(), [1, 2, 3]);
    let body = server.requests()[0].json();
    assert_eq!(body["content"], "one two three");
    assert_eq!(
        (&body["add_special"], &body["parse_special"]),
        (&json!(false), &json!(true))
    );
}

#[test]
fn generation_streams_tokens_and_stops_at_end_of_speech() {
    let speech = speech_tokens(3);
    let server = llama_mock(speech.clone());
    let engine = Maya1LlamaCpp::new(&server.url, Box::new(FakeDecoder(Default::default())));
    let (tokens, stats) = engine.generate_tokens(&[7, 8, 9]).unwrap();
    assert_eq!(
        tokens, speech,
        "tokens after the end-of-speech token must be dropped"
    );
    assert_eq!(
        (stats.prompt_tokens, stats.gen_tokens, stats.truncated),
        (3, 21, false)
    );

    let body = server.requests()[0].json();
    assert_eq!(body["prompt"], json!([7, 8, 9]));
    assert_eq!(
        (&body["stream"], &body["return_tokens"], &body["n_predict"]),
        (&json!(true), &json!(true), &json!(2048))
    );
    assert_eq!(body["temperature"], 0.4);
    assert!(body.get("seed").is_none());
}

#[test]
fn truncation_is_reported_when_the_budget_runs_out() {
    let server = MockServer::start(|_| {
        Resp::sse(&[
            json!({"tokens": [CODE_TOKEN_OFFSET, CODE_TOKEN_OFFSET + 1], "stop": false}),
            json!({"tokens": [], "stop": true}),
        ])
    });
    let mut engine = Maya1LlamaCpp::new(&server.url, Box::new(FakeDecoder(Default::default())));
    engine.n_predict = 2;
    engine.sampling.seed = Some(42);
    let (tokens, stats) = engine.generate_tokens(&[1]).unwrap();
    assert_eq!(tokens.len(), 2);
    assert!(stats.truncated);
    assert_eq!(server.requests()[0].json()["seed"], 42);
}

#[test]
fn synth_builds_the_prompt_unpacks_frames_and_decodes() {
    let seen = Arc::new(Mutex::new(vec![]));
    let server = llama_mock(speech_tokens(3));
    let mut engine = Maya1LlamaCpp::new(&server.url, Box::new(FakeDecoder(seen.clone())));
    let chunk = Chunk {
        text: "<whisper> Don't look.".into(),
        style: "Old man, low".into(),
        pause_ms: 250,
        parts: Vec::new(),
    };
    let pcm = engine.synth(&chunk).unwrap();

    assert_eq!(pcm.len(), 3 * maya1::SAMPLES_PER_FRAME * 2);
    let levels = seen.lock().unwrap()[0].clone();
    assert_eq!((levels[0].len(), levels[1].len(), levels[2].len()), (3, 6, 12));
    assert_eq!(levels[0], [0, 1, 2]);
    assert_eq!(engine.last_stats.gen_tokens, 21);

    let reqs = server.requests();
    assert_eq!(reqs[0].path(), "/tokenize");
    assert_eq!(
        reqs[0].json()["content"],
        "<description=\"Old man, low\"> <whisper> Don't look."
    );
    let prompt: Vec<u64> = reqs[1].json()["prompt"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    // words in `content` -> ids 1..=n, wrapped by the Maya1 header/footer tokens
    assert_eq!(&prompt[..2], [128259, 128000]);
    assert_eq!(&prompt[prompt.len() - 4..], [128009, 128260, 128261, 128257]);
}

struct Scripted;
impl Completer for Scripted {
    fn chat_json(&self, _m: &[Message], _s: &Value) -> Result<String> {
        Ok(json!({"narrator_voice": "Old man, rasping", "events": [
            {"i": 0, "mood": "dread", "pause_ms": 1000},
            {"i": 2, "delivery": "whisper", "beat": "gasp"},
        ]})
        .to_string())
    }
}

#[test]
fn end_to_end_story_to_wav_with_mock_servers() {
    let server = llama_mock(speech_tokens(2));
    let seen = Arc::new(Mutex::new(vec![]));
    let mut engine = Maya1LlamaCpp::new(&server.url, Box::new(FakeDecoder(seen)));
    let dir = tempfile::tempdir().unwrap();
    let out = dir.path().join("story.wav");

    let html = "<p>The house had been empty for years.</p>\
                <p>\"Don't go in,\" she whispered. I went in anyway.</p><p>Something moved upstairs.</p>";
    let sentences = pipeline::prepare(html);
    let res = pipeline::narrate(
        &sentences,
        &mut engine,
        Some(&Scripted),
        &PlanOptions::default(),
        &out,
    )
    .unwrap();

    let texts: Vec<_> = res.chunks.iter().map(|c| c.text.as_str()).collect();
    assert!(
        texts
            .iter()
            .any(|t| t.starts_with("<whisper> <gasp> I went in anyway.")),
        "{texts:?}"
    );
    assert!(res.chunks[0].style.contains("Old man, rasping") && res.chunks[0].style.contains("ominous"));

    let wav = std::fs::read(&out).unwrap();
    assert_eq!(&wav[..4], b"RIFF");
    let data_len = u32::from_le_bytes(wav[40..44].try_into().unwrap()) as usize;
    assert_eq!(wav.len(), 44 + data_len);
    assert!((res.audio_seconds - data_len as f64 / 2.0 / 24000.0).abs() < 1e-9);
    assert!(res.audio_seconds > 1.0, "the 1000 ms pause is in the file");
}
