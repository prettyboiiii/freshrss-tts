//! Runs the compiled binary against mock servers: the argument parsing and glue the user hits first.

mod common;

use common::{fake_freshrss, llama_mock, llm_mock, speech_tokens, MockServer};
use serde_json::{json, Value};
use std::path::Path;
use std::process::{Command, Output};

const STORY: &str = "<p>The house had been empty for years.</p>\
    <p>\"Don't go in,\" she whispered. I went in anyway.</p><p>Something moved upstairs.</p>";

fn run(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_freshrss-tts"))
        .args(args)
        .env_remove("DIRECTOR_URL")
        .env_remove("MAYA1_URL")
        .env_remove("SNAC_DIR")
        .output()
        .expect("spawn binary")
}

fn out(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}
fn err(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn story_file(dir: &Path) -> String {
    let p = dir.join("story.html");
    std::fs::write(&p, STORY).unwrap();
    p.to_str().unwrap().to_string()
}

fn director_reply() -> Value {
    json!({"narrator_voice": "Old man, rasping", "events": [
        {"i": 0, "mood": "dread", "pause_ms": 1000},
        {"i": 2, "delivery": "whisper", "beat": "gasp"},
    ]})
}

#[test]
fn help_lists_the_commands() {
    let o = run(&["--help"]);
    assert!(o.status.success());
    for cmd in ["check", "plan", "tags", "narrate", "bench"] {
        assert!(out(&o).contains(cmd), "{cmd} missing from help");
    }
}

#[test]
fn plan_without_an_llm_prints_plain_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let o = run(&["plan", &story_file(dir.path())]);
    assert!(o.status.success(), "{}", err(&o));
    // one generation per paragraph: the 700 ms paragraph pause ends a chunk so the silence is real
    assert!(out(&o).contains("4 sentences -> 3 generations"), "{}", out(&o));
    assert!(
        out(&o).contains("\"Don't go in,\" she whispered. I went in anyway.\n    (+700 ms)"),
        "{}",
        out(&o)
    );
    assert!(err(&o).contains("neutral narration"));
}

#[test]
fn plan_with_an_llm_inserts_tags_and_sends_the_hint() {
    let dir = tempfile::tempdir().unwrap();
    let llm = llm_mock(director_reply());
    let o = run(&[
        "plan",
        &story_file(dir.path()),
        "--llm-url",
        &llm.url,
        "--hint",
        "hospital setting",
    ]);
    assert!(o.status.success(), "{}", err(&o));
    let text = out(&o);
    assert!(text.contains("narrator: Old man, rasping"), "{text}");
    assert!(text.contains("<whisper> <gasp> I went in anyway."), "{text}");
    assert!(text.contains("ominous"), "{text}");
    let body = llm.requests()[0].json();
    assert!(body["messages"][1]["content"]
        .as_str()
        .unwrap()
        .contains("hospital setting"));
}

#[test]
fn plan_json_and_director_url_from_the_environment() {
    let dir = tempfile::tempdir().unwrap();
    let llm = llm_mock(director_reply());
    let o = Command::new(env!("CARGO_BIN_EXE_freshrss-tts"))
        .args(["plan", &story_file(dir.path()), "--json"])
        .env("DIRECTOR_URL", &llm.url)
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", err(&o));
    let v: Value = serde_json::from_str(&out(&o)).unwrap();
    assert_eq!(v["segments"][0]["mood"], "dread");
    assert_eq!(v["segments"][2]["beat"], "gasp");
    assert_eq!(v["segments"][2]["delivery"], "whisper");
    assert_eq!(v["segments"][3]["text"], "Something moved upstairs.");
}

#[test]
fn plan_reads_stdin() {
    use std::io::Write;
    let mut child = Command::new(env!("CARGO_BIN_EXE_freshrss-tts"))
        .args(["plan", "-"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(STORY.as_bytes()).unwrap();
    let o = child.wait_with_output().unwrap();
    assert!(o.status.success() && out(&o).contains("4 sentences"));
}

#[test]
fn check_logs_in_and_lists_labelled_entries() {
    let fr = MockServer::start(fake_freshrss);
    let o = run(&[
        "check",
        "--url",
        &fr.url,
        "--user",
        "me",
        "--api-password",
        "pw",
        "--label",
        "narrate-scary",
    ]);
    assert!(o.status.success(), "{}", err(&o));
    let text = out(&o);
    assert!(
        text.contains("login ok") && text.contains("label \"narrate-scary\": 1 entries"),
        "{text}"
    );
    assert!(text.contains("The Lighthouse Keeper") && text.contains("https://example.org/story"));
}

#[test]
fn check_fails_clearly_on_a_bad_password() {
    let fr = MockServer::start(|_| common::Resp::status(401, "no"));
    let o = run(&["check", "--url", &fr.url, "--user", "me", "--api-password", "bad"]);
    assert!(!o.status.success());
    assert!(err(&o).contains("401"), "{}", err(&o));
}

#[test]
fn tags_reports_single_token_tags_and_probes() {
    let server = MockServer::start(|req| {
        let tag = req.json()["content"].as_str().unwrap().to_string();
        let n = if tag == "<scream>" || !tag.starts_with('<') {
            3
        } else {
            1
        };
        common::Resp::json(json!({"tokens": (0..n).map(|i| 128300 + i).collect::<Vec<_>>()}))
    });
    let o = run(&["tags", "--server", &server.url]);
    assert!(o.status.success(), "{}", err(&o));
    let text = out(&o);
    assert!(
        text.contains("used  <whisper>") && text.contains("single token"),
        "{text}"
    );
    assert!(
        text.contains("probe <scream>") && text.contains("NOT a single token"),
        "{text}"
    );
}

#[test]
fn tags_fails_when_an_emitted_tag_is_not_a_single_token() {
    let server = MockServer::start(|_| common::Resp::json(json!({"tokens": [1, 2, 3]})));
    let o = run(&["tags", "--server", &server.url]);
    assert!(!o.status.success());
    assert!(err(&o).contains("not single tokens"), "{}", err(&o));
}

#[test]
fn narrate_writes_a_wav_end_to_end() {
    let dir = tempfile::tempdir().unwrap();
    let snac = dir.path().join("snac");
    std::fs::create_dir(&snac).unwrap();
    freshrss_tts::testing::write_zero_weight_snac(&snac).unwrap();
    let tts = llama_mock(speech_tokens(2));
    let llm = llm_mock(director_reply());
    let wav = dir.path().join("o/story.wav");

    let o = run(&[
        "narrate",
        &story_file(dir.path()),
        "--server",
        &tts.url,
        "--snac-dir",
        snac.to_str().unwrap(),
        "--llm-url",
        &llm.url,
        "--seed",
        "7",
        "--out",
        wav.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", err(&o));
    assert!(out(&o).contains("s of audio in"), "{}", out(&o));
    let bytes = std::fs::read(&wav).unwrap();
    assert_eq!(&bytes[..4], b"RIFF");
    assert!(
        bytes.len() > 44 + 2 * 24000,
        "1 s pause alone is 48000 bytes, got {}",
        bytes.len()
    );
    let completions: Vec<_> = tts
        .requests()
        .into_iter()
        .filter(|r| r.path() == "/completion")
        .collect();
    assert!(completions.len() >= 2);
    assert_eq!(completions[0].json()["seed"], 7);
}

#[test]
fn narrate_plain_never_calls_the_llm() {
    let dir = tempfile::tempdir().unwrap();
    let snac = dir.path().join("snac");
    std::fs::create_dir(&snac).unwrap();
    freshrss_tts::testing::write_zero_weight_snac(&snac).unwrap();
    let tts = llama_mock(speech_tokens(1));
    let llm = llm_mock(director_reply());
    let wav = dir.path().join("plain.wav");
    let o = run(&[
        "narrate",
        &story_file(dir.path()),
        "--server",
        &tts.url,
        "--snac-dir",
        snac.to_str().unwrap(),
        "--llm-url",
        &llm.url,
        "--plain",
        "--out",
        wav.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", err(&o));
    assert!(llm.requests().is_empty());
}

#[test]
fn missing_snac_directory_is_a_clear_error() {
    let dir = tempfile::tempdir().unwrap();
    let tts = llama_mock(vec![]);
    let o = run(&[
        "narrate",
        &story_file(dir.path()),
        "--server",
        &tts.url,
        "--snac-dir",
        "/nonexistent/snac",
    ]);
    assert!(!o.status.success());
    assert!(err(&o).contains("loading SNAC"), "{}", err(&o));
}

#[test]
fn bench_writes_wavs_and_metrics() {
    let dir = tempfile::tempdir().unwrap();
    let snac = dir.path().join("snac");
    std::fs::create_dir(&snac).unwrap();
    freshrss_tts::testing::write_zero_weight_snac(&snac).unwrap();
    let tts = llama_mock(speech_tokens(1));
    let llm = llm_mock(director_reply());
    let outdir = dir.path().join("bench");

    let o = run(&[
        "bench",
        "--server",
        &tts.url,
        "--snac-dir",
        snac.to_str().unwrap(),
        "--llm-url",
        &llm.url,
        "--out",
        outdir.to_str().unwrap(),
    ]);
    assert!(o.status.success(), "{}", err(&o));
    let text = out(&o);
    for name in [
        "01_calm_plain",
        "04_dread_directed",
        "06_panic_directed",
        "07_director",
    ] {
        assert!(text.contains(name), "{name} missing:\n{text}");
        assert!(outdir.join(format!("{name}.wav")).exists(), "{name}.wav");
    }
    let metrics: Value =
        serde_json::from_str(&std::fs::read_to_string(outdir.join("metrics.json")).unwrap()).unwrap();
    assert_eq!(metrics["cases"].as_array().unwrap().len(), 7);
    assert!(text.contains("RTF"));
}
