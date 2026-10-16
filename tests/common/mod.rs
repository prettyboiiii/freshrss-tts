//! A tiny blocking HTTP/1.1 server for tests: records requests, answers from a closure.
#![allow(dead_code)]

use freshrss_tts::engines::maya1::{self, CODE_END_ID};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

#[derive(Debug, Clone)]
pub struct Req {
    pub method: String,
    /// Path including the query string.
    pub target: String,
    pub headers: HashMap<String, String>,
    pub body: String,
}

impl Req {
    pub fn path(&self) -> &str {
        self.target.split('?').next().unwrap()
    }
    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap()
    }
}

pub struct Resp {
    pub status: u16,
    pub content_type: &'static str,
    pub body: String,
    /// No Content-Length: the body ends when the connection closes (like an SSE stream).
    pub streaming: bool,
}

impl Resp {
    pub fn text(body: &str) -> Self {
        Self {
            status: 200,
            content_type: "text/plain",
            body: body.into(),
            streaming: false,
        }
    }
    pub fn json(v: serde_json::Value) -> Self {
        Self {
            status: 200,
            content_type: "application/json",
            body: v.to_string(),
            streaming: false,
        }
    }
    pub fn status(status: u16, body: &str) -> Self {
        Self {
            status,
            ..Self::text(body)
        }
    }
    pub fn sse(events: &[serde_json::Value]) -> Self {
        let body: String = events.iter().map(|e| format!("data: {e}\n\n")).collect();
        Self {
            status: 200,
            content_type: "text/event-stream",
            body,
            streaming: true,
        }
    }
}

pub struct MockServer {
    pub url: String,
    pub requests: Arc<Mutex<Vec<Req>>>,
}

impl MockServer {
    pub fn start(handler: impl Fn(&Req) -> Resp + Send + Sync + 'static) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let requests = Arc::new(Mutex::new(Vec::new()));
        let handler = Arc::new(handler);
        let log = requests.clone();
        thread::spawn(move || {
            for stream in listener.incoming().flatten() {
                let (handler, log) = (handler.clone(), log.clone());
                thread::spawn(move || {
                    let _ = serve(stream, &*handler, &log);
                });
            }
        });
        Self { url, requests }
    }

    pub fn requests(&self) -> Vec<Req> {
        self.requests.lock().unwrap().clone()
    }
}

fn serve(stream: TcpStream, handler: &dyn Fn(&Req) -> Resp, log: &Mutex<Vec<Req>>) -> std::io::Result<()> {
    let mut reader = BufReader::new(stream.try_clone()?);
    let mut start = String::new();
    reader.read_line(&mut start)?;
    let mut parts = start.split_whitespace();
    let (method, target) = (
        parts.next().unwrap_or("").to_string(),
        parts.next().unwrap_or("").to_string(),
    );
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line)?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        if let Some((k, v)) = line.split_once(':') {
            headers.insert(k.trim().to_ascii_lowercase(), v.trim().to_string());
        }
    }
    let len: usize = headers
        .get("content-length")
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body)?;
    let req = Req {
        method,
        target,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    };
    log.lock().unwrap().push(req.clone());

    let resp = handler(&req);
    let mut out = stream;
    let mut head = format!(
        "HTTP/1.1 {} X\r\ncontent-type: {}\r\nconnection: close\r\n",
        resp.status, resp.content_type
    );
    if !resp.streaming {
        head.push_str(&format!("content-length: {}\r\n", resp.body.len()));
    }
    head.push_str("\r\n");
    out.write_all(head.as_bytes())?;
    out.write_all(resp.body.as_bytes())?;
    out.flush()
}

/// `frames` frames of fake speech tokens (1 L1 + 2 L2 + 4 L3 codes per frame).
pub fn speech_tokens(frames: u32) -> Vec<u32> {
    let levels = [
        (0..frames).collect::<Vec<u32>>(),
        (0..frames * 2).collect(),
        (0..frames * 4).collect(),
    ];
    maya1::pack_snac_to_7(&levels)
}

/// llama-server stand-in: /tokenize returns one id per whitespace word; /completion streams `speech`
/// then the end-of-speech token, then trailing junk a client must never read.
pub fn llama_mock(speech: Vec<u32>) -> MockServer {
    MockServer::start(move |req| match req.path() {
        "/tokenize" => {
            let n = req.json()["content"].as_str().unwrap().split_whitespace().count();
            Resp::json(json!({"tokens": (1..=n).collect::<Vec<_>>()}))
        }
        "/completion" => {
            let mut events: Vec<Value> = speech
                .chunks(5)
                .map(|c| json!({"content": "", "tokens": c, "stop": false}))
                .collect();
            events.push(json!({"content": "", "tokens": [CODE_END_ID, 999, 998], "stop": false}));
            events.push(json!({"content": "", "tokens": [997], "stop": true}));
            Resp::sse(&events)
        }
        _ => Resp::status(404, ""),
    })
}

/// An OpenAI-style chat endpoint that always answers with `reply` as the message content.
pub fn llm_mock(reply: Value) -> MockServer {
    let content = reply.to_string();
    MockServer::start(move |_| Resp::json(json!({"choices": [{"message": {"content": content}}]})))
}

pub const ITEM_ID: &str = "tag:google.com,2005:reader/item/00000000000000a1";

pub fn fake_freshrss(req: &Req) -> Resp {
    match req.path() {
        "/api/greader.php/accounts/ClientLogin" => Resp::text("SID=me/abc\nLSID=null\nAuth=me/abc\n"),
        "/api/greader.php/reader/api/0/token" => Resp::text("tok123\n"),
        "/api/greader.php/reader/api/0/edit-tag" => Resp::text("OK"),
        p if p.starts_with("/api/greader.php/reader/api/0/stream/contents/") => {
            Resp::json(json!({"items": [{
                "id": ITEM_ID,
                "title": "The Lighthouse Keeper",
                "published": 1760000000,
                "canonical": [{"href": "https://example.org/story"}],
                "summary": {"content": "<p>Hello.</p>"},
                "origin": {"title": "NoSleep"},
                "categories": ["user/-/state/com.google/reading-list", "user/-/label/narrate-scary"],
            }]}))
        }
        _ => Resp::status(404, "no"),
    }
}
