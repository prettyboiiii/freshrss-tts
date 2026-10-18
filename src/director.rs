//! The "director": an LLM decides *how* a story is performed; code applies it.
//!
//! The LLM never gets a chance to rewrite the story. It receives numbered sentences and returns a
//! sparse list of events ("from sentence 12 on the mood is dread", "sentence 15 starts with a gasp").
//! We merge those events with defaults and attach the result to the *unchanged* sentences. Anything
//! the model gets wrong (bad JSON, unknown words, out-of-range indices) degrades to neutral narration
//! instead of failing the job.

use anyhow::Result;
use regex::Regex;
use serde::Serialize;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::sync::LazyLock;

use crate::textprep::{Sentence, Span};

macro_rules! vocab {
    ($(#[$m:meta])* $name:ident { $($var:ident = $s:literal),+ $(,)? }) => {
        $(#[$m])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub enum $name { $($var),+ }
        impl $name {
            pub const ALL: &'static [$name] = &[$($name::$var),+];
            pub fn as_str(self) -> &'static str { match self { $($name::$var => $s),+ } }
            pub fn parse(s: &str) -> Option<Self> { match s { $($s => Some($name::$var),)+ _ => None } }
        }
    };
}

// Closed vocabularies: engine adapters translate these to their own tags/descriptions.
vocab!(Mood {
    Neutral = "neutral", Calm = "calm", Uneasy = "uneasy", Dread = "dread", Tense = "tense",
    Fearful = "fearful", Panicked = "panicked", Sorrowful = "sorrowful", Angry = "angry",
    Relieved = "relieved",
});
vocab!(Delivery { Normal = "normal", Hushed = "hushed", Whisper = "whisper", Raised = "raised" });
vocab!(
    /// Short non-verbal sounds before a sentence. The schema also allows "none".
    Beat {
        Gasp = "gasp", Sigh = "sigh", Cry = "cry", Laugh = "laugh", Chuckle = "chuckle",
        Giggle = "giggle",
    }
);

pub const DEFAULT_NARRATOR: &str = "Adult male narrator, deep warm voice, measured storytelling pace";
pub const MAX_PAUSE_MS: u32 = 1500;
const MAX_SPEAKER_CHARS: usize = 200;
/// A pause this long counts as a dramatic one; two of them need at least `LONG_PAUSE_GAP` sentences between.
const LONG_PAUSE_MS: u32 = 1100;
const LONG_PAUSE_GAP: usize = 5;

/// How one sentence is performed. `mood` and `delivery` persist; `beat` and `pause_ms` do not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Directive {
    pub mood: Mood,
    pub delivery: Delivery,
    pub beat: Option<Beat>,
    pub pause_ms: u32,
}

impl Default for Directive {
    fn default() -> Self {
        Self {
            mood: Mood::Neutral,
            delivery: Delivery::Normal,
            beat: None,
            pause_ms: 250,
        }
    }
}

/// An unchanged sentence plus its directive.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Segment {
    pub idx: usize,
    pub text: String,
    pub para_start: bool,
    pub directive: Directive,
    /// Quoted and unquoted runs of `text`, in order.
    pub spans: Vec<Span>,
    /// Voice of the character whose quoted words are in `spans`; None = the narrator speaks them.
    pub speaker: Option<String>,
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub narrator_voice: String,
    pub segments: Vec<Segment>,
    /// Events the LLM got wrong and we ignored or clamped.
    pub repaired: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct Message {
    pub role: &'static str,
    pub content: String,
}

pub trait Completer {
    /// Returns the model's reply, constrained to `schema` where the server supports it.
    fn chat_json(&self, messages: &[Message], schema: &Value) -> Result<String>;
}

/// JSON schema handed to llama.cpp for grammar-constrained output (kept deliberately plain).
pub fn director_schema() -> Value {
    let moods: Vec<_> = Mood::ALL.iter().map(|m| m.as_str()).collect();
    let deliveries: Vec<_> = Delivery::ALL.iter().map(|d| d.as_str()).collect();
    let mut beats = vec!["none"];
    beats.extend(Beat::ALL.iter().map(|b| b.as_str()));
    json!({
        "type": "object",
        "properties": {
            "narrator_voice": {"type": "string"},
            "events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "i": {"type": "integer"},
                        "mood": {"type": "string", "enum": moods},
                        "delivery": {"type": "string", "enum": deliveries},
                        "beat": {"type": "string", "enum": beats},
                        "pause_ms": {"type": "integer"},
                        "speaker": {"type": "string"},
                    },
                    "required": ["i"],
                },
            },
        },
        "required": ["narrator_voice", "events"],
    })
}

fn join_vocab<T: Copy>(all: &[T], f: impl Fn(T) -> &'static str) -> String {
    all.iter().map(|&v| f(v)).collect::<Vec<_>>().join(", ")
}

pub fn system_prompt() -> String {
    format!(
        "You are the director of an audiobook narration. You are given numbered sentences from a \
story. You do NOT rewrite or repeat the text. You only answer with JSON saying how the narrator \
should perform it.\n\n\
Output: {{\"narrator_voice\": \"...\", \"events\": [{{\"i\": <sentence number>, ...}}, ...]}}\n\n\
- narrator_voice: ONE short description of the narrator's voice for the whole story (age, gender, \
pitch, pace, timbre). Keep it generic and calm unless the listener's direction says otherwise.\n\
- events: only for sentences where something changes. Most sentences need no event.\n\
  - mood: {moods}. Persists until you change it. Build tension gradually; return to neutral or calm \
after a scare.\n\
  - delivery: {deliveries}. Persists until you change it. Use \"whisper\" sparingly (secrets, creeping \
dread, a character whispering). \"raised\" is for shouting.\n\
  - beat: none, {beats}. A short non-verbal sound spoken BEFORE that one sentence. Use rarely, only \
when the text implies the sound.\n\
  - pause_ms: silence after that one sentence, 0-{max_pause}, in steps of 100. The default is already \
right for most sentences; set it only for a few key moments (1200-{max_pause} before a reveal or after \
a scare), never as a steady ramp.\n\
  - speaker: for a sentence with words in quotation marks spoken by a character OTHER than the \
narrator: who that character is, as a voice description of identity only (gender, approximate age, \
pitch), e.g. \"boy, about 12, higher pitch\". Never put emotion in it, that is what mood is for. Copy \
exactly the same words every time that character speaks, so the voice stays the same. Omit speaker when \
the narrator (the \"I\" of the story) is the one speaking, and for sentences without quotes.\n\
Sentence numbers must come from the list. Never invent text.",
        moods = join_vocab(Mood::ALL, Mood::as_str),
        deliveries = join_vocab(Delivery::ALL, Delivery::as_str),
        beats = join_vocab(Beat::ALL, Beat::as_str),
        max_pause = MAX_PAUSE_MS,
    )
}

pub fn build_messages(
    sentences: &[Sentence],
    hint: Option<&str>,
    state: Option<Directive>,
    narrator_voice: Option<&str>,
) -> Vec<Message> {
    let mut lines = Vec::new();
    if let Some(h) = hint.filter(|h| !h.trim().is_empty()) {
        lines.push(format!("Listener's direction (follow it): {}", h.trim()));
    }
    if let Some(v) = narrator_voice {
        lines.push(format!("Narrator voice already chosen (keep it): {v}"));
    }
    if let Some(s) = state {
        lines.push(format!(
            "State carried over: mood={}, delivery={}",
            s.mood.as_str(),
            s.delivery.as_str()
        ));
    }
    lines.push("Sentences:".to_string());
    for s in sentences {
        let mark = if s.para_start { " (new paragraph)" } else { "" };
        lines.push(format!("[{}]{} {}", s.idx, mark, s.text));
    }
    vec![
        Message {
            role: "system",
            content: system_prompt(),
        },
        Message {
            role: "user",
            content: lines.join("\n"),
        },
    ]
}

/// Split into consecutive windows so long stories fit the model's context.
pub fn windows(sentences: &[Sentence], max_chars: usize, max_sentences: usize) -> Vec<&[Sentence]> {
    let mut out = Vec::new();
    let (mut start, mut size) = (0, 0);
    for (n, s) in sentences.iter().enumerate() {
        let len = s.text.chars().count();
        if n > start && (size + len > max_chars || n - start >= max_sentences) {
            out.push(&sentences[start..n]);
            start = n;
            size = 0;
        }
        size += len;
    }
    if start < sentences.len() {
        out.push(&sentences[start..]);
    }
    out
}

fn loads(raw: &str) -> Option<Map<String, Value>> {
    let mut text = raw.trim();
    if let Some(rest) = text.strip_prefix("```") {
        text = rest.strip_prefix("json").unwrap_or(rest).trim_start();
    }
    text = text.strip_suffix("```").unwrap_or(text).trim_end();
    let parse = |s: &str| {
        serde_json::from_str::<Value>(s)
            .ok()
            .and_then(|v| v.as_object().cloned())
    };
    parse(text).or_else(|| {
        let (a, b) = (text.find('{')?, text.rfind('}')?);
        (a < b).then(|| parse(&text[a..=b])).flatten()
    })
}

const SPEECH_VERBS: &str =
    "told|said|asked|replied|answered|whispered|shouted|yelled|screamed|muttered|added|called|begged";

/// True when the narrator ("I") is the one speaking the first quote of a sentence, as in
/// `I told him, "..."` or `"...," I said.` The small director model often hands those to a character.
fn narrator_speaks(spans: &[Span]) -> bool {
    static BEFORE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(r"\bI\s+(?:\w+\s+){{0,2}}(?i:{SPEECH_VERBS})\b[^.!?]*$")).unwrap()
    });
    static AFTER: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(&format!(r"^[\s,.]*I\s+(?:\w+\s+){{0,2}}(?i:{SPEECH_VERBS})\b")).unwrap()
    });
    let Some(q) = spans.iter().position(|p| p.quoted) else {
        return false;
    };
    let before = q
        .checked_sub(1)
        .and_then(|i| spans.get(i))
        .is_some_and(|p| BEFORE.is_match(&p.text));
    let after = spans
        .get(q + 1)
        .is_some_and(|p| !p.quoted && AFTER.is_match(&p.text));
    before || after
}

pub fn default_pause_ms(text: &str, last_in_paragraph: bool) -> u32 {
    if last_in_paragraph {
        900
    } else if text.ends_with('\u{2026}') || text.ends_with("...") {
        500
    } else if text
        .trim_end_matches(['"', '\'', '\u{201d}', '\u{2019}', ')'])
        .ends_with(['?', '!'])
    {
        350
    } else {
        250
    }
}

pub struct Resolved {
    pub segments: Vec<Segment>,
    pub narrator_voice: Option<String>,
    pub state: Directive,
    pub repaired: usize,
}

/// Merge the LLM's events with defaults. `sentences` is one window (indices may start above 0);
/// events outside it are ignored.
pub fn resolve(sentences: &[Sentence], raw: Option<&str>, state: Directive) -> Resolved {
    let mut repaired = 0;
    let mut voice = None;
    let mut events: HashMap<usize, Map<String, Value>> = HashMap::new();
    match raw.filter(|r| !r.trim().is_empty()).map(loads) {
        Some(None) => repaired += 1,
        Some(Some(data)) => {
            if let Some(v) = data.get("narrator_voice").and_then(Value::as_str) {
                let v = v.trim();
                if !v.is_empty() {
                    voice = Some(v.chars().take(300).collect::<String>());
                }
            }
            let valid: std::collections::HashSet<usize> = sentences.iter().map(|s| s.idx).collect();
            for ev in data.get("events").and_then(Value::as_array).into_iter().flatten() {
                let idx = ev
                    .as_object()
                    .and_then(|o| o.get("i"))
                    .and_then(Value::as_u64)
                    .map(|i| i as usize)
                    .filter(|i| valid.contains(i));
                match idx {
                    Some(i) => events
                        .entry(i)
                        .or_default()
                        .extend(ev.as_object().unwrap().clone()),
                    None => repaired += 1,
                }
            }
        }
        None => {}
    }

    let (mut mood, mut delivery) = (state.mood, state.delivery);
    let mut speaker: Option<String> = None;
    let mut since_long = LONG_PAUSE_GAP;
    let mut segments = Vec::with_capacity(sentences.len());
    for (pos, s) in sentences.iter().enumerate() {
        let empty = Map::new();
        let ev = events.get(&s.idx).unwrap_or(&empty);
        match ev.get("mood").map(|v| v.as_str().and_then(Mood::parse)) {
            Some(Some(m)) => mood = m,
            Some(None) => repaired += 1,
            None => {}
        }
        match ev.get("delivery").map(|v| v.as_str().and_then(Delivery::parse)) {
            Some(Some(d)) => delivery = d,
            Some(None) => repaired += 1,
            None => {}
        }
        let beat = match ev.get("beat") {
            None => None,
            Some(v) => match v.as_str() {
                Some("none") => None,
                Some(b) => Beat::parse(b).or_else(|| {
                    repaired += 1;
                    None
                }),
                None => {
                    repaired += 1;
                    None
                }
            },
        };
        let last_in_par = pos + 1 == sentences.len() || sentences[pos + 1].para_start;
        let mut pause = default_pause_ms(&s.text, last_in_par);
        if let Some(p) = ev.get("pause_ms") {
            match p.as_f64().filter(|_| !p.is_boolean()) {
                Some(f) => {
                    let wanted = f as i64;
                    let clamped = wanted.clamp(0, MAX_PAUSE_MS as i64);
                    repaired += (clamped != wanted) as usize;
                    pause = clamped as u32;
                }
                None => repaired += 1,
            }
        }
        // Small models ramp pauses up toward a climax; keep long ones rare so the story keeps moving.
        if pause >= LONG_PAUSE_MS {
            if since_long < LONG_PAUSE_GAP {
                pause = default_pause_ms(&s.text, last_in_par);
                repaired += 1;
                since_long += 1;
            } else {
                since_long = 0;
            }
        } else {
            since_long += 1;
        }
        // A character's voice is only meaningful for sentences with quoted words; a quote that runs
        // over several sentences keeps its speaker.
        let has_quote = s.spans.iter().any(|p| p.quoted);
        speaker = match ev.get("speaker").and_then(Value::as_str).map(str::trim) {
            _ if narrator_speaks(&s.spans) => None,
            Some(v) if has_quote && !v.is_empty() => Some(v.chars().take(MAX_SPEAKER_CHARS).collect()),
            _ if has_quote && s.continues_quote => speaker.take(),
            _ => None,
        };
        segments.push(Segment {
            idx: s.idx,
            text: s.text.clone(),
            para_start: s.para_start,
            directive: Directive {
                mood,
                delivery,
                beat,
                pause_ms: pause,
            },
            spans: s.spans.clone(),
            speaker: speaker.clone(),
        });
    }
    Resolved {
        segments,
        narrator_voice: voice,
        state: Directive {
            mood,
            delivery,
            ..Directive::default()
        },
        repaired,
    }
}

#[derive(Debug, Clone)]
pub struct PlanOptions {
    pub hint: Option<String>,
    /// Fixes the narrator voice instead of letting the LLM pick one.
    pub narrator_voice: Option<String>,
    pub window_chars: usize,
    pub window_sentences: usize,
}

impl Default for PlanOptions {
    fn default() -> Self {
        Self {
            hint: None,
            narrator_voice: None,
            window_chars: 5000,
            window_sentences: 120,
        }
    }
}

/// Direct a whole story window by window. `llm = None` gives plain neutral narration.
pub fn plan_story(sentences: &[Sentence], llm: Option<&dyn Completer>, opts: &PlanOptions) -> Plan {
    let voice_locked = opts.narrator_voice.is_some();
    let mut plan = Plan {
        narrator_voice: opts
            .narrator_voice
            .clone()
            .unwrap_or_else(|| DEFAULT_NARRATOR.to_string()),
        segments: Vec::new(),
        repaired: 0,
    };
    let mut state = Directive::default();
    for window in windows(sentences, opts.window_chars, opts.window_sentences) {
        let first_window = plan.segments.is_empty();
        let mut raw = None;
        if let Some(llm) = llm {
            let msgs = build_messages(
                window,
                opts.hint.as_deref(),
                (!first_window).then_some(state),
                (!first_window || voice_locked).then_some(plan.narrator_voice.as_str()),
            );
            for attempt in 1..=2 {
                // network, 5xx, timeout: retry once, then go neutral
                match llm.chat_json(&msgs, &director_schema()) {
                    Ok(r) => {
                        raw = Some(r);
                        break;
                    }
                    Err(e) => eprintln!("warning: director call failed (attempt {attempt}): {e:#}"),
                }
            }
        }
        let r = resolve(window, raw.as_deref(), state);
        state = r.state;
        plan.repaired += r.repaired;
        plan.segments.extend(r.segments);
        if first_window && !voice_locked {
            if let Some(v) = r.narrator_voice {
                plan.narrator_voice = v; // the first window picks the voice, later ones keep it
            }
        }
    }
    plan
}

/// Strip all direction but keep pauses (used to A/B the director against plain narration).
pub fn neutral(segments: &[Segment]) -> Vec<Segment> {
    segments
        .iter()
        .map(|s| Segment {
            directive: Directive {
                pause_ms: s.directive.pause_ms,
                ..Directive::default()
            },
            ..s.clone()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::textprep::split_sentences;
    use std::cell::RefCell;
    use std::collections::VecDeque;

    const STORY: &str = "The house had been empty for years. Nobody in town talked about it.\n\n\
        \"Don't go in,\" she whispered. I went in anyway. The stairs groaned.\n\n\
        Something moved upstairs. I ran.";

    fn sents() -> Vec<Sentence> {
        split_sentences(STORY)
    }

    struct Stub {
        replies: RefCell<VecDeque<Result<String, String>>>,
        calls: RefCell<Vec<Vec<Message>>>,
    }
    impl Stub {
        fn new(replies: Vec<Result<String, String>>) -> Self {
            Self {
                replies: RefCell::new(replies.into()),
                calls: RefCell::new(vec![]),
            }
        }
        fn ok(v: Value) -> Result<String, String> {
            Ok(v.to_string())
        }
    }
    impl Completer for Stub {
        fn chat_json(&self, messages: &[Message], _schema: &Value) -> Result<String> {
            self.calls.borrow_mut().push(messages.to_vec());
            self.replies
                .borrow_mut()
                .pop_front()
                .unwrap()
                .map_err(|e| anyhow::anyhow!(e))
        }
    }

    #[test]
    fn text_is_never_altered_even_with_hostile_output() {
        let s = sents();
        let llm = Stub::new(vec![Stub::ok(json!({
            "narrator_voice": "Gravel-voiced old man",
            "events": [
                {"i": 2, "mood": "dread", "text": "REWRITTEN", "replace": "evil"},
                {"i": 99, "mood": "panicked"},
                {"i": "x", "mood": "panicked"},
                {"i": 4, "mood": "not-a-mood", "delivery": "yodel", "beat": "explode"},
            ],
        }))]);
        let plan = plan_story(&s, Some(&llm), &PlanOptions::default());
        let got: Vec<_> = plan.segments.iter().map(|x| x.text.clone()).collect();
        let want: Vec<_> = s.iter().map(|x| x.text.clone()).collect();
        assert_eq!(got, want);
        assert_eq!(plan.narrator_voice, "Gravel-voiced old man");
        assert!(plan.repaired >= 5, "repaired = {}", plan.repaired);
    }

    #[test]
    fn mood_and_delivery_persist_but_beat_and_pause_do_not() {
        let plan = plan_story(
            &sents(),
            Some(&Stub::new(vec![Stub::ok(json!({
                "narrator_voice": "v",
                "events": [
                    {"i": 2, "mood": "dread", "delivery": "whisper", "beat": "gasp", "pause_ms": 1500},
                    {"i": 5, "mood": "neutral", "delivery": "normal"},
                ],
            }))])),
            &PlanOptions::default(),
        );
        let d: Vec<_> = plan.segments.iter().map(|s| s.directive).collect();
        assert_eq!((d[0].mood, d[1].mood), (Mood::Neutral, Mood::Neutral));
        assert!(d[2..5]
            .iter()
            .all(|x| x.mood == Mood::Dread && x.delivery == Delivery::Whisper));
        assert_eq!((d[2].beat, d[3].beat), (Some(Beat::Gasp), None));
        assert_eq!(d[2].pause_ms, 1500);
        assert_ne!(d[3].pause_ms, 1500);
        assert_eq!((d[5].mood, d[5].delivery), (Mood::Neutral, Delivery::Normal));
    }

    #[test]
    fn pause_is_clamped_and_defaults_follow_punctuation_and_paragraphs() {
        let plan = plan_story(
            &sents(),
            Some(&Stub::new(vec![Stub::ok(
                json!({"narrator_voice": "v", "events": [{"i": 0, "pause_ms": 999999}]}),
            )])),
            &PlanOptions::default(),
        );
        assert_eq!(plan.segments[0].directive.pause_ms, MAX_PAUSE_MS);
        assert_eq!(plan.segments[1].directive.pause_ms, 900); // last sentence of its paragraph
        assert_eq!(default_pause_ms("What?", false), 350);
        assert_eq!(default_pause_ms("He said \"run!\"", false), 350);
        assert_eq!(default_pause_ms("Wait...", false), 500);
        assert_eq!(default_pause_ms("Fine.", false), 250);
    }

    #[test]
    fn long_pauses_are_kept_rare() {
        let sentences = split_sentences(
            "One ran. Two ran. Three ran. Four ran. Five ran. Six ran. Seven ran. Eight ran.",
        );
        // a steady ramp of long pauses, as small models like to produce
        let events: Vec<_> = (0..8)
            .map(|i| json!({"i": i, "pause_ms": 1200 + 100 * i}))
            .collect();
        let raw = json!({"narrator_voice": "v", "events": events}).to_string();
        let r = resolve(&sentences, Some(&raw), Directive::default());
        let long: Vec<_> = r
            .segments
            .iter()
            .map(|s| s.directive.pause_ms >= LONG_PAUSE_MS)
            .collect();
        assert_eq!(long, [true, false, false, false, false, false, true, false]);
        assert!(r.repaired >= 6);
    }

    #[test]
    fn narrators_own_quotes_never_get_a_character_voice() {
        let sentences = split_sentences(
            "My brother turned and I told him, \"Someone is waiting.\" \"Go away,\" I said. \"Stop,\" he said.",
        );
        let raw = json!({"narrator_voice": "v", "events": [
            {"i": 0, "speaker": "boy, about 12"}, {"i": 1, "speaker": "boy, about 12"},
            {"i": 2, "speaker": "boy, about 12"},
        ]})
        .to_string();
        let r = resolve(&sentences, Some(&raw), Directive::default());
        let who: Vec<_> = r.segments.iter().map(|s| s.speaker.as_deref()).collect();
        assert_eq!(who, [None, None, Some("boy, about 12")]);
    }

    #[test]
    fn speaker_applies_to_quoted_sentences_and_carries_over_a_quote() {
        let sentences = split_sentences("He asked, \"Who is that? In the kitchen.\" Then he ran. I sat.");
        // 0: He asked, "Who is that?   1: In the kitchen."   2: Then he ran.   3: I sat.
        let raw = json!({"narrator_voice": "v", "events": [
            {"i": 0, "speaker": "young boy, scared"}, {"i": 2, "speaker": "ignored: no quote here"},
        ]})
        .to_string();
        let r = resolve(&sentences, Some(&raw), Directive::default());
        let who: Vec<_> = r.segments.iter().map(|s| s.speaker.as_deref()).collect();
        assert_eq!(
            who,
            [Some("young boy, scared"), Some("young boy, scared"), None, None]
        );
    }

    #[test]
    fn garbage_json_degrades_to_neutral() {
        let plan = plan_story(
            &sents(),
            Some(&Stub::new(vec![Ok("this is not json at all".into())])),
            &PlanOptions::default(),
        );
        assert!(plan.repaired >= 1);
        assert_eq!(plan.narrator_voice, DEFAULT_NARRATOR);
        assert!(plan.segments.iter().all(|s| s.directive.mood == Mood::Neutral));
    }

    #[test]
    fn code_fences_and_surrounding_prose_are_tolerated() {
        let s = sents();
        let raw =
            "Sure! ```json\n{\"narrator_voice\": \"v\", \"events\": [{\"i\": 1, \"mood\": \"tense\"}]}\n```";
        let r = resolve(&s, Some(raw), Directive::default());
        assert_eq!(r.narrator_voice.as_deref(), Some("v"));
        assert_eq!(r.segments[1].directive.mood, Mood::Tense);
        assert_eq!(r.state.mood, Mood::Tense);
        let fenced = "```json\n{\"narrator_voice\": \"w\", \"events\": []}\n```";
        assert_eq!(
            resolve(&s, Some(fenced), Directive::default())
                .narrator_voice
                .as_deref(),
            Some("w")
        );
    }

    #[test]
    fn llm_failure_is_retried_once_then_neutral() {
        let llm = Stub::new(vec![Err("boom".into()), Err("boom again".into())]);
        let plan = plan_story(&sents(), Some(&llm), &PlanOptions::default());
        assert_eq!(llm.calls.borrow().len(), 2);
        assert!(plan.segments.iter().all(|s| s.directive.mood == Mood::Neutral));
    }

    #[test]
    fn retry_recovers_after_one_failure() {
        let llm = Stub::new(vec![
            Err("flaky".into()),
            Stub::ok(json!({"narrator_voice": "v", "events": [{"i": 0, "mood": "calm"}]})),
        ]);
        let plan = plan_story(&sents(), Some(&llm), &PlanOptions::default());
        assert_eq!(plan.segments[0].directive.mood, Mood::Calm);
    }

    #[test]
    fn no_llm_gives_plain_narration() {
        let plan = plan_story(&sents(), None, &PlanOptions::default());
        assert_eq!(plan.segments.len(), sents().len());
        assert!(plan.segments.iter().all(|s| s.directive.beat.is_none()));
        assert!(plan_story(&[], None, &PlanOptions::default()).segments.is_empty());
    }

    #[test]
    fn user_voice_override_wins() {
        let opts = PlanOptions {
            narrator_voice: Some("Mine".into()),
            ..Default::default()
        };
        let llm = Stub::new(vec![Stub::ok(
            json!({"narrator_voice": "LLM voice", "events": []}),
        )]);
        assert_eq!(plan_story(&sents(), Some(&llm), &opts).narrator_voice, "Mine");
    }

    #[test]
    fn windows_carry_state_and_voice_between_calls() {
        let text: Vec<String> = (0..10).map(|n| format!("Sentence number {n} here.")).collect();
        let many = split_sentences(&text.join(" "));
        let llm = Stub::new(vec![
            Stub::ok(json!({"narrator_voice": "First voice", "events": [{"i": 0, "mood": "dread"}]})),
            Stub::ok(json!({"narrator_voice": "Second voice", "events": []})),
        ]);
        let opts = PlanOptions {
            window_sentences: 5,
            hint: Some("slow and creepy".into()),
            ..Default::default()
        };
        let plan = plan_story(&many, Some(&llm), &opts);
        assert_eq!(plan.narrator_voice, "First voice");
        assert_eq!(plan.segments[7].directive.mood, Mood::Dread); // persisted into the second window
        let user2 = &llm.calls.borrow()[1][1].content;
        assert!(user2.contains("State carried over: mood=dread"), "{user2}");
        assert!(user2.contains("Narrator voice already chosen (keep it): First voice"));
        assert!(user2.contains("slow and creepy"));
    }

    #[test]
    fn windows_split_by_size() {
        let text: Vec<String> = (0..30).map(|n| format!("Sentence number {n} here.")).collect();
        let many = split_sentences(&text.join(" "));
        let ws = windows(&many, 200, 100);
        assert_eq!(ws.iter().map(|w| w.len()).sum::<usize>(), 30);
        assert!(ws.len() > 1);
        assert!(windows(&[], 200, 100).is_empty());
    }

    #[test]
    fn messages_list_numbered_sentences_and_hint() {
        let msgs = build_messages(&sents(), Some("old man whispering"), None, None);
        assert_eq!((msgs[0].role, msgs[1].role), ("system", "user"));
        assert!(msgs[1]
            .content
            .contains("[0] (new paragraph) The house had been empty for years."));
        assert!(msgs[1].content.contains("[1] Nobody in town talked about it."));
        assert!(msgs[1].content.contains("old man whispering"));
        assert!(msgs[0].content.contains("dread") && msgs[0].content.contains("whisper"));
    }

    #[test]
    fn schema_enumerates_the_vocabulary() {
        let schema = director_schema();
        let props = &schema["properties"]["events"]["items"]["properties"];
        assert!(props["mood"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("dread")));
        assert!(props["delivery"]["enum"]
            .as_array()
            .unwrap()
            .contains(&json!("whisper")));
        assert_eq!(props["beat"]["enum"][0], "none");
    }

    #[test]
    fn neutral_strips_direction_but_keeps_pauses() {
        let plan = plan_story(
            &sents(),
            Some(&Stub::new(vec![Stub::ok(
                json!({"narrator_voice": "v", "events": [{"i": 0, "mood": "dread", "pause_ms": 900}]}),
            )])),
            &PlanOptions::default(),
        );
        let stripped = neutral(&plan.segments);
        assert_eq!(
            stripped[0].directive,
            Directive {
                pause_ms: 900,
                ..Directive::default()
            }
        );
        assert_eq!(stripped[0].text, plan.segments[0].text);
    }

    #[test]
    fn vocab_roundtrip() {
        for m in Mood::ALL {
            assert_eq!(Mood::parse(m.as_str()), Some(*m));
        }
        assert_eq!(Beat::parse("none"), None);
        assert_eq!(Delivery::parse("yodel"), None);
    }
}
