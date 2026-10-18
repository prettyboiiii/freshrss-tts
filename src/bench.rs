//! Bake-off helpers: fixed A/B samples (plain vs hand-directed), timing, and peak GPU memory.
//!
//! The question this answers on your hardware: how fast is Maya1 on the 1050, how much VRAM does it
//! take, and does tag/description direction audibly beat plain narration?

use anyhow::Result;
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::audio;
use crate::director::{Beat, Delivery, Mood, DEFAULT_NARRATOR};
use crate::engines::maya1::{self, beat_tag, describe};
use crate::engines::maya1_llamacpp::Maya1LlamaCpp;
use crate::engines::{Chunk, Engine};

/// A named list of chunks to synthesise.
pub struct Case {
    pub name: &'static str,
    pub chunks: Vec<Chunk>,
}

const P1: &str = "The night shift at the old county hospital was supposed to be quiet. I had worked it \
for six years, and in all that time the only sounds were the hum of the vending machines and the soft \
squeak of my shoes on the third floor corridor.";
const P2A: &str = "That night, the squeak had a second voice. Every time I stopped walking, something \
behind me stopped too.";
const P2B: &str = "I counted to ten. I didn't turn around. I couldn't.";
const P3A: &str = "The lights died all at once.";
const P3B: &str =
    "Somewhere in the dark, a door that had been welded shut for twenty years slowly swung open.";
const P3C: &str = "And a small voice said my name.";

fn chunk(style: String, text: String, pause_ms: u32) -> Chunk {
    Chunk {
        text,
        style,
        pause_ms,
        seed: None,
        parts: Vec::new(),
    }
}

/// Six cases: each of three passages once plain and once hand-directed.
pub fn samples() -> Vec<Case> {
    let v = DEFAULT_NARRATOR;
    let plain = |name, text: &str| Case {
        name,
        chunks: vec![chunk(v.to_string(), text.to_string(), 0)],
    };
    let whisper = "<whisper>";
    vec![
        plain("01_calm_plain", P1),
        Case {
            name: "02_calm_directed",
            chunks: vec![chunk(
                describe(v, Mood::Calm, Delivery::Normal),
                P1.to_string(),
                0,
            )],
        },
        plain("03_dread_plain", &format!("{P2A} {P2B}")),
        Case {
            name: "04_dread_directed",
            chunks: vec![
                chunk(describe(v, Mood::Dread, Delivery::Normal), P2A.to_string(), 1200),
                chunk(
                    describe(v, Mood::Dread, Delivery::Hushed),
                    format!("{whisper} {P2B}"),
                    0,
                ),
            ],
        },
        plain("05_panic_plain", &format!("{P3A} {P3B} {P3C}")),
        Case {
            name: "06_panic_directed",
            chunks: vec![
                chunk(
                    describe(v, Mood::Panicked, Delivery::Normal),
                    format!("{} {P3A}", beat_tag(Beat::Gasp)),
                    900,
                ),
                chunk(describe(v, Mood::Dread, Delivery::Normal), P3B.to_string(), 1200),
                chunk(
                    describe(v, Mood::Fearful, Delivery::Hushed),
                    format!("{whisper} {P3C}"),
                    0,
                ),
            ],
        },
    ]
}

/// The three passages as one story, for the director to annotate.
pub fn story_text() -> String {
    [
        P1.to_string(),
        format!("{P2A} {P2B}"),
        format!("{P3A} {P3B} {P3C}"),
    ]
    .join("\n\n")
}

/// Candidate tags worth probing in a GGUF: confirmed ones plus names that appear in some Maya1 docs.
pub fn tag_probe_list() -> Vec<String> {
    let mut tags: Vec<String> = maya1::all_tags().into_iter().map(String::from).collect();
    for extra in [
        "scream",
        "laugh_harder",
        "sing",
        "snort",
        "exhale",
        "gulp",
        "sarcastic",
        "curious",
        "disappointed",
        "excited",
        "cough",
        "sniff",
        "sob",
        "groan",
        "yawn",
        "clear_throat",
        "whisper_fearfully",
    ] {
        tags.push(format!("<{extra}>"));
    }
    tags
}

#[derive(Debug, Clone)]
pub struct CaseResult {
    pub name: String,
    pub audio_s: f64,
    pub wall_s: f64,
    pub gen_tokens: usize,
    pub gen_seconds: f64,
    pub truncated: bool,
}

impl CaseResult {
    pub fn rtf(&self) -> f64 {
        if self.audio_s > 0.0 {
            self.wall_s / self.audio_s
        } else {
            f64::INFINITY
        }
    }
    pub fn tok_per_s(&self) -> f64 {
        if self.gen_seconds > 0.0 {
            self.gen_tokens as f64 / self.gen_seconds
        } else {
            0.0
        }
    }
    pub fn to_json(&self) -> Value {
        json!({
            "case": self.name, "audio_seconds": self.audio_s, "wall_seconds": self.wall_s,
            "real_time_factor": self.rtf(), "generated_tokens": self.gen_tokens,
            "tokens_per_second": self.tok_per_s(), "truncated": self.truncated,
        })
    }
}

/// Synthesise `chunks` into `<out_dir>/<name>.wav` and measure.
pub fn run_chunks(
    engine: &mut Maya1LlamaCpp,
    name: &str,
    chunks: &[Chunk],
    out_dir: &Path,
) -> Result<CaseResult> {
    let t0 = Instant::now();
    let mut parts = Vec::new();
    let (mut tokens, mut secs, mut truncated) = (0, 0.0, false);
    for (n, c) in chunks.iter().enumerate() {
        eprintln!(
            "[{name}] chunk {}/{}: {} chars",
            n + 1,
            chunks.len(),
            c.text.chars().count()
        );
        parts.push((engine.synth(c)?, c.pause_ms));
        let s = &engine.last_stats;
        tokens += s.gen_tokens;
        secs += s.gen_seconds;
        truncated |= s.truncated;
    }
    let sr = engine.sample_rate();
    let pcm = audio::join(parts, sr);
    audio::write_wav(&out_dir.join(format!("{name}.wav")), &pcm, sr)?;
    Ok(CaseResult {
        name: name.to_string(),
        audio_s: audio::duration_s(&pcm, sr),
        wall_s: t0.elapsed().as_secs_f64(),
        gen_tokens: tokens,
        gen_seconds: secs,
        truncated,
    })
}

pub fn print_table(results: &[CaseResult], peak_vram_mib: Option<u32>) {
    println!(
        "{:<22} {:>8} {:>8} {:>6} {:>7} {:>7}  truncated",
        "case", "audio_s", "wall_s", "RTF", "tok/s", "tokens"
    );
    for r in results {
        println!(
            "{:<22} {:>8.1} {:>8.1} {:>6.2} {:>7.1} {:>7}  {}",
            r.name,
            r.audio_s,
            r.wall_s,
            r.rtf(),
            r.tok_per_s(),
            r.gen_tokens,
            if r.truncated { "YES (cut off)" } else { "no" }
        );
    }
    match peak_vram_mib {
        Some(m) => println!("peak GPU memory during the run: {m} MiB (of 4096 on a GTX 1050 4 GB)"),
        None => println!("peak GPU memory: unknown (nvidia-smi not available)"),
    }
    println!("RTF = wall seconds per audio second; above 1.0 is slower than real time.");
}

/// Parse one line of `nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits`.
pub fn parse_nvidia_smi(output: &str) -> Option<u32> {
    output.lines().filter_map(|l| l.trim().parse::<u32>().ok()).max()
}

/// Polls `nvidia-smi` in the background and remembers the highest memory use.
pub struct VramSampler {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU32>,
    seen: Arc<AtomicBool>,
    handle: Option<JoinHandle<()>>,
}

impl VramSampler {
    pub fn start() -> Self {
        let (stop, peak, seen) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU32::new(0)),
            Arc::new(AtomicBool::new(false)),
        );
        let (s, p, k) = (stop.clone(), peak.clone(), seen.clone());
        let handle = thread::spawn(move || {
            while !s.load(Ordering::Relaxed) {
                let out = Command::new("nvidia-smi")
                    .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                    .output();
                if let Some(mib) = out
                    .ok()
                    .and_then(|o| parse_nvidia_smi(&String::from_utf8_lossy(&o.stdout)))
                {
                    p.fetch_max(mib, Ordering::Relaxed);
                    k.store(true, Ordering::Relaxed);
                } else {
                    break; // no nvidia-smi: stop trying
                }
                thread::sleep(Duration::from_millis(400));
            }
        });
        Self {
            stop,
            peak,
            seen,
            handle: Some(handle),
        }
    }

    pub fn stop(mut self) -> Option<u32> {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        self.seen
            .load(Ordering::Relaxed)
            .then(|| self.peak.load(Ordering::Relaxed))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn six_cases_in_plain_directed_pairs() {
        let cases = samples();
        assert_eq!(cases.len(), 6);
        for pair in cases.chunks(2) {
            assert!(pair[0].name.ends_with("_plain") && pair[1].name.ends_with("_directed"));
            // same words, only direction differs
            let words = |c: &Case| {
                c.chunks
                    .iter()
                    .map(|k| k.text.as_str())
                    .collect::<Vec<_>>()
                    .join(" ")
                    .split_whitespace()
                    .filter(|w| !w.starts_with('<'))
                    .collect::<Vec<_>>()
                    .join(" ")
            };
            assert_eq!(words(&pair[0]), words(&pair[1]), "{}", pair[0].name);
        }
        assert!(cases[5].chunks[0].text.starts_with("<gasp> "));
        assert!(cases
            .iter()
            .flat_map(|c| &c.chunks)
            .all(|c| c.text.chars().count() <= maya1::DEFAULT_MAX_CHARS));
    }

    #[test]
    fn story_text_has_three_paragraphs_and_probe_list_has_confirmed_tags() {
        assert_eq!(story_text().split("\n\n").count(), 3);
        let tags = tag_probe_list();
        assert!(tags.contains(&"<whisper>".to_string()) && tags.contains(&"<scream>".to_string()));
    }

    #[test]
    fn nvidia_smi_parsing() {
        assert_eq!(parse_nvidia_smi("1234\n"), Some(1234));
        assert_eq!(parse_nvidia_smi("100\n900\n"), Some(900));
        assert_eq!(parse_nvidia_smi("No devices were found\n"), None);
    }

    #[test]
    fn rtf_and_rates() {
        let r = CaseResult {
            name: "x".into(),
            audio_s: 10.0,
            wall_s: 25.0,
            gen_tokens: 800,
            gen_seconds: 20.0,
            truncated: false,
        };
        assert_eq!(r.rtf(), 2.5);
        assert_eq!(r.tok_per_s(), 40.0);
        assert_eq!(r.to_json()["real_time_factor"], 2.5);
    }
}
