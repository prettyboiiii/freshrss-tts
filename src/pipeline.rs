//! Story HTML -> sentences -> directed segments -> TTS chunks -> one audio file.

use anyhow::Result;
use serde::Serialize;
use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use crate::audio;
use crate::director::{plan_story, Completer, Plan, PlanOptions, Segment};
use crate::engines::{maya1, Chunk, Engine};
use crate::textprep::{html_to_text, split_sentences, Sentence};

pub fn prepare(html_or_text: &str) -> Vec<Sentence> {
    split_sentences(&html_to_text(html_or_text))
}

pub fn direct_and_render(
    sentences: &[Sentence],
    llm: Option<&dyn Completer>,
    opts: &PlanOptions,
    max_chars: usize,
) -> (Plan, Vec<Chunk>) {
    let plan = plan_story(sentences, llm, opts);
    let chunks = maya1::render(&plan.segments, &plan.narrator_voice, max_chars);
    (plan, chunks)
}

pub struct Narrated {
    pub plan: Plan,
    pub chunks: Vec<Chunk>,
    pub timings: Vec<Timing>,
    pub audio_seconds: f64,
    pub wall_seconds: f64,
}

/// Where one sentence sits in the finished audio (for a read-along view).
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Timing {
    pub idx: usize,
    pub text: String,
    pub speaker: Option<String>,
    pub start: f64,
    pub end: f64,
}

/// Place every sentence on the audio timeline. `speech_s[n]` is the length of chunk `n` after
/// trimming. A chunk is shared out among its sentences by character count, so times inside a chunk
/// are estimates; chunk boundaries are exact.
pub fn timings(segments: &[Segment], chunks: &[Chunk], speech_s: &[f64]) -> Vec<Timing> {
    let mut span: HashMap<usize, (f64, f64)> = HashMap::new();
    let mut t = 0.0;
    for (chunk, &dur) in chunks.iter().zip(speech_s) {
        let total = chunk.parts.iter().map(|&(_, n)| n).sum::<usize>().max(1) as f64;
        let mut at = t;
        for &(idx, n) in &chunk.parts {
            let end = at + dur * n as f64 / total;
            let e = span.entry(idx).or_insert((at, end));
            e.1 = end;
            at = end;
        }
        t += dur + f64::from(chunk.pause_ms) / 1000.0;
    }
    segments
        .iter()
        .filter_map(|s| {
            let &(start, end) = span.get(&s.idx)?;
            Some(Timing {
                idx: s.idx,
                text: s.text.clone(),
                speaker: s.speaker.clone(),
                start,
                end,
            })
        })
        .collect()
}

/// Direct, render and synthesise a whole story into `out_wav`.
pub fn narrate(
    sentences: &[Sentence],
    engine: &mut dyn Engine,
    llm: Option<&dyn Completer>,
    opts: &PlanOptions,
    out_wav: &Path,
) -> Result<Narrated> {
    let t0 = Instant::now();
    let (plan, chunks) = direct_and_render(sentences, llm, opts, maya1::DEFAULT_MAX_CHARS);
    let sr = engine.sample_rate();
    let mut parts = Vec::with_capacity(chunks.len());
    let mut speech_s = Vec::with_capacity(chunks.len());
    for (n, chunk) in chunks.iter().enumerate() {
        eprintln!(
            "synth {}/{} ({} chars)",
            n + 1,
            chunks.len(),
            chunk.text.chars().count()
        );
        let pcm = engine.synth(chunk)?;
        speech_s.push(audio::duration_s(audio::trim_silence(&pcm, sr), sr));
        parts.push((pcm, chunk.pause_ms));
    }
    let pcm = audio::join(parts, sr);
    audio::write_wav(out_wav, &pcm, sr)?;
    Ok(Narrated {
        timings: timings(&plan.segments, &chunks, &speech_s),
        plan,
        chunks,
        audio_seconds: audio::duration_s(&pcm, sr),
        wall_seconds: t0.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::director::Directive;
    use crate::textprep::Span;

    fn seg(idx: usize, text: &str) -> Segment {
        Segment {
            idx,
            text: text.into(),
            para_start: false,
            directive: Directive::default(),
            spans: vec![Span {
                text: text.into(),
                quoted: false,
            }],
            speaker: None,
        }
    }

    fn chunk(parts: &[(usize, usize)], pause_ms: u32) -> Chunk {
        Chunk {
            text: String::new(),
            style: String::new(),
            pause_ms,
            seed: None,
            parts: parts.to_vec(),
        }
    }

    #[test]
    fn sentences_are_placed_on_the_timeline() {
        let segs = [seg(0, "One."), seg(1, "Two!"), seg(2, "Three.")];
        // chunk 0 = sentences 0+1 (10 + 30 chars, 4 s of speech, 0.5 s pause); chunk 1 = sentence 2 (2 s)
        let chunks = [chunk(&[(0, 10), (1, 30)], 500), chunk(&[(2, 5)], 0)];
        let t = timings(&segs, &chunks, &[4.0, 2.0]);
        let got: Vec<_> = t.iter().map(|x| (x.idx, x.start, x.end)).collect();
        assert_eq!(got, [(0, 0.0, 1.0), (1, 1.0, 4.0), (2, 4.5, 6.5)]);
    }

    #[test]
    fn a_sentence_split_over_chunks_spans_both() {
        let segs = [seg(0, "He said \"hi\".")];
        let chunks = [chunk(&[(0, 7)], 120), chunk(&[(0, 4)], 0)];
        let t = timings(&segs, &chunks, &[1.0, 0.5]);
        assert_eq!((t[0].start, t[0].end), (0.0, 1.62));
    }
}
