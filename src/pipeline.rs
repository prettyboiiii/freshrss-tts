//! Story HTML -> sentences -> directed segments -> TTS chunks -> one audio file.

use anyhow::Result;
use std::path::Path;
use std::time::Instant;

use crate::audio;
use crate::director::{plan_story, Completer, Plan, PlanOptions};
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
    pub audio_seconds: f64,
    pub wall_seconds: f64,
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
    let mut parts = Vec::with_capacity(chunks.len());
    for (n, chunk) in chunks.iter().enumerate() {
        eprintln!(
            "synth {}/{} ({} chars)",
            n + 1,
            chunks.len(),
            chunk.text.chars().count()
        );
        parts.push((engine.synth(chunk)?, chunk.pause_ms));
    }
    let sr = engine.sample_rate();
    let pcm = audio::join(parts, sr);
    audio::write_wav(out_wav, &pcm, sr)?;
    Ok(Narrated {
        plan,
        chunks,
        audio_seconds: audio::duration_s(&pcm, sr),
        wall_seconds: t0.elapsed().as_secs_f64(),
    })
}
