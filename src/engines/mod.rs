//! TTS engine adapters. Each turns directed segments into whatever its model understands.

pub mod maya1;
pub mod maya1_llamacpp;

use anyhow::Result;

/// One TTS generation: `text` (with any inline tags) in a voice `style`, then `pause_ms` of silence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub style: String,
    pub pause_ms: u32,
    /// Sampling seed shared by every generation of one voice; `None` leaves it to the engine.
    pub seed: Option<i64>,
    /// (segment idx, characters) of each piece of story text in `text`, in order. Used to time
    /// sentences inside the finished audio; empty for hand-written chunks.
    pub parts: Vec<(usize, usize)>,
}

pub trait Engine {
    fn sample_rate(&self) -> u32;
    /// Mono 16-bit little-endian PCM at `sample_rate()`.
    fn synth(&mut self, chunk: &Chunk) -> Result<Vec<u8>>;
}
