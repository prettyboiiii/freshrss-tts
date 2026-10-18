//! Maya1 through llama.cpp's `llama-server` (GGUF; runs on Pascal via CUDA or Vulkan) + SNAC decode.
//!
//! Why not PyTorch for the language model: the 1050 is Pascal, current PyTorch wheels dropped it, and
//! fp16 there is slow. llama.cpp has no such problem. Only the small SNAC decoder runs in-process, on
//! the CPU via candle, which also keeps the GPU's 4 GB free for the model.
//!
//! Server API used (llama.cpp `tools/server/README.md`): `POST /tokenize {content, add_special,
//! parse_special}` and `POST /completion {prompt: [token ids], stream, return_tokens, ...}`.

use anyhow::{bail, Context, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::snac;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::time::{Duration, Instant};

use super::maya1::{self, CODE_END_ID, CODE_TOKEN_OFFSET, SNAC_MAX_ID, SNAC_SAMPLE_RATE};
use super::{Chunk, Engine};

/// Turns the three SNAC code levels into mono s16le PCM at 24 kHz.
pub trait Decoder {
    fn decode(&mut self, levels: &[Vec<u32>; 3]) -> Result<Vec<u8>>;
}

pub const SNAC_CONFIG_FILE: &str = "config.json";
pub const SNAC_WEIGHTS_FILE: &str = "snac_24khz.safetensors";
/// The reference code drops the first 2048 samples (decoder warm-up).
const WARMUP_SAMPLES: usize = 2048;

pub struct SnacDecoder {
    model: snac::Model,
}

impl SnacDecoder {
    /// `dir` must contain `config.json` (hubertsiuzdak/snac_24khz) and `snac_24khz.safetensors`
    /// (lmz/candle-snac); see docs/BENCH.md for the download commands.
    pub fn load(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join(SNAC_CONFIG_FILE);
        let cfg: snac::Config = serde_json::from_reader(
            std::fs::File::open(&cfg_path).with_context(|| format!("opening {}", cfg_path.display()))?,
        )?;
        let weights = dir.join(SNAC_WEIGHTS_FILE);
        if !weights.exists() {
            bail!("missing {}", weights.display());
        }
        // SAFETY: the file is only read, and not modified while mapped.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, &Device::Cpu)? };
        Ok(Self {
            model: snac::Model::new(&cfg, vb)?,
        })
    }

    pub fn from_model(model: snac::Model) -> Self {
        Self { model }
    }
}

impl Decoder for SnacDecoder {
    fn decode(&mut self, levels: &[Vec<u32>; 3]) -> Result<Vec<u8>> {
        if levels[0].is_empty() {
            return Ok(Vec::new());
        }
        let dev = Device::Cpu;
        let codes: Vec<Tensor> = levels
            .iter()
            .map(|l| Tensor::new(l.as_slice(), &dev)?.unsqueeze(0))
            .collect::<candle_core::Result<_>>()?;
        let pcm = self.model.decode(&[&codes[0], &codes[1], &codes[2]])?;
        let wave = pcm.i(0)?.i(0)?.to_vec1::<f32>()?;
        let skip = if wave.len() > WARMUP_SAMPLES {
            WARMUP_SAMPLES
        } else {
            0
        };
        Ok(wave[skip..]
            .iter()
            .flat_map(|&x| ((x.clamp(-1.0, 1.0) * 32767.0) as i16).to_le_bytes())
            .collect())
    }
}

#[derive(Debug, Clone, Default)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub gen_seconds: f64,
    /// Hit `n_predict` without an end-of-speech token: the audio is cut off.
    pub truncated: bool,
}

impl GenStats {
    pub fn tok_per_s(&self) -> f64 {
        if self.gen_seconds > 0.0 {
            self.gen_tokens as f64 / self.gen_seconds
        } else {
            0.0
        }
    }
}

#[derive(Debug, Clone)]
pub struct Sampling {
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u32,
    pub repeat_penalty: f64,
    pub seed: Option<i64>,
}

impl Default for Sampling {
    fn default() -> Self {
        // Starting points, not gospel: tune them with `freshrss-tts bench`.
        Self {
            temperature: 0.4,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            seed: None,
        }
    }
}

pub struct Maya1LlamaCpp {
    url: String,
    agent: ureq::Agent,
    pub sampling: Sampling,
    pub n_predict: u32,
    decoder: Box<dyn Decoder>,
    pub last_stats: GenStats,
}

impl Maya1LlamaCpp {
    pub fn new(server_url: &str, decoder: Box<dyn Decoder>) -> Self {
        Self {
            url: server_url.trim_end_matches('/').to_string(),
            agent: ureq::AgentBuilder::new()
                .timeout_connect(Duration::from_secs(10))
                .timeout_read(Duration::from_secs(600))
                .build(),
            sampling: Sampling::default(),
            n_predict: 2048,
            decoder,
            last_stats: GenStats::default(),
        }
    }

    pub fn tokenize(&self, text: &str) -> Result<Vec<u32>> {
        let reply: Value = self
            .agent
            .post(&format!("{}/tokenize", self.url))
            .send_json(json!({"content": text, "add_special": false, "parse_special": true}))
            .context("llama-server /tokenize failed")?
            .into_json()?;
        reply["tokens"]
            .as_array()
            .context("/tokenize reply has no tokens array")?
            .iter()
            .map(|t| t.as_u64().map(|n| n as u32).context("non-integer token id"))
            .collect()
    }

    /// Stream tokens until the end-of-speech token (checked client-side: it is not an EOS to llama.cpp).
    pub fn generate_tokens(&self, prompt: &[u32], seed: Option<i64>) -> Result<(Vec<u32>, GenStats)> {
        let mut body = json!({
            "prompt": prompt,
            "n_predict": self.n_predict,
            "stream": true,
            "return_tokens": true,
            "temperature": self.sampling.temperature,
            "top_p": self.sampling.top_p,
            "top_k": self.sampling.top_k,
            "repeat_penalty": self.sampling.repeat_penalty,
        });
        if let Some(seed) = self.sampling.seed.or(seed) {
            body["seed"] = json!(seed);
        }
        let resp = self
            .agent
            .post(&format!("{}/completion", self.url))
            .send_json(body)
            .context("llama-server /completion failed")?;

        let mut stats = GenStats {
            prompt_tokens: prompt.len(),
            ..Default::default()
        };
        let mut out: Vec<u32> = Vec::new();
        let (mut first, mut last): (Option<Instant>, Option<Instant>) = (None, None);
        for line in BufReader::new(resp.into_reader()).lines() {
            let line = line.context("reading the token stream")?;
            let Some(data) = line.strip_prefix("data:") else {
                continue;
            };
            let ev: Value = serde_json::from_str(data.trim()).context("bad event in token stream")?;
            let new: Vec<u32> = ev["tokens"]
                .as_array()
                .map(|a| a.iter().filter_map(|t| t.as_u64().map(|n| n as u32)).collect())
                .unwrap_or_default();
            if !new.is_empty() {
                let now = Instant::now();
                first.get_or_insert(now);
                last = Some(now);
            }
            if let Some(pos) = new.iter().position(|&t| t == CODE_END_ID) {
                out.extend(&new[..pos]);
                break; // dropping the reader closes the connection, which stops generation
            }
            out.extend(new);
            if ev["stop"].as_bool() == Some(true) {
                stats.truncated = out.len() >= self.n_predict as usize;
                break;
            }
        }
        stats.gen_tokens = out.len();
        if let (Some(f), Some(l)) = (first, last) {
            stats.gen_seconds = l.duration_since(f).as_secs_f64();
        }
        Ok((out, stats))
    }
}

impl Engine for Maya1LlamaCpp {
    fn sample_rate(&self) -> u32 {
        SNAC_SAMPLE_RATE
    }

    fn synth(&mut self, chunk: &Chunk) -> Result<Vec<u8>> {
        let ids = maya1::build_prompt_ids(&mut |t| self.tokenize(t), &chunk.style, &chunk.text)?;
        let (tokens, stats) = self.generate_tokens(&ids, chunk.seed)?;
        self.last_stats = stats;
        let snac_tokens: Vec<u32> = tokens
            .into_iter()
            .filter(|t| (CODE_TOKEN_OFFSET..=SNAC_MAX_ID).contains(t))
            .collect();
        self.decoder.decode(&maya1::unpack_snac_from_7(&snac_tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Real SNAC 24 kHz architecture with zero weights: the audio is meaningless, shapes and cost are real.
    fn zero_weight_decoder() -> SnacDecoder {
        SnacDecoder::from_model(crate::testing::zero_weight_model().unwrap())
    }

    #[test]
    fn each_seven_token_frame_is_2048_samples_minus_warmup() {
        let mut dec = zero_weight_decoder();
        let frames = 4;
        let levels = [vec![1; frames], vec![2; frames * 2], vec![3; frames * 4]];
        let pcm = dec.decode(&levels).unwrap();
        assert_eq!(pcm.len() / 2, frames * maya1::SAMPLES_PER_FRAME - WARMUP_SAMPLES);
    }

    /// Run on your own machine: `cargo test --release -- --ignored --nocapture snac_cpu_decode_speed`.
    /// Zero weights cost the same FLOPs as real ones, so this measures the SNAC decode cost per second
    /// of audio on your CPU without downloading anything. RTF should stay flat as the length grows.
    #[test]
    #[ignore = "benchmark; run explicitly in release mode"]
    fn snac_cpu_decode_speed() {
        let mut dec = zero_weight_decoder();
        dec.decode(&[vec![1; 4], vec![2; 8], vec![3; 16]]).unwrap(); // warm up
        for frames in [30usize, 60, 120, 240] {
            let levels = [vec![1; frames], vec![2; frames * 2], vec![3; frames * 4]];
            let t = std::time::Instant::now();
            let pcm = dec.decode(&levels).unwrap();
            let audio_s = pcm.len() as f64 / 2.0 / 24000.0;
            let wall = t.elapsed().as_secs_f64();
            println!(
                "SNAC decode: {audio_s:5.1} s of audio in {wall:6.2} s (RTF {:.3})",
                wall / audio_s
            );
        }
    }

    #[test]
    fn empty_input_decodes_to_nothing() {
        let mut dec = zero_weight_decoder();
        assert!(dec.decode(&[vec![], vec![], vec![]]).unwrap().is_empty());
    }

    #[test]
    fn load_reads_config_and_safetensors_from_a_directory() {
        let dir = tempfile::tempdir().unwrap();
        crate::testing::write_zero_weight_snac(dir.path()).unwrap();
        let mut dec = SnacDecoder::load(dir.path()).expect("load");
        let pcm = dec.decode(&[vec![1; 2], vec![2; 4], vec![3; 8]]).unwrap();
        assert_eq!(pcm.len() / 2, 2 * maya1::SAMPLES_PER_FRAME - WARMUP_SAMPLES);
    }

    #[test]
    fn missing_model_files_give_a_clear_error() {
        let dir = tempfile::tempdir().unwrap();
        let err = SnacDecoder::load(dir.path())
            .err()
            .expect("should fail")
            .to_string();
        assert!(err.contains("config.json"), "{err}");
    }
}
