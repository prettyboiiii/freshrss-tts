//! Join PCM chunks with silences, write WAV, encode with ffmpeg. Mono 16-bit little-endian PCM.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

pub fn silence(ms: u32, sample_rate: u32) -> Vec<u8> {
    vec![0u8; 2 * (sample_rate as usize * ms as usize / 1000)]
}

/// Samples quieter than this (about -40 dBFS) count as silence when trimming chunk edges.
const SILENCE_LEVEL: i16 = 330;
/// Quiet audio kept next to the speech so word endings do not get clipped.
const TRIM_MARGIN_MS: usize = 40;

/// Cut the dead air a TTS leaves before and after speech, so the pauses in a story are the ones
/// the director asked for. A chunk that is all silence becomes empty.
pub fn trim_silence(pcm: &[u8], sample_rate: u32) -> &[u8] {
    let loud = |s: &[u8; 2]| i16::from_le_bytes(*s).saturating_abs() >= SILENCE_LEVEL;
    let (samples, _) = pcm.as_chunks::<2>();
    let Some(first) = samples.iter().position(loud) else {
        return &[];
    };
    let last = samples.iter().rposition(loud).unwrap_or(first);
    let margin = sample_rate as usize * TRIM_MARGIN_MS / 1000;
    let start = first.saturating_sub(margin);
    let end = (last + 1 + margin).min(pcm.len() / 2);
    &pcm[2 * start..2 * end]
}

/// `parts` is (pcm, pause_ms_after) per chunk. Each chunk is trimmed of leading/trailing silence.
pub fn join<I: IntoIterator<Item = (Vec<u8>, u32)>>(parts: I, sample_rate: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (pcm, pause_ms) in parts {
        out.extend(trim_silence(&pcm, sample_rate));
        out.extend(silence(pause_ms, sample_rate));
    }
    out
}

pub fn duration_s(pcm: &[u8], sample_rate: u32) -> f64 {
    pcm.len() as f64 / 2.0 / sample_rate as f64
}

pub fn wav_bytes(pcm: &[u8], sample_rate: u32) -> Vec<u8> {
    let data_len = pcm.len() as u32;
    let mut v = Vec::with_capacity(44 + pcm.len());
    v.extend(b"RIFF");
    v.extend((36 + data_len).to_le_bytes());
    v.extend(b"WAVEfmt ");
    v.extend(16u32.to_le_bytes()); // fmt chunk size
    v.extend(1u16.to_le_bytes()); // PCM
    v.extend(1u16.to_le_bytes()); // mono
    v.extend(sample_rate.to_le_bytes());
    v.extend((sample_rate * 2).to_le_bytes()); // byte rate
    v.extend(2u16.to_le_bytes()); // block align
    v.extend(16u16.to_le_bytes()); // bits per sample
    v.extend(b"data");
    v.extend(data_len.to_le_bytes());
    v.extend(pcm);
    v
}

pub fn write_wav(path: &Path, pcm: &[u8], sample_rate: u32) -> Result<()> {
    std::fs::write(path, wav_bytes(pcm, sample_rate)).with_context(|| format!("writing {}", path.display()))
}

/// Loudness-normalise and encode (format from the extension: `.mp3`, `.opus` or `.ogg`).
pub fn encode(wav: &Path, out: &Path, bitrate: &str) -> Result<()> {
    let codec = match out.extension().and_then(|e| e.to_str()) {
        Some("mp3") => "libmp3lame",
        Some("opus" | "ogg") => "libopus",
        other => bail!("unsupported output format {other:?}"),
    };
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(wav)
        .args([
            "-af",
            "loudnorm=I=-18:TP=-2:LRA=11",
            "-ac",
            "1",
            "-c:a",
            codec,
            "-b:a",
            bitrate,
        ])
        .arg(out)
        .status()
        .context("running ffmpeg (is it installed?)")?;
    if !status.success() {
        bail!("ffmpeg exited with {status}");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_and_join_lengths() {
        assert_eq!(silence(500, 8000).len(), 2 * 4000);
        let loud = |v: i16| v.to_le_bytes().repeat(8000);
        let pcm = join([(loud(1000), 500), (loud(2000), 0)], 8000);
        assert_eq!(duration_s(&pcm, 8000), 2.5);
    }

    #[test]
    fn trim_cuts_dead_air_but_keeps_a_margin() {
        let sr = 8000; // margin = 40 ms = 320 samples
        let mut pcm = vec![0u8; 2 * 4000]; // 0.5 s of silence
        pcm.extend(1000i16.to_le_bytes().repeat(800)); // 0.1 s of speech
        pcm.extend(vec![0u8; 2 * 4000]);
        let trimmed = trim_silence(&pcm, sr);
        assert_eq!(trimmed.len() / 2, 800 + 2 * 320);
        assert!(trim_silence(&[0u8; 100], sr).is_empty());
        assert!(trim_silence(&[], sr).is_empty());
    }

    #[test]
    fn wav_header_is_valid() {
        let pcm = vec![0u8; 480];
        let w = wav_bytes(&pcm, 24000);
        assert_eq!(&w[..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(w[4..8].try_into().unwrap()), 36 + 480);
        assert_eq!(&w[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(w[24..28].try_into().unwrap()), 24000);
        assert_eq!(u32::from_le_bytes(w[28..32].try_into().unwrap()), 48000);
        assert_eq!(&w[36..40], b"data");
        assert_eq!(u32::from_le_bytes(w[40..44].try_into().unwrap()), 480);
        assert_eq!(w.len(), 44 + 480);
    }

    #[test]
    fn unsupported_format_is_rejected() {
        let err = encode(Path::new("a.wav"), Path::new("a.flac"), "64k").unwrap_err();
        assert!(err.to_string().contains("unsupported"));
    }

    #[test]
    fn ffmpeg_encoding_smoke() {
        if Command::new("ffmpeg").arg("-version").output().is_err() {
            return; // ffmpeg not installed here
        }
        let dir = tempfile::tempdir().unwrap();
        let (wav, mp3) = (dir.path().join("a.wav"), dir.path().join("a.mp3"));
        // A tone, not silence: ffmpeg's loudnorm hands LAME an invalid level for all-zero input.
        let tone: Vec<u8> = (0..16000)
            .flat_map(|i| {
                let x = (i as f32 * 440.0 * std::f32::consts::TAU / 8000.0).sin();
                ((x * 8000.0) as i16).to_le_bytes()
            })
            .collect();
        write_wav(&wav, &tone, 8000).unwrap();
        encode(&wav, &mp3, "64k").unwrap();
        assert!(std::fs::metadata(&mp3).unwrap().len() > 0);
    }
}
