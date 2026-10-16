//! Join PCM chunks with silences, write WAV, encode with ffmpeg. Mono 16-bit little-endian PCM.

use anyhow::{bail, Context, Result};
use std::path::Path;
use std::process::Command;

pub fn silence(ms: u32, sample_rate: u32) -> Vec<u8> {
    vec![0u8; 2 * (sample_rate as usize * ms as usize / 1000)]
}

/// `parts` is (pcm, pause_ms_after) per chunk.
pub fn join<I: IntoIterator<Item = (Vec<u8>, u32)>>(parts: I, sample_rate: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for (pcm, pause_ms) in parts {
        out.extend(pcm);
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
        let pcm = join([([1u8, 0].repeat(8000), 500), ([2u8, 0].repeat(8000), 0)], 8000);
        assert_eq!(duration_s(&pcm, 8000), 2.5);
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
