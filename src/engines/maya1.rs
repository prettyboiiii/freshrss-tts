//! Maya1 adapter: voice description + inline emotion tags -> chunks and prompt token ids.
//!
//! Prompt layout and the 7-token SNAC frame unpacking follow the reference code in
//! <https://github.com/gokhaneraslan/maya1-finetuning> (Apache-2.0), whose `maya/constants.py` and
//! `maya/utils.py` define them for the Maya1 model. Not yet verified against a real Maya1 GGUF run:
//! that is what `freshrss-tts bench` is for.

use anyhow::Result;

use super::Chunk;
use crate::director::{Beat, Delivery, Mood, Segment};
use crate::textprep::split_long;

// Token ids (maya/constants.py).
pub const BOS_ID: u32 = 128000;
pub const TEXT_EOT_ID: u32 = 128009;
pub const SOH_ID: u32 = 128259; // start of header
pub const EOH_ID: u32 = 128260; // end of header
pub const SOA_ID: u32 = 128261; // start of audio
pub const CODE_START_ID: u32 = 128257;
pub const CODE_END_ID: u32 = 128258;
pub const CODE_TOKEN_OFFSET: u32 = 128266;
pub const SNAC_MAX_ID: u32 = 156937;
pub const SNAC_TOKENS_PER_FRAME: usize = 7;
pub const SNAC_SAMPLE_RATE: u32 = 24000;
/// Samples produced by one 7-token frame (one coarse SNAC step: 24000 / 11.72 Hz).
pub const SAMPLES_PER_FRAME: usize = 2048;

pub const DEFAULT_MAX_CHARS: usize = 300; // ~20 s of speech; one generation tops out near 25 s at 2048 tokens
/// A pause at least this long ends a chunk so the silence is real.
pub const BREAK_PAUSE_MS: u32 = 600;

/// Only tags confirmed on the model card / README. The card says "12+ more" exist; add them here
/// once `freshrss-tts tags` shows they are single tokens in your GGUF.
pub fn beat_tag(b: Beat) -> &'static str {
    match b {
        Beat::Gasp => "<gasp>",
        Beat::Sigh => "<sigh>",
        Beat::Cry => "<cry>",
        Beat::Laugh => "<laugh>",
        Beat::Chuckle => "<chuckle>",
        Beat::Giggle => "<giggle>",
    }
}

pub fn delivery_tag(d: Delivery) -> Option<&'static str> {
    match d {
        Delivery::Whisper => Some("<whisper>"),
        Delivery::Raised => Some("<angry>"),
        Delivery::Normal | Delivery::Hushed => None,
    }
}

/// Every tag this adapter can emit (used by the `tags` diagnostic).
pub fn all_tags() -> Vec<&'static str> {
    let mut t: Vec<_> = Beat::ALL.iter().map(|&b| beat_tag(b)).collect();
    t.extend(Delivery::ALL.iter().filter_map(|&d| delivery_tag(d)));
    t
}

/// Mood has no tag; it is steered through the natural-language voice description.
pub fn mood_descriptor(m: Mood) -> &'static str {
    match m {
        Mood::Neutral => "",
        Mood::Calm => "calm, steady",
        Mood::Uneasy => "hesitant, uneasy, slightly lowered pitch",
        Mood::Dread => "low pitch, slow, ominous, hushed",
        Mood::Tense => "tight, controlled, quickening pace",
        Mood::Fearful => "shaky, frightened, breathy",
        Mood::Panicked => "fast, breathless, panicked",
        Mood::Sorrowful => "heavy, sad, slow",
        Mood::Angry => "sharp, harsh, intense",
        Mood::Relieved => "softening, exhaling, relieved",
    }
}

fn delivery_descriptor(d: Delivery) -> &'static str {
    match d {
        Delivery::Hushed => "quiet, close to the microphone",
        _ => "",
    }
}

pub fn describe(narrator_voice: &str, mood: Mood, delivery: Delivery) -> String {
    let mut parts = vec![narrator_voice.trim().trim_end_matches('.').to_string()];
    for extra in [mood_descriptor(mood), delivery_descriptor(delivery)] {
        if !extra.is_empty() {
            parts.push(extra.to_string());
        }
    }
    parts.retain(|p| !p.is_empty());
    parts.join(", ")
}

/// Group directed sentences into generations. The text of every sentence is preserved verbatim.
pub fn render(segments: &[Segment], narrator_voice: &str, max_chars: usize) -> Vec<Chunk> {
    struct Open {
        style: String,
        tag: Option<&'static str>,
        parts: Vec<String>,
        len: usize,
        pause: u32,
    }
    let mut chunks = Vec::new();
    let mut cur: Option<Open> = None;

    let flush = |cur: &mut Option<Open>, chunks: &mut Vec<Chunk>| {
        if let Some(o) = cur.take() {
            let body = o.parts.join(" ");
            let text = match o.tag {
                Some(t) => format!("{t} {body}"),
                None => body,
            };
            chunks.push(Chunk {
                text: text.trim().to_string(),
                style: o.style,
                pause_ms: o.pause,
            });
        }
    };

    for seg in segments {
        let d = &seg.directive;
        let style = describe(narrator_voice, d.mood, d.delivery);
        let tag = delivery_tag(d.delivery);
        let pieces = split_long(&seg.text, max_chars);
        for (n, piece) in pieces.iter().enumerate() {
            let last_piece = n == pieces.len() - 1;
            let text = match d.beat.filter(|_| n == 0) {
                Some(b) => format!("{} {piece}", beat_tag(b)),
                None => piece.clone(),
            };
            let tlen = text.chars().count();
            let must_flush = cur
                .as_ref()
                .is_some_and(|o| o.style != style || o.tag != tag || o.len + tlen + 1 > max_chars);
            if must_flush {
                flush(&mut cur, &mut chunks);
            }
            let o = cur.get_or_insert_with(|| Open {
                style: style.clone(),
                tag,
                parts: vec![],
                len: 0,
                pause: 0,
            });
            o.parts.push(text);
            o.len += tlen + 1;
            o.pause = if last_piece { d.pause_ms } else { 120 };
            if o.pause >= BREAK_PAUSE_MS {
                flush(&mut cur, &mut chunks);
            }
        }
    }
    flush(&mut cur, &mut chunks);
    chunks
}

/// Token ids for one generation. `tokenize` must parse special tokens (emotion tags) as single ids.
pub fn build_prompt_ids(
    tokenize: &mut dyn FnMut(&str) -> Result<Vec<u32>>,
    description: &str,
    text: &str,
) -> Result<Vec<u32>> {
    let body = tokenize(&format!("<description=\"{description}\"> {text}"))?;
    let mut ids = vec![SOH_ID, BOS_ID];
    ids.extend(body);
    ids.extend([TEXT_EOT_ID, EOH_ID, SOA_ID, CODE_START_ID]);
    Ok(ids)
}

/// Split generated ids into the three SNAC codebook levels (1 + 2 + 4 codes per 7-token frame).
/// A trailing end-of-speech token and any incomplete frame are dropped.
pub fn unpack_snac_from_7(tokens: &[u32]) -> [Vec<u32>; 3] {
    let toks = match tokens.last() {
        Some(&CODE_END_ID) => &tokens[..tokens.len() - 1],
        _ => tokens,
    };
    let (mut l1, mut l2, mut l3) = (vec![], vec![], vec![]);
    for frame in toks.as_chunks::<SNAC_TOKENS_PER_FRAME>().0 {
        // each slot carries its own 4096 offset on top of CODE_TOKEN_OFFSET
        let c: Vec<u32> = frame
            .iter()
            .map(|&t| t.wrapping_sub(CODE_TOKEN_OFFSET) % 4096)
            .collect();
        l1.push(c[0]);
        l2.extend([c[1], c[4]]);
        l3.extend([c[2], c[3], c[5], c[6]]);
    }
    [l1, l2, l3]
}

/// Inverse of [`unpack_snac_from_7`] (used by tests and mock servers).
pub fn pack_snac_to_7(levels: &[Vec<u32>; 3]) -> Vec<u32> {
    let [l1, l2, l3] = levels;
    let mut out = Vec::new();
    for i in 0..l1.len() {
        let codes = [
            l1[i],
            l2[2 * i],
            l3[4 * i],
            l3[4 * i + 1],
            l2[2 * i + 1],
            l3[4 * i + 2],
            l3[4 * i + 3],
        ];
        out.extend(
            codes
                .iter()
                .enumerate()
                .map(|(slot, &c)| c + CODE_TOKEN_OFFSET + slot as u32 * 4096),
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::director::Directive;

    const VOICE: &str = "Adult male narrator, deep warm voice";

    fn seg(i: usize, text: &str, d: Directive) -> Segment {
        Segment {
            idx: i,
            text: text.into(),
            para_start: false,
            directive: d,
        }
    }
    fn plain(i: usize, text: &str) -> Segment {
        seg(i, text, Directive::default())
    }

    #[test]
    fn describe_adds_mood_and_delivery_descriptors() {
        assert_eq!(describe("Voice.", Mood::Neutral, Delivery::Normal), "Voice");
        assert!(describe(VOICE, Mood::Dread, Delivery::Hushed)
            .ends_with("ominous, hushed, quiet, close to the microphone"));
    }

    #[test]
    fn same_style_sentences_share_a_chunk_and_text_is_verbatim() {
        let chunks = render(
            &[plain(0, "One."), plain(1, "Two."), plain(2, "Three.")],
            VOICE,
            300,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "One. Two. Three.");
        assert_eq!(chunks[0].style, VOICE);
    }

    #[test]
    fn mood_change_starts_a_new_generation_with_new_description() {
        let dread = Directive {
            mood: Mood::Dread,
            ..Default::default()
        };
        let chunks = render(
            &[
                plain(0, "Calm."),
                seg(1, "Dread now.", dread),
                seg(2, "Still.", dread),
            ],
            VOICE,
            300,
        );
        let texts: Vec<_> = chunks.iter().map(|c| c.text.as_str()).collect();
        assert_eq!(texts, ["Calm.", "Dread now. Still."]);
        assert!(chunks[1].style.contains("ominous") && !chunks[0].style.contains("ominous"));
    }

    #[test]
    fn whisper_tag_leads_the_chunk_and_beat_tag_precedes_its_sentence() {
        let w = Directive {
            delivery: Delivery::Whisper,
            ..Default::default()
        };
        let chunks = render(
            &[
                seg(
                    0,
                    "Don't turn around.",
                    Directive {
                        beat: Some(Beat::Gasp),
                        ..w
                    },
                ),
                seg(1, "It's behind you.", w),
            ],
            VOICE,
            300,
        );
        assert_eq!(chunks.len(), 1);
        assert_eq!(
            chunks[0].text,
            "<whisper> <gasp> Don't turn around. It's behind you."
        );
    }

    #[test]
    fn raised_delivery_uses_angry_tag() {
        let d = Directive {
            delivery: Delivery::Raised,
            ..Default::default()
        };
        assert_eq!(
            render(&[seg(0, "Get out!", d)], VOICE, 300)[0].text,
            "<angry> Get out!"
        );
    }

    #[test]
    fn long_pause_ends_a_chunk_and_is_carried() {
        let long = Directive {
            pause_ms: 1500,
            ..Default::default()
        };
        let chunks = render(&[seg(0, "Before.", long), plain(1, "After.")], VOICE, 300);
        let got: Vec<_> = chunks.iter().map(|c| (c.text.as_str(), c.pause_ms)).collect();
        assert_eq!(got, [("Before.", 1500), ("After.", 250)]);
    }

    #[test]
    fn chunks_respect_max_chars_and_keep_every_word() {
        let segs: Vec<_> = (0..20)
            .map(|i| plain(i, "This is a sentence of moderate length, honestly."))
            .collect();
        let chunks = render(&segs, VOICE, 120);
        assert!(chunks.len() > 3);
        assert!(chunks.iter().all(|c| c.text.chars().count() <= 120));
        let joined = chunks
            .iter()
            .map(|c| c.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        let want = segs.iter().map(|s| s.text.as_str()).collect::<Vec<_>>().join(" ");
        assert_eq!(joined, want);
    }

    #[test]
    fn overlong_sentence_is_split_and_beat_only_on_first_piece() {
        let long = format!("{}end.", "word, ".repeat(100));
        let d = Directive {
            beat: Some(Beat::Sigh),
            ..Default::default()
        };
        let chunks = render(&[seg(0, &long, d)], VOICE, 100);
        assert!(chunks.len() > 1 && chunks[0].text.starts_with("<sigh> "));
        assert!(chunks[1..].iter().all(|c| !c.text.contains("<sigh>")));
        assert!(chunks
            .iter()
            .all(|c| c.text.chars().count() <= 100 + "<sigh> ".len()));
    }

    #[test]
    fn prompt_ids_layout_and_text_given_to_tokenizer() {
        let mut seen = vec![];
        let ids = build_prompt_ids(
            &mut |s| {
                seen.push(s.to_string());
                Ok(vec![1, 2, 3])
            },
            "deep voice",
            "<whisper> Hi.",
        )
        .unwrap();
        assert_eq!(ids, [128259, 128000, 1, 2, 3, 128009, 128260, 128261, 128257]);
        assert_eq!(seen, ["<description=\"deep voice\"> <whisper> Hi."]);
    }

    #[test]
    fn snac_roundtrip_and_slot_layout() {
        let levels = [
            vec![10, 11, 12],
            vec![20, 21, 22, 23, 24, 25],
            (30..42).collect::<Vec<u32>>(),
        ];
        let mut tokens = pack_snac_to_7(&levels);
        tokens.push(CODE_END_ID);
        assert_eq!(unpack_snac_from_7(&tokens), levels);
        // slot order is L1, L2a, L3a, L3b, L2b, L3c, L3d
        let first: Vec<u32> = tokens[..7]
            .iter()
            .map(|t| (t - CODE_TOKEN_OFFSET) % 4096)
            .collect();
        assert_eq!(first, [10, 20, 30, 31, 21, 32, 33]);
        assert!(tokens[..tokens.len() - 1].iter().all(|&t| t <= SNAC_MAX_ID));
    }

    #[test]
    fn unpack_drops_incomplete_trailing_frame_and_handles_empty() {
        let mut t = pack_snac_to_7(&[vec![1], vec![2, 3], vec![4, 5, 6, 7]]);
        t.push(CODE_TOKEN_OFFSET + 5);
        assert_eq!(unpack_snac_from_7(&t), [vec![1], vec![2, 3], vec![4, 5, 6, 7]]);
        assert_eq!(unpack_snac_from_7(&[]), [Vec::<u32>::new(), vec![], vec![]]);
    }

    #[test]
    fn every_emitted_tag_is_listed() {
        let tags = all_tags();
        assert!(tags.contains(&"<whisper>") && tags.contains(&"<gasp>") && tags.contains(&"<angry>"));
    }
}
