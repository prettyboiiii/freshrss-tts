//! HTML -> plain text -> sentences.
//!
//! The director and the TTS engines both work on whole sentences, so this module is the single place
//! that decides what a "sentence" is. Nothing downstream rewrites the text.

use regex::Regex;
use std::sync::LazyLock;

const BLOCK_TAGS: &[&str] = &[
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "blockquote",
    "pre",
    "tr",
    "hr",
    "section",
    "article",
    "figure",
    "figcaption",
    "table",
    "header",
    "footer",
];
/// Content of these is dropped. `script`/`style` are skipped as raw text up to their closing tag.
const SKIP_TAGS: &[&str] = &["noscript", "template", "svg", "head"];
const RAW_TEXT_TAGS: &[&str] = &["script", "style"];

/// Words that end in "." without ending the sentence.
const ABBREV: &[&str] = &[
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc", "no", "mt", "fig", "inc", "ltd", "co",
    "e.g", "i.e", "a.m", "p.m", "approx", "dept", "est", "gen", "col", "lt", "capt",
];
const OPENERS: &str = "\"'\u{201c}\u{2018}([\u{2014}";

static PARA_SPLIT: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n[ \t]*\n+").unwrap());
static BLANKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").unwrap());
// Sentence-ending punctuation, optional closing quotes/brackets, then whitespace.
static BOUNDARY: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"([.!?]+|\x{2026}+)(["'\x{201d}\x{2019})\]]*)(\s+)"#).unwrap());
static LAST_WORD: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[A-Za-z][A-Za-z.]*$").unwrap());

/// Strip markup and return paragraphs separated by a blank line.
pub fn html_to_text(html: &str) -> String {
    let mut out = String::new();
    let mut text = String::new();
    let mut skip_depth = 0usize;
    let lower = html.to_ascii_lowercase();
    let bytes = html.as_bytes();
    let mut i = 0;

    let flush = |text: &mut String, out: &mut String, skip: usize| {
        if skip == 0 && !text.is_empty() {
            out.push_str(&html_escape::decode_html_entities(text));
        }
        text.clear();
    };

    while i < bytes.len() {
        if bytes[i] != b'<' {
            // copy one UTF-8 character
            let ch_len = html[i..].chars().next().map_or(1, char::len_utf8);
            text.push_str(&html[i..i + ch_len]);
            i += ch_len;
            continue;
        }
        // A '<' that does not start a tag/comment is literal text.
        let next = bytes.get(i + 1).copied().unwrap_or(b' ');
        if !(next.is_ascii_alphabetic() || next == b'/' || next == b'!' || next == b'?') {
            text.push('<');
            i += 1;
            continue;
        }
        if lower[i..].starts_with("<!--") {
            i = lower[i..].find("-->").map_or(bytes.len(), |p| i + p + 3);
            continue;
        }
        // Find the end of the tag, honouring quoted attribute values.
        let mut j = i + 1;
        let mut quote: Option<u8> = None;
        while j < bytes.len() {
            match (quote, bytes[j]) {
                (None, b'>') => break,
                (None, q @ (b'"' | b'\'')) => quote = Some(q),
                (Some(q), c) if c == q => quote = None,
                _ => {}
            }
            j += 1;
        }
        let tag_src = &lower[i + 1..j.min(bytes.len())];
        let closing = tag_src.starts_with('/');
        let name: String = tag_src
            .trim_start_matches('/')
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric())
            .collect();
        i = (j + 1).min(bytes.len());
        flush(&mut text, &mut out, skip_depth);

        if RAW_TEXT_TAGS.contains(&name.as_str()) {
            if !closing {
                let end_pat = format!("</{name}");
                i = lower[i..]
                    .find(&end_pat)
                    .and_then(|p| lower[i + p..].find('>').map(|q| i + p + q + 1))
                    .unwrap_or(bytes.len());
            }
        } else if SKIP_TAGS.contains(&name.as_str()) {
            if closing {
                skip_depth = skip_depth.saturating_sub(1);
            } else if !tag_src.ends_with('/') {
                skip_depth += 1;
            }
        } else if skip_depth == 0 {
            if name == "br" {
                out.push('\n');
            } else if BLOCK_TAGS.contains(&name.as_str()) {
                out.push_str("\n\n");
            }
        }
    }
    flush(&mut text, &mut out, skip_depth);
    normalize_text(&out)
}

/// Collapse whitespace; keep paragraph breaks (blank lines), join soft line breaks.
pub fn normalize_text(text: &str) -> String {
    let cleaned: String = text
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .chars()
        .filter(|c| {
            !matches!(
                c,
                '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{2060}' | '\u{feff}' | '\u{ad}'
            )
        })
        .map(|c| if c == '\u{a0}' { ' ' } else { c })
        .collect();
    let paragraphs: Vec<String> = PARA_SPLIT
        .split(&cleaned)
        .map(|block| {
            block
                .split('\n')
                .map(|line| BLANKS.replace_all(line, " ").trim().to_string())
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" ")
        })
        .filter(|p| !p.is_empty())
        .collect();
    paragraphs.join("\n\n")
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Sentence {
    pub idx: usize,
    pub text: String,
    pub para_start: bool,
}

fn starts_sentence(ch: char) -> bool {
    ch.is_uppercase() || ch.is_numeric() || OPENERS.contains(ch)
}

fn split_paragraph(par: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut start = 0;
    for caps in BOUNDARY.captures_iter(par) {
        let whole = caps.get(0).unwrap();
        let Some(next) = par[whole.end()..].chars().next() else {
            continue;
        };
        if !starts_sentence(next) {
            continue;
        }
        if &caps[1] == "." {
            if let Some(word) = LAST_WORD.find(&par[start..whole.start()]) {
                let w = word.as_str();
                let lw = w.to_lowercase();
                let single_initial = w.len() == 1 && w.chars().all(|c| c.is_uppercase());
                if ABBREV.contains(&lw.trim_end_matches('.')) || single_initial {
                    continue;
                }
            }
        }
        out.push(par[start..caps.get(2).unwrap().end()].trim().to_string());
        start = whole.end();
    }
    let tail = par[start..].trim();
    if !tail.is_empty() {
        out.push(tail.to_string());
    }
    out
}

/// Split text into sentences, remembering where paragraphs start.
pub fn split_sentences(text: &str) -> Vec<Sentence> {
    let mut sentences = Vec::new();
    for par in normalize_text(text).split("\n\n").filter(|p| !p.is_empty()) {
        for (n, s) in split_paragraph(par).into_iter().enumerate() {
            sentences.push(Sentence {
                idx: sentences.len(),
                text: s,
                para_start: n == 0,
            });
        }
    }
    sentences
}

/// Cut an over-long sentence at clause boundaries, then at spaces. Never alters characters.
pub fn split_long(text: &str, max_chars: usize) -> Vec<String> {
    if text.chars().count() <= max_chars {
        return vec![text.to_string()];
    }
    let mut pieces = Vec::new();
    let mut rest: Vec<char> = text.chars().collect();
    while rest.len() > max_chars {
        let window = &rest[..=max_chars];
        let min_cut = max_chars / 3;
        // (is cut point end index) for each rule, in priority order
        let rules: [&dyn Fn(char) -> bool; 3] = [
            &|c| matches!(c, ';' | ':' | '\u{2014}' | '\u{2013}'),
            &|c| c == ',',
            &|_| false, // plain whitespace handled below
        ];
        let mut cut = None;
        for (rank, rule) in rules.iter().enumerate() {
            let hits: Vec<usize> = (0..window.len())
                .filter_map(|k| {
                    if rank < 2 {
                        (rule(window[k]) && window.get(k + 1).is_some_and(|c| c.is_whitespace()))
                            .then_some(k + 2)
                    } else {
                        window[k].is_whitespace().then_some(k + 1)
                    }
                })
                .filter(|&h| h >= min_cut)
                .collect();
            if let Some(&h) = hits.last() {
                cut = Some(h);
                break;
            }
        }
        let cut = cut.unwrap_or(max_chars);
        let piece: String = rest[..cut].iter().collect();
        pieces.push(piece.trim().to_string());
        let remainder: String = rest[cut..].iter().collect();
        rest = remainder.trim().chars().collect();
    }
    if !rest.is_empty() {
        pieces.push(rest.iter().collect());
    }
    pieces.into_iter().filter(|p| !p.is_empty()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(s: &str) -> Vec<String> {
        split_sentences(s).into_iter().map(|x| x.text).collect()
    }

    #[test]
    fn html_paragraphs_and_noise() {
        let html = "<style>p{color:red}</style><h1>Title</h1><p>First&nbsp;para &amp; more.</p>\
                    <script>if (a<b) alert(1)</script><p>Second<br>line.</p><div>Third</div>";
        assert_eq!(
            html_to_text(html),
            "Title\n\nFirst para & more.\n\nSecond line.\n\nThird"
        );
    }

    #[test]
    fn double_br_is_a_paragraph_break() {
        assert_eq!(html_to_text("a<br><br>b"), "a\n\nb");
    }

    #[test]
    fn comments_attributes_and_literal_lt() {
        let html =
            r#"<p class="a>b" data-x='<'>1 < 2 and <!-- hidden --> visible</p><noscript>no</noscript>"#;
        assert_eq!(html_to_text(html), "1 < 2 and visible");
    }

    #[test]
    fn multibyte_text_survives() {
        assert_eq!(
            html_to_text("<p>Café – “quoted” 日本語</p>"),
            "Café – “quoted” 日本語"
        );
    }

    #[test]
    fn normalize_joins_soft_wraps_and_drops_invisible() {
        assert_eq!(
            normalize_text("one\ntwo\u{200b}  three\n\n\n\nfour"),
            "one two three\n\nfour"
        );
    }

    #[test]
    fn simple_sentences_and_paragraph_flags() {
        let out = split_sentences("It was dark. Nobody came!\n\nThen a sound?");
        let t: Vec<_> = out.iter().map(|s| s.text.as_str()).collect();
        assert_eq!(t, ["It was dark.", "Nobody came!", "Then a sound?"]);
        assert_eq!(
            out.iter().map(|s| s.para_start).collect::<Vec<_>>(),
            [true, false, true]
        );
        assert_eq!(out.iter().map(|s| s.idx).collect::<Vec<_>>(), [0, 1, 2]);
    }

    #[test]
    fn dialogue_tag_stays_attached() {
        assert_eq!(
            texts("\"Why?\" she asked. \"Please go.\""),
            ["\"Why?\" she asked.", "\"Please go.\""]
        );
    }

    #[test]
    fn closing_quote_after_period_ends_sentence() {
        assert_eq!(
            texts("He said \"stop.\" Then he left."),
            ["He said \"stop.\"", "Then he left."]
        );
    }

    #[test]
    fn abbreviations_and_initials_do_not_split() {
        assert_eq!(
            texts("Dr. Smith met J. Doe at 5 p.m. today. It rained."),
            ["Dr. Smith met J. Doe at 5 p.m. today.", "It rained."]
        );
    }

    #[test]
    fn ellipsis_before_lowercase_does_not_split() {
        assert_eq!(
            texts("I thought... maybe not. Fine."),
            ["I thought... maybe not.", "Fine."]
        );
    }

    #[test]
    fn text_is_preserved_exactly() {
        let src = "The door creaked. \"Hello?\" Nothing. Then, from the dark: a whisper.\n\nShe ran.";
        let joined = texts(src).join(" ");
        assert_eq!(joined, src.split_whitespace().collect::<Vec<_>>().join(" "));
    }

    #[test]
    fn split_long_respects_limit_and_keeps_words() {
        let s = format!("{}end.", "word ".repeat(80));
        let s = s.trim();
        let parts = split_long(s, 100);
        assert!(parts.iter().all(|p| p.chars().count() <= 100));
        assert_eq!(parts.join(" "), s);
    }

    #[test]
    fn split_long_prefers_clause_boundaries() {
        let s = "The corridor stretched on and on, and every door was shut, except the last one.";
        let parts = split_long(s, 50);
        assert!(parts[0].ends_with(','), "{parts:?}");
        assert_eq!(parts.join(" "), s);
    }

    #[test]
    fn split_long_handles_unbroken_text_and_unicode() {
        let s = "é".repeat(250);
        let parts = split_long(&s, 100);
        assert_eq!(
            parts.iter().map(|p| p.chars().count()).collect::<Vec<_>>(),
            [100, 100, 50]
        );
        assert_eq!(parts.concat(), s);
    }

    #[test]
    fn short_text_untouched() {
        assert_eq!(split_long("short", 50), ["short"]);
    }
}
