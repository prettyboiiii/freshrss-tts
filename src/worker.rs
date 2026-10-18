//! Pieces of the label worker that need no network: file names, per-story metadata and the podcast feed.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::time::{Duration, Instant};

pub const EPISODE_SUFFIX: &str = ".episode.json";

/// What the feed needs to know about one narrated story. Stored next to the audio as
/// `<slug>.episode.json`, so the feed can be rebuilt from the output directory alone.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Episode {
    pub slug: String,
    pub title: String,
    /// The story's own link.
    pub url: String,
    /// Name of the feed the story came from.
    pub source: String,
    /// Unix seconds.
    pub published: i64,
    pub audio_file: String,
    pub audio_bytes: u64,
    pub duration_s: f64,
}

/// A file-name-safe id for a FreshRSS entry id (`tag:google.com,2005:reader/item/<hex>`).
pub fn slug(entry_id: &str) -> String {
    let tail = entry_id.rsplit('/').next().unwrap_or(entry_id);
    let clean: String = tail
        .chars()
        .filter(char::is_ascii_alphanumeric)
        .collect::<String>()
        .to_lowercase();
    if clean.is_empty() {
        let h = entry_id.bytes().fold(0xcbf29ce484222325u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x100000001b3)
        });
        format!("entry-{h:016x}")
    } else {
        clean
    }
}

pub fn load_episodes(dir: &Path) -> Result<Vec<Episode>> {
    let mut episodes = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        if !path
            .file_name()
            .and_then(|n| n.to_str())
            .is_some_and(|n| n.ends_with(EPISODE_SUFFIX))
        {
            continue;
        }
        let text = std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
        match serde_json::from_str(&text) {
            Ok(ep) => episodes.push(ep),
            Err(e) => eprintln!("# skipping {}: {e}", path.display()),
        }
    }
    Ok(episodes)
}

/// Wait until every llama-server answers `/health` with something other than 503 (still loading) or a
/// connection error. Scaled up from zero, the servers need minutes to load their models.
pub fn wait_ready(urls: &[String], timeout: Duration) -> Result<()> {
    let start = Instant::now();
    for url in urls {
        let health = format!("{}/health", url.trim_end_matches('/'));
        while !answers(&health) {
            if start.elapsed() > timeout {
                anyhow::bail!("{health} was not ready after {timeout:?}");
            }
            std::thread::sleep(Duration::from_secs(5));
        }
    }
    Ok(())
}

fn answers(health: &str) -> bool {
    match ureq::get(health).timeout(Duration::from_secs(5)).call() {
        Ok(_) => true,
        Err(ureq::Error::Status(code, _)) => code != 503,
        Err(_) => false,
    }
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// RFC 2822 date in UTC, e.g. `Thu, 01 Jan 1970 00:00:00 +0000`.
pub fn rfc2822(unix: i64) -> String {
    const DAYS: [&str; 7] = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let days = unix.div_euclid(86_400);
    let secs = unix.rem_euclid(86_400);
    // civil date from a day count (Howard Hinnant's algorithm)
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    format!(
        "{}, {day:02} {} {year} {:02}:{:02}:{:02} +0000",
        DAYS[(days + 4).rem_euclid(7) as usize],
        MONTHS[(month - 1) as usize],
        secs / 3600,
        secs % 3600 / 60,
        secs % 60
    )
}

/// Podcast RSS 2.0 with iTunes duration; newest episode first. `base_url` is where the audio files are
/// served (no trailing slash needed).
pub fn render_feed(title: &str, base_url: &str, episodes: &[Episode]) -> String {
    let base = base_url.trim_end_matches('/');
    let mut sorted: Vec<&Episode> = episodes.iter().collect();
    sorted.sort_by(|a, b| b.published.cmp(&a.published).then_with(|| a.slug.cmp(&b.slug)));
    let mut xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
<rss version=\"2.0\" xmlns:itunes=\"http://www.itunes.com/dtds/podcast-1.0.dtd\">\n<channel>\n\
<title>{t}</title>\n<link>{b}/</link>\n<description>{t}</description>\n<language>en</language>\n",
        t = xml_escape(title),
        b = xml_escape(base),
    );
    for ep in sorted {
        xml.push_str(&format!(
            "<item>\n<title>{title}</title>\n<link>{link}</link>\n\
<guid isPermaLink=\"false\">{guid}</guid>\n<pubDate>{date}</pubDate>\n\
<description>{desc}</description>\n\
<enclosure url=\"{base}/{file}\" length=\"{len}\" type=\"audio/ogg\"/>\n\
<itunes:duration>{dur}</itunes:duration>\n</item>\n",
            title = xml_escape(&ep.title),
            link = xml_escape(&ep.url),
            guid = xml_escape(&ep.slug),
            date = rfc2822(ep.published),
            desc = xml_escape(&format!("From {}. Original: {}", ep.source, ep.url)),
            base = xml_escape(base),
            file = xml_escape(&ep.audio_file),
            len = ep.audio_bytes,
            dur = ep.duration_s.round() as u64,
        ));
    }
    xml.push_str("</channel>\n</rss>\n");
    xml
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ep(slug: &str, published: i64) -> Episode {
        Episode {
            slug: slug.into(),
            title: format!("Story {slug} & <more>"),
            url: "https://example.com/a?x=1&y=2".into(),
            source: "r/nosleep".into(),
            published,
            audio_file: format!("{slug}.opus"),
            audio_bytes: 1234,
            duration_s: 99.6,
        }
    }

    #[test]
    fn slug_is_the_hex_tail_or_a_stable_hash() {
        assert_eq!(
            slug("tag:google.com,2005:reader/item/000000AbC123"),
            "000000abc123"
        );
        assert_eq!(slug("12345"), "12345");
        assert!(slug("///").starts_with("entry-"));
        assert_eq!(slug("///"), slug("///"));
    }

    #[test]
    fn dates_are_rfc2822() {
        assert_eq!(rfc2822(0), "Thu, 01 Jan 1970 00:00:00 +0000");
        assert_eq!(rfc2822(1_760_000_000), "Thu, 09 Oct 2025 08:53:20 +0000");
        assert_eq!(rfc2822(951_782_400), "Tue, 29 Feb 2000 00:00:00 +0000");
        assert_eq!(rfc2822(-1), "Wed, 31 Dec 1969 23:59:59 +0000");
    }

    #[test]
    fn feed_escapes_orders_and_points_at_the_audio() {
        let xml = render_feed(
            "Narrated <stories>",
            "https://tts.example/podcast/",
            &[ep("a", 100), ep("b", 200)],
        );
        assert!(xml.contains("<title>Narrated &lt;stories&gt;</title>"));
        assert!(xml.contains(
            "<enclosure url=\"https://tts.example/podcast/b.opus\" length=\"1234\" type=\"audio/ogg\"/>"
        ));
        assert!(xml.contains("Story a &amp; &lt;more&gt;"));
        assert!(xml.contains("https://example.com/a?x=1&amp;y=2"));
        assert!(xml.contains("<itunes:duration>100</itunes:duration>"));
        assert!(
            xml.find("Story b").unwrap() < xml.find("Story a").unwrap(),
            "newest first"
        );
        assert!(xml.ends_with("</channel>\n</rss>\n"));
    }

    #[test]
    fn episodes_round_trip_through_the_output_directory() {
        let dir = tempfile::tempdir().unwrap();
        let e = ep("abc", 5);
        std::fs::write(
            dir.path().join(format!("abc{EPISODE_SUFFIX}")),
            serde_json::to_string(&e).unwrap(),
        )
        .unwrap();
        std::fs::write(dir.path().join("abc.opus"), b"x").unwrap();
        std::fs::write(dir.path().join(format!("bad{EPISODE_SUFFIX}")), "{").unwrap();
        assert_eq!(load_episodes(dir.path()).unwrap(), [e]);
    }
}
