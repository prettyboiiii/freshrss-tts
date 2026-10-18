//! Narrate FreshRSS stories with an LLM "director" and a local TTS engine.
//!
//! Flow: FreshRSS label -> HTML -> sentences ([`textprep`]) -> directions from a small LLM
//! ([`director`], [`llm`]) -> engine chunks ([`engines`]) -> audio ([`audio`]); [`worker`] ties it to FreshRSS labels and a podcast feed.

pub mod audio;
pub mod bench;
pub mod director;
pub mod engines;
pub mod freshrss;
pub mod llm;
pub mod pipeline;
#[cfg(any(test, feature = "testing"))]
pub mod testing;
pub mod textprep;
pub mod worker;
