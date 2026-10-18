use anyhow::{bail, Context, Result};
use clap::{Args, Parser, Subcommand};
use serde_json::json;
use std::io::Read;
use std::path::{Path, PathBuf};

use freshrss_tts::bench;
use freshrss_tts::director::{Completer, PlanOptions};
use freshrss_tts::engines::maya1::{self, DEFAULT_MAX_CHARS};
use freshrss_tts::engines::maya1_llamacpp::{Maya1LlamaCpp, SnacDecoder};
use freshrss_tts::freshrss::FreshRss;
use freshrss_tts::llm::ChatClient;
use freshrss_tts::pipeline;

#[derive(Parser)]
#[command(
    name = "freshrss-tts",
    version,
    about = "Narrate FreshRSS stories with an LLM director and a local TTS"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Log in to FreshRSS and list the entries under the narrate labels
    Check {
        #[arg(long, env = "FRESHRSS_URL")]
        url: String,
        #[arg(long, env = "FRESHRSS_USER")]
        user: String,
        #[arg(long, env = "FRESHRSS_API_PASSWORD", hide_env_values = true)]
        api_password: String,
        /// Repeatable; default: narrate, narrate-scary
        #[arg(long = "label")]
        labels: Vec<String>,
        #[arg(long, default_value_t = 10)]
        limit: usize,
    },
    /// Direct a text/HTML file and print what the TTS would receive (no TTS needed)
    Plan {
        /// Path, or - for stdin
        file: String,
        #[command(flatten)]
        director: DirectorArgs,
        /// Print the full plan as JSON
        #[arg(long)]
        json: bool,
    },
    /// Check which emotion tags are single tokens in your Maya1 GGUF
    Tags {
        #[arg(long, env = "MAYA1_URL", default_value = "http://127.0.0.1:8080")]
        server: String,
    },
    /// Narrate a text/HTML file to a WAV
    Narrate {
        file: String,
        #[command(flatten)]
        tts: TtsArgs,
        #[command(flatten)]
        director: DirectorArgs,
        #[arg(long, default_value = "out/story.wav")]
        out: PathBuf,
        /// Skip the director: plain neutral narration
        #[arg(long)]
        plain: bool,
    },
    /// Bake-off: fixed A/B samples (plain vs directed) with timing and peak GPU memory
    Bench {
        #[command(flatten)]
        tts: TtsArgs,
        /// Also run the full director pipeline (needs --llm-url)
        #[command(flatten)]
        director: DirectorArgs,
        #[arg(long, default_value = "out/bench")]
        out: PathBuf,
    },
}

#[derive(Args)]
struct DirectorArgs {
    /// OpenAI-compatible base URL of the director LLM, e.g. http://127.0.0.1:8081
    #[arg(long = "llm-url", env = "DIRECTOR_URL")]
    llm_url: Option<String>,
    #[arg(long = "llm-model", env = "DIRECTOR_MODEL")]
    llm_model: Option<String>,
    #[arg(long = "llm-api-key", env = "DIRECTOR_API_KEY", hide_env_values = true)]
    llm_api_key: Option<String>,
    /// Extra direction, e.g. "old man whispering, hospital setting"
    #[arg(long)]
    hint: Option<String>,
    /// Fix the narrator voice description instead of letting the LLM pick one
    #[arg(long)]
    voice: Option<String>,
}

#[derive(Args)]
struct TtsArgs {
    /// llama-server running the Maya1 GGUF
    #[arg(long, env = "MAYA1_URL", default_value = "http://127.0.0.1:8080")]
    server: String,
    /// Directory with config.json + snac_24khz.safetensors (see docs/BENCH.md)
    #[arg(long, env = "SNAC_DIR", default_value = "models/snac")]
    snac_dir: PathBuf,
    #[arg(long)]
    seed: Option<i64>,
    #[arg(long, default_value_t = 0.4)]
    temperature: f64,
    #[arg(long, default_value_t = 0.9)]
    top_p: f64,
    #[arg(long, default_value_t = 1.1)]
    repeat_penalty: f64,
}

impl DirectorArgs {
    fn client(&self) -> Option<ChatClient> {
        self.llm_url.as_deref().map(|u| {
            ChatClient::new(u)
                .with_model(self.llm_model.clone())
                .with_api_key(self.llm_api_key.clone())
        })
    }
    fn options(&self) -> PlanOptions {
        PlanOptions {
            hint: self.hint.clone(),
            narrator_voice: self.voice.clone(),
            ..Default::default()
        }
    }
}

impl TtsArgs {
    fn engine(&self) -> Result<Maya1LlamaCpp> {
        let decoder = SnacDecoder::load(&self.snac_dir)
            .with_context(|| format!("loading SNAC from {}", self.snac_dir.display()))?;
        let mut e = Maya1LlamaCpp::new(&self.server, Box::new(decoder));
        e.sampling.seed = self.seed;
        e.sampling.temperature = self.temperature;
        e.sampling.top_p = self.top_p;
        e.sampling.repeat_penalty = self.repeat_penalty;
        Ok(e)
    }
}

fn read_input(file: &str) -> Result<String> {
    if file == "-" {
        let mut s = String::new();
        std::io::stdin().read_to_string(&mut s)?;
        Ok(s)
    } else {
        std::fs::read_to_string(file).with_context(|| format!("reading {file}"))
    }
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Check {
            url,
            user,
            api_password,
            labels,
            limit,
        } => check(&url, &user, &api_password, labels, limit),
        Cmd::Plan { file, director, json } => plan(&file, &director, json),
        Cmd::Tags { server } => tags(&server),
        Cmd::Narrate {
            file,
            tts,
            director,
            out,
            plain,
        } => narrate(&file, &tts, &director, &out, plain),
        Cmd::Bench { tts, director, out } => run_bench(&tts, &director, &out),
    }
}

fn check(url: &str, user: &str, pw: &str, labels: Vec<String>, limit: usize) -> Result<()> {
    let mut fr = FreshRss::new(url, user, pw);
    fr.login()?;
    println!("login ok");
    let labels = if labels.is_empty() {
        vec!["narrate".into(), "narrate-scary".into()]
    } else {
        labels
    };
    for label in labels {
        let entries = fr.label_entries(&label, limit)?;
        println!("label {label:?}: {} entries", entries.len());
        for e in entries {
            let title: String = e.title.chars().take(70).collect();
            println!("  - {title:?} ({} chars of html) {}", e.html.len(), e.url);
        }
    }
    Ok(())
}

fn plan(file: &str, d: &DirectorArgs, as_json: bool) -> Result<()> {
    let sentences = pipeline::prepare(&read_input(file)?);
    let llm = d.client();
    if llm.is_none() {
        eprintln!("# no --llm-url / DIRECTOR_URL: neutral narration, no tags");
    }
    let (plan, chunks) = pipeline::direct_and_render(
        &sentences,
        llm.as_ref().map(|c| c as &dyn Completer),
        &d.options(),
        DEFAULT_MAX_CHARS,
    );
    if as_json {
        let segs: Vec<_> = plan
            .segments
            .iter()
            .map(|s| {
                json!({"idx": s.idx, "text": s.text, "para_start": s.para_start,
                    "mood": s.directive.mood.as_str(), "delivery": s.directive.delivery.as_str(),
                    "beat": s.directive.beat.map(|b| b.as_str()), "pause_ms": s.directive.pause_ms,
                    "speaker": s.speaker})
            })
            .collect();
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "narrator_voice": plan.narrator_voice, "repaired": plan.repaired, "segments": segs,
            }))?
        );
        return Ok(());
    }
    println!("narrator: {}", plan.narrator_voice);
    println!(
        "{} sentences -> {} generations ({} LLM fixes)\n",
        sentences.len(),
        chunks.len(),
        plan.repaired
    );
    for (n, c) in chunks.iter().enumerate() {
        println!(
            "[{}] <description={:?}>\n    {}\n    (+{} ms)",
            n + 1,
            c.style,
            c.text,
            c.pause_ms
        );
    }
    Ok(())
}

fn tags(server: &str) -> Result<()> {
    // No SNAC needed: only llama-server's /tokenize.
    struct NoDecoder;
    impl freshrss_tts::engines::maya1_llamacpp::Decoder for NoDecoder {
        fn decode(&mut self, _: &[Vec<u32>; 3]) -> Result<Vec<u8>> {
            bail!("tags does not decode audio")
        }
    }
    let engine = Maya1LlamaCpp::new(server, Box::new(NoDecoder));
    let confirmed = maya1::all_tags();
    let mut bad = 0;
    for tag in bench::tag_probe_list() {
        let ids = engine.tokenize(&tag)?;
        let mark = if confirmed.contains(&tag.as_str()) {
            "used "
        } else {
            "probe"
        };
        if ids.len() == 1 {
            println!("{mark} {tag:<20} ok, single token {}", ids[0]);
        } else {
            println!(
                "{mark} {tag:<20} NOT a single token ({} pieces) - not a real tag in this model",
                ids.len()
            );
            bad += confirmed.contains(&tag.as_str()) as usize;
        }
    }
    if bad > 0 {
        bail!("{bad} tag(s) this tool emits are not single tokens in your GGUF; check the model/tokenizer");
    }
    println!("\nAdd any 'probe' tags that are single tokens to src/engines/maya1.rs to use them.");
    Ok(())
}

fn narrate(file: &str, tts: &TtsArgs, d: &DirectorArgs, out: &Path, plain: bool) -> Result<()> {
    if let Some(dir) = out.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let sentences = pipeline::prepare(&read_input(file)?);
    let mut engine = tts.engine()?;
    let llm = if plain { None } else { d.client() };
    if llm.is_none() {
        eprintln!("# no director: neutral narration");
    }
    let vram = bench::VramSampler::start();
    let res = pipeline::narrate(
        &sentences,
        &mut engine,
        llm.as_ref().map(|c| c as &dyn Completer),
        &d.options(),
        out,
    )?;
    println!(
        "{}: {:.1} s of audio in {:.1} s (RTF {:.2}), {} generations, {} LLM fixes; peak GPU {}",
        out.display(),
        res.audio_seconds,
        res.wall_seconds,
        res.wall_seconds / res.audio_seconds.max(0.001),
        res.chunks.len(),
        res.plan.repaired,
        vram.stop().map_or("unknown".into(), |m| format!("{m} MiB")),
    );
    Ok(())
}

fn run_bench(tts: &TtsArgs, d: &DirectorArgs, out: &Path) -> Result<()> {
    std::fs::create_dir_all(out)?;
    let mut engine = tts.engine()?;
    let vram = bench::VramSampler::start();
    let mut results = Vec::new();
    for case in bench::samples() {
        results.push(bench::run_chunks(&mut engine, case.name, &case.chunks, out)?);
    }
    if let Some(llm) = d.client() {
        let sentences = pipeline::prepare(&bench::story_text());
        let (plan, chunks) =
            pipeline::direct_and_render(&sentences, Some(&llm), &d.options(), DEFAULT_MAX_CHARS);
        println!(
            "director chose: {} ({} fixes)",
            plan.narrator_voice, plan.repaired
        );
        for c in &chunks {
            println!("  <description={:?}> {} (+{} ms)", c.style, c.text, c.pause_ms);
        }
        results.push(bench::run_chunks(&mut engine, "07_director", &chunks, out)?);
    } else {
        eprintln!("# no --llm-url: skipping the director case");
    }
    let peak = vram.stop();
    println!();
    bench::print_table(&results, peak);
    let metrics =
        json!({"peak_vram_mib": peak, "cases": results.iter().map(|r| r.to_json()).collect::<Vec<_>>()});
    std::fs::write(out.join("metrics.json"), serde_json::to_string_pretty(&metrics)?)?;
    println!("\nWAVs and metrics.json are in {}", out.display());
    Ok(())
}
