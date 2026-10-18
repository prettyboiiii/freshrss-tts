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
use freshrss_tts::textprep::Sentence;
use freshrss_tts::worker::{self, Episode};

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

// parsed once at start-up, so the size of the largest variant does not matter
#[allow(clippy::large_enum_variant)]
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
    /// Narrate the stories under the narrate labels, publish a podcast feed, relabel them `narrated`
    Work {
        #[arg(long, env = "FRESHRSS_URL")]
        url: String,
        #[arg(long, env = "FRESHRSS_USER")]
        user: String,
        #[arg(long, env = "FRESHRSS_API_PASSWORD", hide_env_values = true)]
        api_password: String,
        /// Repeatable; default: narrate, narrate-scary
        #[arg(long = "label")]
        labels: Vec<String>,
        /// Label put on a story once it is narrated (the story keeps its other labels)
        #[arg(long, default_value = "narrated")]
        done_label: String,
        /// Label put on a story that could not be narrated, so it is not retried forever
        #[arg(long, default_value = "narrate-failed")]
        failed_label: String,
        /// Most stories to narrate in one pass
        #[arg(long, default_value_t = 3)]
        max: usize,
        /// Keep running: look again every this many seconds instead of exiting after one pass
        #[arg(long)]
        watch: Option<u64>,
        /// ntfy server to announce finished stories on, e.g. http://ntfy
        #[arg(long, env = "NTFY_URL")]
        ntfy_url: Option<String>,
        #[arg(long, env = "NTFY_TOPIC", default_value = "narrated")]
        ntfy_topic: String,
        #[arg(long, env = "NTFY_TOKEN", hide_env_values = true)]
        ntfy_token: Option<String>,
        /// Seconds to wait for the model servers to finish loading before narrating
        #[arg(long, default_value_t = 1200)]
        ready_timeout: u64,
        /// Directory the audio, episode files and feed.xml go to
        #[arg(long, default_value = "out/podcast")]
        out: PathBuf,
        /// Where `out` is served from, used for the audio links in the feed
        #[arg(long, env = "PUBLIC_URL")]
        public_url: String,
        #[arg(long, default_value = "Narrated stories")]
        feed_title: String,
        #[arg(long, default_value = "32k")]
        bitrate: String,
        /// List what would be narrated; change nothing
        #[arg(long)]
        dry_run: bool,
        #[command(flatten)]
        tts: TtsArgs,
        #[command(flatten)]
        director: DirectorArgs,
    },
    /// Serve `{"queue": N}` (stories waiting under the narrate labels) for KEDA to scale on
    Queue {
        #[arg(long, env = "FRESHRSS_URL")]
        url: String,
        #[arg(long, env = "FRESHRSS_USER")]
        user: String,
        #[arg(long, env = "FRESHRSS_API_PASSWORD", hide_env_values = true)]
        api_password: String,
        /// Repeatable; default: narrate, narrate-scary
        #[arg(long = "label")]
        labels: Vec<String>,
        #[arg(long, default_value = "0.0.0.0:8080")]
        listen: String,
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
        Cmd::Work {
            url,
            user,
            api_password,
            labels,
            done_label,
            failed_label,
            max,
            watch,
            ntfy_url,
            ntfy_topic,
            ntfy_token,
            ready_timeout,
            out,
            public_url,
            feed_title,
            bitrate,
            dry_run,
            tts,
            director,
        } => {
            let opts = WorkOpts {
                labels,
                done_label,
                failed_label,
                max,
                ready_timeout,
                ntfy: ntfy_url.map(|url| worker::Ntfy {
                    url,
                    topic: ntfy_topic,
                    token: ntfy_token,
                }),
                out,
                public_url,
                feed_title,
                bitrate,
                dry_run,
            };
            loop {
                let pass = work(&opts, FreshRss::new(&url, &user, &api_password), &tts, &director);
                let Some(secs) = watch else { break pass };
                if let Err(e) = pass {
                    eprintln!("pass failed: {e:#}");
                }
                std::thread::sleep(std::time::Duration::from_secs(secs));
            }
        }
        Cmd::Queue {
            url,
            user,
            api_password,
            labels,
            listen,
        } => queue_server(&listen, FreshRss::new(&url, &user, &api_password), &labels),
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
    let name = out
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let res = narrate_story(
        &mut engine,
        &sentences,
        llm.as_ref().map(|c| c as &dyn Completer),
        &d.options(),
        out,
        &name,
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

/// Narrate `sentences` to `out_wav` and write the read-along data (`.timings.json`) next to it.
/// `audio_name` is the file the timings refer to (the WAV may be replaced by an encoded copy).
fn narrate_story(
    engine: &mut Maya1LlamaCpp,
    sentences: &[Sentence],
    llm: Option<&dyn Completer>,
    opts: &PlanOptions,
    out_wav: &Path,
    audio_name: &str,
) -> Result<pipeline::Narrated> {
    let res = pipeline::narrate(sentences, engine, llm, opts, out_wav)?;
    std::fs::write(
        out_wav.with_extension("timings.json"),
        serde_json::to_string_pretty(&json!({"audio": audio_name, "sentences": res.timings}))?,
    )?;
    Ok(res)
}

struct WorkOpts {
    labels: Vec<String>,
    done_label: String,
    failed_label: String,
    max: usize,
    ready_timeout: u64,
    ntfy: Option<worker::Ntfy>,
    out: PathBuf,
    public_url: String,
    feed_title: String,
    bitrate: String,
    dry_run: bool,
}

fn work(w: &WorkOpts, mut fr: FreshRss, tts: &TtsArgs, d: &DirectorArgs) -> Result<()> {
    fr.login()?;
    let todo = waiting(&mut fr, &w.labels, w.max.max(10))?;
    let (new, finished): (Vec<_>, Vec<_>) = todo.into_iter().partition(|(_, e)| {
        !w.out
            .join(format!("{}{}", worker::slug(&e.id), worker::EPISODE_SUFFIX))
            .exists()
    });
    println!(
        "{} stories to narrate (at most {} now), {} already narrated but still labelled",
        new.len(),
        w.max,
        finished.len()
    );
    if w.dry_run {
        for (label, e) in new.iter().take(w.max) {
            println!(
                "  would narrate [{label}] {:?} -> {}.opus",
                e.title,
                worker::slug(&e.id)
            );
        }
        for (label, e) in &finished {
            println!("  would relabel [{label}] {:?}", e.title);
        }
        return Ok(());
    }
    std::fs::create_dir_all(&w.out)?;
    // A previous run narrated these but could not relabel them.
    for (label, e) in &finished {
        relabel(&mut fr, e, label, &w.done_label);
    }
    if !new.is_empty() {
        let servers: Vec<String> = [Some(tts.server.clone()), d.llm_url.clone()]
            .into_iter()
            .flatten()
            .collect();
        worker::wait_ready(&servers, std::time::Duration::from_secs(w.ready_timeout))?;
        let mut engine = tts.engine()?;
        let llm = d.client();
        for (label, e) in new.iter().take(w.max) {
            match narrate_entry(
                w,
                &mut engine,
                llm.as_ref().map(|c| c as &dyn Completer),
                d,
                label,
                e,
            ) {
                Ok(ep) => {
                    println!(
                        "narrated {:?}: {:.0} s, {} bytes",
                        e.title, ep.duration_s, ep.audio_bytes
                    );
                    relabel(&mut fr, e, label, &w.done_label);
                    let base = w.public_url.trim_end_matches('/');
                    notify(
                        w,
                        &format!("Narrated: {}", e.title),
                        &format!("{:.0} min from {}", (ep.duration_s / 60.0).max(1.0), e.feed),
                        Some(&format!("{base}/")),
                        Some(&format!("{base}/{}", ep.audio_file)),
                        &["headphones"],
                    );
                }
                Err(err) => {
                    eprintln!("could not narrate {:?}: {err:#}", e.title);
                    relabel(&mut fr, e, label, &w.failed_label);
                    notify(
                        w,
                        &format!("Could not narrate: {}", e.title),
                        &format!("{err:#}"),
                        None,
                        None,
                        &["warning"],
                    );
                }
            }
        }
    }
    let episodes = worker::load_episodes(&w.out)?;
    std::fs::write(
        w.out.join("feed.xml"),
        worker::render_feed(&w.feed_title, &w.public_url, &episodes),
    )?;
    println!("feed.xml: {} episodes", episodes.len());
    Ok(())
}

/// Stories waiting under any of `labels` (default: narrate, narrate-scary), newest first, each once.
fn waiting(
    fr: &mut FreshRss,
    labels: &[String],
    per_label: usize,
) -> Result<Vec<(String, freshrss_tts::freshrss::Entry)>> {
    let labels = if labels.is_empty() {
        vec!["narrate".to_string(), "narrate-scary".to_string()]
    } else {
        labels.to_vec()
    };
    let mut found: Vec<(String, freshrss_tts::freshrss::Entry)> = Vec::new();
    for label in &labels {
        for e in fr.label_entries(label, per_label)? {
            if !found.iter().any(|(_, f)| f.id == e.id) {
                found.push((label.clone(), e));
            }
        }
    }
    Ok(found)
}

/// Swap `from` for `to` on a story. Failing is not fatal: for a narrated story the episode file
/// exists, so the next pass only retries this step.
fn relabel(fr: &mut FreshRss, e: &freshrss_tts::freshrss::Entry, from: &str, to: &str) {
    if let Err(err) = fr.edit_labels(std::slice::from_ref(&e.id), &[to], &[from]) {
        eprintln!("could not relabel {:?}: {err:#}", e.title);
    }
}

/// Tell the user (ntfy, if configured). A failed notification never fails the work.
fn notify(w: &WorkOpts, title: &str, message: &str, click: Option<&str>, play: Option<&str>, tags: &[&str]) {
    if let Some(ntfy) = &w.ntfy {
        if let Err(e) = ntfy.publish(title, message, click, play, tags) {
            eprintln!("could not notify: {e:#}");
        }
    }
}

/// Minimal HTTP endpoint for KEDA's `metrics-api` scaler: every request gets `{"queue": N}`.
fn queue_server(listen: &str, mut fr: FreshRss, labels: &[String]) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind(listen).with_context(|| format!("listening on {listen}"))?;
    println!("serving the queue length on {listen}");
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let _ = stream.set_read_timeout(Some(std::time::Duration::from_secs(5)));
        // drain the request head; the answer does not depend on it
        let mut reader = BufReader::new(&stream);
        let mut line = String::new();
        while reader.read_line(&mut line).is_ok_and(|n| n > 2) {
            line.clear();
        }
        let answer = fr
            .login()
            .and_then(|()| waiting(&mut fr, labels, 50))
            .map(|found| found.len());
        let (status, body) = match answer {
            Ok(n) => ("200 OK", json!({"queue": n}).to_string()),
            Err(e) => {
                eprintln!("queue lookup failed: {e:#}");
                (
                    "503 Service Unavailable",
                    json!({"error": "lookup failed"}).to_string(),
                )
            }
        };
        let _ = write!(
            stream,
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
    }
    Ok(())
}

fn narrate_entry(
    w: &WorkOpts,
    engine: &mut Maya1LlamaCpp,
    llm: Option<&dyn Completer>,
    d: &DirectorArgs,
    label: &str,
    e: &freshrss_tts::freshrss::Entry,
) -> Result<Episode> {
    let sentences = pipeline::prepare(&e.html);
    if sentences.is_empty() {
        bail!("the story has no text");
    }
    let slug = worker::slug(&e.id);
    let mut opts = d.options();
    if label == "narrate-scary" && opts.hint.is_none() {
        opts.hint = Some("a scary story: build dread, hushed delivery in tense moments".into());
    }
    let wav = w.out.join(format!("{slug}.wav"));
    let audio_file = format!("{slug}.opus");
    let res = narrate_story(engine, &sentences, llm, &opts, &wav, &audio_file)?;
    let opus = w.out.join(&audio_file);
    freshrss_tts::audio::encode(&wav, &opus, &w.bitrate)?;
    std::fs::remove_file(&wav)?;
    let ep = Episode {
        slug: slug.clone(),
        title: e.title.clone(),
        url: e.url.clone(),
        source: e.feed.clone(),
        published: e.published,
        audio_file,
        audio_bytes: std::fs::metadata(&opus)?.len(),
        duration_s: res.audio_seconds,
    };
    // written last: its presence means the story is done
    std::fs::write(
        w.out.join(format!("{slug}{}", worker::EPISODE_SUFFIX)),
        serde_json::to_string_pretty(&ep)?,
    )?;
    Ok(ep)
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
