# freshrss-tts

Narrate [FreshRSS](https://freshrss.org) stories on your own hardware, with emotion and no usage caps.

A small local LLM reads the story and decides *how* it should be performed (mood, whispers, gasps,
pauses). Code inserts those directions into the **unchanged** text, and a local expressive TTS
([Maya1](https://huggingface.co/maya-research/maya1)) reads it. Built for a modest GPU (GTX 1050, 4 GB),
batch-style: it is slower than real time, so think "narrated overnight", not "read aloud now".

> **Status: pre-bake-off.** Everything below is covered by tests against mock servers, but it has not
> yet been run against a real Maya1 model, real SNAC weights or a live FreshRSS. Run the bake-off
> ([docs/BENCH.md](docs/BENCH.md)) before building on it. [docs/DESIGN.md](docs/DESIGN.md) lists exactly
> what is and is not verified.

## How it works

```
FreshRSS label ──► HTML ──► sentences ──► director LLM ──► directed sentences ──► Maya1 chunks ──► WAV
 (narrate,         textprep   (numbered)   (sparse JSON      (text untouched;     llama-server      SNAC
  narrate-scary)                            events, schema-   tags + voice         (GPU, GGUF)       decode
                                            constrained)      description added)                     (CPU)
```

- **Selection** is a FreshRSS label (or star) applied from any client; the worker polls the Google Reader
  API. Labels name the style, e.g. `narrate` and `narrate-scary`.
- **The director never rewrites the story.** It returns a sparse list of events from a closed vocabulary;
  anything invalid degrades to neutral narration instead of failing the job.
- **Two processes share the load:** the language model runs on the GPU in `llama-server` (llama.cpp, so
  Pascal cards work), the SNAC audio decoder runs on the CPU in this program, and the director LLM runs
  on the CPU in a second `llama-server`.
- Optional `--hint "old man whispering, hospital setting"` adds extra direction; otherwise it is automatic.

## Commands

| Command | What it does |
| --- | --- |
| `freshrss-tts check` | Log in to FreshRSS and list entries under the narrate labels |
| `freshrss-tts plan FILE` | Show what the TTS would receive for a story (director only, no TTS needed) |
| `freshrss-tts tags` | Check which emotion tags are real single tokens in your Maya1 GGUF |
| `freshrss-tts narrate FILE` | Narrate a text/HTML file to a WAV, with timing and peak GPU memory |
| `freshrss-tts bench` | Fixed plain-vs-directed samples to compare by ear, plus speed and VRAM |

Settings come from flags or environment variables (`FRESHRSS_URL`, `FRESHRSS_USER`,
`FRESHRSS_API_PASSWORD`, `DIRECTOR_URL`, `DIRECTOR_MODEL`, `MAYA1_URL`, `SNAC_DIR`); see `--help`.

## Build and test

```sh
cargo test                      # ~1 min the first time; needs ffmpeg for one smoke test
cargo build --release
```

Rust 1.80+ features are used (tested with a current stable toolchain). The SNAC decoder is
[candle](https://github.com/huggingface/candle)'s, and `vendor/candle-core` is a patched copy of
candle-core 0.10.2: every `zip` release before 9.0.3 is yanked on crates.io, so unpatched candle cannot
be resolved. See [vendor/candle-core/PATCHES.md](vendor/candle-core/PATCHES.md); delete it once candle
releases against a non-yanked `zip`.

## Roadmap

1. Bake-off on the target machine (this repo, now).
2. Worker: poll labels, narrate, publish a podcast feed, swap `narrate` for `narrated`.
3. Container image and Kubernetes manifests (GPU worker scaled to zero by KEDA when the queue is empty).

## License

Apache-2.0, see [LICENSE](LICENSE). Model weights are not included and keep their own licenses (Maya1:
Apache-2.0; SNAC: MIT).
