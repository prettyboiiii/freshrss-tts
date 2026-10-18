# Design

## Goals and constraints

- Uncapped, local narration of FreshRSS stories with real emotional range (scary stories are the test).
- Hardware: debian node, i7-7700HQ (4C/8T), 35 GB RAM, GTX 1050 with 4 GB VRAM (Pascal). The host needs NVIDIA driver 580 or newer: 550 only supports CUDA 12.4, and the CUDA 12.8 llama.cpp image needs 12.8.
- Batch is fine. Real-time is not required.
- Rust, no Python in the product.

## Pipeline

1. **Select**: FreshRSS label (`narrate`, `narrate-scary`) via the Google Reader API.
2. **Prepare** (`textprep`): HTML to paragraphs to numbered sentences. Abbreviations, dialogue tags and
   ellipses are handled so a sentence is a real unit of speech.
3. **Direct** (`director`): a local LLM (OpenAI-compatible, JSON-schema-constrained) returns
   `{"narrator_voice": "...", "events": [{"i": 12, "mood": "dread", ...}]}`.
   - Events are **sparse**: `mood` and `delivery` persist until changed; `beat` and `pause_ms` apply to one
     sentence. A 100-sentence story needs a few hundred output tokens, which keeps a CPU-only LLM viable.
   - Closed vocabulary: moods (`neutral calm uneasy dread tense fearful panicked sorrowful angry relieved`),
     deliveries (`normal hushed whisper raised`), beats (`gasp sigh cry laugh chuckle giggle`).
   - Long stories are windowed; mood, delivery and narrator voice carry across windows.
   - **Invariant:** the model never produces story text. Sentences are copied from the input, so the
     narration cannot drift from the story. Bad JSON, unknown words, out-of-range indices and absurd
     pauses are ignored or clamped (`Plan::repaired` counts them). A dead LLM yields neutral narration.
4. **Render** (`engines::maya1`): group sentences into generations (about 300 characters, about 20 s).
   Mood becomes words in the voice description (`<description="...">`); whisper/raised become inline
   `<whisper>`/`<angry>` tags; beats become `<gasp>` etc. A pause of 600 ms or more ends a generation so
   the silence is real.
   - Quoted speech: the director may name a `speaker` (a voice description of identity only, such as
     "boy, about 12, higher pitch") for a sentence with quotes; those spans are generated in that voice.
     Quotes the narrator speaks (`I told him, "..."`) always stay in the narrator's voice (code rule).
   - Maya1 cannot clone a voice, so every generation of one voice uses the same sampling seed
     (`voice_seed`) to keep it steadier across chunks. This helps but does not remove drift.
   - Pauses: defaults come from punctuation (250 ms sentence end, 350 ms `?`/`!`, 500 ms ellipsis, 900 ms
     paragraph end). Gaps inside one generation are the model's own. Dramatic pauses (1100 ms or more) are
     kept rare in code.
5. **Synthesise** (`engines::maya1_llamacpp`): prompt token ids go to `llama-server /completion` (GGUF on
   the GPU), tokens stream back until the end-of-speech token, 7-token frames are unpacked into the three
   SNAC levels and decoded to 24 kHz audio by candle on the CPU.
6. **Assemble** (`audio`): trim dead air at the edges of each chunk, join with silences, write WAV and a
   `.timings.json` (sentence start/end, estimated inside a chunk) that the output page uses for a
   read-along (ffmpeg encoding to mp3/opus exists in
   `audio::encode`, not yet wired into a worker).

## What is verified, and how

| Claim | Evidence |
| --- | --- |
| Sentence splitting, HTML cleaning, text preservation | Unit tests |
| Director merges events correctly and cannot alter text | Unit tests including hostile LLM output |
| Maya1 prompt layout and 7-token SNAC layout | Ported from the reference code ([maya1-finetuning](https://github.com/gokhaneraslan/maya1-finetuning)); round-trip tests |
| SNAC decoder: 7-token frame to 2048 samples, shapes, file loading | Tests with the real architecture and zero weights |
| llama-server client | Tests against a mock built from llama.cpp's documented API |
| FreshRSS client | Tests against a mock built from reading FreshRSS's `greader.php` |
| CLI wiring | Tests that run the binary against mocks, SNAC directory fabricated |
| Maya1 GGUF runs on the GTX 1050 (llama.cpp `server-cuda`, CUDA 12.8, driver 580.178.04) | Live run in the cluster: about 21 tokens/s, about 2.7 GB VRAM, no truncation in `bench` |
| Emotion tags exist in the GGUF | `tags` probe against the live server |
| Real SNAC weights decode to audio, real llama-server streaming (`return_tokens`) | Full stories narrated end to end |
| FreshRSS login, label listing, story fetch (`check`) | Run against the real instance; the API password lives in a SOPS secret |
| Reddit "submitted by" footer is dropped, quotes split into spans | Unit tests plus a real story |
| Speed | Whole story: real-time factor about 6-9 including about 4 minutes of CPU director time (generation dominates; SNAC decode does not) |
| Voice and pacing quality | The user listened to full stories: "pretty good, not perfect" after the seed, pause and narrator-quote fixes |

## What is NOT verified

- Voice drift between generations is reduced but not gone, and character voices can still sound
  different from one line to the next. Judged by ear only.
- Whether 4-bit quantization hurts audio quality (compare Q4 with a larger quant).
- Whether llama-server flags (`-np 1`, flash attention, KV quantisation) raise tokens/s. GPU temperature
  reached 87 C under load.
- The 3-4B director's judgement is mediocre: it rarely uses whisper or beats and sometimes mislabels
  speakers. Code rules (narrator quotes, rare long pauses) cover the worst cases.
- FreshRSS `edit-tag` creating a missing label and the label id formats (needed by the worker).
- The worker, podcast feed and scale-to-zero are not built yet.

## Decisions and alternatives

- **Maya1 via llama.cpp, not PyTorch**: current PyTorch wheels dropped Pascal (2.14 is the last with
  them) and fp16 on a 1050 is slow. llama.cpp supports it and quantized GGUFs fit in 4 GB.
- **Ruled out**: Fish S1-mini (reported about 5 GB, being retired), Chatterbox (sound tags only, no tone
  control), Orpheus (no whisper tag), Kokoro (no emotion control; fine as a plain fallback).
- **Backup if Maya1 disappoints**: Qwen3-TTS 1.7B (instruction-controlled), via its llama.cpp port.
- **No FreshRSS extension in v1**: labels and stars work from every client. An inline player can come later.
