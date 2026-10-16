# Design

## Goals and constraints

- Uncapped, local narration of FreshRSS stories with real emotional range (scary stories are the test).
- Hardware: debian node, i7-7700HQ (4C/8T), 35 GB RAM, GTX 1050 with 4 GB VRAM (Pascal, driver 550).
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
5. **Synthesise** (`engines::maya1_llamacpp`): prompt token ids go to `llama-server /completion` (GGUF on
   the GPU), tokens stream back until the end-of-speech token, 7-token frames are unpacked into the three
   SNAC levels and decoded to 24 kHz audio by candle on the CPU.
6. **Assemble** (`audio`): join chunks with silences, write WAV (ffmpeg encoding to mp3/opus exists in
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

## What is NOT verified (the bake-off decides these)

- Maya1 quality on a story: tag adherence, whether mood descriptions change delivery, **voice drift
  between generations**, long-form stability.
- Speed and VRAM of Maya1 Q4 on the GTX 1050 (arithmetic suggests 2-3x slower than real time).
- Whether 4-bit quantization hurts audio quality (compare Q4 with a larger quant).
- Real SNAC weights (`lmz/candle-snac`) loading and sounding right.
- SNAC CPU decode speed on your CPU: about 1.1-1.5x real time on a 4 vCPU Xeon in this project's tests
  (`cargo test --release -- --ignored --nocapture snac_cpu_decode_speed` measures yours). It may
  dominate total time; options are overlapping it with generation, or running it on the GPU.
- Real llama-server streaming (`tokens` field with `return_tokens`), real FreshRSS (`edit-tag` creating
  a missing label, id formats). `freshrss-tts check` is the live test for FreshRSS.
- Quality of a 3-4B director LLM's choices.

## Decisions and alternatives

- **Maya1 via llama.cpp, not PyTorch**: current PyTorch wheels dropped Pascal (2.14 is the last with
  them) and fp16 on a 1050 is slow. llama.cpp supports it and quantized GGUFs fit in 4 GB.
- **Ruled out**: Fish S1-mini (reported about 5 GB, being retired), Chatterbox (sound tags only, no tone
  control), Orpheus (no whisper tag), Kokoro (no emotion control; fine as a plain fallback).
- **Backup if Maya1 disappoints**: Qwen3-TTS 1.7B (instruction-controlled), via its llama.cpp port.
- **No FreshRSS extension in v1**: labels and stars work from every client. An inline player can come later.
