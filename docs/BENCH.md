# Bake-off: does Maya1 work on your GTX 1050?

Goal: find out, before building anything more, whether local Maya1 gives you a narrator you want to
listen to, how fast it is, and whether it fits in 4 GB. About an hour including downloads.

You need the debian machine with: Rust (`rustup`), this repo built (`cargo build --release`), and
`llama-server` from [llama.cpp](https://github.com/ggml-org/llama.cpp).

## 1. Get llama-server with GPU support

Any one of these (see llama.cpp's `docs/build.md` and `docs/docker.md`):

- **Docker + NVIDIA container toolkit**: image `ghcr.io/ggml-org/llama.cpp:server-cuda`. I could not
  confirm that the prebuilt image includes the Pascal architecture (`sm_61`); if it fails to use the
  GPU, build from source instead.
- **Build from source with CUDA** (needs the CUDA toolkit, not just the driver):
  `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build build --config Release -j 8`
- **Build with Vulkan** (works with the plain NVIDIA driver; needs Vulkan headers and `glslc`):
  `cmake -B build -DGGML_VULKAN=ON && cmake --build build --config Release -j 8`
- A prebuilt Linux binary from the [releases page](https://github.com/ggml-org/llama.cpp/releases), if
  one matches your setup.

## 2. Download the models

```sh
mkdir -p models/snac && cd models/snac
curl -LO https://huggingface.co/hubertsiuzdak/snac_24khz/resolve/main/config.json
curl -L -o snac_24khz.safetensors https://huggingface.co/lmz/candle-snac/resolve/main/snac_24khz.safetensors
cd ..
```

Maya1 GGUF: [Mungert/maya1-GGUF](https://huggingface.co/Mungert/maya1-GGUF). I could not list its files
from my environment; download a `Q4_K_M` and a higher-bit one (`Q6_K` or `Q5_K_M`) to compare. Larger
than about 3 GB will not fit your card with its context.

## 3. Check the emotion tags are real

```sh
llama-server -m models/maya1-Q4_K_M.gguf -ngl 99 -c 4096 --port 8080     # leave running
freshrss-tts tags
```

Every `used` tag must say `single token`. If one does not, the GGUF/tokenizer does not know it as a tag
(the program exits non-zero). `probe` lines show other tag names; any that are single tokens can be added
to `src/engines/maya1.rs`.

## 4. Run the bake-off

```sh
freshrss-tts bench --out out/bench
```

It writes six WAVs, three passages each **plain** and **hand-directed**, plus `metrics.json`, and prints
a table: audio seconds, wall seconds, real-time factor (RTF, above 1.0 is slower than real time),
tokens per second, and peak GPU memory. Watch `nvidia-smi` too.

Listen to the pairs (`01`/`02`, `03`/`04`, `05`/`06`) and answer:

1. Does the directed one sound more like a scary-story narrator than the plain one? (Whispers audible?
   Gasp present? Mood change from the description?)
2. Voice consistency: within `04` and `06` the voice changes between generations. Is the drift tolerable?
3. Any `YES (cut off)` in the table, garbled words, or silence glitches?
4. Speed: RTF and tokens/s. About 83 tokens per second of audio are needed for real time.
5. Memory: peak GPU MiB (the card has 4096).
6. Repeat with the higher-bit GGUF: is the audio audibly better?

Sampling knobs if something is off: `--temperature`, `--top-p`, `--repeat-penalty`, `--seed` (fixed seed
for A/B comparisons).

## 5. Try the director (no GPU needed)

Start a second llama-server on the CPU with a small instruct model (I have not benchmarked these;
a Qwen3 4B Instruct, Phi-4-mini or Llama 3.2 3B GGUF are the candidates to try):

```sh
llama-server -m models/qwen3-4b-instruct-Q4_K_M.gguf -ngl 0 -c 8192 -t 4 --port 8081
freshrss-tts plan story.html --llm-url http://127.0.0.1:8081 --hint "old man whispering, hospital setting"
```

`plan` prints exactly what Maya1 would receive. Judge: are the moods sensible, are whispers rare and
placed well, is the narrator voice description reasonable, and what does "LLM fixes" say (how often the
model broke the rules)? Then the whole pipeline:

```sh
freshrss-tts narrate story.html --llm-url http://127.0.0.1:8081 --out out/story.wav
```

Adding `--llm-url` to `bench` also produces `07_director.wav` from the same passages.

## 6. Measure SNAC decoding on your CPU

```sh
cargo test --release -- --ignored --nocapture snac_cpu_decode_speed
```

Prints the RTF of the CPU audio decoder for 2.5 to 20 s of audio. In this project's tests on a 4 vCPU
Xeon it was about 1.1 to 1.5, i.e. slower than real time, so it may be the slowest stage. Building for
AVX2, which your i7-7700HQ supports, was 15-20% faster there (RTF 0.9 to 1.3):

```sh
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release    # will not run on CPUs without AVX2
```

Tell me your number.

## What to send back

The `tags` output, the bench table, peak VRAM, your listening verdict on questions 1-3, your SNAC RTF,
and the `plan` output for a real story. That decides: go ahead with Maya1, try Qwen3-TTS 1.7B, or
rethink (a bigger GPU or a capped cloud service for stories only).
