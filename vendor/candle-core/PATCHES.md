# Patched copy of candle-core 0.10.2

Upstream: https://github.com/huggingface/candle (MIT OR Apache-2.0, see LICENSE).

Why this exists: every published `zip` release before 9.0.3 is yanked on crates.io, and every released
`candle-core` (0.9.x to 0.11.0) depends on one of them, so a fresh `cargo build` cannot resolve. Changes
against the 0.10.2 crate from crates.io:

* `Cargo.toml`: `zip = "7.2.0"` -> `zip = "9.0.4"`; examples, tests, benches and dev-dependencies removed.
* `src/npy.rs`, `src/pickle.rs`: adapt to zip 9, where `ZipFile::name()` and `ZipArchive::file_names()`
  return `Result`s (three call sites). These code paths read `.npz`/`.pth` files, which this project
  never uses; it only loads safetensors.

Delete this directory and the `[patch.crates-io]` entry in the root `Cargo.toml` once a candle release
depends on a non-yanked `zip`.
