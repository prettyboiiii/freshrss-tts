# syntax=docker/dockerfile:1
FROM rust:1-bookworm AS build
WORKDIR /src
# x86-64-v3 (AVX2/FMA) makes the CPU SNAC decoder ~15-20% faster; every x86 node here has it.
# Build with --build-arg TARGET_CPU=x86-64 for a fully portable binary.
ARG TARGET_CPU=x86-64-v3
ENV RUSTFLAGS="-C target-cpu=${TARGET_CPU}"
COPY Cargo.toml Cargo.lock ./
COPY vendor ./vendor
COPY src ./src
RUN cargo build --release --locked --bin freshrss-tts

FROM debian:bookworm-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates ffmpeg \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/target/release/freshrss-tts /usr/local/bin/freshrss-tts
USER 65532:65532
ENTRYPOINT ["freshrss-tts"]
