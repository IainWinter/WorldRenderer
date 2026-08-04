#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

export PATH="/opt/homebrew/opt/rustup/bin:$HOME/.cargo/bin:$PATH"

cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir pkg target/wasm32-unknown-unknown/release/worldrenderer.wasm
cargo fmt

lsof -ti:8080 | xargs kill -9 2>/dev/null || true
exec python3 -m http.server 8080
