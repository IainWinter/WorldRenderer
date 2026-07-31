param([switch]$Serve)

$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot

cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --target web --no-typescript --out-dir pkg target/wasm32-unknown-unknown/release/worldrenderer.wasm
cargo fmt

if ($Serve) {
    python -m http.server 8080
}
