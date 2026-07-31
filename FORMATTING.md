# Formatting

Code style is not a concern while writing. The formatter is the single authority. Run it once, after all code for a task is written.

## Language

Rust (compiled to `wasm32-unknown-unknown`), WGSL shaders.

## Formatter

- Tool: `rustfmt`
- Config: `rustfmt.toml`
- Command: `cargo fmt`

WGSL files have no formatter. Match the surrounding style by hand.

## Build

```
./build.ps1            # cargo build --release + wasm-bindgen + cargo fmt
./build.ps1 -Serve     # same, then serves on :8080
```

## Rule

Formatter output is final. Do not hand-tune what it produces. Do not argue with it.
