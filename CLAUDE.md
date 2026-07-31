# WorldRenderer

High-performance 3D globe renderer. Rust → wasm → WebGPU.

## Rules

1. **Save every prompt.** Before doing anything else, write my message verbatim to `ai/prompts/<YYYYMMDD><HHMM>.md`. Exact text, no edits, no wrapper, no commentary.
2. **No comments in code.** Ever. Unless I ask for them.
3. **No tests.** Ever. Unless I ask. If a test is needed for in-development verification, write it, use it, then delete it.
4. **Ignore code style while writing.** The formatter fixes it after. See `FORMATTING.md`.
5. **Run the formatter** as the last step, after all code is written. `cargo fmt`.
6. **Keep it simple.** No speculative abstraction, no extra files, no scope creep.

## Language

Rust, `wasm32-unknown-unknown`, wgpu on the WebGPU backend. Shaders are WGSL in `src/shaders/`.

## Layout

| File | Role |
|---|---|
| `src/lib.rs` | wasm entry, app state, render loop, JS API |
| `src/gpu.rs` | device/surface/depth, upload budget, slot allocator |
| `src/camera.rs` | globe camera, relative-to-eye view, reverse-Z projection |
| `src/math.rs` | WGS84 ellipsoid, oct encoding, frustum, horizon culling |
| `src/tiling.rs` | Web Mercator tile keys, bounds, data source URLs |
| `src/quadtree.rs` | LOD selection, tile cache, eviction, upload integration |
| `src/stream.rs` | worker pool, job queue, inbox |
| `src/worker.rs` | worker-side entry: fetch, decode, mesh build |
| `src/terrain_mesh.rs` | grid topology, terrarium decode, mesh build, jpeg decode |
| `src/terrain_gpu.rs` | terrain pipeline, vertex arena, imagery atlas |
| `src/vector.rs` | polygon / line / icon pipelines and batches |
| `src/model.rs` | glTF (.glb) parsing and the worker transfer blob |
| `src/model_gpu.rs` | model pipeline, per-model buffers, instance transforms |
| `tests.html` | frontend behaviour suite - open it, it must stay all-green |
| `sources.js` | imagery presets and demo places |

## Controls

Left drag pans (the grabbed point stays under the cursor at any heading or tilt).
Wheel zooms toward the cursor. Right or shift drag tilts and rotates. WASD flies,
QE rotates, RF tilts. North is up at tilt 0.

## Invariants

- Nothing that touches the network or decodes an image runs on the main thread. Workers only.
- GPU uploads go through `UploadBudget`. Never upload unbounded work in one frame.
- Terrain vertex positions are relative to their tile centre; tile centres are relative to the eye. Never put absolute ECEF in an f32.
- Depth is reverse-Z: clear to 0.0, compare `Greater`.
- `textureSample` must sit in uniform control flow. Sample unconditionally, then `select`.
- WGSL has reserved keywords that are easy to hit (`target`, `filter`, `sample`). A shader
  that fails to compile invalidates the whole render pass and the frame goes black with no
  error unless a logger is installed - `console_log` is initialised in `start` for that.
- LOD never pops: each vertex carries a morph target toward the parent grid and tiles blend
  into their parent as they shrink. Imagery cross-fades over `IMAGERY_FADE_FRAMES`.
- The camera eye is clamped above the terrain under it every frame.
- Anything that allocates a mesh slot or imagery layer must free the old one on replace,
  or the arena leaks and the globe silently stops loading.

## Data

- Terrain: AWS Terrain Tiles, terrarium PNG, max z15.
- Imagery: EOX Sentinel-2 cloudless WMTS, max z14. CC BY 4.0 — keep the attribution in `index.html`.
