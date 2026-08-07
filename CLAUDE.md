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

Rust, `wasm32-unknown-unknown`, wgpu on the WebGPU backend, falling back to WebGL2 when
`navigator.gpu.requestAdapter()` returns null. Shaders are WGSL in `src/shaders/`.

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

Tools take over the pointer when active (`set_tool`): `place` drops balls, `select` drags a
rectangle. The selection is an overlay — one constant height taken from the anchor's terrain
hit, depth compare `Always`, so it draws over relief instead of draping on it, and the shader
drops vertices past the horizon so it never shows through the far side of the globe. Its axes
come from the camera's screen axes flattened onto the tangent plane, so it is camera-oriented,
not north-aligned; the dragged corner is the ray hit on the overlay's own height surface
(`ellipsoid_entry_at`), which is what keeps the corner under the cursor at any tilt.

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
- `map_async` never completes on the WebGL2 backend unless the device is polled. Any readback
  loop must call `device.poll(PollType::Poll)` while it waits; it is a no-op under WebGPU.
- The imagery atlas layer count must never be a multiple of 6. It is square, so wgpu-hal's
  GL backend calls that "cube compatible" and binds `GL_TEXTURE_CUBE_MAP_ARRAY`, which does
  not exist in WebGL2 - every upload fails with `INVALID_ENUM` and the globe goes black with
  no other symptom. `TerrainRenderer::new` drops the cap by one when it lands on a multiple.
- Anything that allocates a mesh slot or imagery layer must free the old one on replace,
  or the arena leaks and the globe silently stops loading.

## Data

- Terrain: AWS Terrain Tiles, terrarium PNG, max z15.
- Imagery: EOX Sentinel-2 cloudless WMTS, max z14. CC BY 4.0 — keep the attribution in `index.html`.
- Every worker fetch goes through the Cache Storage bucket `worldrenderer-tiles-v1` (persists across
  reloads, secure context only - `http://localhost` or https). Budget defaults to 16384 MB, set with
  `set_cache_limit_mb` (0 disables); over budget it evicts oldest-first down to 90%. `clear_cache`,
  `cache_usage_mb`.
