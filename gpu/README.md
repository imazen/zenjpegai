# zenjpegai-gpu

WebGPU (`wgpu` 30) synthesis backend for `zenjpegai`. The entropy stage, hyper-decoder and context
model stay on the CPU (the entropy stage has to be bit-exact and the whole lot is a few
milliseconds); both synthesis transforms (SOP, BOP, HOP; luma and chroma) run as WGSL compute
shaders, tile by tile, with no readback between layers.

Builds for native targets (Vulkan / Metal / DX12) and `wasm32-unknown-unknown` (browser WebGPU).
The core crate does not depend on `wgpu`.

Status, measured parity and what is missing: `PORTING.md` (row `gpu/`). Timings:
`benchmarks/gpu_decode_*.tsv`.

## API for the browser build

Everything that waits for the GPU is `async` and never blocks (in the browser `map_async`
resolves from the event loop; on native the same futures drive `device.poll`). Blocking wrappers
(`GpuContext::new`, `GpuDecoder::decode`) exist on native only.

```rust
use std::sync::Arc;
use zenjpegai::model::ModelBundle;
use zenjpegai_gpu::{Blitter, ContextOptions, GpuContext, GpuDecoder, GpuError};

// Once per page. Err(GpuError::NoAdapter) = no WebGPU (or only a software adapter): use the CPU
// decoder. For strict mode (bit-identical to the CPU engine) also use the CPU decoder: the GPU
// path is within the same bounds against the reference but not bit-identical to the CPU path.
let ctx = Arc::new(GpuContext::new_async(&ContextOptions::default()).await?);
let decoder = GpuDecoder::new(ctx.clone(), Box::new(bundle /* ModelBundle */));
decoder.preload(model_id, op)?; // optional: parse + upload weights before the first picture

// (a) pixels on the CPU: RGB (or YUV planes), same output stage as the CPU decoder.
let picture = decoder.decode_picture_async(&stream).await?;

// (b) no float readback: keep the picture on the GPU and draw it.
let decoded = decoder.decode_to_gpu(&stream)?;      // CPU stages + GPU work queued; returns at once
if decoded.presentable_on_gpu() {                   // 8-bit 4:4:4 BT.709, no post-filters
    let tex = decoded.to_rgba_texture()?;           // rgba8unorm, display size
    // canvas: surface = instance.create_surface(SurfaceTarget::Canvas(canvas)), configured with
    // ctx.device(); then each frame:
    let frame = surface.get_current_texture();
    Blitter::new(&ctx, surface_format).blit(&ctx, &tex, &frame.texture.create_view(&Default::default()));
    frame.present();
    // or, without a WebGPU canvas: zenjpegai_gpu::read_rgba8(&ctx, &tex).await -> RGBA bytes
    // for ImageData (4 bytes per pixel read back instead of 12).
} else {
    let (picture, _, _) = decoder.finish(decoded).await?; // post-filters / subsampled / 10 bit
}
```

If the surface must share the device, create the adapter yourself
(`instance.request_adapter` with `compatible_surface`) and pass it to `GpuContext::from_adapter`,
or wrap an existing device with `GpuContext::from_device`.

Notes for the wasm build:

- Depend with `default-features = false` (the defaults turn on `zenjpegai/parallel`, i.e. rayon,
  and `avx512`); add your own thread-pool feature wiring if the page is cross-origin isolated.
- `GpuContext` and everything holding `wgpu` handles is `!Send` on wasm. WebGPU works in dedicated
  workers (`WorkerNavigator.gpu`, enabled in the `wgpu` features here), so the whole decoder can
  live in a worker; presenting from a worker needs an `OffscreenCanvas`.
- Errors other than `GpuError::Codec` mean "fall back to the CPU decoder" (`NoAdapter`, `Device`,
  `TooLarge`: a feature map exceeds the adapter's `maxStorageBufferBindingSize`; the context
  requests up to 1 GiB. Measured largest map for a 1024 x 1024 tile: SOP 8 MiB, BOP 24 MiB, HOP
  128 MiB, i.e. WebGPU's default limit; total GPU memory held after that tile: 52 / 100 / 537 MiB).
- The first picture of each size compiles pipelines and builds bind groups (`Timing::plans_built`);
  later pictures of the same size replay cached plans.

## Status (2026-09-17, stopped early on a budget change)

Done and gated (`just gpu-test`, numbers in `PORTING.md`): kernel set, SOP / BOP / HOP synthesis
with tiling and regions, `GpuDecoder`, presentation path, wasm32 build, benchmark harness.

**Open, in order:**

1. **No hardware GPU has run this code.** On the dev box the render nodes (`/dev/dri/renderD12*`,
   an RTX 2080 on nouveau/NVK and the Radeon iGPU) are not accessible to the login user (not in
   the `render` group), so Vulkan only offers llvmpipe. All parity numbers and the committed
   `benchmarks/gpu_decode_2026-09-17_llvmpipe.tsv` come from that software rasteriser: they prove
   correctness and that the harness works, and say nothing about GPU speed or the CPU / GPU
   crossover size. Next: `sudo usermod -aG render,video $USER` (new login), then
   `ZENJPEGAI_GPU_ADAPTER=<name> just gpu-test` and
   `cargo run --release -p zenjpegai-gpu --example gpu_bench -- --adapter <name> --out benchmarks/gpu_decode_<date>.tsv`
   (+ a `.meta` with GPU, driver, wgpu version, commit).
2. **Kernel tuning has not started** (it needs item 1): `OB` (output blocks per invocation,
   `layers.rs::pick_ob`, now at most 4), workgroup size (`kernels.rs::WG`), a specialised
   stride-2 transposed convolution without the parity branches, fusing the ResAU gate into the
   1x1 convolution, `nsys` / timestamp breakdown per layer. `shader-f16` is not implemented.
3. **Browser**: never run in a browser (no WebGPU-capable browser on the box). The web build has to
   call the API below; nothing in `wasm/` or `web/` references this crate yet.
4. Latent upload converts planar to HWC4 on the CPU per picture (0.4 ms for 560 x 888); the
   float readback is 12 bytes per pixel (use the texture path to avoid it).
5. No cancellation (`enough::Stop`) on the GPU path; `GpuDecoder` has no `max_channels`
   (progressive decode) option.

## Tests and benchmarks

```
just gpu-test                          # needs an adapter, the checkpoints and the reference vectors
ZENJPEGAI_GPU_ADAPTER=nvidia just gpu-test
ZENJPEGAI_GPU_ALLOW_SOFTWARE=1 just gpu-test   # llvmpipe / SwiftShader / WARP
cargo run --release -p zenjpegai-gpu --example gpu_bench -- --out benchmarks/gpu_decode_<date>.tsv
```

`tests/kernels.rs` checks every kernel against the core crate's plain-loop oracle;
`tests/decode_ref.rs` runs whole reference streams (same gate as the CPU engine) and the
presentation path.

## Layout

- `src/kernels.rs`: WGSL generators. Feature maps are `array<vec4<f32>>` in HWC4 order, weights
  are `mat4x4` blocks in the order the shader walks them, one invocation computes up to 16 output
  channels of one pixel.
- `src/plan.rs`: records a network into a list of dispatches, assigns tensors to pooled buffers by
  lifetime, replays it as one compute pass.
- `src/synthesis.rs`: the transforms (names and crops mirror `zenjpegai::model::synthesis`),
  tiling, picture-level buffers, timing.
- `src/decoder.rs`, `src/present.rs`: whole-stream decoding, texture output, blit.
