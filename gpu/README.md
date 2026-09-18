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

## Status (2026-09-17, first hardware run and first tuning pass)

Done and gated (`just gpu-test` on an RTX 2080, numbers in `PORTING.md`): kernel set,
SOP / BOP / HOP synthesis with tiling and regions, `GpuDecoder`, presentation path, wasm32
build, benchmark harness, per-dispatch profiling.

### What the hardware says

`benchmarks/gpu_decode_2026-09-17_rtx2080.{tsv,meta}` (RTX 2080, NVIDIA 580.178.04),
`..._rtx2080_pretuning.tsv` for the same sweep before the tuning commit, `..._radv_igpu.tsv` for
the machine's integrated Radeon, `gpu_reference_2026-09-17.{tsv,meta}` for the reference
software on the same card. The box is shared with other agents; medians over seven interleaved
rounds absorb most of that but not all, and the `.meta` files name the rows that are visibly off.

- **The driver is the single biggest factor.** The same card on Mesa NVK (open) needs 20.4 ms of
  device time for a 64 x 64 SOP synthesis — i.e. per-dispatch overhead only — against 1.1 ms on
  the proprietary driver, and 28.8 ms against 3.7 ms at 560 x 888. Conclusions about this code
  cannot be drawn from an NVK run.
- **Crossover against the crate's own CPU engine** (8 rayon threads on a 9950X3D, so a
  conservative CPU): SOP at about 384 x 384 (3.08 ms against 3.17), BOP at about 256 x 256
  (3.29 against 4.25), HOP between 128 x 128 (10.40 against 9.92, CPU) and 256 x 256 (18.90
  against 35.19, GPU) — the heavier the transform, the smaller the picture at which the GPU
  wins. Below that the GPU's fixed cost (a whole-picture synthesis is 25 to 160 dispatches
  however small the picture) dominates: SOP 64 x 64 is 1.5 ms on the GPU against 0.4 ms on the
  CPU.
- **The win is modest, not a rout**: at 1024 x 1024 the GPU is 2.1x the 8-thread CPU for SOP,
  1.5x for BOP, 1.3x for HOP. Against a single CPU thread it is 6.0x / 6.0x / 4.3x. Whole
  reference streams at 560 x 888, codestream to 8-bit RGB (the `decode` rows, where the CPU-only
  entropy stage is included): 16.8 vs 17.9 ms (SOP), 22.7 vs 24.6 (BOP), 273.8 vs 387.2 (HOP).
- **Cold start is Vulkan, not shader compilation.** Creating the instance, adapter and device
  costs 180-210 ms once per process (884 ms on the very first process after a driver load);
  compiling all the pipelines of an operating point and building its plans costs 2.8 ms (BOP,
  9 pipelines) to 8.3 ms (HOP, 23). A pipeline cache would therefore buy single-digit
  milliseconds, and `wgpu::Device::create_pipeline_cache` is `unsafe`, which this crate forbids.

### Tuning done, with numbers

Four changes, chosen from the per-dispatch profile (`gpu_bench --profile`, committed as
`benchmarks/gpu_profile_2026-09-17_rtx2080.{tsv,meta}` for both sides). The numbers are device
time from a before / after profile pair run back to back on an idle card, each the second run of
its binary (the first run after a build measures low clocks and is not comparable).

| change | what it does | HOP 1024 x 1024 |
| --- | --- | --- |
| `Graph::conv_res` | computes the ResAU gate and the residual add in the convolution's store instead of a pointwise dispatch over the whole map | dispatches per picture SOP 36 -> 30, BOP 33 -> 27, HOP 172 -> 158 |
| transposed convolution | visits only the taps whose divisibility test used to pass: 4 of 16 at k=4 stride 2, at most 4 of 9 at k=3 | `convt_k3_s2` 36.7 -> 28.1 ms |
| depthwise 3x3 | stages its `(WG+2)^2` halo in workgroup memory, 1600 bytes, one channel block per workgroup: 9 reads per output pixel become 1.56 | `depthwise3x3` 134.4 -> 64.8 ms |
| workgroup edge per layer | 16 x 16 instead of 8 x 8 where the layer has at least 16 input blocks and both output dimensions are at least 128, so each broadcast weight block is shared by four times as many invocations | 1x1 convolutions 162.4 -> 117.3 ms, the 3x3 family 231.6 -> 204.0 ms |

Whole-picture device time, same pair:

| | SOP 560x888 | SOP 1024² | BOP 560x888 | BOP 1024² | HOP 560x888 | HOP 1024² |
| --- | --- | --- | --- | --- | --- | --- |
| before | 3.76 ms | 7.35 ms | 10.80 ms | 23.91 ms | 259.4 ms | 635.3 ms |
| after | 3.76 ms | 7.41 ms | 9.67 ms | 21.31 ms | 223.5 ms | 498.0 ms |
| | 0% | 0% | **-10%** | **-11%** | **-14%** | **-22%** |

**SOP does not move**, and saying so is the point of measuring: its network has no depthwise
layer, its upsampling is a 2x2 convolution plus a pixel shuffle rather than a transposed
convolution, and its convolutions run on maps too small for the wider workgroup, so the only
change that reaches it is the gate / residual fusion — six dispatches out of thirty-six, none of
them the expensive ones.

Every one of the four keeps each output's summation order, so `just gpu-test` reports exactly
the same parity numbers before and after — that is the check that they are speed-ups and not
approximations.

Also fixed on hardware: the presentation path now rounds in the shader (half to even, like the
CPU output stage) instead of trusting the driver's float-to-`rgba8unorm` rounding, which on
NVIDIA disagreed on 46533 of 1491840 samples (28 after the fix, all by one step). llvmpipe had
agreed, so this only appeared on hardware.

**Falsified, do not re-try as-is:** register tiling over *pixels* in the convolution (2x2 and
2x1 outputs per invocation, which amortises the `4 * ob` weight loads of each tap). It helps SOP
(1024 x 1024: 8.48 -> 7.23 ms) and loses where it matters — BOP 22.4 -> 26.3 ms, HOP 555 -> 711
ms — and a rule that only tiles convolutions with few input blocks is worse than not tiling at
all. The cost tracks `ob * px * py` accumulators, i.e. register pressure, so the next thing to
try is a *smaller* `ob` together with a pixel tile, not a bigger tile.

**Open, in order:**

1. **The convolutions run at a few per cent of the card's f32 peak, and the tuning did not
   change that** — HOP's biggest single dispatch, the 3x3 stride-2 of the CAB, is 9.66 G
   multiply-adds in 49.1 ms before and 44.6 ms after, i.e. 0.39 then 0.43 TFLOP/s against about
   10. After the tuning the profile says HOP 1024x1024 is 24% 1x1 convolution, 13% depthwise and
   34% the rest of the 3x3 family. Neither weight bandwidth nor arithmetic explains the gap, so
   the next step is to find what does — occupancy, warp stall reasons — before writing another
   kernel. The CUDA profilers cannot: these are Vulkan compute shaders, `ncu` prints "No kernels
   were profiled" on this binary and the `nsys` installed here cannot load its Vulkan importer.
   Nsight Graphics (GPU Trace) is the tool and is not installed on this box; until it is, the
   measurements available are `gpu_bench --profile` and A/B runs of kernel variants.
   Candidates named but not measured: workgroup-memory tiling for the 3x3 (it needs
   input-channel chunking, which changes the summation order — the parity gate would have to be
   re-measured), a GEMM-shaped 1x1 with workgroup-staged weights, `OB` other than
   `layers.rs::pick_ob`'s at-most-4, `shader-f16` (not implemented; parity would have to be
   re-measured).
2. **Readback dominates large pictures**: 4096 x 4096 SOP is 139 ms of device time and 227 ms of
   wall, the difference being 201 MB of `f32` planes over PCIe. The texture path
   (`decode_to_gpu` + `to_rgba_texture`) avoids it; the plane path could read back 8-bit instead.
3. **Browser**: wired into `wasm/` + `web/` (2026-09-18, `pkg-webgpu`): `initGpu`/`decode`/`present`
   work end to end under Chromium's WebGPU (verified through Dawn's SwiftShader adapter;
   no hardware WebGPU run yet). **Blocking API note**: `layers.rs::storage` uses
   `device.create_buffer_init`, which on the web backend calls `createBuffer` with
   `mappedAtCreation: true`; wgpu then `unwrap()`s the JS result (wgpu 30.0.1
   `backend/webgpu.rs:2461`). Dawn rejects `mappedAtCreation` buffers that exceed its staging
   limit, and on SwiftShader the failure is device-state dependent (a 921600-byte weight buffer
   failed on the decode following a canvas `present`, while the same call succeeded earlier).
   With `panic = "abort"` the unwrap trap escapes wasm-bindgen's executor: it cannot become a
   `GpuError`, so in-worker fallback never runs. `wasm/src/gpu.rs::disableGpu` + a worker
   `onerror` -> pool retry now recover into the CPU path, but the real fix is here: replace
   `create_buffer_init` with `create_buffer` + `queue.write_buffer` (as `plan.rs` already does
   for its own buffers) or return an error from buffer creation instead of letting wgpu unwrap it.
4. Latent upload converts planar to HWC4 on the CPU per picture (0.4 ms for 560 x 888).

5. No cancellation (`enough::Stop`) on the GPU path; `GpuDecoder` has no `max_channels`
   (progressive decode) option.
6. **A workspace that has synthesised a very large picture stays slow for small ones.** In the
   size sweep the 560 x 888 BOP row, which runs last, measures 32.3 ms of device time against
   9.7 ms for the same picture in a fresh context, reproducibly across runs; capping the sweep
   at 2048 instead of 4096 makes it 9.4 ms. The pooled activation buffers only ever grow
   (`plan.rs::Pool`), so after a 4096 x 4096 BOP picture every later plan binds slots sized for
   that one. Shrinking or bucketing the pool would fix it; nothing else in the file is affected
   (SOP and HOP show the same row unchanged).
7. The integrated Radeon (RADV) loses the device during a BOP size sweep; see
   `benchmarks/gpu_decode_2026-09-17_radv_igpu.meta`. It is 8x slower than the crate's own CPU
   engine anyway, so the useful fix is probably to decline integrated adapters by default rather
   than to chase the hang.

## Tests and benchmarks

```
just gpu-test                          # needs an adapter, the checkpoints and the reference vectors
ZENJPEGAI_GPU_ADAPTER=GeForce just gpu-test
ZENJPEGAI_GPU_ALLOW_SOFTWARE=1 just gpu-test   # llvmpipe / SwiftShader / WARP
cargo run --release -p zenjpegai-gpu --example gpu_bench -- --out benchmarks/gpu_decode_<date>.tsv
just gpu-profile GeForce               # per-dispatch device time, one compute pass per dispatch
scripts/bench/reference_gpu.sh 3       # the reference software, CPU and GPU, on the same streams
```

`--adapter` matches a substring of the adapter *name*: the integrated Radeon of a Ryzen calls
itself "AMD Ryzen 9 9950X3D 16-Core Processor (RADV RAPHAEL_MENDOCINO)", so `--adapter RADV`,
not `--adapter Radeon`. Its render node is `root:render` while the NVIDIA nodes are
world-readable, so a run on it needs the `render` group
(`sudo -u <user> -g render env HOME=... gpu_bench ...` works without a new login).

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
