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

## Status (2026-09-18, second tuning pass)

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
try is a *smaller* `ob` together with a pixel tile, not a bigger tile. (2026-09-18: retested
one dimension, px=2 on top of the channel-group unrolls — still slower, 293 vs 226 ms on HOP
560x888.)

### Second pass (2026-09-18, `benchmarks/gpu_decode_2026-09-18_rtx2080_kernels.tsv` +
`gpu_profile_2026-09-18_rtx2080.tsv`)

The profile TSV now carries each dispatch's uniform parameters, so
`scripts/bench/gpu_roofline.py` can classify every kernel by achieved TFLOP/s or GB/s against
the card's ~10 TFLOP/s f32 / ~448 GB/s. What it said, and what was done about it:

| change | mechanism | effect (HOP 560x888) |
| --- | --- | --- |
| `conv_tiled`: workgroup-staged input halo + weight chunk | the 3x3/stride-2 convolutions re-read each input texel 9 times through global memory; staging the `(WG+KH-1)^2` input tile and the `(ICH,OCB)` weight chunk in workgroup memory makes each lane's tap reads cheap. Selected per layer only when the tile fits 16 KiB of workgroup storage (`plan.rs::conv_full`), input-channel blocking `ICH` keeps the accumulation order identical so parity counts do not move | k3x3 s1 ic32 kernels 19.4 -> 8.5 ms, s2 ic32 21.8 -> 6.9 ms, k1x1 ic32 36.3 -> 19.1 ms; ~8% -> 25-37% of f32 peak |
| `depthwise3x3_z{2,4}` | channel block is now the fastest lane dimension: a warp loads/stores contiguous `vec4` runs instead of striding `c4*16B` per pixel | 27.0 -> 7.8 ms, 135 GB/s (30% of peak) |
| `elu_gate` flattened over `(pixel, channel-block)` | same strided-access fix | 11.4 -> 1.4 ms, ~395 GB/s (88% of peak) |
| channel-group dims baked into the conv key (`icg4`, `ocg4`; `ic4` for `convt`) | lets the compiler fully unroll the inner channel loops; required adding `ocg4` to the pipeline key — two layers that shared a key but differed in `ocg4` produced wrong chroma (caught by `decode_ref`, regression test in `tests/kernels.rs`) | flat on its own; enables the staging above |

Whole-picture device time — per-dispatch profile totals, one compute pass per dispatch run
back-to-back so clocks stay boosted (HOP 4096² is from the sweep; the profile only runs
560x888 and 1024²):

| | HOP 560x888 | HOP 1024² | HOP 4096² | BOP 560x888 | BOP 1024² | SOP 560x888 | SOP 1024² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| after pass 1 | 224.1 ms | 498.1 ms | 8919 ms | 9.70 ms | 21.4 ms | 3.82 ms | 7.47 ms |
| after pass 2 | 103.4 ms | 208.5 ms | 4045 ms | 7.32 ms | 14.5 ms | 3.37 ms | 6.41 ms |
| | **-54%** | **-58%** | **-55%** | **-25%** | **-32%** | **-12%** | **-14%** |

SOP moved least: its convolutions are small-channel single-tile work where staging overhead
roughly cancels reuse. The full sweep's synthesis rows at >=1024² carry the downclock artifact
(the GPU idles at ~435 MHz during each interleaved CPU round; NVIDIA 580.178.04, no permission
to lock clocks), so e.g. BOP 1024² reads 70 ms there against 14.5 ms in the profile — the
profile totals are the trustworthy comparison.

Whole-stream decode (codestream to 8-bit, CPU entropy stage included): BOP 560x888 22.8 -> 17.4 ms
wall, HOP 560x888 326 -> 109 ms, img01 2096x1400 164 -> 93 ms.

Parity is unchanged: every optimisation above keeps each output's summation order (the staged
tiles only reorder *which lane* computes a tap, never the multiply-add sequence per output), and
`just gpu-test` reports the same counts `PORTING.md` records.

Also fixed in this pass:

- **Pool / plan-cache ping-pong.** `Pool::fit` used to bump `generation` on every slot
  replacement, and the plan cache drops everything when the stamp moves — so a run of
  heterogenous tiles (2048x2048 = 9 tiles of very different sizes) shrank the pool on each
  small tile and regrew it on each large one, evicting every plan built so far and rebuilding
  all 9 on the next warm run (`gpu_bench` asserts `plans_built == 0`; it caught this).
  Growth no longer bumps the generation — a plan's slot binds are self-contained scratch, so a
  stale-but-valid buffer is still correct — and shrink is allowed only at the first `fit` of a
  run (`Pool::begin_pass`, called by `run_tailed`). Small-after-large still releases memory:
  568 -> ~195 MiB for 4096² then 560x888.
- **Readback** (item 2 of the 2026-09-18 work): `emit_output` converts to the output format on
  the GPU (8/10-bit RGB and YUV at the coded subsampling, half-to-even like the CPU output
  stage) and reads back packed u16 pairs — 201 MB of f32 planes at 4096x4096 become ~100 MB,
  decode wall −20 to −29%.
- **Cancellation / `max_channels`** (item 3): `decode_picture_async_with`,
  `decode_to_gpu_with`, `decode_with` take the same `enough::Stop` token the CPU path uses,
  checked between submits, and a channel limit for progressive decode; tests in
  `tests/decode_ref.rs`.

**Open, in order:**

1. **`shader-f16`** (opt-in, default off — not implemented). The RTX 2080 exposes
   `shaderFloat16` + `storageBuffer16BitAccess`, so `wgpu::Features::SHADER_F16` is
   requestable. What it needs: an f16 weight copy (or a second buffer) per layer, f16 kernel
   variants generated beside the f32 ones, feature detection with an f32 fallback, and a
   separately measured parity bound — f16 accumulation will move the 8-bit quantisation counts,
   so the feature must record its own numbers rather than inherit the f32 bound. Expected win
   on Turing is the ~2x fp16 FMA rate on the compute-bound convs (the bandwidth-bound kernels
   would need f16 *storage* to benefit, a bigger change).
2. `convt` (transposed conv) is now the top kernel on HOP (12.3 ms of 103). Its reduction is
   over *input* bandwidth — the weight tile needed is `(ic4 * oc4 * k * k)` which does not fit
   16 KiB of workgroup memory at the useful channel counts, so it still gathers. Options:
   split the workgroup-memory budget between input and weights, or restructure as small GEMM.
3. `attention_apply` / `gram_chunks` / `pixel_shuffle_2` each run at ~10-45% of bandwidth peak
   on scalar or strided access patterns; together ~10 ms of HOP 560x888.
4. Latent upload converts planar to HWC4 on the CPU per picture (0.4 ms for 560 x 888).
5. The integrated Radeon (RADV) loses the device during a BOP size sweep; see
   `benchmarks/gpu_decode_2026-09-17_radv_igpu.meta`. It is 8x slower than the crate's own CPU
   engine anyway, so the useful fix is probably to decline integrated adapters by default rather
   than to chase the hang.
6. **Browser**: shipped and re-verified on hardware after this pass (2026-09-18): the
   `benchmark-gpu` Playwright spec ran the new kernels under Chromium WebGPU on the RTX 2080
   (`WEBGPU_ADAPTER=hardware`, adapter `nvidia/turing`, non-fallback, `path=gpu` rows in
   `benchmarks/wasm_decode_2026-09-18_gpu.tsv`). **Software-adapter caveat unchanged**: under
   `WEBGPU_ADAPTER=swiftshader` Dawn loses the device at model-2 load; decodes fall back per
   call to the CPU engine (correct, slower).

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
