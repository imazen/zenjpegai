# zenjpegai ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpegai/ci.yml?style=flat-square&label=CI)

Not yet published to crates.io (version 0.0.1) — the crates.io / lib.rs / docs.rs / license
badges land at first publish.

zenjpegai is a JPEG AI (ISO/IEC 6048-1 | ITU-T T.840.1) image codec in safe Rust: a port of
the [JPEG AI reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
with `#![forbid(unsafe_code)]`, SIMD through [archmage](https://lib.rs/crates/archmage) and
[magetypes](https://lib.rs/crates/magetypes), and no Python or PyTorch at runtime.

**Status: decoder complete for the reference's feature set; encoder covers RGB and YUV
sources.** Decoder: SOP / BOP / HOP operating points, any picture size (synthesis tiling,
dependent and independent regions), RGB (BT.709) and YUV streams in 4:4:4 / 4:2:2 / 4:2:0 at
8 and 10 bit, RVS, GRFS, LSBS, quality maps, cube flags, progressive decode, all four
post-filters (EFE linear, EFE non-linear, eICCI, LEF), UDI. Encoder: interleaved-RGB (8/16 bit)
and planar-YUV (4:4:4 / 4:2:2 / 4:2:0, 8/10 bit) sources, the same coding tools and regions,
analysis tiling above 1 MP, fixed-model and target-bpp rate control with the reference's
`ECLibLH` likelihood measure or the coded stream (`RateEstimate`), resource limits and a
memory estimate (`EncodeLimits`, `estimate_encode_memory`); it reproduces the reference
encoder's streams byte for byte on 25 of 32 reference encodes (same length on the rest)
and the `tools_on` tool set (`--tools-on`, or the individual `--rvs`/`--grfs`/`--lsbs`/
`--lef`/`--efe-linear`/`--eicci`/`--efe-nonlinear` flags) with the post-filter decisions
the reference's searches would make — modulo MKL `lstsq` nondeterminism that costs the
reference up to 0.12 dB (`PORTING.md`). `num_chs` below the model's channel count is
supported too (`--num-chs y,uv`). Not ported: the hyperopt UV displacement search and the
user-defined colour transform (both are dead code in the pinned reference commit — see
`PORTING.md`), eICCI on subsampled pictures. Anything unsupported is rejected with `Error::Unsupported`; nothing is silently
approximated. The entropy stage is bit-exact; the reconstructed 8-bit picture differs from the
reference decoder's in about 0.005 % of samples, each by one step (convolution summation
order). Also here: a WebAssembly build with a browser polyfill and demo (`web/`, live at
https://imazen.github.io/zenjpegai/ and https://zenjpegai.pages.dev/) and a wgpu compute
backend (`gpu/`, verified in-browser on hardware WebGPU).
`PORTING.md` tracks every module, what is missing, and which test proves each piece.

New here? Start with [`CONTRIBUTING.md`](CONTRIBUTING.md): setup, code map, how parity is proven, what to do next.

## Use

```rust
let decoder = zenjpegai::Decoder::new("path/to/jpeg-ai-reference-software/models");
let image = decoder.decode(&std::fs::read("picture.bits")?)?; // interleaved RGB
```

```
cargo build --release --features cli
ZENJPEGAI_MODELS=path/to/models target/release/zenjpegai decode picture.bits picture.png --time
```

## Embedding / resource control

Everything below is safe Rust and works the same in the browser build (wasm exports named in
the last rows).

| What | API |
| --- | --- |
| Pre-flight limits (dimensions, codestream size, estimated heap) checked before any picture-sized allocation | `Limits` / `EncodeLimits` (`Decoder::limits`, `Encoder::limits`; `max_memory_bytes` rejects by estimate) |
| Heap estimate of one job, from headers alone | `estimate_memory` / `estimate_encode_memory`, `Decoder::estimate_memory`, `Encoder::estimate_memory` → `MemoryEstimate { live_bytes, peak_bytes() }` |
| Admission control: a shared byte budget several decoders/encoders draw `live_bytes` grants from — FIFO wait, fail-fast (`Error::ResourceBusy`), `allow_oversize` runs one too-big job alone | `MemoryBudget` (`new`, `with_policy(BudgetPolicy::Wait|FailFast)`, `with_max_jobs`, `allow_oversize`, `acquire`, `held`, `jobs`, `waiting`) wired with `Decoder::budget` / `Encoder::budget`; grants release on drop, error or cancel |
| Cancellation of waits and decodes | `enough::Stop` (`decode_with`, `decode_picture_with`, `BudgetGuard` waiters → `Error::Cancelled`) |
| Measured heap of one call and of the process | `MemoryReport { tracked_peak_bytes, tracked_live_bytes, pool_bytes }` via `Decoder::memory_report` / `Encoder::memory_report` (cumulative), `decode_picture_report`, `decode_with_report`, `encode_report` (per call), `MemoryWatch` (arbitrary spans) |
| Recycle pool bound and drain | `nn::fast::set_pool_limit` / `pool_limit` (default 24 buffers / 1 GiB), `Decoder::release_buffers` / `Encoder::release_buffers` |
| GPU working set | `gpu` pool slots are bounded and evict by plan fit (`gpu::plan::Pool`) — no external knob |
| CLI | `zenjpegai decode|encode --memory-budget <bytes> --memory-report` |
| Browser | wasm `estimateMemory`, `memoryReport`, `releaseBuffers`; per-decode `timings.memory`/`wasmBytes`; `DecoderPool({ maxBytesInFlight })` byte-based admission on the shared queue |

The estimate's calibration lives in `src/decoder/limits.rs` / `src/encoder/limits.rs`
(picture + synthesis-tile terms per operating point, plus the cold model-load peak per
`model_id`); `tests/memory_ref.rs` asserts it covers the tracked peak of every reference
stream, and `scripts/bench/memory_tracked.sh` reconciles the tracked numbers with
heaptrack (`benchmarks/memory_tracked_*.tsv`).

## Speed

Decode time on a Ryzen 9 9950X3D, same streams, same machine, no `-C target-cpu=native`
(`scripts/bench/decode_end_to_end.sh`, raw data in `benchmarks/decode_end_to_end_2026-09-18_r2.tsv`
and `benchmarks/decode_end_to_end_2026-09-18_d4serial.tsv` for the current zen columns).
Reference = upstream `b9e573f` on PyTorch 1.10.2 CPU; its number is its own `TOTAL`
(model load excluded, median of 3). zenjpegai numbers are steady state with models loaded
(min of 3); "process" is one whole command-line run including start-up, model load and
PNG output.

| stream | ref, 1 thread | ref, 8 threads | zen, 1 thread | zen, 8 threads | ref GPU (CUDA) | zen GPU (wgpu) | ref process | zen process |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 560x888, simple profile (SOP), 0.50 bpp | 146 ms | 110 ms | 52 ms | 14 ms | 87 ms | 12 ms | 1183 ms | 108 ms |
| 560x888, base profile (BOP), 0.12 bpp | 226 ms | 136 ms | 82 ms | 19 ms | — | — | 1204 ms | 116 ms |
| 560x888, base profile (BOP), 0.50 bpp | 230 ms | 128 ms | 83 ms | 22 ms | 93 ms | 17 ms | 1192 ms | 118 ms |
| 560x888, base profile (BOP), 1.00 bpp | 217 ms | 124 ms | 83 ms | 21 ms | — | — | 1120 ms | 145 ms |
| 560x888, high profile (HOP), 0.50 bpp | 2313 ms | 660 ms | 1066 ms | ~320 ms | 160 ms | 109 ms | 1645 ms | 441 ms |
| 2096x1400, base profile (BOP), 0.50 bpp | 1228 ms | 512 ms | 494 ms | 120 ms | 288 ms | 93 ms | 1585 ms | 599 ms |

The GPU columns are the same box's RTX 2080 (the CPU columns above are its Ryzen 9
9950X3D): the reference's CUDA path (torch 1.10.2+cu113, `decode_stream` TOTAL, median
of 3, synchronised — `benchmarks/gpu_reference_2026-09-17.tsv`) against zenjpegai's WGSL
compute shaders through wgpu/Vulkan, whole codestream to 8-bit, CPU entropy stage included
(`benchmarks/gpu_decode_2026-09-18_rtx2080_kernels.tsv`). On SOP/BOP at 560x888 the GPU
total is dominated by the CPU entropy stage, so zen's GPU and 8-thread numbers nearly
coincide; HOP is the kernel-bound operating point, where the reference's cuDNN path stays
1.5x ahead of the WGSL kernels while its CPU path is 3.8x behind this crate's. The GPU
decode's CPU stages run on the rayon pool too — CPU-side decode time in the GPU path
dropped ~13-39 % after the entropy/latent parallelisation (HOP unchanged — its GPU wall
dominates; `benchmarks/gpu_decode_2026-09-18_d4serial.tsv` vs `_rtx2080`).

So: ~2.2x to 2.8x faster than the reference on one thread. Threads help the reference
too (up to ~3.5x on HOP), but zenjpegai's eight-thread decode still beats its
eight-thread decode by 2x to 6.4x — and the margin grows under machine load, because
the convolution accumulators no longer spill to the stack (see
`benchmarks/conv_kernels_2026-09-18.md`). Process wall — what a batch pipeline feels —
is 4x to 11x faster end to end.
Two picture sizes is a thin sample: no tiny pictures, nothing above 3 MP yet.

Encode, all tools on (`--tools-on`; `benchmarks/encode_tools_on_2026-09-18.tsv`): the
reference's post-filter searches dominate its encode — a rate-matched `tools_on` encode of
the 2096x1400 picture takes ~35 s of its own TOTAL where zenjpegai's takes ~7.0 s threaded
(~21 s on one thread), about 5x; at 560x888, 5.0 s vs 0.96 s. At a fixed operating point
the gap is 2.3x (560x888) and 3.4x (2096x1400) threaded, roughly par on one thread at the
smaller size.

In the browser (Chromium 153, RTX 2080 via Dawn/Vulkan, ~1 MP demo corpus, medians from
`benchmarks/wasm_decode_2026-09-18_gpu.tsv`): the WebGPU package decodes a warm stream in
~46 ms vs ~89 ms on the threaded CPU package and ~642 ms on the single-threaded SIMD one,
so `auto` uses the GPU when a hardware adapter exists (details: `web/README.md` §7).
Whichever package runs, headers + entropy decode (integer me-tANS) + hyper-scale decoder
(int8) + latent reconstruction always run on the CPU — bit-exact/integer stages plus a few ms
of float work, multi-threaded on the `*-threads` packages — while synthesis (the heavy conv
net) + output conversion + presentation run on the GPU only when the GPU path is taken.
Post-filters, when a stream signals them, are CPU after synthesis. The demo page reports
startup cost (wasm fetch/instantiate, thread pool, GPU init, per-model-bundle load rows with
network/cache/prefetch provenance) separately from each picture's own `ttr` split
(stream/wait/decode/present with the CPU/GPU stage split above); the polyfill keeps the
authored `<img>`/`<picture>` fallback on `no-wasm-simd`, `too-large` (`maxPixels`), or any
decode error, and can prefetch model bundles into the worker's Cache API namespace. Full
timing schema and degradation knobs (`maxChannels` progressive decode, `maxPixels`, prefetch
modes): `web/README.md` §9.

## Running the checks

```
cargo test --lib --tests                                    # public API only; no reference data needed
cargo clippy --all-targets --all-features -- -D warnings
ZENJPEGAI_REF=~/work/zen/jpeg-ai-reference-software \
  cargo test --lib --tests --all-features                   # full parity gates vs the reference dumps
cargo test --lib --tests --all-features tiers_and_threads   # every SIMD tier x thread count bit-identical
```

Reference-oracle commands (need the upstream checkout and vectors — setup in
[`CONTRIBUTING.md`](CONTRIBUTING.md)):

```
scripts/bench/decode_end_to_end.sh > benchmarks/decode_end_to_end_$(date +%F).tsv
scripts/bench/encode_end_to_end.sh > benchmarks/encode_end_to_end_$(date +%F).tsv
HOST_LABEL=dev scripts/wasm/parity.sh > benchmarks/wasm_parity_$(date +%F).tsv   # wasm vs native vs reference
cd web && npm ci && npm run build && npx playwright test    # browser suite (demo assets via scripts/fetch-demo-assets.mjs)
```

The decoder's untrusted-input surface is fuzzed with cargo-fuzz (libFuzzer): `just fuzz`
builds three nightly targets — `container_headers`, `entropy_stage`, and `decode_full` —
and runs them with the `fuzz/zenjpegai.dict` dictionary and the `fuzz/seeds/` corpus.
Every input from `fuzz/regression/` is replayed on stable by `cargo test --test
fuzz_regression`; crashes and corpora live outside git and sync to block storage via
`~/work/zen-workspace/fuzz-sync.sh`.

Status detail, the honest table of what is and is not ported, and every measured parity
bound: [`PORTING.md`](PORTING.md). How to work here: [`CONTRIBUTING.md`](CONTRIBUTING.md).

## What JPEG AI is

JPEG AI is a learned image codec. The transform is a set of trained convolutional networks; the
rest is conventional codec machinery: marker-delimited substreams, a table-based ANS entropy
coder (me-tANS), an integer hyper-scale network that must be bit-exact across implementations,
and a stack of post-filters. The reference software implements all of it in Python and PyTorch
with two small C++ extensions.

## Goals

1. Decode every conforming stream the reference decoder accepts, with a bit-exact entropy
   stage and a reconstruction that matches the reference to within float rounding.
2. Encode streams the reference decoder accepts.
3. Be faster than the reference on the same CPU, measured and committed under `benchmarks/`.

## Model weights

The networks' weights are the reference software's trained checkpoints (`models/*.pth`,
git-lfs in the upstream repository). They are not redistributed in this repository. zenjpegai
reads the upstream `.pth` files directly; no Python is needed.

## License

Dual-licensed: [AGPL-3.0](LICENSE-AGPL3) or [commercial](LICENSE-COMMERCIAL).

This is a port of the JPEG AI reference software, Copyright (c) 2010-2026 ITU/ISO/IEC,
distributed under the BSD 3-clause license in [`upstream-notices/LICENSE`](upstream-notices/LICENSE).
That notice also says the software "may be subject to other third party and contributor rights,
including patent rights, and no such rights are granted under this license." The same applies here.

## AI disclosure

This port is written with heavy use of AI coding tools (Claude). Parity with the reference is
established by tests against reference-produced bitstreams and intermediate tensors, not by
inspection; see `PORTING.md`.
