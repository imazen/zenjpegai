# Wasm decode profile and speed work — 2026-09-18

Host: dev (Ryzen 9 9950X3D, 16 cores / 32 threads), Node v26.7.0, Wasmtime 40.0.1,
Chromium 153, Firefox 155, WebKit 26.6 (playwright). Stream for single-threaded numbers:
`img30_base_off_bpp050` (560x888, ~0.5 MP). Browser numbers: the 8 demo streams at
0.25 + 0.75 bpp (~1 MP each), threads package, isolated page.

## Per-function profile (wasmtime --profile=guest, single thread, Wasm128 tier)

`wasm32-wasip1` CLI build (`--features cli`, release, debug names kept), decode of the
560x888 stream: 949 samples over 1068.8 ms total (includes one model load).

Self time, top 15:

| self ms | self % | function |
|--------:|-------:|----------|
|   939.1 |  87.9% | nn::fast::conv::run_row |
|    45.6 |   4.3% | nn::fast::conv::for_each_row (PackedConv row setup) |
|    16.5 |   1.5% | model::synthesis::ResAu::forward |
|    12.7 |   1.2% | for_each_row\<i32\> (int_conv row loop) |
|     6.8 |   0.6% | __rust_no_alloc_shim |
|     6.3 |   0.6% | nn::fast::conv::pack (weight packing at model load) |
|     4.6 |   0.4% | std::io::default_read_to_end (stream + model file IO) |
|     3.2 |   0.3% | BTensor::cat |
|     2.9 |   0.3% | decoder::output::quantize |
|     2.8 |   0.3% | for_each_row\<f32\> (second conv site) |
|     2.2 |   0.2% | Vec\<f32\>::clone |
|     2.1 |   0.2% | dlmalloc |
|     2.1 |   0.2% | hsd::HyperScaleDecoder::run |
|     2.1 |   0.2% | PackedIntConv::new (weight packing at model load) |
|     2.1 |   0.2% | layers::pixel_shuffle_to_planar |

Inclusive: PackedConv::forward 78.7%, synthesize_with 74.2% (SynthesisPrimary 52.4%,
ResAu 23.2%, SynthesisSecondary 21.4%), reconstruct_latent_with 22.0%,
PackedConvTranspose 14.5%, hyper decoder 12.9%, context model 8.9%.

The float convolution micro-kernel is ~88% of single-threaded decode. Everything else
is glue; depthwise conv, int8 conv, layout copies, allocation, and the bindgen boundary
are each under ~1% and cannot move the total by more than a few ms.

## Attribution of the native gap (same 560x888 stream, warm decode)

| engine / tier                        | decode ms | note |
|--------------------------------------|----------:|------|
| native, Tier::V4 (AVX-512), 32 thr   |        26 | benchmark mean ~27 at 8 thr |
| native, Scalar, 1 thr                |     ~12000 | baseline x86-64 has no FMA: `fmaf` libcalls |
| node 26, Wasm128, 1 thr              |       ~375 | V8 JIT |
| node 26, Scalar (Lanes8), 1 thr      |       ~372 | == SIMD: kernel is issue/latency-bound, not width-bound |
| wasmtime 40, Wasm128, 1 thr          |       ~920 | Cranelift lowers the v128 loop ~2.4x worse than V8 |
| wasmtime 40, Scalar, 1 thr           |       ~433 | ditto (scalar path wins under Cranelift) |
| Chromium worker, threads pkg, 1 thr  |       ~760 | per ~1 MP stream (bigger image) |
| Chromium worker, threads pkg, 16 thr |       ~132 | " |

(i) Engine/JIT quality: same wasm bytes run 2.4x slower under Cranelift than V8 — this is
the largest single factor after width and is not fixable from our side.
(ii) Kernel shape: at V=8/B=4 the inner loop is 16 f32x4 mul+add + ~6 memory ops +
~8 bookkeeping ops per 32 lane-updates. Scalar tier matching the SIMD tier means the loop
is issue/latency-bound, not flop-width-bound — wider tiles that fit 16 v128 registers
give at most ~25% fewer instructions, and every wider tile measured *slower* (register
spills): B=5 −20%, B=6 −4% under V8. B=4 stands.
(iii) Missing FMA: unfused `f32x4.mul` + `f32x4.add` is the policy on wasm32 (2 instrs
per update instead of 1 native `vfmadd`) — about half the per-core gap vs AVX-512 is
width, a chunk of the rest is unfusing.
(iv) Threading: see below — rayon scales ~5.3x at 8 threads, then saturates.

## Thread scaling (Chromium, threads package, 16 stream rows each, mean decode ms)

| rayon workers | mean decode ms | speedup vs 1 |
|--------------:|---------------:|--------------|
| simd pkg, 1 worker (non-isolated baseline) | 730.6 | — |
|  1 | 760.0 | 1.0x |
|  2 | 419.0 | 1.8x |
|  4 | 228.6 | 3.3x |
|  8 | 144.0 | 5.3x |
| 16 | 131.8 | 5.8x |
| 32 | 162.4 | 4.7x |

Raw TSV: `benchmarks/wasm_threads_2026-09-18.tsv` (+ .meta).

Two conclusions:

- The curve is Amdahl-limited: the data fit ~12% effectively-serial work (entropy decode,
  model-latent reconstruction are sequential), so the floor at infinite threads is
  ~90 ms for these streams and 16 workers already sits on it. 9950X3D has 16 physical
  cores; 32 children + the calling thread also oversubscribe (34 > 32 hw threads).
- **32 workers is measurably *worse* than 16** — and 32 was the shipped default
  (`navigator.hardwareConcurrency`). The worker now caps the default rayon pool at
  `min(hwc, 16)` child workers; `?threads=N` still overrides exactly
  (web/src/worker.js `rayonThreads`, web/tests/threads.spec.ts).

## What changed (accepted)

1. `worker.js`: default rayon pool `min(hwc, 16)` instead of `hwc` — measured best on
   this host (above); also fixes the N+1 oversubscription (caller runs rayon jobs too).
2. `worker.js` + `build-wasm.sh`: wasm-bindgen init now uses the single-object form
   `init({ module_or_path, memory })` — kills the "deprecated parameters" console
   warning in both the module worker and wasm-bindgen-rayon's spawned child workers
   (generated snippet is rewritten at build time).
3. `conv.rs`: per-row border-column scratch moved from `alloc::vec![0.0; n]` (two
   allocator hits per output row per conv layer) to a `thread_local` scratch reused
   across rows — under shared memory every malloc is an atomic-linked allocator call;
   ~5% on the single-thread wasip1 decode (375 -> 348 ms min-of-8 under V8).
   Bit-identical: same zeros + copied windows, `tiers_and_threads` passes.

## Evaluated and rejected (measured, kept for the record)

- conv kernel tile `B=5,V=8`: −20% (odd tail fragmentation + spills). `B=6,V=8`: −4%
  (15 v128s is past the register budget). `V=4` shapes need more splats per output and
  are worse on paper; not tried.
- Hoisting the `r[j*S..]` reslices out of the `(channel, tap)` loop: no gain under V8 —
  the loop got *larger* (extra fat-pointer table). Reverted.
- `wasm-opt -O4 --converge` on pkg-threads: statistically identical to `-O3`
  (157-165 ms vs 160-167 ms on a 6-decode probe). Kept `-O3`.
- `--initial-memory` preallocation: shared memory min is 18 pages today; raising it to
  cover a ~1 MP decode (~1 GB peak) commits real RAM per worker for a one-time
  first-decode saving of a few tens of ms. Not worth it.
- wasm-bindgen boundary: stream in / RGBA out are each already one copy + transfer;
  `decode()` needs the pixel-conversion pass regardless. Nothing to cut.
- pad_par copy in transposed convs, int8 conv, depthwise, cat/layout copies: all
  <1.2% self time in the profile — left alone.
- `+tail-call`: no hot tail calls. `opt-level="s"/"z"`: previously measured slower
  (size-only). Not retried.

## Before/after, browser decode (median of the 16 stream rows, threads package)

`benchmarks/wasm_decode_2026-09-18.tsv` holds four appended runs; run 0 is the repo
baseline (default = all hw threads), run 3 is this change set. Runs 1-2 are prior
intermediate sessions on the same box.

| browser  | before (run 0) | after (run 3) | ratio |
|----------|---------------:|--------------:|------:|
| chromium |        204.4 ms |       104.1 ms | 0.51x |
| firefox  |        227.0 ms |       107.3 ms | 0.47x |
| webkit   |        238.3 ms |       131.3 ms | 0.55x |

The controlled variable (thread count alone) accounts for ~1.23x on this box
(162 -> 132 ms mean in the scaling table); the rest of the end-to-end gain is the
allocator-contention fix plus run-to-run machine state — the box is shared and these
runs are not perfectly controlled. Non-isolated simd package: ~731 ms mean over the 16
streams (unchanged — it is single-threaded and the kernel did not change).

## Remaining headroom

- ~12% serial fraction (entropy + latent reconstruction) caps threaded decode at
  ~90-130 ms for these streams regardless of kernel work.
- The Wasm128 conv loop is at its bit-identity floor: unfused mul+add, fixed
  accumulation order, 16 v128 registers. Only a numeric-order change (not allowed) or
  a faster engine removes it.
- WebGPU (the sibling `webgpu` work) is the remaining large multiplier.
