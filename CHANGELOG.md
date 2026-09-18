# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->

### Added
- Encoder: chroma-subsampled, 10-bit and YUV sources. `SourceImage::{Rgb, Yuv}` with
  `SourceImage::read_yuv` (planar YUV whose `WxH_Nbit_{420,422,444}` file name carries the
  geometry, like the reference's `extract_info`), `SourceMeta` with `s_ver`/`s_hor` from the
  chroma plane size and `c_ver`/`c_hor` from `EncodeParams` (`-c_ver_value`/`-c_hor_value`;
  4:2:2 coded as 4:2:0 is `Unsupported`, as upstream), `EncodeParams::diff_display` for the
  non-displayed border, and 16-bit PNG input (`bit_depth_idc` 4). Chroma down-sampling is
  `encoder::resample::resize_bilinear`, bit-identical to PyTorch's bilinear
  `align_corners=True` kernel (`tests/vectors/nn/bilinear_align_corners.bin`). CLI:
  `zenjpegai encode in.yuv …`, `--c-ver`/`--c-hor`/`--diff-display`. On the nine `formats`
  vectors the analysis-transform inputs are bit-for-bit the reference's, seven streams are
  byte-identical, and the two 10-bit ones are the same length with 5 of 1,254,400 luma
  residual symbols moved by 1; the reference decoder reads all nine.
- eICCI on chroma-subsampled pictures (4:2:0 and 4:2:2): the reference decoder's
  `Image.to_444_` bicubic chroma up-sampling, eICCI at luma size, then `to_format_` bilinear
  down-sampling back (`filters::icci::resize_bilinear` reproduces PyTorch's channels-last
  kernel bit for bit — weight products rounded once, then nested FMAs). Oracle vectors are
  forced encodes (`scripts/ref_vectors/force_icci_encode.py`, `make_reference_streams.sh`
  set `icci420`): the reference encoder never enables eICCI for subsampled sources. The 4:2:2
  stream is conformant; the 4:2:0 one deliberately carries `icci_enable_flag` in a position
  the grammar does not have, so the dump scripts patch the reference decoder at runtime
  (`--patch-icci420`) and the tests rebuild the header — PORTING.md has the details.
- Cube-flag decode coverage: the encoder-set streams `enc_img30_bop_m0_bm1069` and
  `enc_img30_hop_m3_bm1069` (beta displacement -1069, `use_cube_flags = 1` with false flags,
  incl. a cleared `cube_group_flag`) now carry reference decoder dumps
  (`make_reference_streams.sh cubeflags`) and gate `tests/entropy_ref.rs` bit-exact;
  `tests/decode_ref.rs` decodes the BOP stream end to end.
- `tools::gain::scaler_from_log` verified bit-identical to the reference's `torch.exp` on every
  reachable input (`tests/vectors/gain_scaler.bin`, `gen_gain_scaler_vectors.py`): all
  `gain_vector_log` entries of the eight `VM_common_int` checkpoints times every
  `beta_displacement_log` the header can signal (-2048..=2047).
- Investigated `colour_transform_idx = 2` (`scripts/ref_vectors/probe_colour_transform2.py`):
  dead, self-inconsistent code upstream — the decision is "not ported", such streams stay
  `Error::Unsupported` (PORTING.md "Reference dead code").
- CI (`.github/workflows/ci.yml`): fmt, clippy `-D warnings` + tests on ubuntu-latest /
  windows-11-arm / macos-15-intel / macos-latest, `i686-unknown-linux-gnu` via `cross`, a
  no_std check (native + `wasm32-unknown-unknown`) with a dedicated "public API only" lint step,
  `zenjpegai-gpu` / `zenjpegai-wasm` type-checks, and an MSRV job pinned to the verified `1.89.0`.
  README CI badge.
- `no_std` + `alloc` verified clean (`cargo check`/`clippy --no-default-features`, native and
  `wasm32-unknown-unknown`); `just no-std`.
- `zencodec` feature (`src/codec.rs`): `JpegAiDecoderConfig` / `JpegAiDecodeJob` / `JpegAiDecoder`
  implement `zencodec::decode::{DecoderConfig, DecodeJob, Decode}` over `Decoder` (RGB output;
  format registers as `ImageFormat::Custom`, magic `FF 80 FF 82`).
- `unstable-internals` feature: `nn`/`model`/`tools`/`mans`/`bitio`/`container`/`tensor`/
  `weights`/`filters` (most of `decoder`) are `pub(crate)` by default, `pub` under this feature
  (implied by `cli` and `zencodec`). `#![warn(missing_docs)]` on the committed public API without
  it.
- `enough::Stop` checked in the post-filters (`filters::apply`, once per enabled filter; eICCI
  additionally once per tile).
- `benchmarks/build_time_2026-09-18.md`: clean/incremental build times + `cargo llvm-lines`
  survey (no build-time issue found).
- **Encoder** (`encoder::Encoder`, `zenjpegai encode in.png out.bits --model N --beta-disp B [--op sop|bop|hop]`): colour pre-processing, analysis transforms, hyper-encoder, `z` quantisation, the shared scale derivation, the context model's compress direction (per-stage quantise / cube flag / requantise), skip mode, residual quantisation, header and container assembly, one `AnsEncoder` per substream. Fixed model and operating point, one tile, one region, one ANS thread, tools off, 8-bit RGB 4:4:4. On all seven reference vectors the integer stage (`z_hat`, both scale maps, cube flags) equals the reference encoder's, six of seven streams are **byte-identical** to the reference's (the seventh, 178 KB at beta +400, moves 42 of 501,760 residual symbols by 1), and the reference decoder reads every one of them, differing from our own decode in 51-59 of 1,491,840 samples, all by 1. 0.12 s of process wall time against the reference encoder's 2.0 s at 560x888.
- Encoder: the quality map (`Encoder::encode_with_quality_map`, `zenjpegai encode --quality-map mask.png`): the ROI-mask generator, the entropy-index choice, the DPCM deltas and their SOQ substream. Both reference encodes that use it come out byte-identical.
- Encoder: coding tools - RVS, GRFS (the gain flags derived as `analyzeCWG` does), LSBS, 1..16 ANS threads per substream, and region partitioning in both modes (`EncodeParams`, `zenjpegai encode --rvs --grfs --lsbs --ans-threads N --regions dependent|independent`). All five tool vectors come out byte-identical; the dependent-region one is byte-identical given the reference's own latents and both region streams are its length from the PNG.
- Encoder: analysis tiling (`encoder::tiles`) - above 1 MP the analysis transform and the hyper-encoder run per tile (1024 / overlap 64 luma, 512 / 32 chroma) and the picture header carries the synthesis tiling; the 2096x1400 reference encode comes out at the reference's byte count with one residual symbol moved.
- Encoder: rate matching (`Encoder::encode_to_bpp`, `zenjpegai encode --bpp R`), mirroring `bitrate_matcher`'s model pre-selection + log-rate interpolation + bisection but coding every trial instead of estimating it. Same model as the reference at all five CTC rates and closer to the target at each (worst -3.5 % against the reference's -10.1 %); 12-15 trial encodes, 0.55 s wall at 560x888.
- Encode benchmark against the reference encoder (`scripts/bench/{encode_end_to_end.sh,ref_encode.py}`, `benchmarks/encode_end_to_end_2026-09-18.{tsv,meta}`): 9.3-13.6x its one-thread `TOTAL` at the same fixed model.
- `decoder::entropy::{ComponentScales, component_scales, dequantize_residual}`: the scale derivation both directions share (factored out of `decode_component`; decoder output unchanged).
- `gpu/` crate (`zenjpegai-gpu`): SOP / BOP / HOP synthesis as WGSL compute shaders through wgpu 30 (native + wasm32), `GpuDecoder`, readback-free RGBA presentation, `just gpu-test`, `gpu_bench` example.
- GPU backend on hardware (711233ad): `just gpu-test` passes on an RTX 2080 (NVIDIA 580.178.04) with the parity bounds unchanged from llvmpipe, and the first real timings are committed - `benchmarks/gpu_decode_2026-09-17_rtx2080.{tsv,meta}` (64 to 4096 plus the reference streams; `..._pretuning.tsv` is the same sweep before the tuning, `..._radv_igpu.{tsv,meta}` the integrated Radeon), `gpu_profile_2026-09-17_rtx2080.{tsv,meta}` (per kernel, before and after), `gpu_reference_2026-09-17.{tsv,meta}` (the reference software on the same card, via the new `scripts/bench/reference_gpu.sh`). The GPU overtakes this crate's 8-thread CPU engine at about 384x384 (SOP), 256x256 (BOP) and between 128x128 and 256x256 (HOP).
- GPU kernel tuning (711233ad, 0120c800), all of them keeping every output's summation order so the parity numbers do not move: the ResAU gate and the residual add are computed in the convolution's store rather than a pointwise dispatch (dispatches per picture SOP 36 -> 30, BOP 33 -> 27, HOP 172 -> 158); the transposed convolution visits only the taps its divisibility test used to keep (4 of 16 at k=4 stride 2; `convt_k3_s2` in HOP 36.5 -> 28.1 ms); the depthwise 3x3 stages its halo in workgroup memory (134.4 -> 64.8 ms); the convolution's workgroup edge is 16x16 instead of 8x8 on layers with at least 16 input blocks and both output dimensions at least 128 (HOP's 1x1 convolutions 162.4 -> 117.3 ms). Device time for one picture: BOP 560x888 10.80 -> 9.67 ms, BOP 1024x1024 23.91 -> 21.31 ms, HOP 560x888 259.4 -> 223.5 ms, HOP 1024x1024 635.3 -> 498.0 ms; SOP unchanged, because none of the four reaches its network except the fusion.
- GPU per-dispatch profiling (711233ad): `Workspace::set_profile`, `GpuPicture::dispatch_ns`, `gpu_bench --profile`, `just gpu-profile`; `gpu_bench` also writes its TSV after every row.
- WebGPU in the browser: `zenjpegai-wasm`'s new optional `gpu` feature (`wasm/src/gpu.rs`: async `decode` with CPU fallback on any `GpuError`, `initGpu`/`gpuStatus`/`present`/`disableGpu`) builds a third package, `pkg-webgpu` (`web/scripts/build-wasm.sh webgpu`; `pkg-simd`/`pkg-threads` unchanged). `DecoderPool`'s `gpu` option (`auto|software|force-software|off`) loads it only when `navigator.gpu` yields a non-software adapter; `decodeToCanvas` transfers an `OffscreenCanvas` to the worker once and presents through `Blitter` with no CPU readback (`presented: 'gpu'`), falling back to worker-side `putImageData` (`'2d'`). Worker-level trap recovery (`self.onerror` → `disableGpu` → one transparent CPU retry) covers wgpu-30's `createBuffer` `unwrap` panic, documented in `gpu/README.md` "Known limitations". Verified in-browser via SwiftShader (`chromium-webgpu` Playwright project, `webgpu-decode.spec.ts`: 201/3,000,000 samples differ, all by 1 — GPU parity reported separately); per-path timings in `benchmarks/wasm_decode_2026-09-18_gpu.{tsv,meta}`, size cost in `benchmarks/wasm_size_2026-09-18.md` §7. The browser path has never run on a hardware WebGPU adapter yet — SwiftShader only (`web/README.md` §7).
- Encoder groundwork only (no encoder yet): analysis transforms (`model::analysis`, BOP + HOP) and hyper-encoder (`model::hyper_encoder`) with loaders, within 1.1e-4 of the reference encoder's tensors and `z_hat` symbol-exact (`tests/encode_ref.rs`); `dump_encode.py --enc2` and the `encoder` reference-vector set.
- Browser: worker pool + `<img>`/`<picture>` JPEG AI polyfill (`web/src/{worker,pool,polyfill}.js`), a 36-test Playwright suite across chromium/firefox/webkit and three CSP/isolation profiles, a demo (`web/demo/`) with 8 imazen-26 images published as the `demo-assets-v1` release, and `.github/workflows/pages.yml` (Pages deploy not yet enabled on the repo — human decision, see `web/README.md`). Fixed the `threads` wasm package, which never actually linked/ran before this: missing `--shared-memory`/`--import-memory`/`--export=__wasm_init_tls` linker flags, plus a worker.js message-queue race and a reentrant-`decode()` deadlock. New additive `pth` cargo feature (default on) keeps the `.pth`/pickle reader out of the wasm build (`pkg-simd` 389,178 → 362,789 bytes after `wasm-opt`); full size audit in `benchmarks/wasm_size_2026-09-18.md`.
- `wasm/` crate (`zenjpegai-wasm`, wasm-bindgen: `addModels` / `info` / `decode`, single-thread SIMD128 build and a rayon `threads` build) and `web/scripts/build-wasm.sh`; `pkg-simd` is 389 KB (131 KB brotli). Polyfill, tests and demo are not written yet (`web/README.md`).
- `Limits` (pixels, dimensions, input bytes, estimated memory; safe defaults 120 MP / 4 GiB), judged on the picture header before any model load or picture-sized allocation; `estimate_memory` / `Decoder::estimate_memory` (model calibrated on heaptrack measurements, `benchmarks/memory_2026-09-17.*`); `Decoder::preload`; `nn::fast::set_pool_limit` and `zenjpegai --pool-mb / --discard / preload`.
- Decode memory: feature maps are freed as soon as the next layer has them, synthesis tiles go straight into the cropped / subsampled output planes, entropy-stage tensors are shed after use. Measured peak heap (heaptrack, decode only, default 1 GiB buffer pool): 560x888 SOP 83.7 to 64.7 MB, BOP 145.0 to 116.0 MB, HOP 1260 to 1190 MB, 2096x1400 BOP 389.5 to 309.1 MB; without buffer recycling (`--pool-mb 0`) 39.5 / 64.6 / 442.2 / 145.7 MB. Pixels identical, no slowdown (`benchmarks/memory_2026-09-17.*`, `decode_before_after_2026-09-17.tsv`).
- Cooperative cancellation: `Decoder::decode_with(stream, &dyn enough::Stop)`, checked per region / channel chunk / network layer / synthesis tile; `Error::Cancelled`.
- UDI (user-defined information) substream exposed as `Headers::user_data`.
- Progressive decode: `Decoder::max_channels` / `--max-channels` read only a prefix of the latent channels, matching the reference's `num_decode_chs` on three settings.
- Packed model bundles: `weights::packed` (`ZJM1` checkpoints, `ZJB1` bundles, `PackedBundle` model source, `Recorder`), `zenjpegai pack-models`, and `--models <bundle file>`; a (model, operating point) pair shrinks to 15-29 MB with pixels bit-identical to the `.pth` path. `ModelSource::accessed` + `model::with_checkpoint` (additive).
- Chroma-subsampled and 10-bit pictures: bicubic chroma up-sampling (bit-identical to PyTorch's kernel), YUV output for YUV sources (`Decoder::decode_picture`, `Picture`, `YuvImage`), 10-bit quantisation, non-displayed border; CLI writes `.yuv` and 16-bit PNG.
- Quality map (spatially varying quantisation), bit-exact on three reference streams.
- WebAssembly: `Wasm128` SIMD tier and the wasm32 numeric policy (unfused multiply-add on every wasm tier: bit-identical across wasm engines, within 1 of native in under 1/48000 samples); the whole test suite runs on `wasm32-wasip1` under node (`just test-wasi`, `scripts/wasm/`).
- EFE linear and EFE non-linear post-filters (decoder side) for every chroma format the reference software can produce; checked filter-by-filter against 17 reference streams (non-linear filter bit-identical, linear within 9.2e-5 on a 0..255 scale) and through the whole decoder on 7.
- Post-filters LEF (bit-identical to the reference on its own input) and eICCI (4:4:4; per-tile network selection, own overlapping tiling, networks on `nn::fast`, cached in `Decoder`), checked per filter and through whole decodes (hand-assembled and through `Decoder`) of five reference streams, upstream's all-tools configuration included; filter benchmark in `benchmarks/filters_lef_icci_2026-09-17.tsv`.
- Coding tools: residual variance scaling (RVS), channel gain flags (GRFS) and latent scaling before synthesis (LSBS), bit-exact in the entropy stage on six reference streams.
- HOP synthesis (high operating point): residual blocks, CAB, TAM transformer blocks; depthwise convolution kernel and attention math that are bit-identical on every CPU tier.
- Convolutions read their input in place (virtual zero padding) and tensors recycle their storage through a bounded pool; `Decoder::release_buffers`.
- `Decoder` (one-call decode with a model cache, errors traced with whereat) and the `zenjpegai` command line tool (`cli` feature, PNG output through zenpng).
- End-to-end decode benchmark against the reference software (`scripts/bench/`, `benchmarks/decode_end_to_end_2026-09-17{,b}.tsv`).
- Region-partitioned reconstruction (dependent and independent regions, region-aware synthesis tiles), checked on three reference region streams.
- `tools::tiles` + tiled synthesis: pictures of any size decode (overlapping synthesis tiles, overlap halves discarded), checked against the reference on a 2096x1400 stream.
- `nn::fast`: blocked (NCHWc) SIMD convolution engine on archmage/magetypes (AVX-512, AVX2, NEON, scalar; rayon over rows), bit-identical to the reference layers on every tier; int8 `madd` convolution for the hyper-scale decoder. All models now run on it.
- Repository skeleton.
- `bitio`, `container`: MSB-first bit IO, Exp-Golomb, marker/region/thread container parsing and writing.
- `nn` (reference layers defining the numeric contract), `model::{hyper_decoder, mcm, synthesis}`, `decoder::{reconstruct, output}`: first complete decode path (SOP/BOP, single tile, 4:4:4 → RGB). Correct but not yet fast.
- `decoder::entropy` with `model::{hsd, common}`, `tools::{gain, skip, regions}`: entropy stage of the decoder (z, integer hyper-scale decoder, sigma indices, skip mask, residual), bit-exact on 12 reference streams.
- `header`: picture header, tool header flags and rendering information, parse and write, with profile/level conformance checks.
- `weights`: reader for the upstream PyTorch `.pth` checkpoints (no Python needed; the pickle interpreter is data-only).
- `mans`: me-tANS entropy coder (tables, decoder, encoder), bit-exact against the reference C++ extension.

### Fixed
- Demo asset staleness (`web/`): stream/bundle files keep their names across `demo-assets-v1`
  swaps, so URL-keyed caches (HTTP + Cache API) served old bytes indefinitely — the live demo
  showed a stale image after an asset swap. `manifest.json` now carries each variant's
  `file`/`sha256` and a per-bundle digest map (`web/scripts/update-demo-manifest.mjs`,
  uploaded 2026-09-18); `demo.js` fetches `streams/<file>?v=<sha256>` and the manifest itself
  with `cache: 'no-cache'`; `worker.js` keys Cache API entries on `?v=` digests
  (`zenjpegai-models-v2`, the unversioned v1 cache is deleted). `DecoderPool`/`polyfill` take
  an optional `bundleVersions` map. `web/tests/cache-swap.spec.ts` proves a same-name swap
  renders new content on reload without clearing storage. `playwright.config.ts` ports are
  shiftable via `JAI_PORT_BASE` so sibling workspaces stop testing each other's `dist/site`.
- Browser decode scheduling (`web/`): `DecoderPool` now runs all decodes through one shared
  priority queue instead of posting every job to a worker at once — at most
  `min(navigator.hardwareConcurrency - 1, N)` in-flight on the simd build (one hardware thread
  left for the page's main thread) and exactly one on the
  threads build — and the polyfill + demo only enqueue an image once it is within one viewport
  height of the viewport, in visibility order (`IntersectionObserver`, re-evaluated function
  priorities), with a pre-sized placeholder while queued. `pool.stats()` and
  `timings.queued` expose the scheduling for tests; `web/tests/scheduling.spec.ts` covers the
  bounds (first-image bound, `maxInflight`, per-image decode <= 2x solo).
- Demo: the `coi-loader.js` service-worker promotion wait is bounded at 1.5 s so a stalled or
  blocked SW install can no longer hang the page's top-level await.
- Demo: the Art Institute of Chicago slot is now corpus item 3008 (a Van der Spelt
  flower-garland trompe-l'oeil, CC0), replacing the removed dead-chicken still life
  (`web/demo/IMAGES.md`, `demo-assets-v1` release updated and hash-verified).
- GPU presentation on hardware (711233ad): the `rgba8unorm` texture is rounded in the shader (half to even, like the CPU output stage) instead of relying on the driver's float-to-unorm conversion, which on NVIDIA disagreed with the CPU output stage on 46,533 of 1,491,840 samples (28 after the fix, all by one step). llvmpipe had agreed, so it only showed on hardware.
- `Decoder::decode` rejected LSBS streams although LSBS was ported (a stale check; the staged test path did not go through `Decoder`). `tests/decode_ref.rs` now also decodes every stream through `Decoder` and requires identical samples.
- Tool header: `icci_enable_flag` is not coded for 4:2:0 sources; streams with EFE non-linear or LEF data after it parsed wrongly.
