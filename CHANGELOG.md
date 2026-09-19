# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->
- `EncodeParams` no longer derives `Eq` (the new `eicci` config carries `f32` loss weights).

### Added

#### Encoder
- Encoder: `num_chs` below the model's channel count (`E10`) — `EncodeParams::num_chs` /
  CLI `--num-chs y,uv` port the reference's
  `-model.CCS_SGMM.tools_common.model_{y,uv}.common_modules.num_chs`: the picture header
  signals the clamped count per component, the residual substream codes only the first
  `num_chs` latent channels and the tail keeps a zero quantised residual, with the
  reference's cube-flag semantics on uncoded channels (inert on the context-module luma,
  the full `|y - psi|` error on the context-free chroma) and `grfs_channel_flag`
  truncated to the coded count. Five new reference vectors (`make_reference_streams.sh
  numchs`): four streams byte-identical to the reference encoder's (the tiled 2096x1400
  one moves 1 symbol by 1, same length), all decode through our decoder and the
  reference decoder. The rate matcher's `find_UV_beta_with_hyperopt` UV search is not
  ported — it is unreachable in the pinned reference commit (`hyperopt` is commented
  out of `requirements.txt` and absent from the environment; `MSSSIM` is never imported
  in `bitrate_matcher.py`), documented under "Reference dead code" in `PORTING.md`.
- Encoder: the `tools_on` preset (`E5`) — `EncodeParams::tools_on` / CLI `--tools-on`
  enable the whole `cfg/tools_on.json` tool set at once (RVS, GRFS, LSBS, EFE linear,
  eICCI, EFE non-linear, LEF); the individual flags `--efe-linear`, `--efe-nonlinear`
  and `--eicci` work alongside the existing `--lef` / `--rvs` / `--grfs` / `--lsbs`.
  Gate on all ten CTC points (`img30`/`img01` x 0.12-1.00 bpp, new `toolson` reference
  vector set): the reference's `(model, beta)` pick every time, every tool signalled,
  entropy payload symbol-identical except one-step rounding-boundary moves. The
  post-filter decisions differ from the reference's (deterministic f64 least-squares
  vs MKL's nondeterministic f32 `lstsq`) and reconstruct *better* on every vector
  (+0.017..+0.122 dB through our decoder and the reference's alike) at +0.07..+1.93 %
  size, all of it the larger TON header — measured numbers and the mechanism in
  `PORTING.md` → "Encoder parity → tools_on".
- Encoder: the EFE non-linear post-filter (`E2`) — `EncodeParams::efe_nonlinear` / CLI
  `--efe-nonlinear` run the reference's `EFEnonlinear.compress` on the post-eICCI
  reconstruction (`src/encoder/filters/efe_nonlinear.rs`): the 1200-px tile grid snapped to
  64, per-tile `lumaMin`/`lumaMax`, the eight ReLU-hinge weight solve per tile per chroma
  plane (`integerize`, raw u16 codes — the header's `minSymbol`/`maxSymbol` fold pins
  0/65535), per-plane enable by `lossModifier`-scaled loss, and the W5 on/off-mask search
  (`block_sizes`/`base_model_beta` by `model_id`, `candNum`, strictly-positive keep rule)
  against EFE linear's up-sampled picture. The solve reuses E1's deterministic f64
  pivoted-QR `gelsy`: **272/272 tile weight codes exact** against the oracle's f64
  re-solve (`dgelsy`); MKL's f32 draw is nondeterministic on these rank-borderline hinge
  matrices (the coded stream and a `compress` replay disagree on enable flags on identical
  inputs), so enable/mask/output parity is measured, not asserted exactly (11/34 enable
  outliers vs both reference draws, 3 mask keep/drop diffs, 289/1234 mask blocks, max
  output diff 19.2/255). The stock reference decoder accepts our
  `efe_linear + efe_nonlinear` stream (decoded output within 1).
- Encoder: the eICCI model-selection search (`E3`) — `EncodeParams::eicci` / CLI
  `--eicci [--eicci-loss mse|ms-ssim|mixed] [--eicci-long-list] [--eicci-tile-samples N]`
  run the reference's `icci_filter.py::compress` + `model_idxes.py::encode_header` once on
  the final reconstruction (`src/encoder/filters/icci.rs`, `src/encoder/msssim.rs`):
  per-tile luma/chroma candidate search over the decoder's own eICCI networks, the shipped
  `mixed` loss or pure `mse` / `ms-ssim` with configurable weights, the short lists or the
  whole bank, the filter's `numSamplesPerTile` tiling, and the 4:4:4-source gate. `ms_ssim`
  is a fixed-order `pytorch_msssim 0.2.1` port (f32 elementwise ops, f64 reductions) —
  max abs error 2.0e-6 against the reference's own scores and bit-identical on every SIMD
  tier. Selection parity: 0 of 9 tile selections differ from the reference on the four
  eICCI vectors; our streams carry the identical `IcciHeader`, decode with worst diff 0,
  and the reference decoder reads them.
- Encoder: the EFE linear post-filter (`E1`) — `EncodeParams::efe_linear` / CLI
  `--efe-linear` run the reference's `EFElinear.compress` on the final reconstruction
  (`src/encoder/filters/efe_linear.rs`): region-split candidate search x filter lengths,
  `LumaAidedUpsampler_encoder` least-squares solves (coded 4:4:4 / 4:2:2 / 4:2:0, the
  luma-aided refine, the >2 M-sample subsample-averaging), `integerize`,
  `minSymbol`/`maxSymbol` and the second "up-sampled" filter set, all behind the existing
  TON header. The solver is a deterministic f64 pivoted-QR `gelsy`: identical integerised
  codes to the reference wherever its MKL f32 `gelsy` is well-posed (64/146 dumped solves;
  every design matrix bit-identical), and documented deviation where MKL's rank estimate
  is nondeterministic — on `img30_base_efelin_bpp050` our stream is +0.28 % and **+0.099 dB**
  through the reference decoder.
- Encoder: `RateEstimate::Likelihood` (`EncodeParams::rate_estimate`, CLI
  `--rate-estimate likelihood`) — the reference bitrate matcher's `ECLibLH` trial measure
  (`src/encoder/lh.rs`: the factorized `z` model's float `forward` as per-channel likelihood
  tables, plus `GMProbModel.forward`'s unquantised Gaussian on the residual, with the
  reference's effective -10 %/+5 % tolerances and unclamped bisection window). It picks the
  reference's `(model, beta)` exactly at all five CTC rates on both test pictures; the coded
  measure stays the default.
- `zencodec` encode bridge (`E9`): `JpegAiEncoderConfig` → `JpegAiEncodeJob` →
  `JpegAiEncoder` behind the `zencodec` feature, mirroring the decoder side — the config
  carries `Arc<dyn ModelSource>` + `Engine` + `EncodeParams` + `EncodeLimits` and shares one
  `Encoder`'s model cache across clones/jobs. `with_generic_quality(q)` rate-matches to
  `q / 100` bits per pixel (`Encoder::encode_to_bpp`, the `RateMatch` reported back as an
  output extension); `estimate_encode_resources` reports the calibrated
  `estimate_encode_memory`; per-job `with_limits`/`with_stop` layer onto the config's.
  Accepted input: full-range RGB8/RGB16 `PixelSlice` (sRGB-tagged or untagged) and
  padding-alpha RGBA8 via `encode_srgba8`; other descriptors, row-push and animation are
  rejected with `UnsupportedOperation`. Proven byte-identical to `Encoder::encode` /
  `encode_to_bpp` (`tests/codec_ref.rs`).
- Encoder resource limits and a memory estimate, mirroring the decoder's (`E8`):
  `EncodeLimits` (max pixels / dimensions / estimated heap; default 120 MP and 4 GiB),
  `Encoder::limits`, `estimate_encode_memory` / `Encoder::estimate_memory` (fixed-model or
  rate-matched) and `Encoder::release_buffers`. Bounds are checked on the source size before
  any network runs; `max_memory_bytes` also shrinks the shared recycled-buffer pool, and
  `--pool-mb` now applies to `zenjpegai encode` too. Rate matching drops losing models'
  latents during selection (no more deep clones; only the best-so-far set is kept), trial
  streams die once measured, each component's residual payload is coded before the next
  component's compress runs, and non-traced encodes shed scales / masks / quantised residuals
  right after their payload is coded. heaptrack: `benchmarks/memory_encode_2026-09-18.tsv` —
  the estimate is 1.30x–1.44x of the measured pool-free peak heap at 560x888 and 2096x1400,
  fixed and rate-matched (gate: within 1x–2x).
- Encoder: chroma-subsampled, 10-bit and YUV sources. `SourceImage::{Rgb, Yuv}` with
  `SourceImage::read_yuv` (planar YUV whose `WxH_Nbit_{420,422,444}` file name carries the
  geometry, like the reference's `extract_info`), `SourceMeta` with `s_ver`/`s_hor` from the
  chroma plane size and `c_ver`/`c_hor` from `EncodeParams` (`-c_ver_value`/`-c_hor_value`;
  4:2:2 coded as 4:2:0 is `Unsupported`, as upstream), `EncodeParams::diff_display` for the
  non-displayed border, and 16-bit PNG input (`bit_depth_idc` 4). Chroma down-sampling is
  `nn::resize_bilinear`, bit-identical to PyTorch's bilinear
  `align_corners=True` kernel (`tests/vectors/nn/bilinear_align_corners.bin`). CLI:
  `zenjpegai encode in.yuv …`, `--c-ver`/`--c-hor`/`--diff-display`. On the nine `formats`
  vectors the analysis-transform inputs are bit-for-bit the reference's, seven streams are
  byte-identical, and the two 10-bit ones are the same length with 5 of 1,254,400 luma
  residual symbols moved by 1; the reference decoder reads all nine.
- Encoder: the LEF post-filter's share (`EncodeParams::lef`, `zenjpegai encode --lef`) — the
  reference channel `LEF_chIdx` is derived from the luma scale map as `LEF.analyze` does it
  (`encoder::filters::lef`; the filter itself stays decoder-side, the reference runs it after
  the rate loop). The new fixed-model vector `enc_img30_bop_m1_b0_lef` encodes byte-identical
  to the reference's, and the chosen channel matches the reference on all five LEF streams.
- **Encoder** (`encoder::Encoder`, `zenjpegai encode in.png out.bits --model N --beta-disp B [--op sop|bop|hop]`): colour pre-processing, analysis transforms, hyper-encoder, `z` quantisation, the shared scale derivation, the context model's compress direction (per-stage quantise / cube flag / requantise), skip mode, residual quantisation, header and container assembly, one `AnsEncoder` per substream. Fixed model and operating point, one tile, one region, one ANS thread, tools off, 8-bit RGB 4:4:4. On all seven reference vectors the integer stage (`z_hat`, both scale maps, cube flags) equals the reference encoder's, six of seven streams are **byte-identical** to the reference's (the seventh, 178 KB at beta +400, moves 42 of 501,760 residual symbols by 1), and the reference decoder reads every one of them, differing from our own decode in 51-59 of 1,491,840 samples, all by 1. 0.12 s of process wall time against the reference encoder's 2.0 s at 560x888.
- Encoder: the quality map (`Encoder::encode_with_quality_map`, `zenjpegai encode --quality-map mask.png`): the ROI-mask generator, the entropy-index choice, the DPCM deltas and their SOQ substream. Both reference encodes that use it come out byte-identical.
- Encoder: coding tools - RVS, GRFS (the gain flags derived as `analyzeCWG` does), LSBS, 1..16 ANS threads per substream, and region partitioning in both modes (`EncodeParams`, `zenjpegai encode --rvs --grfs --lsbs --ans-threads N --regions dependent|independent`). All five tool vectors come out byte-identical; the dependent-region one is byte-identical given the reference's own latents and both region streams are its length from the PNG.
- Encoder: analysis tiling (`encoder::tiles`) - above 1 MP the analysis transform and the hyper-encoder run per tile (1024 / overlap 64 luma, 512 / 32 chroma) and the picture header carries the synthesis tiling; the 2096x1400 reference encode comes out at the reference's byte count with one residual symbol moved.
- Encoder: rate matching (`Encoder::encode_to_bpp`, `zenjpegai encode --bpp R`), mirroring `bitrate_matcher`'s model pre-selection + log-rate interpolation + bisection but coding every trial instead of estimating it. Same model as the reference at all five CTC rates and closer to the target at each (worst -3.5 % against the reference's -10.1 %); 12-15 trial encodes, 0.55 s wall at 560x888.
- Encode benchmark against the reference encoder (`scripts/bench/{encode_end_to_end.sh,ref_encode.py}`, `benchmarks/encode_end_to_end_2026-09-18.{tsv,meta}`): 9.3-13.6x its one-thread `TOTAL` at the same fixed model.
- Encoder groundwork only (no encoder yet): analysis transforms (`model::analysis`, BOP + HOP) and hyper-encoder (`model::hyper_encoder`) with loaders, within 1.1e-4 of the reference encoder's tensors and `z_hat` symbol-exact (`tests/encode_ref.rs`); `dump_encode.py --enc2` and the `encoder` reference-vector set.

#### Decoder
- eICCI on chroma-subsampled pictures (4:2:0 and 4:2:2): the reference decoder's
  `Image.to_444_` bicubic chroma up-sampling, eICCI at luma size, then `to_format_` bilinear
  down-sampling back (`nn::resize_bilinear` reproduces PyTorch's channels-last
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
- `decoder::entropy::{ComponentScales, component_scales, dequantize_residual}`: the scale derivation both directions share (factored out of `decode_component`; decoder output unchanged).
- Native decode speed pass (d2native), same machine-code method as the wasm pass: generated
  AVX-512 showed the V4 tile `B = 28` spilled all 28 accumulators to the stack every
  `(input-channel, tap)` iteration — ~27 dead `vmovaps` stores per 28 FMAs. Retuned to
  `B = 24` (spill-free), replaced the B=8/4/1 tail cascade with one overlapping full-width
  block, unrolled the common tap counts (`ntaps` 1/4/9), made the int8 HSD conv keep its
  accumulators in registers across all input pairs (was a store+load per pair) at `B = 24`,
  gave the depthwise 3x3 a constant-trip tap loop, made `pad_par` write-once, and merged the
  two border scratch fetches into one. Single-thread decode: ~40 % faster (560x888 BOP
  138 -> 83 ms, HOP 2346 -> 1323 ms, 2096x1400 BOP 1041 -> 618 ms, interleaved A/B);
  ~30-35 % at 8 threads. Bit-identical output on every tier and thread count;
  `benchmarks/conv_kernels_2026-09-18.md`, `decode_end_to_end_2026-09-18_d2native{,_base}.tsv`.
  VNNI (`vpdpwssd`) examined and not used: archmage/magetypes expose no safe op for the
  packed layout.
- Decode-stage parallelism (d4serial): the two components' entropy bodies, the residual
  streams inside a multi-threaded (`ECThread8`) ANS substream (compact per-thread decode
  + scatter), independent latent-reconstruction regions, the per-region scale/mask/gather/
  dequantise helpers, the two synthesis transforms and output conversion now run on the
  rayon pool; `z` + quality-map decode overlaps both component chains, and the luma
  synthesis transform runs inside the luma chain so it overlaps the chroma chain's tail.
  Cancellation keeps its exact check sequence through a `Gate` wrapper that serialises
  `Stop::check` and memoises the trip. Bit-identical output on every tier and thread
  count (all `decode_ref`/`entropy_ref` vectors, `tiers_and_threads_agree_bit_for_bit`).
  Native interleaved A/B mins vs the pre-pass build: 1 thread ~unchanged, 8 threads
  −17-23 %, 16 threads −14-30 % (BOP 560x888 25→21 ms, img01 157→120 ms at 8T;
  `benchmarks/decode_end_to_end_2026-09-18_d4serial.tsv`). Browser threads package scales
  to 16 workers (~642→93 ms mean over 16 streams; appended block of
  `benchmarks/wasm_threads_2026-09-18.tsv`); GPU-path CPU decode time −13-39 %
  (`benchmarks/gpu_decode_2026-09-18_d4serial.tsv`). New `Decoder::decode_picture_stats`
  returning `DecodeStats` per-stage timings, CLI `decode --stats` (TSV on stderr), and
  `scripts/bench/decode_stages.sh`; stage breakdowns in
  `benchmarks/decode_stages_2026-09-18.{tsv,meta}` (native + wasip1 — on Wasm128 the
  remaining single-thread cost is synthesis kernels, 75-90 % of decode, not scheduling).
  `decode_picture` (and therefore `Decoder::decode`) output unchanged.
- `Limits` (pixels, dimensions, input bytes, estimated memory; safe defaults 120 MP / 4 GiB), judged on the picture header before any model load or picture-sized allocation; `estimate_memory` / `Decoder::estimate_memory` (model calibrated on heaptrack measurements, `benchmarks/memory_2026-09-17.*`); `Decoder::preload`; `nn::fast::set_pool_limit` and `zenjpegai --pool-mb / --discard / preload`.
- Decode memory: feature maps are freed as soon as the next layer has them, synthesis tiles go straight into the cropped / subsampled output planes, entropy-stage tensors are shed after use. Measured peak heap (heaptrack, decode only, default 1 GiB buffer pool): 560x888 SOP 83.7 to 64.7 MB, BOP 145.0 to 116.0 MB, HOP 1260 to 1190 MB, 2096x1400 BOP 389.5 to 309.1 MB; without buffer recycling (`--pool-mb 0`) 39.5 / 64.6 / 442.2 / 145.7 MB. Pixels identical, no slowdown (`benchmarks/memory_2026-09-17.*`, `decode_before_after_2026-09-17.tsv`).
- Cooperative cancellation: `Decoder::decode_with(stream, &dyn enough::Stop)`, checked per region / channel chunk / network layer / synthesis tile; `Error::Cancelled`.
- UDI (user-defined information) substream exposed as `Headers::user_data`.
- Progressive decode: `Decoder::max_channels` / `--max-channels` read only a prefix of the latent channels, matching the reference's `num_decode_chs` on three settings.
- Packed model bundles: `weights::packed` (`ZJM1` checkpoints, `ZJB1` bundles, `PackedBundle` model source, `Recorder`), `zenjpegai pack-models`, and `--models <bundle file>`; a (model, operating point) pair shrinks to 15-29 MB with pixels bit-identical to the `.pth` path. `ModelSource::accessed` + `model::with_checkpoint` (additive).
- Chroma-subsampled and 10-bit pictures: bicubic chroma up-sampling (bit-identical to PyTorch's kernel), YUV output for YUV sources (`Decoder::decode_picture`, `Picture`, `YuvImage`), 10-bit quantisation, non-displayed border; CLI writes `.yuv` and 16-bit PNG.
- Quality map (spatially varying quantisation), bit-exact on three reference streams.
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
- `bitio`, `container`: MSB-first bit IO, Exp-Golomb, marker/region/thread container parsing and writing.
- `nn` (reference layers defining the numeric contract), `model::{hyper_decoder, mcm, synthesis}`, `decoder::{reconstruct, output}`: first complete decode path (SOP/BOP, single tile, 4:4:4 → RGB). Correct but not yet fast.
- `decoder::entropy` with `model::{hsd, common}`, `tools::{gain, skip, regions}`: entropy stage of the decoder (z, integer hyper-scale decoder, sigma indices, skip mask, residual), bit-exact on 12 reference streams.
- `header`: picture header, tool header flags and rendering information, parse and write, with profile/level conformance checks.
- `weights`: reader for the upstream PyTorch `.pth` checkpoints (no Python needed; the pickle interpreter is data-only).
- `mans`: me-tANS entropy coder (tables, decoder, encoder), bit-exact against the reference C++ extension.

#### Browser & WebAssembly
- Browser startup fast path + Cloudflare Pages (`cfpages`, 2026-09-19): the demo now also
  deploys to `zenjpegai.pages.dev` — a `deploy-cloudflare` job in `pages.yml` runs
  `wrangler pages deploy` on the same `dist/site` artifact (classic Pages project, secrets
  `CLOUDFLARE_API_TOKEN` / var `CLOUDFLARE_ACCOUNT_ID`). `web/demo/_headers` sends
  COOP/COEP/CORP + immutable caching for the content-addressed paths, so the page is
  cross-origin isolated on the FIRST load and `coi-loader.js` never registers `sw-coi.js`
  there (GitHub Pages still needs its one reload). Model bundles are prefetched in
  parallel into the worker's Cache API keys while wasm still downloads (`web/src/prefetch.js`:
  first card's `common`+`op` pair first, every other needed model after, handed to the
  worker by transfer via `DecoderPool`'s `modelBuffers`; `ensureModels` fetches its halves
  concurrently), the polyfill prefetches the visible `<img>` set's models off a plain-JS
  PIH probe (`web/src/stream-info.js`), and `build-site.mjs` stamps preload links for the
  first pair from the manifest digests. `web/scripts/pack-rayon-child.mjs` collapses
  wasm-bindgen-rayon's per-child snippet+glue requests into one self-contained
  `rayon-child.js` cloned into every child through a shared blob URL. Measured
  (`benchmarks/wasm_startup_2026-09-19.tsv`, headless Chromium, cold profile): github.io
  pre 75 req / 58 JS / first-image 1464-2162 ms -> CF preview 29 req / 8 JS / 1351-1902 ms,
  `webgpu-threads`, no reload, zero service workers. Detail: `web/README.md` §8.
- WebGPU in the browser: `zenjpegai-wasm`'s optional `gpu` feature (`wasm/src/gpu.rs`: async `decode` with CPU fallback on any `GpuError`, `initGpu`/`gpuStatus`/`present`/`disableGpu`) builds a third package, `pkg-webgpu` (`web/scripts/build-wasm.sh webgpu`; `pkg-simd`/`pkg-threads` unchanged). `DecoderPool`'s `gpu` option (`auto|on|software|force-software|off`): `on` loads `pkg-webgpu` when `navigator.gpu` yields a non-software adapter; `auto` (the default) uses the GPU on a non-software adapter — measured faster than both CPU packages on an RTX 2080 after the b3gpu pass (web/README §7). `decodeToCanvas` transfers an `OffscreenCanvas` to the worker once and presents through `Blitter` with no CPU readback (`presented: 'gpu'`), falling back to worker-side `putImageData` (`'2d'`). The worker's `ready` message carries `adapterProbe` (JS-side `GPUAdapter.info`: vendor/architecture/`isFallbackAdapter`) because wgpu's web backend leaves the adapter name empty under Dawn. Verified on hardware (`WEBGPU_ADAPTER=hardware` → Chromium + `--use-angle=vulkan --ignore-gpu-blocklist`, adapter `nvidia/turing`, `isFallbackAdapter: false`; the specs fail loudly if no hardware adapter appears) and via SwiftShader (`chromium-webgpu` Playwright project, `webgpu-decode.spec.ts`: 203/3,000,000 samples differ, all by 1 — GPU parity reported separately); per-path timings in `benchmarks/wasm_decode_2026-09-18_gpu.{tsv,meta}`, size cost in `benchmarks/wasm_size_2026-09-18.md` §7. Worker-level trap recovery (`self.onerror` → `disableGpu` → one transparent CPU retry) remains as defence in depth.
- Browser: worker pool + `<img>`/`<picture>` JPEG AI polyfill (`web/src/{worker,pool,polyfill}.js`), a ~90-test Playwright suite across chromium/firefox/webkit/chromium-webgpu and three CSP/isolation profiles, a demo (`web/demo/`) with 8 imazen-26 images published as the `demo-assets-v1` release, and `.github/workflows/pages.yml` (Pages enabled, site live — see `web/README.md`). Fixed the `threads` wasm package, which never actually linked/ran before this: missing `--shared-memory`/`--import-memory`/`--export=__wasm_init_tls` linker flags, plus a worker.js message-queue race and a reentrant-`decode()` deadlock. New additive `pth` cargo feature (default on) keeps the `.pth`/pickle reader out of the wasm build (`pkg-simd` 389,178 → 362,789 bytes after `wasm-opt`); full size audit in `benchmarks/wasm_size_2026-09-18.md`.
- Browser decode speed pass (d3wasm): ~2x faster on the demo streams — median 204 → 104 ms
  (chromium), 227 → 107 ms (firefox), 238 → 131 ms (webkit) on a 32-hwc Ryzen 9 9950X3D
  (`benchmarks/wasm_decode_2026-09-18.tsv`, `wasm_profile_2026-09-18.md`). The conv micro-kernel
  is 88% of decode and already at its bit-identity floor, so the wins came from outside it:
  the rayon pool now defaults to `min(hardwareConcurrency, 16)` children (a 32-child pool
  regressed past the 16-core knee), the per-row conv border scratch is `thread_local` (shared
  memory makes every malloc atomic-linked), and wasm-bindgen init uses the single-object form
  (deprecation warning gone, incl. wasm-bindgen-rayon's spawned workers via a build-time patch
  in `build-wasm.sh`). `DecoderPool`/`worker.js` take a `threads` override used by
  `web/tests/threads.spec.ts` for the scaling table.
- Wasm conv kernel pass 2 (`nn::fast::conv`): TurboFan machine-code inspection showed the
  previous pass's "floor" was wrong — accumulators were spilled to memory per input block and
  every tap carried bounds-check branches. Restructured `block()` so the B accumulators stay in
  registers across the whole input-channel loop (bias initialises them directly, tap rows are
  sliced once to `(B-1)*S+1` with a single provable `assert!`, the common `vin == V` channel
  loop unrolls so splat offsets fold into load instructions, padding taps alias a static
  `ZERO_PAD` — the per-call zero `Vec` is gone). Single-threaded decode of the 560x888 profile
  stream under node: 356.6 -> 307.1 ms (-13.9%); wasmtime 40: ~920 -> 365 ms (-60%). Output is
  bit-identical across wasm tiers and thread counts; `benchmarks/wasm_kernel_2026-09-18.md` has
  the instruction counts and the rejected variants (B=2/6, paired taps, ntaps==9 unroll,
  iterator positions).
- `wasm/` crate (`zenjpegai-wasm`, wasm-bindgen: `addModels` / `info` / `decode`, single-thread SIMD128 build and a rayon `threads` build) and `web/scripts/build-wasm.sh`; `pkg-simd` is 389 KB (131 KB brotli). Polyfill, tests and demo are not written yet (`web/README.md`).

- WebAssembly: `Wasm128` SIMD tier and the wasm32 numeric policy (unfused multiply-add on every wasm tier: bit-identical across wasm engines, within 1 of native in under 1/48000 samples); the whole test suite runs on `wasm32-wasip1` under node (`just test-wasi`, `scripts/wasm/`).

#### GPU
- `gpu/` crate (`zenjpegai-gpu`): SOP / BOP / HOP synthesis as WGSL compute shaders through wgpu 30 (native + wasm32), `GpuDecoder`, readback-free RGBA presentation, `just gpu-test`, `gpu_bench` example.
- GPU backend on hardware (711233ad): `just gpu-test` passes on an RTX 2080 (NVIDIA 580.178.04) with the parity bounds unchanged from llvmpipe, and the first real timings are committed - `benchmarks/gpu_decode_2026-09-17_rtx2080.{tsv,meta}` (64 to 4096 plus the reference streams; `..._pretuning.tsv` is the same sweep before the tuning, `..._radv_igpu.{tsv,meta}` the integrated Radeon), `gpu_profile_2026-09-17_rtx2080.{tsv,meta}` (per kernel, before and after), `gpu_reference_2026-09-17.{tsv,meta}` (the reference software on the same card, via the new `scripts/bench/reference_gpu.sh`). The GPU overtakes this crate's 8-thread CPU engine at about 384x384 (SOP), 256x256 (BOP) and between 128x128 and 256x256 (HOP).
- GPU kernel tuning (711233ad, 0120c800), all of them keeping every output's summation order so the parity numbers do not move: the ResAU gate and the residual add are computed in the convolution's store rather than a pointwise dispatch (dispatches per picture SOP 36 -> 30, BOP 33 -> 27, HOP 172 -> 158); the transposed convolution visits only the taps its divisibility test used to keep (4 of 16 at k=4 stride 2; `convt_k3_s2` in HOP 36.5 -> 28.1 ms); the depthwise 3x3 stages its halo in workgroup memory (134.4 -> 64.8 ms); the convolution's workgroup edge is 16x16 instead of 8x8 on layers with at least 16 input blocks and both output dimensions at least 128 (HOP's 1x1 convolutions 162.4 -> 117.3 ms). Device time for one picture: BOP 560x888 10.80 -> 9.67 ms, BOP 1024x1024 23.91 -> 21.31 ms, HOP 560x888 259.4 -> 223.5 ms, HOP 1024x1024 635.3 -> 498.0 ms; SOP unchanged, because none of the four reaches its network except the fusion.
- GPU per-dispatch profiling (711233ad): `Workspace::set_profile`, `GpuPicture::dispatch_ns`, `gpu_bench --profile`, `just gpu-profile`; `gpu_bench` also writes its TSV after every row.
- GPU work queue B3 (b3gpu2, 2026-09-18): bounded buffer retention (the activation `Pool` and the workspace's latent / picture / staging buffers release the excess of a large picture on the next small run — 568 -> ~195 MiB for 4096x4096 then 560x888 — and `GpuDecoder::release_buffers` / `Workspace::release_buffers` drop them explicitly); GPU-side output conversion (`GpuOut::Quantized` runs the output stage — BT.709 or coded-subsampling YUV, 8/10 bit, round-half-to-even — in an `emit_output` kernel and reads back packed u16 pairs, halving readback; decode wall −20 to −29%); cancellation and `max_channels` on the GPU path (`decode_with` / `decode_to_gpu_with` / `decode_picture_async_with` take the same `enough::Stop` token, checked between submits); and a second kernel-efficiency pass driven by `gpu_bench --profile` + `scripts/bench/gpu_roofline.py` roofline classification: workgroup-staged input halos and weight chunks for stride-1/2 convolutions (selected per layer where the tile fits 16 KiB of workgroup storage; its `ic0`-chunk-outer accumulation reorders `KH > 1` sums, so the measured parity counts moved slightly within their bounds — 8-bit vs reference now 47 / 59 / 33 of 1,491,840 and 345 / 368 of 8,803,200 samples, all one step), channel-contiguous depthwise workgroups (`depthwise3x3` 27.0 -> 7.8 ms), a flattened `elu_gate` (11.4 -> 1.4 ms, ~88% of bandwidth peak), and channel-group constants baked into the conv / conv-transpose pipelines. HOP 560x888 device 224 -> 103 ms, HOP 1024x1024 498 -> 208 ms, HOP 4096x4096 8919 -> 4045 ms, BOP 1024x1024 21.4 -> 14.5 ms, SOP ~−13%; whole-stream decode BOP 560x888 22.8 -> 17.4 ms, HOP 326 -> 109 ms, img01 2096x1400 164 -> 93 ms. Numbers: `benchmarks/gpu_decode_2026-09-18_rtx2080_kernels.tsv`, `benchmarks/gpu_profile_2026-09-18_rtx2080.tsv`; `gpu/README.md` "Status" has the analysis, the falsified approaches (pixel-tiled convolutions, still slower), and the remaining work (`shader-f16`, `convt` staging, attention kernels).

#### Library & infrastructure
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
- Decoder fuzzing (`fuzz/`, cargo-fuzz + libFuzzer): three targets over the untrusted-input
  surface — `container_headers` (marker scan, `Codestream::parse`, header/TON/RDI/UDI
  parsing, region and thread split helpers), `entropy_stage` (`decode_entropy_stage*` on
  arbitrary bytes against a fixed synthetic zero-weight `ZJM1` model, with a counting
  `Stop`), and `decode_full` (`Decoder::decode_with` under tight `Limits` and
  cancellation, on arbitrary bytes and mutated real streams). Synthetic checkpoints let
  the full pipeline run without committing model weights. `fuzz/zenjpegai.dict`
  dictionary, `fuzz/seeds/` corpus (47.9 KB of real + truncated reference streams),
  `tests/fuzz_regression.rs` replays `fuzz/regression/*` through the same entry points on
  stable, `just fuzz` recipe, additive CI job (nightly target build + stable replay),
  corpora/artifacts sync to `/mnt/v/fuzzes/zenjpegai/` via
  `~/work/zen-workspace/fuzz-sync.sh`. First campaign: no crash, timeout, OOM, or
  slow-unit findings.
- Repository skeleton.

### Fixed
- Fuzz infrastructure (review pass): `tests/fuzz_regression.rs` panics on an
  unreadable `fuzz/regression/` (it previously replayed as an empty success) and
  now also replays the committed `fuzz/seeds/` streams through every entry
  point; the CI fuzz job and `just fuzz` pin `--target x86_64-unknown-linux-gnu`
  (cargo-fuzz 0.13 otherwise defaults to musl, installed on neither). Review
  coverage added for the d4serial helpers: `Gate` determinism/stickiness
  tests under rayon, and a serial-vs-pooled full-decode bit-equality test over
  the region/qmap/thread vectors (`serial_and_pooled_decoders_agree_bit_for_bit`).
- GPU plan cache invalidated mid-run on heterogenous tiles (`gpu/`): `Pool::fit` bumped
  `generation` on every slot replacement, and the plan cache drops everything when the
  workspace stamp moves — so a multi-tile run like 2048x2048 (tiny edge tiles alternating with
  large interior ones) shrank the pool on each small tile and regrew it on each large one,
  evicting every plan built so far and rebuilding all 9 on the next warm run (`gpu_bench`'s
  `plans_built == 0` assertion caught it). Growth no longer bumps the generation — a plan's
  slot binds are self-contained scratch — and shrink runs only at the first `fit` of a run
  (`Pool::begin_pass`, called by `run_tailed`); correctness was never affected, only plan
  reuse and the pinned buffers' lifetime.
- Browser WebGPU per-call overhead (b3gpu): per-phase timing
  (`benchmarks/wasm_gpu_phases_2026-09-18.tsv`) showed the ~150 ms gap between GPU wall decode
  and device time was ~120 ms of serial single-threaded CPU latent work in `pkg-webgpu`, not
  upload/submit cost. New `pkg-webgpu-threads` package (`build-wasm.sh webgpu-threads`: webgpu
  + rayon; `worker.js` picks it on cross-origin isolated pages) parallelises the CPU stages;
  `gpu/` now records all tile plans + timestamp resolve + a caller-chosen tail (planes copy,
  `rgba8unorm` convert, staged RGBA readback — `GpuOut` on `GpuDecoder::decode_to_gpu_with`)
  into one command encoder, and `GpuContext::map_read2` maps the staging and timestamp buffers
  in one device-drain round-trip. Warm ~1 MP `decode_ms` on the RTX 2080 through Dawn/Vulkan:
  ~179 -> ~46 ms (vs `threads` ~88 ms, `simd` ~640 ms medians on
  `benchmarks/wasm_decode_2026-09-18_gpu.tsv`), so `auto` now takes a hardware
  adapter on isolated and non-isolated pages alike. Parity measured per pass (203 of
  3,000,000 samples differ by 1 in-browser after `conv_tiled`'s reassociation, was 195).
- WebGPU upload trap (`gpu/`): `create_buffer_init` created buffers with `mappedAtCreation`,
  which over Dawn's staging limit throws a JS `RangeError` that wgpu-30's web backend
  `unwrap()`s into an uncaught wasm trap (`panic = "abort"` — it escaped the decode promise
  entirely and the device died mid-benchmark). All buffer uploads now go through unmapped
  `create_buffer` + `GpuContext::write_buffer` (chunked `queue.write_buffer`, 16 MiB) — no
  `mappedAtCreation` call remains in the crate. Native `gpu-tests` parity counts are
  byte-identical before/after; the browser hardware run (RTX 2080, Dawn/Vulkan) decodes every
  demo stream on the GPU path with zero traps.
- Demo asset staleness (`web/`): stream/bundle files keep their names across `demo-assets-v1`
  swaps, so URL-keyed caches (HTTP + Cache API) served old bytes indefinitely — the live demo
  showed a stale image after an asset swap. `manifest.json` now carries each variant's
  `file`/`sha256` and a per-bundle digest map (`web/scripts/update-demo-manifest.mjs`,
  uploaded 2026-09-18); `demo.js` fetches `streams/<file>?v=<sha256>` and the manifest itself
  with `cache: 'no-cache'`; `worker.js` keys Cache API entries on `?v=` digests
  (`zenjpegai-models-v2`, the unversioned v1 cache is deleted). `DecoderPool`/`polyfill` take
  an optional `bundleVersions` map. `web/tests/cache-swap.spec.ts` proves a same-name swap
  renders new content on reload without clearing storage. `playwright.config.ts` ports are
  shiftable via `JAI_PORT_BASE` and default to a port derived from the workspace path, so
  sibling workspaces stop testing each other's `dist/site`.
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
