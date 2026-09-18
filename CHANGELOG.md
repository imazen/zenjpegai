# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->

### Added
- **Encoder** (`encoder::Encoder`, `zenjpegai encode in.png out.bits --model N --beta-disp B [--op sop|bop|hop]`): colour pre-processing, analysis transforms, hyper-encoder, `z` quantisation, the shared scale derivation, the context model's compress direction (per-stage quantise / cube flag / requantise), skip mode, residual quantisation, header and container assembly, one `AnsEncoder` per substream. Fixed model and operating point, one tile, one region, one ANS thread, tools off, 8-bit RGB 4:4:4. On all seven reference vectors the integer stage (`z_hat`, both scale maps, cube flags) equals the reference encoder's, six of seven streams are **byte-identical** to the reference's (the seventh, 178 KB at beta +400, moves 42 of 501,760 residual symbols by 1), and the reference decoder reads every one of them, differing from our own decode in 51-59 of 1,491,840 samples, all by 1. 0.12 s of process wall time against the reference encoder's 2.0 s at 560x888.
- Encoder: analysis tiling (`encoder::tiles`) - above 1 MP the analysis transform and the hyper-encoder run per tile (1024 / overlap 64 luma, 512 / 32 chroma) and the picture header carries the synthesis tiling; the 2096x1400 reference encode comes out at the reference's byte count with one residual symbol moved.
- Encoder: rate matching (`Encoder::encode_to_bpp`, `zenjpegai encode --bpp R`), mirroring `bitrate_matcher`'s model pre-selection + log-rate interpolation + bisection but coding every trial instead of estimating it. Same model as the reference at all five CTC rates and closer to the target at each (worst -3.5 % against the reference's -10.1 %); 12-15 trial encodes, 0.55 s wall at 560x888.
- Encode benchmark against the reference encoder (`scripts/bench/{encode_end_to_end.sh,ref_encode.py}`, `benchmarks/encode_end_to_end_2026-09-18.{tsv,meta}`): 9.3-13.6x its one-thread `TOTAL` at the same fixed model.
- `decoder::entropy::{ComponentScales, component_scales, dequantize_residual}`: the scale derivation both directions share (factored out of `decode_component`; decoder output unchanged).
- `gpu/` crate (`zenjpegai-gpu`): SOP / BOP / HOP synthesis as WGSL compute shaders through wgpu 30 (native + wasm32), `GpuDecoder`, readback-free RGBA presentation, `just gpu-test`, `gpu_bench` example. Parity gated on llvmpipe only; not yet run or timed on a hardware GPU (`gpu/README.md`).
- Encoder groundwork only (no encoder yet): analysis transforms (`model::analysis`, BOP + HOP) and hyper-encoder (`model::hyper_encoder`) with loaders, within 1.1e-4 of the reference encoder's tensors and `z_hat` symbol-exact (`tests/encode_ref.rs`); `dump_encode.py --enc2` and the `encoder` reference-vector set.
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
- `Decoder::decode` rejected LSBS streams although LSBS was ported (a stale check; the staged test path did not go through `Decoder`). `tests/decode_ref.rs` now also decodes every stream through `Decoder` and requires identical samples.
- Tool header: `icci_enable_flag` is not coded for 4:2:0 sources; streams with EFE non-linear or LEF data after it parsed wrongly.
