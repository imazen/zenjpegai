# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->

### Added
- `gpu/` crate (`zenjpegai-gpu`): SOP / BOP / HOP synthesis as WGSL compute shaders through wgpu 30 (native + wasm32), `GpuDecoder`, readback-free RGBA presentation, `just gpu-test`, `gpu_bench` example. Parity gated on llvmpipe only; not yet run or timed on a hardware GPU (`gpu/README.md`).
- `wasm/` crate (`zenjpegai-wasm`, wasm-bindgen: `addModels` / `info` / `decode`, single-thread SIMD128 build and a rayon `threads` build) and `web/scripts/build-wasm.sh`; `pkg-simd` is 389 KB (131 KB brotli). Polyfill, tests and demo are not written yet (`web/README.md`).
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
