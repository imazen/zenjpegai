# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that will ship together in the next major (or minor for 0.x) release. -->

### Added
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
