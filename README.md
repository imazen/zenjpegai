# zenjpegai ![CI](https://img.shields.io/github/actions/workflow/status/imazen/zenjpegai/ci.yml?style=flat-square&label=CI)

Not yet published to crates.io (version 0.0.1) — the crates.io / lib.rs / docs.rs / license
badges land at first publish.

zenjpegai is a JPEG AI (ISO/IEC 6048-1 | ITU-T T.840.1) image codec in safe Rust: a port of
the [JPEG AI reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
with `#![forbid(unsafe_code)]`, SIMD through [archmage](https://lib.rs/crates/archmage) and
[magetypes](https://lib.rs/crates/magetypes), and no Python or PyTorch at runtime.

**Status: decoder complete for the reference's feature set, encoder complete for RGB 4:4:4
sources.** Decoder: SOP / BOP / HOP operating points, any picture size (synthesis tiling, dependent
and independent regions), RGB (BT.709) and YUV streams in 4:4:4 / 4:2:2 / 4:2:0 at 8 and 10 bit,
RVS, GRFS, LSBS, quality maps, cube flags, progressive decode, all four post-filters (EFE linear,
EFE non-linear, eICCI, LEF), UDI. Encoder: the same coding tools and regions, tiling above 1 MP,
fixed-model and target-bpp rate control; it reproduces the reference encoder's streams byte for
byte on 13 of 17 fixed-model vectors (same length on the rest). Not ported: post-filter decisions
on the encode side, subsampled / 10-bit / YUV encoder inputs, the user-defined colour transform
(the reference's own implementation is inconsistent — see `PORTING.md`), eICCI on subsampled
pictures. Anything unsupported is rejected with `Error::Unsupported`; nothing is silently
approximated. The entropy stage is bit-exact; the reconstructed 8-bit picture differs from the
reference decoder's in about 0.005 % of samples, each by one step (convolution summation order).
Also here: a WebAssembly build with a browser polyfill and demo (`web/`, live at
https://imazen.github.io/zenjpegai/) and a wgpu compute backend (`gpu/`).
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

## Speed

Decode time on a Ryzen 9 9950X3D, same streams, same machine, no `-C target-cpu=native`
(`scripts/bench/decode_end_to_end.sh`, raw data in `benchmarks/decode_end_to_end_2026-09-17b.tsv`).
Reference = upstream `b9e573f` on PyTorch 1.10.2 CPU as shipped, which pins PyTorch to one thread;
its number is its own `TOTAL` (model load excluded). zenjpegai numbers are steady state with models
loaded; "process" is one whole command-line run including start-up, model load and PNG output.

| stream | reference | zenjpegai, 1 thread | zenjpegai, threaded | reference process | zenjpegai process |
| --- | --- | --- | --- | --- | --- |
| 560x888, simple profile (SOP), 0.50 bpp | 132 ms | 56 ms | 21 ms | 950 ms | 109 ms |
| 560x888, base profile (BOP), 0.12 bpp | 212 ms | 93 ms | 25 ms | 1023 ms | 117 ms |
| 560x888, base profile (BOP), 0.50 bpp | 211 ms | 93 ms | 27 ms | 1034 ms | 121 ms |
| 560x888, base profile (BOP), 1.00 bpp | 214 ms | 95 ms | 27 ms | 1037 ms | 156 ms |
| 560x888, high profile (HOP), 0.50 bpp | 2279 ms | 1231 ms | 308 ms | 3118 ms | 437 ms |
| 2096x1400, base profile (BOP), 0.50 bpp | 1227 ms | 533 ms | 151 ms | 2172 ms | 600 ms |

So: 1.9x to 2.4x faster on one thread, 6x to 8x with threads. Letting the reference use all 32
hardware threads did not help it on this box (its best single run was never better than 0.9x of
its one-thread time, and the median was 2x to 6x worse); those rows are in the TSV.
Two picture sizes is a thin sample: no tiny pictures, nothing above 3 MP yet.

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
