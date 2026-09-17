# zenjpegai

zenjpegai is a JPEG AI (ISO/IEC 6048-1 | ITU-T T.840.1) image codec in safe Rust: a port of
the [JPEG AI reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
with `#![forbid(unsafe_code)]`, SIMD through [archmage](https://lib.rs/crates/archmage) and
[magetypes](https://lib.rs/crates/magetypes), and no Python or PyTorch at runtime.

**Status: work in progress — a partial decoder, no encoder.** Missing first: the high operating
point (HOP synthesis), all four post-filters, latent scaling (LSBS), RVS/GRFS, quality maps,
chroma-subsampled and 10-bit pictures, custom colour transforms, progressive decode, and the
entire encoder above the entropy coder. Streams that need any of these are rejected with
`Error::Unsupported`; nothing is silently approximated.

What works: streams at the simple and base operating points with the tools above switched off
(4:4:4, 8-bit, BT.709), at any picture size, with or without region partitioning and multiple
ANS threads. The entropy stage is bit-exact; the reconstructed 8-bit picture differs from the
reference decoder's in about 0.005 % of samples, each by one step (convolution summation order).
`PORTING.md` tracks every module, what is missing, and which test proves each piece.

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
(`scripts/bench/decode_end_to_end.sh`, raw data in `benchmarks/decode_end_to_end_2026-09-17.tsv`).
Reference = upstream `b9e573f` on PyTorch 1.10.2 CPU as shipped, which pins PyTorch to one thread;
its number is its own `TOTAL` (model load excluded). zenjpegai numbers are steady state with models
loaded; "process" is one whole command-line run including start-up, model load and PNG output.

| stream | reference | zenjpegai, 1 thread | zenjpegai, threaded | reference process | zenjpegai process |
| --- | --- | --- | --- | --- | --- |
| 560x888, simple profile, 0.50 bpp | 127 ms | 58 ms | 29 ms | 947 ms | 118 ms |
| 560x888, base profile, 0.12 bpp | 210 ms | 109 ms | 51 ms | 1027 ms | 136 ms |
| 560x888, base profile, 0.50 bpp | 236 ms | 97 ms | 42 ms | 1208 ms | 133 ms |
| 560x888, base profile, 1.00 bpp | 212 ms | 99 ms | 46 ms | 1035 ms | 171 ms |
| 2096x1400, base profile, 0.50 bpp | 1178 ms | 570 ms | 263 ms | 2119 ms | 673 ms |

Letting the reference use all 32 hardware threads did not help it on this box (281–2590 ms on the
small picture, 849 ms on the large one, with large run-to-run spread); those rows are in the TSV.
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
