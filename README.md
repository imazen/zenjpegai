# zenjpegai

zenjpegai is a JPEG AI (ISO/IEC 6048-1 | ITU-T T.840.1) image codec in safe Rust: a port of
the [JPEG AI reference software](https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software)
with `#![forbid(unsafe_code)]`, SIMD through [archmage](https://lib.rs/crates/archmage) and
[magetypes](https://lib.rs/crates/magetypes), and no Python or PyTorch at runtime.

**Status: work in progress.** Nothing here decodes an image yet. `PORTING.md` tracks what is
ported, what is missing, and which test proves each piece matches the reference.

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
