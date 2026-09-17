# Working on zenjpegai

A map for someone (human or agent) picking this repo up cold. Status lives in `PORTING.md`
(what is ported, what is missing, which test proves each piece); rules live in `CLAUDE.md`.

## What this is

A safe-Rust port of the JPEG AI reference software (upstream commit `b9e573f`). JPEG AI is a
learned image codec: a conventional container + table ANS entropy coder around a handful of
convolutional networks whose weights come from upstream's PyTorch checkpoints.

## Setup (once)

1. Rust stable; `just` optional.
2. The upstream checkout with models (git-lfs, 3.4 GB) and a Python 3.8 + torch 1.10.2 venv:
   recipe and gotchas in `CLAUDE.md` ("Reference environment"). Default location
   `~/work/zen/jpeg-ai-reference-software`, override with `ZENJPEGAI_REF`.
3. Reference vectors (streams + tensor dumps from the reference, not in git):
   `scripts/ref_vectors/make_reference_streams.sh all` writes them to
   `/mnt/v/output/zenjpegai/reference/vectors` (override `ZENJPEGAI_VECTORS`). Sets: `smoke
   regions tools filters efe filtertiles qmap formats`. Minutes each.

## Everyday commands

```
cargo test --lib --tests                                   # no reference data needed
just test-ref                                              # everything, needs steps 2 + 3
cargo clippy --all-targets --all-features -- -D warnings
cargo build --release --features cli
ZENJPEGAI_MODELS=$ZENJPEGAI_REF/models target/release/zenjpegai decode X.bits out.png --time
target/release/zenjpegai info X.bits                        # every header field
scripts/bench/decode_end_to_end.sh > benchmarks/decode_end_to_end_$(date +%F).tsv
```

## Code map (decode order)

| Stage | Where | Must match the reference |
| --- | --- | --- |
| markers, substreams | `src/container.rs`, `src/bitio.rs` | exactly |
| picture / tool / rendering headers | `src/header.rs` | exactly (parse + write) |
| me-tANS entropy coder | `src/mans/` | exactly (vectors in `tests/vectors/mans`) |
| checkpoints (`.pth`, packed bundles) | `src/weights/`, `src/model/mod.rs` | exactly |
| entropy stage: z, integer hyper-scale decoder, sigma, RVS/GRFS, quality map, residual | `src/decoder/entropy.rs`, `src/model/hsd.rs`, `src/tools/` | exactly |
| hyper-decoder, context model (MCM), LSBS | `src/decoder/reconstruct.rs`, `src/model/{hyper_decoder,mcm}.rs` | bounded float error |
| synthesis SOP / BOP / HOP, tiling, regions | `src/model/{synthesis,attention}.rs`, `src/tools/{tiles,regions}.rs` | bounded float error |
| post-filters | `src/filters/` | bounded float error |
| chroma format, colour, bit depth | `src/decoder/output.rs` | bicubic resampler bit-identical to PyTorch |
| one-call API, CLI | `src/decoder/api.rs`, `src/bin/zenjpegai.rs` | - |
| SIMD engine (all float networks run on it) | `src/nn/fast/` | every tier / thread count bit-identical to `src/nn/reference.rs` |
| browser build, GPU backend | `wasm/`, `web/`, `gpu/` | see their READMEs |

## How parity is proven

- `tests/entropy_ref.rs`: integer stage equal to the reference tensor dumps, bit for bit.
- `tests/decode_ref.rs`: latents within 5e-4, planes within 3e-3 (0..255), final samples differ
  by at most 1 in fewer than 1/5000; also decodes through `Decoder` and demands identical output.
- `tests/fast_vs_reference.rs`, `tests/math_tiers.rs`: SIMD tiers vs plain loops, bit-identical.
- `tests/nn_vectors.rs`: layer definitions vs PyTorch on tiny tensors.
- Never loosen a bound to get green. If the reference itself is wrong (it is, in places: see
  "Reference ... bug" sections in `PORTING.md`), patch the defect at runtime in the dump script
  and say so; never edit the upstream checkout.

## Porting a new piece: the loop that worked

1. Read the upstream Python for the decoder direction only; note every constant that is not in
   the bitstream (several live in `cfg/pipeline.json` or Python defaults).
2. Make a reference stream that exercises exactly that piece (add a set to
   `make_reference_streams.sh`; configs of ours go in `scripts/ref_vectors/cfg/`).
3. Dump the tensors around it (`scripts/ref_vectors/dump_decode.py`, additive).
4. Port, add the vector to `tests/entropy_ref.rs` / `tests/decode_ref.rs`, record measured
   errors in `PORTING.md`, update `CHANGELOG.md`, push (small commits, `jj`, see `CLAUDE.md`).

## What to do next

See the "Work queue" at the end of `PORTING.md`.
