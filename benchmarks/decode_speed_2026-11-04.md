# Decode speed vs reference — 560x888 img30 BOP (2026-11-04)

Measured on `dev` (Ryzen 9 9950X3D, Zen 5, AVX-512). Commit: `708c8e6d` (+ pin).
Command: `./target/release/zenjpegai decode stream.bits /dev/null --discard --repeat 12 --models $REF/models`
(cargo build --release --features cli). Reference baseline from CLAUDE.md: decoder "TOTAL"
~0.22 s for the same 560x888 BOP test image (torch 1.10.2, 16 threads, CPU).

## Result

| run | decode ms |
|-----|-----------|
| 0 (incl. model load) | 76.6 |
| 1–11 steady-state | 28.3 – 33.0 (median ~31.7) |

**~31.7 ms steady-state vs ~220 ms reference → ~7x faster** (same bit-exact output —
`decode_ref` 50/50 across all streams). The first decode's model-load cost (~77 ms) is
one-time; `pack-models` bundles shrink it further for shipped artefacts. The reference's
own Python process wall (~1.4 s incl. model load) makes the end-to-end gap larger still.

## Encode (fixed model 1, beta-disp 0, BOP — single-pass, no rate matching / RDLR)

`./target/release/zenjpegai encode img30_560x888_8bit_444.yuv out.bits --model 1 --beta-disp 0 --op bop`:
**0.165 s wall** (0.80 s user across threads) → 21445 bytes / 0.3450 bpp. The reference's
Python pipeline takes multiple seconds for the same fixed-point encode (model load +
oneDNN analysis + Python orchestration), so the single-pass encoder is an order of
magnitude faster end-to-end even before the optimizer paths.

## Per-operating-point (same 560x888 img30 base stream, `--op`)

| op | decode ms (steady-state) |
|----|--------------------------|
| sop | ~22 – 25 |
| bop | ~32 – 36 |
| hop | n/a — `img30_base_off` is a BOP stream (no HOP synthesis present) |

SOP (luma synthesis only, coarser model) is ~30% cheaper than BOP; HOP needs a HOP-coded
stream to measure.
