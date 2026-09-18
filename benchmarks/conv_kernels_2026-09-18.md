# Native conv kernel — machine-code-driven pass (d2native) — 2026-09-18

Host: dev (Ryzen 9 9950X3D, 16c/32T, shared box — load was 4-20 during these runs;
interleaved A/B runs are the trustworthy numbers). rustc 1.98.1, no
`-C target-cpu=native` (runtime dispatch, as shipped). Method copied from
`wasm_kernel_2026-09-18.md`: read the generated assembly, find the spill /
dead-store / loop-overhead defects, fix the tile shape, re-measure.

## What the machine code showed

`__arcane_conv_row_v4` (AVX-512, V = 16 lanes) at the old tile `B = 28`:
every `(input-channel-block, tap)` iteration emitted **27 `vmovaps zmm→[rsp]`
spill stores** plus the matching reloads — the 28 f32x16 accumulators did not
fit the register budget, so each FMA's result round-tripped through the stack.
The AVX2 (`B = 12`, 16 ymm registers) and Wasm kernels were already clean.

Fixes, all verified in the generated code of the final build:

| change | asm evidence (final build) |
| --- | --- |
| V4 tile `B = 28 → 24` (24 accs + weight + temporaries = 26-27 zmm) | every dense FMA window (24-FMA inner bodies) has **zero** `[rsp]`/`[rbp]` vector traffic; base build had 27 dead stores per window |
| Overlapping tail block (`n % B != 0` → one full-width block at `n - B` instead of a B=8/4/1 cascade) | tail cascade only reachable for `n < B`; eliminates up to 13 narrow full-cost passes (int conv) / ~10.9% of conv time spent in `tap_loop<16,8,1>` tails (float, perf) |
| `tap_loop_n` const-`NT` specialization for `ntaps ∈ {1, 4, 9}` (1x1, convT 2x2 phases, 3x3) | tap loop fully unrolled, weight offsets constant-folded, ~1 cmp per 24-FMA window instead of per tap |
| `int_row` accumulators hoisted: `[F::Acc; B]` live across all `(pair, tap)` iterations, stored once | inner loop is `vmovdqu64` + 24×(`vpbroadcastd` + `vpmaddwd` + `vpaddd`), accs in zmm10-24, **0** stack stores; base reloaded/stored `out[j..j+B]` per input pair |
| int V4 tile `B = 14 → 24` | same register budget argument as float |
| `dw_row` `ntaps == 9` constant-trip specialization | 9-FMA unrolled body, bounds checks hoisted to a per-row prologue |
| `border_row` contiguous-run copy | per-column `copy_from_slice` calls → one slice copy per `(icb, ky)`; both border sides share one TLS scratch buffer |
| `pad_par` write-once (pooled scratch + explicit pad fills, no calloc-zero-then-copy) | removes a full-tensor zero pass per padded conv input |

## Numbers

End-to-end decode, steady-state min of repeats, interleaved base/new runs
(`45dc8af2` baseline binary vs this branch, same load window):

| stream | 1T base | 1T new | Δ | 8T base | 8T new | Δ |
| --- | --- | --- | --- | --- | --- | --- |
| img30_base_off_bpp050 (BOP, 560x888) | 138.4 ms | 83.2 ms | **-40 %** | 49.0 ms | 31.7 ms | **-35 %** |
| img30_high_off_bpp050 (HOP) | 2346 ms | 1323 ms | **-44 %** | | | |
| img01_base_off_bpp050 (BOP, 2096x1400) | 1041 ms | 618 ms | **-41 %** | 266 ms | 190 ms | **-29 %** |

PNG output bit-identical (`sha256 d14630b9…` on both builds). Full-stream TSV:
`decode_end_to_end_2026-09-18_d2native.tsv` vs `…_d2native_base.tsv` (the TSV run
landed during a load spike — its per-row mins are inflated; the interleaved
table above is the honest delta).

`conv_kernels` micro-bench, interleaved base/new, v4_1t (loaded box, MAD ±5 ms):

| group | base | new | Δ |
| --- | --- | --- | --- |
| `res_conv3x3_160_160@1/16` | 51.3-55.3 ms | 52.3-57.3 ms | ~par in noise |
| `up1_convT4x4_160_64@1/16` | 56.7 ms | 40.0 ms | **-30 %** |
| `conv4_1x1_96_16@1/4` | 10.06 ms | 9.91 ms | ~par |

v3_1t unchanged on all three (B stays 12). The end-to-end win comes from all
decode shapes together — smaller width rows feel the tail-cascade removal more
than the isolated 131-wide bench shape — plus the int conv, pad, depthwise and
border changes.

## Not taken

- **`iter_mut().zip(step_by)` for the accumulator loop**: compiled, but forced
  the accumulator array to memory — decode regressed ~86 → ~140-165 ms.
  Reverted to indexed `acc[b]`; the code carries a comment saying so.
- **AVX-512 VNNI (`vpdpwssd`) for the int conv**: would fuse madd+add (~25 %
  of that inner loop), but `archmage`/`magetypes` 0.9.29 expose no safe VNNI
  op for the packed `xp` layout and no interleave op to build one — would
  need unsafe or a new crate API. Documented, not implemented.
- **Hoisting the `r.len()` assert out of the `(v, tap)` loop**: LLVM still
  re-checks per iteration (can't GVN the fat-pointer len across the call).
  The remaining per-iteration cost is one fused cmp+branch — ~1 uop.
- **B = 26/27 on V4**: measured marginally faster in single runs but sit at
  the spill cliff; B = 24 chosen for stability.
- **V3 stays B = 12**: 16 ymm registers; B = 14 spills one accumulator per
  FMA (verified in asm).

## Parity

`tests/fast_vs_reference.rs` all 8 tests, `tests/decode_ref.rs`
`tiers_and_threads_agree_bit_for_bit` (140 s, bit-identical across tiers and
thread counts), full `cargo test --lib --tests --all-features` — all green.
The overlap tail recomputes positions `n - B..j` with the identical
`(v, tap)` accumulation order and stores the same values twice, so output
is bit-identical by construction.
