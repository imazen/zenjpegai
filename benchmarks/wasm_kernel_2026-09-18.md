# Wasm conv kernel — second pass (machine-code driven) — 2026-09-18

Host: dev (Ryzen 9 9950X3D, 16 cores / 32 threads), Node v26.7.0, Wasmtime 40.0.1,
rustc stable + nightly-2026-09-02 (web packages). Stream: `img30_base_off_bpp050`
(560x888, ~0.5 MP), single thread. Follow-up to `wasm_profile_2026-09-18.md`.

## Why a second pass

The first pass measured Scalar ≈ Wasm128 under V8 (~372 vs ~375 ms) and concluded
"issue/latency-bound, floor reached" without looking at generated code. This pass
inspected the TurboFan machine code (`node --print-wasm-code --code-comments
--print-wasm-code-function-index=N --no-liftoff --wasm-tier-up`) and the wasm text
(`wasm-tools print`).

## Baseline machine code (func 284, before)

Body 57.6 KB. WAT op counts (static): 192 `f32x4.mul`, 152 `v128.store`, 186
`i32.load`, ~112 call sites to `slice_index_fail` bounds-check paths. Inner
`(channel, tap) x B` body: 2 bounds-check branches, fat-pointer loads, index
arithmetic, 4 `v128.load32_splat`, 8 mul + 8 add per weight pair — ~45 machine
instrs of which ~22 are the useful mul/add path. On top of that, `block()`
re-loaded and re-stored the accumulators for every input-channel block and the
bias was written to memory then read back.

## What changed (`src/nn/fast/conv.rs`, generic kernel — all tiers re-verified)

`block()` restructured so the `B` accumulators live across the *entire* input
channel-block loop, not per-block:

- `acc` is initialised straight from the bias (`[F::load(t, bias); B]`) and
  stored once at the end — the old code wrote bias to `out`, then loaded/stored
  `acc` through memory for every `icb`.
- Per `(icb, j-block)` the `ntaps` tap rows are cut down once to
  `L = (B-1)*S + 1` positions (`&xp[..][..l]`, fat slices; `None` taps point at a
  shared `ZERO_PAD` static — the per-call `vec![0.0; ...]` alloc is gone, and so
  is `RowJob::zero`).
- The inner body is `tap_loop`: `wrow.iter().zip(sr.iter())` for weights x rows,
  `assert!(r.len() > (B-1)*S)` once per tap so LLVM's constraint elimination
  drops every `r[b*S]` bounds check from the unrolled position loop.
- `vin == V` (every input block but the last) gets a constant-trip `0..V` loop
  that fully unrolls — each splat offset `v*4` folds into the load instruction.
- The `all-taps-Some` test is hoisted out of the icb loop (interior rows).

Numeric contract unchanged: same `(ic, tap)` accumulation order, unfused
`mul`+`add` on wasm (`nn::fmadd` policy), so output is bit-identical.

## Machine code after (func ~297)

The B=4 hot loop is now, per `(v, tap)` under V8:

```
safepoint poll        cmpb [r13-0x4f],0 / jnz     (V8 loop interrupt check)
len assert            movl + cmpl + jna           (the assert! — always true)
row base              movl sr[t].ptr + leal +v*4
weights               2 x vmovdqu                 (f32x8 = two v128 halves)
positions b=0..3      4 x (leaq + vbroadcastss)   static +0x00/0x20/0x40/0x60
update                8 x vmulps + 8 x vaddps     (both v128 halves x 4 accs)
loop control          ~4
```

~32 instrs per `(v,tap)` for 32 lane-updates, **all 8 accumulator halves in xmm
registers** (xmm2-14, zero `vmovups` spill traffic in the body). The dead B=8
tail instantiation still shows spills but never runs with B=4.

## Measurements (min of warm decodes, wasip1 CLI under node)

| variant | min ms | note |
|---------|-------:|------|
| baseline (prev pass) | 356.6 | |
| acc-resident + thin-row restructure | 333.5 | first rewrite |
| + `vin==V` unrolled v-loop | 310.4 | kept |
| + interior-tap hoist | 313.2 | kept (neutral, cleaner) |
| **final** | **307.1** | **-13.9%** |
| B=2 | 414 | rejected: halves work per weight load |
| B=6 | 384 | rejected: spills |
| ntaps==9 full unroll | 377-650 | rejected: register-pressure catastrophe |
| taps paired 2x | 361 | rejected: chunk machinery > saved polls |
| `step_by`+`zip` position iter | 322 | rejected: iterator overhead > the assert |

Wasmtime 40 (same binary): ~920 ms -> **365 ms** (-60%) — Cranelift amplified
every bounds check; removing them helps it disproportionately.

Browser (threads package, Playwright `benchmark.spec.ts`, mean of the 16 stream
rows): chromium ~101 ms, firefox ~104 ms, webkit ~144 ms — appended to
`wasm_decode_2026-09-18.tsv` (latest block per browser).

## Correctness

- `tiers_and_threads_agree_bit_for_bit`: PASS (native, all tiers, 1..32 threads)
- wasip1 decode PNG: 73/1,491,840 samples differ by 1 vs reference (gate <298)
- wasip1 `--scalar` vs `Wasm128` PNG: byte-identical
- Playwright suite (chromium, firefox, webkit, chromium-webgpu): 87 tests pass
  after clearing three orphaned `serve.mjs` processes from deleted workspaces
  that were squatting on the test ports (JAI_PORT_BASE=3071). All earlier
  failures traced to that stale server, none to the kernel.
- Native V4 decode: 28.1 ms min (baseline ~26-27, shared box) — no regression.

## Remaining floor

Per `(v,tap)` the ideal is ~22 instrs (2 weight loads + 4 splats + 8 mul + 8
add); V8 emits ~32. The difference is V8 artifacts unreachable from safe Rust:
the per-loop safepoint poll, the len reload for the assert, the wasted `leaq`
before each `vbroadcastss` (x86 base+index+disp could fold), and loop counters.
The B=8/V=16 shapes spill; B=4/V=8 remains the sweet spot for V8.
