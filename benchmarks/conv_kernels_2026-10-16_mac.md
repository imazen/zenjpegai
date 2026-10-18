# NEON conv/int-conv tile check on Apple M4 Pro — 2026-10-16

Host: Apple M4 Pro (macOS 27.0, build 26A428, 24 GiB RAM). rustc 1.99.0, no
`-C target-cpu=native` — runtime dispatch, as shipped. All runs through
`run-heavy --mem 12G --jobs 4` (nice + process-tree RSS monitor; macOS has no
cgroup cap).

## Measurement caveat (read first)

`zenbench` refuses CPU placement on this machine: `cpu-info` sees the M4 Pro's
asymmetric Performance/Efficiency classes but reports empty CPU ID lists, and
`--cpu-policy=performance` cannot bind threads on macOS (Mach exposes only QoS
hints). Every number below comes from `--cpu-policy=unrestricted`, which
zenbench marks **exploratory**: `UNVERIFIED_UNRESTRICTED`, no core-class or
frequency verification, and `--save-baseline` is refused ("exploratory mode;
missing planned independent process repetitions"). The numbers are consistent
across six back-to-back sweeps (±1-2 %), so relative comparisons between tile
sizes measured in the same run are trustworthy; treat absolute Gops/s as
approximate, not as verified baselines.

Command (float + int groups):

    cargo bench --bench conv_kernels --features unstable-internals -- \
        --cpu-policy=unrestricted [--group=hsd_]

## Scope

Model-free kernels only — no weights or reference vectors exist on this Mac, so
nothing here says anything about end-to-end codec speed. The kernel under test
is the `conv_row`/`int_row` row micro-kernel in `src/nn/fast/` (the arithmetic
core of `PackedConv`, `PackedConvTranspose`, and the HSD `PackedIntConv`), held
to the exact-output oracle `tests/fast_vs_reference.rs` (bitwise `to_bits`
equality against `nn::reference` / scalar loops, every tier × threads on/off).

## What was checked

The tile constant `B` (output positions per micro-kernel block) was tuned on
x86 (`benchmarks/conv_kernels_2026-09-18.md`): `conv_row` B=12 for AVX2 (16
ymm registers), B=24 for AVX-512. NEON has 32 × 128-bit vector registers and
each `f32x8`/`i32x8` accumulator occupies two, so the inherited constants were
worth re-measuring on aarch64:

- `conv_row_neon`: `f32x8`, B swept over {6, 8, 10, 12, 14, 16}
- `int_row_neon`: `i16x16`/`i32x8`, shared the AVX2 value B=6 — under-using the
  32-register file on its face; swept over {6, 8, 10, 12, 14}

### Assembly (otool, release bench binary, `__arcane_conv_row_neon`)

LLVM already emits the good forms: `fmla.4s vAcc, vW, vXb[lane]` (multiply-add
against a lane of an input vector register) and `ld1r.4s` broadcast loads — the
"broadcast input lane" trick the x86 pass had to hand-arrange is already chosen
by LLVM here. The dense ~128-FMA blocks keep all ~24 accumulator registers
resident; the residual cost is the input-scalar pipeline: each of the B
positions needs one scalar per `(in-channel, tap)`, and at B ≥ 8 LLVM stages
some of them through a small stack FIFO (`ldr sN; str q; ldr q; fmla[lane]`,
~79 `[sp]` accesses per 128-FMA block) rather than keeping them all in
registers. That stack traffic is why smaller tiles were expected to win — they
didn't.

## Result: both constants already optimal — no change landed

`conv_row_neon` (float), per-group neon_1t medians, ms (Gops/s in parens):

| group | B=6 | B=8 | B=10 | B=12 | B=14 | B=16 |
| --- | --- | --- | --- | --- | --- | --- |
| res_conv3x3_160_160@1/16 | 65.1 (40.8) | 57.7 (46.0) | 55.9 (47.5) | **54.1 (49.1)** | 60.5 (43.9) | 71.2 (37.3) |
| up1_convT4x4_160_64@1/16 | 47.4 (39.9) | 38.8 (48.7) | 37.3 (50.6) | **35.1 (53.9)** | 36.6 (51.6) | 47.3 (39.9) |
| resau_conv3x3_g4_64@1/8 | 10.0 (42.5) | 8.5 (50.0) | 7.8 (54.3) | 7.9 (53.8) | 8.0 (53.0) | 9.4 (45.2) |
| resau_conv1x1_64_64@1/8 | 5.6 (33.9) | 6.0 (31.7) | 4.5 (42.4) | **4.0 (46.7)** | 4.9 (38.8) | 4.8 (39.7) |
| up2_convT4x4_64_64@1/8 | 77.3 (39.1) | 60.4 (50.0) | 57.5 (52.6) | **54.5 (55.5)** | 55.1 (54.9) | 70.1 (43.1) |
| conv3_3x3_64_96@1/4 | 234.3 (43.5) | 202.0 (50.5) | 200.8 (50.8) | 200.5 (50.9) | 200.5 (50.9) | 294.6 (34.6) |
| conv4_1x1_96_16@1/4 | 8.4 (33.9) | 8.7 (32.5) | 6.4 (44.4) | **5.9 (48.2)** | 7.0 (40.3) | 6.4 (44.5) |

B=12 wins five of seven groups outright and ties (within noise) the other two.
Multithreaded NEON results show the same ordering (B=12: 174-216 Gops/s vs
B=10: 160-208, B=14: 147-215, B=16: 141-171). Smaller tiles amortize the tap
sweep over fewer positions; larger tiles spill accumulators. The committed
value is already the optimum; `conv_row_neon` is unchanged.

`int_row_neon`, new HSD-shaped groups added to `benches/conv_kernels.rs`
(conv1x1 chs→chs, conv3x3 chs→chs, conv1x1 chs→16·chs at latent 88x131,
chs=128 representative):

| group | B=6 | B=8 | B=10 | B=12 | B=14 |
| --- | --- | --- | --- | --- | --- |
| hsd_conv1x1_128_128@1/16 (1t) | **12.8 ms (14.8 G)** | 13.1 (14.4) | 13.5 (14.0) | 12.8 (14.7) | 13.5 (14.0) |
| hsd_conv3x3_128_128@1/16 (1t) | **109.0 (15.6)** | 112.2 (15.1) | 115.4 (14.7) | 110.7 (15.4) | 119.3 (14.2) |
| hsd_conv1x1_128_2048@1/16 (1t) | **202.0 (15.0)** | 206.2 (14.7) | 212.5 (14.2) | 201.7 (15.0) | 212.5 (14.2) |
| (mt, same order) | **48.5 / 59.7 / 52.2 G** | 47.4/58.0/50.9 | 46.1/56.4/49.6 | 48.2/58.7/52.2 | 46.0/54.7/49.6 |

B=6 wins or ties everywhere. The int inner op is not one FMA: `madd` lowers to
a widening `smull`/`smull2` pair plus `saddlp` — several instructions per
accumulator — so each accumulator already hides its own dependency chain and
more accumulators only cost register pressure. Int throughput is ~15 Gops/s
1t / ~50-60 Gops/s mt vs the float path's ~50/200 (the macs unit counts one
multiply-add pair each way; the int kernel retires several NEON ops per pair).

## What is not reachable from safe Rust here

- A lane-indexed broadcast-FMA is already emitted by LLVM (`fmla.4s …, vXb[k]`);
  the remaining load-side cost would need a gather or a vector-of-positions
  kept live per tap — no magetypes lane-extract op exists (checked 0.9.30), and
  the register budget for B live input vectors does not fit anyway.
- NEON `sdot`-style or deeper int fusions are not exposed by the
  `I16x16Backend`/`I32x8Backend` abstraction; adding one is an archmage API
  change, out of this lane's scope.

## Coz on macOS (probe, scratch only — nothing added to the crate)

Requirement was to separate "the `coz` crate compiles" from "native Coz
produces a nonempty profile". Exact results:

- **`coz` crate 0.1.3** (crates.io; deps `once_cell` 1.21.4 + `libc` 0.2.191):
  `cargo build --release` succeeds on aarch64-apple-darwin; `coz::progress!()`,
  `coz::scope!()`, `coz::begin!()`/`end!()` all expand and run — but every
  macro is a **silent no-op on macOS**: `Counter::create_counter` resolves
  `_coz_get_counter` via `dlsym` only under `#[cfg(target_os = "linux")]`;
  other targets get a stub returning null, so no progress point is ever
  registered with the profiler.
- **Probe**: `~/tmp/coz-probe` (release + `debug = 1`), 12 s run under
  `coz run` → `profile.coz` contains only `{"type":"startup"}` +
  `{"type":"runtime"}` (78 bytes, zero samples/experiments). Verified: Rust
  macros compile, but coz-rs **cannot produce a profile on macOS as
  published**.
- **Native coz**, master `256e3c389424` (2026-07-10): no prebuilt macOS
  package exists (0.2.5 ships Linux only; no Homebrew formula). Built from
  source: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build
  build -j4` (fetches LIEF 0.17.0 + libelfin; ~5 min). `coz run` injects
  `build/libcoz/libcoz.so` via `DYLD_INSERT_LIBRARIES`; the macOS backend
  samples with Mach `task_threads` + `thread_get_state`, **not** the private
  kperf framework — so no sudo/SIP/entitlement is needed for sampling itself.
  A C probe (`coz.h` `COZ_PROGRESS`/`COZ_BEGIN`/`COZ_END`, which has no Linux
  gate) produced a real causal profile: 5 experiments with virtual speedups
  0-75 %, throughput-point deltas, latency-point arrivals/departures, and
  3.8k PC samples attributed to the two hot lines.
- **Verdict**: native coz works on this Mac, but the Rust integration is
  dead on arrival — `coz_get_counter` is Linux-gated in the crate, so
  instrumented Rust programs register no progress points and profile runs
  stay empty. Usable only after a (one-crate) patch widening the `cfg` —
  not done; coz is not added to this project.

## Gates run (rustc 1.99.0)

- `cargo fmt --check` — clean.
- `cargo clippy --all-targets --all-features -- -D warnings` — clean after
  fixing three **pre-existing** deprecations surfaced by this newer toolchain
  (the gate failed on the base commit too):
  - `src/nn/fast/{simd,int_conv}.rs`: `load`/`splat`/`from_array` →
    `load_t`/`splat_t`/`from_array_t` (magetypes 0.9.30 marks the old spelling
    deprecated; the deprecated names are pure forwarders — verified zero
    semantic delta).
  - `fuzz/fuzz_targets/common.rs`: kept `AtomicU32::fetch_update` under
    `#[allow(deprecated)]`; the renamed `try_update` needs Rust 1.95 > MSRV
    1.89.
  - `tests/codec_ref.rs`: `ComputeEnvironment::new().with_cores(1)` →
    `conservative()` (identical value; `new` is a deprecated forwarder).
- `cargo test --lib --tests --all-features` — every model-free oracle passes:
  `fast_vs_reference` 8/8 (incl. `conv2d_bit_identical`,
  `convt_bit_identical`, `int_conv_matches_scalar_loops`, tier × threads),
  `math_tiers` 1/1, `mans_vectors` 10/10, `nn_vectors` 5/5,
  `fuzz_regression` 3/3, `memory_budget` 9/9, `memory_tracked` 6/6; lib 81/85.
  The remaining failures are all missing-oracle-environment: `*_ref.rs`
  integration tests and the four `efe_linear::tests::oracle` unit tests need
  `~/work/zen/jpeg-ai-reference-software` and `/mnt/v/output/zenjpegai/`
  (`No such file or directory`), neither of which exists on this Mac —
  pre-existing environmental gap, not a regression.
- `tiers_and_threads_agree_bit_for_bit` could not run (needs stream vectors
  from `/mnt/v`). The kernel changes in this commit are token-spelling renames
  verified as pure forwarders, and `fast_vs_reference` exercises the same
  tier × threading matrix on the kernels touched.

### Addendum 2026-10-17: reference checkout + vectors now present on this Mac

After this report was first written the reference environment was fetched:

- `~/work/zen/jpeg-ai-reference-software` — clone at pinned `b9e573f`
  (`GIT_LFS_SKIP_SMUDGE=1` + `git lfs pull --include="models/,data/"`; 3.4 GB:
  80 `.pth` checkpoints 974 MB, `data/test` PNGs 721 MB).
- `~/work/zen/zenjpegai-refdata/{vectors,inputs}` — 74 vector dirs rsynced
  from `dev:/mnt/v/output/zenjpegai/reference/` (6.4 GB; set
  `ZENJPEGAI_VECTORS=/Users/lilith/work/zen/zenjpegai-refdata/vectors`).

Re-run with both env vars: **all model-free suites still green, plus**
`weights_ref` 4/4 (all 80 upstream `.pth` checkpoints parse, FNV-1a verified),
`filters_efe_ref` 17/17, `memory_ref` 6/6, `packed_bundle` 4/4, and the
previously-unrunnable **`tiers_and_threads_agree_bit_for_bit` passes**
(decode_ref). `decode_ref` 44/47, `entropy_ref` 35/37, `encode_ref` 11/22,
`f16_weights` 5/6: every remaining failure is a `vector_dir` missing-file
panic at `tests/common/mod.rs:45` — never a numeric diff.

Unsynced for disk headroom (container was at 100%; ~0.7 GiB free):
`img01_base_off_{indregions,indregions_threads8}_m1` (~640 MB),
`img01_base_off_bpp{012,025,075,100}` (~1.1 GB), `enc_img01_bop_m1_b0{,_y80u40,
_indregions,_depregions}` (~765 MB), `img30_base_on_bpp{012,050,075}`
(~745 MB), `img01_base_on_*` (~7.3 GB). The reference *Python* env still
cannot run here — pinned `torch==1.10.2+cpu` has no arm64 macOS wheels — so
vectors can only be synced, not regenerated, on this host.

Peak RSS (run-heavy): bench sweeps 0.48 GiB each; test suite 1.02 GiB.

## Files

- `benches/conv_kernels.rs`: +3 model-free HSD int-conv groups (new coverage —
  the int path had no benchmark).
- Kernel sources: **unchanged** — the sweep verified both tile constants are
  already at their NEON optimum on M4 Pro.
- Sweep logs + zenbench evidence JSON: `~/tmp/zenjpegai-mac-bench-B{6,8,10,12,14,16}.log`,
  `~/tmp/zenjpegai-mac-intbench-B{6,8,10,12,14}.log`, `.zenbench/data/runs/`
  (kept outside git).
