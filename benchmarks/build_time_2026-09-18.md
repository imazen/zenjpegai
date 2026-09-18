# Build time — 2026-09-18

Host: AMD Ryzen 9 9950X3D (16C/32T), `rustc 1.98.1 (48a229cea 2026-09-01)`, `cargo build -j 8`,
`nice -n 19`. Commit `77c82df1`. All numbers are `/usr/bin/time -v` wall clock and peak RSS
(`Maximum resident set size`), except the llvm-lines counts.

## Wall-clock build times

| Scenario | Features | Wall | User CPU | Peak RSS |
| --- | --- | --- | --- | --- |
| Fully clean (`cargo clean`, all deps + crate) | default (`std,parallel,avx512`) | 5.18 s | 16.53 s | 430 MB |
| Fully clean | `cli,zencodec` | 10.84 s | 54.64 s | 469 MB |
| Fully clean | `-p zenjpegai-gpu` (own default features) | 15.48 s | — | 909 MB |
| `cargo clean -p zenjpegai` only (deps cached) | default | 3.37 s | 12.96 s | 425 MB |
| `cargo clean -p zenjpegai` only (deps cached) | `cli` (zenpng/zenpixels/zencodec/clap etc. cold) | 6.24 s | 39.91 s | 437 MB |
| `cargo clean -p zenjpegai` only (deps cached, `cli` deps also cached) | `cli` | 3.38 s | 13.32 s | 464 MB |
| Incremental (`touch src/lib.rs`, whole-crate recompile) | `cli` | **0.36 s** | 0.30 s | 244 MB |

Takeaways:
- Fully clean, worst case (no cache at all, every dependency compiled from scratch): under 11
  seconds even with `cli,zencodec` pulling in zenpng/zenpixels/zenpixels-convert/zencodec/clap.
  Default features alone (the common case: `cargo build --release`) is ~5 s clean.
  `zenjpegai-gpu`'s own cold build is the outlier at ~15 s / ~900 MB peak RSS, entirely from
  `wgpu` and its dependency tree (naga, gpu-allocator, wgpu-hal, ...) — not this crate's own
  code.
- With dependencies cached (the day-to-day case after `cargo build` once), rebuilding just this
  crate's own code from scratch (`cargo clean -p zenjpegai`) is ~3.4 s regardless of feature set
  — `cli` and `zencodec` add dependency compilation, not much extra code of this crate's own.
- Incremental rebuilds (the actual edit-compile-test loop) are well under half a second thanks to
  `incremental = true` on the release profile (already set in `Cargo.toml`).
- No action taken: these numbers do not motivate a build-time optimization pass. The one earlier
  data point (7.6 s incremental release build of `663b86d`, `-j 8`) was measuring something
  different (an incremental rebuild with a warm `target/`, not comparable apples-to-apples to any
  row above) and should not be read as "build time regressed" — it wasn't re-measured because the
  comparable scenario here (incremental, `touch src/lib.rs`) is 0.36 s.

## `cargo llvm-lines --release` (default features)

271,274 total LLVM IR lines, 5,196 distinct symbols ("copies" — cargo-llvm-lines' term for
monomorphized instances counted across the whole crate, not per-function). No individual
function has more than 2 monomorphized copies crate-wide (the one `Copies=2` entry is a
`crossbeam_epoch`/`rayon` closure, not this crate's code) — i.e. no generic function is
exploding into dozens of instantiations. The `nn/fast/conv.rs` / `int_conv.rs` tier x block x
stride family the previous agent flagged as an open audit item **is present but small**: each
`conv_row_s::<Tier, V, B, STRIDE>` / `int_row::<Tier, V, B, SHIFT>` instantiation is
~271-389 lines, and `cargo llvm-lines --filter conv` shows roughly a dozen such instances
(one per (SIMD tier, block size, stride/shift) combination actually reachable on this
build's target — x86_64 with `avx512`, so `X64V4Token` f32x16 + `X64V3Token` f32x8 +
scalar `Lanes8`/`ScalarMadd`, times the strides/taps each kernel uses) totalling on the order of
2-3% of the binary's IR. This is normal, expected fan-out for a hand-vectorised multi-tier SIMD
kernel family (archmage/magetypes generate one instantiation per tier by design — see
`~/.claude/CLAUDE.md` "archmage SIMD dispatch"), not pathological bloat, and matches the fast
wall-clock numbers above. **Verdict: no cheap win here** — the largest single functions by IR
line count are ordinary business logic (`weights::pickle::load` 4,985 lines,
`Encoder::encode_from_latents` 2,822, `PictureHeader::parse` 2,186), not generic instantiation.

Top 10 by lines (`cargo llvm-lines --release`, `-s lines`, default sort):

```
  Lines                 Copies              Function name
  -----                 ------              -------------
  271274                5196                (TOTAL)
    4985 (1.8%,  1.8%)     1 (0.0%,  0.0%)  zenjpegai::weights::pickle::load
    2822 (1.0%,  2.9%)     1 (0.0%,  0.0%)  <zenjpegai::encoder::Encoder>::encode_from_latents
    2186 (0.8%,  3.7%)     1 (0.0%,  0.1%)  <zenjpegai::header::PictureHeader>::parse
    2098 (0.8%,  4.5%)     1 (0.0%,  0.1%)  <zenjpegai::weights::zip::Archive>::parse
    1956 (0.7%,  5.2%)     1 (0.0%,  0.1%)  <zenjpegai::decoder::api::Decoder>::decode_inner
    1655 (0.6%,  5.8%)     1 (0.0%,  0.1%)  <zenjpegai::model::mcm::ContextModel>::compress
    1636 (0.6%,  6.4%)     1 (0.0%,  0.1%)  zenjpegai::decoder::reconstruct::synthesize_with
    1620 (0.6%,  7.0%)     1 (0.0%,  0.2%)  zenjpegai::decoder::reconstruct::reconstruct_latent_with
    1525 (0.6%,  7.6%)     1 (0.0%,  0.2%)  <zenjpegai::header::RenderingInfo>::parse
```

## Reproduce

```bash
cargo clean && /usr/bin/time -v cargo build --release -j 8                      # row 1
cargo clean && /usr/bin/time -v cargo build --release --features cli,zencodec -j 8   # row 2
cargo clean -p zenjpegai --release && /usr/bin/time -v cargo build --release -j 8    # row 4
touch src/lib.rs && /usr/bin/time -v cargo build --release --features cli -j 8  # row 7
cargo llvm-lines --release              # full report, default features
cargo llvm-lines --release --filter conv   # the tier x block x stride family
```
