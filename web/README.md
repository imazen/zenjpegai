# zenjpegai in the browser — status and next steps

Goal (user, 2026-09-17): wasm fully optimised including wgpu and workers, a fallback for strict
mode (no cross-origin isolation), a polyfill so `<img>` JPEG AI pictures decode in-browser fast,
proper Playwright testing, a demo pushed to GitHub Pages from CI, demo images from imazen-26.

Work stopped early on a budget change. What follows is exact: missing things first.

## Not done

| # | Deliverable | State |
| --- | --- | --- |
| 4 | `web/src/worker.js`, `web/src/polyfill.js` (worker pool, `crossOriginIsolated` switch, header-first bundle fetch through the Cache API, `blob:` swap, MutationObserver, CSP-safe) | **not started**; no file extension / MIME type chosen yet (check ISO/IEC 6048-1 and the reference software for an official one first) |
| 5 | Playwright tests (`web/tests/*.spec.ts`), static servers (plain / COOP+COEP / strict CSP), browser decode timings, `benchmarks/wasm_decode_<date>.tsv` | **not started**. `@playwright/test` 1.63.0 is in `package.json`; browsers are not installed (`npx playwright install chromium firefox`) |
| 6 | Demo page, imazen-26 demo images, `demo-assets-v1` release, `.github/workflows/pages.yml`, wiring of the `gpu/` backend | **not started**. `gh api repos/imazen/zenjpegai/pages` was never run; repository visibility untouched |
| 3 | wasm crate: **threads build never executed** | it compiles, links and passes `wasm-opt`, but `initThreadPool` has only a browser to run in and no browser test exists yet. Treat as unproven |
| 3 | no-SIMD build | not built and the caniuse check for wasm SIMD was not done. The engine supports it (scalar tier, measured 1.42 s against 0.35 s) |

## Done and measured (2026-09-17, Ryzen 9 9950X3D, node 26; the box was loaded, load average 17)

- **Numeric policy + `Wasm128` tier** (`src/nn/fast`, `src/nn/mod.rs::fmadd`): see `PORTING.md`
  "WebAssembly numeric policy". 560x888 BOP stream, one thread: 6.0 s before (soft-float FMA),
  0.35 s now. Wasm output is identical across tiers and engines; against the reference decoder
  43..443 samples differ per picture, all by 1 (`benchmarks/wasm_parity_2026-09-17.tsv`).
  `just test-wasi` runs the whole test suite on `wasm32-wasip1` under node.
- **Model bundles** (`src/weights/packed.rs`, `zenjpegai pack-models`): SOP 15.05 MB, BOP
  16.22 MB, HOP 28.76 MB per (model, operating point); or 12.36 MB common part per model +
  2.69 / 3.86 / 16.40 MB synthesis part (`--only common|synthesis`). Byte-identical pixels
  (`tests/packed_bundle.rs`). gzip saves 9 %: these are float weights.
- **`wasm/` crate** (`zenjpegai-wasm`): `addModels(bundle)`, `hasModels(modelId, op)`,
  `info(stream)`, `decode(stream) -> {width, height, rgba}`, `buildMode()`, `simdTier()`,
  `releaseBuffers()`, and `initThreadPool` in the threads build.
  `web/scripts/build-wasm.sh [simd] [threads]` writes `web/dist/pkg-simd` and
  `web/dist/pkg-threads` (wasm-bindgen `--target web`, `wasm-opt -O3`; profile `wasm-release`:
  opt-level 3, fat LTO, one codegen unit, panic=abort).

  | package | after wasm-bindgen | after `wasm-opt -O3` | gzip -9 | brotli -q11 |
  | --- | --- | --- | --- | --- |
  | `pkg-simd` | 464,146 | 389,178 | 158,867 | 131,036 |
  | `pkg-threads` | 744,972 | 506,863 | 189,080 | 152,520 |

  `pkg-simd` under node (`web/scripts/node-smoke.mjs`): pixels identical to the wasip1 build;
  best runs 0.37-0.39 s per 0.5 MP decode, medians 0.43-0.52 s under the load above.
  Both packages need the pinned nightly and `-Z build-std`: with the prebuilt std (no
  bulk-memory `memcpy`/`memset`) the same decode took 0.50 s instead of 0.37 s.
  `wasm-opt -O3` against no `wasm-opt`: no speed difference resolvable under that load
  (minima 0.44-0.51 s against 0.38-0.48 s); it is kept for the 16 % size cut. Re-measure on a
  quiet box before trusting either number.

## Next steps, in order

1. `cd web && npm ci && scripts/build-wasm.sh` (needs `rustup toolchain install nightly-2026-09-02
   -c rust-src -t wasm32-unknown-unknown`; override with `ZENJPEGAI_NIGHTLY`). Smoke test:
   `target/release/zenjpegai pack-models --models $ZENJPEGAI_REF/models --model 1 --op bop --out ~/tmp/m1_bop.zjb`,
   then `node web/scripts/node-smoke.mjs ~/tmp/m1_bop.zjb <vector>/stream.bits <native decode>.png`.
2. `web/src/worker.js`: `import init, * as zj from '../pkg-{simd,threads}/zenjpegai.js'`; pick
   `pkg-threads` iff `self.crossOriginIsolated`, then `await zj.initThreadPool(navigator.hardwareConcurrency)`.
   Message protocol: `{id, stream}` -> `zj.info` -> ask the page for the bundle if
   `!zj.hasModels(modelId, op)` -> `zj.addModels` -> `zj.decode` -> transfer `rgba.buffer`.
   Bundle naming proposal: `models/m<id>_common.zjb` + `models/m<id>_<op>.zjb`.
3. `web/src/polyfill.js`, `web/scripts/serve.mjs` (three servers on ports 3000-3999), Playwright
   specs comparing `rgba` with native CLI PNGs through `scripts/wasm/png.mjs` (`diffSamples`).
   Expected from the wasip1 measurements: 18..183 samples differ from native, all by 1.
4. Pages: `gh api repos/imazen/zenjpegai/pages`; GitHub Pages cannot send COOP/COEP, so the
   threaded build there depends on a `coi-serviceworker`-style shim (0.1.7 on npm; not evaluated).
   Publishing would make public: the demo streams, the packed upstream weights (BSD, the notice
   is embedded in every `.zjb` and must also be shown on the site: `upstream-notices/LICENSE`),
   and the wasm build of this AGPL/commercial crate.
5. Speed: the `Wasm128` micro-kernel is 3.8x slower than native AVX-512 on one thread
   (0.35 s against 0.093 s). Untried: relaxed-simd is ruled out by the numeric policy, but a
   wider register block for 1x1 convolutions and a profile (`node --cpu-prof`) are not.
