# zenjpegai in the browser — status and next steps

Goal (user, 2026-09-17): wasm fully optimised including wgpu and workers, a fallback for strict
mode (no cross-origin isolation), a polyfill so `<img>` JPEG AI pictures decode in-browser fast,
proper Playwright testing, a demo pushed to GitHub Pages from CI, demo images from imazen-26.

Updated 2026-09-18. What follows is exact: missing things first.

## Not done / open

| # | Deliverable | State |
| --- | --- | --- |
| 6 | GitHub Pages | **enabled** (checked 2026-09-18: `gh api repos/imazen/zenjpegai/pages` returns `build_type: workflow`, site live at `https://imazen.github.io/zenjpegai/`; repo is public). The Pages deploy makes PUBLIC: the 16 demo `.jai` streams, the 8 packed model-bundle halves (`m{0,1,2,3}_{common,bop}.zjb`, BSD-licensed upstream weights — notice text embedded in every bundle and shown on the demo page + `upstream-notices/LICENSE`), the reference-decoder PNGs used as the test oracle, and this wasm build of the AGPL/commercial-licensed crate. Nothing else. |
| 6 | GPU (`gpu/`) wiring into the web build | **wired and verified on hardware** (2026-09-18, RTX 2080 via Dawn/Vulkan in headless Chromium — §7): the `mappedAtCreation` trap is fixed at the root in `gpu/` (unmapped `create_buffer` + chunked `queue.write_buffer`); worker-side trap recovery stays as defence in depth but no longer fires. Remaining limitation: Dawn's SwiftShader adapter loses its device at model-2 (bpp75) load — the GPU path then falls back per call to the CPU engine (§7). `auto` deliberately stays on the CPU engine: measured GPU vs threads wall decode at demo sizes is parity-or-slower (§7). |

| 1 | Bundle fetched **exactly** once per (model, op) actually needed | Done at the pool/worker level (`ensureModels` in `web/src/worker.js` checks `hasModels` before fetching, and the Cache API means a repeat visit skips the network entirely) but each `simd`-variant Worker in the pool has its OWN wasm memory, so if the pool spawns >1 simd worker (it does, up to `hardwareConcurrency`, capped at 4) and two images on the page need the same model, **that model's bundle is fetched+parsed once per worker that needs it**, not once globally. Acceptable (HTTP/Cache-API means only the first worker pays the network cost; the rest hit cache) but not "exactly once" in the strict sense — documented, not fixed. |
| — | wasm SIMD128 caniuse | not independently re-checked this session (Firefox 155 / Chrome 153 / Safari 26 — the three actually installed and tested below — all ran the `simd` package correctly, which is itself evidence, but no browsers older than "currently installed by Playwright" were probed). |

## Done and measured

### 1. Worker pool + polyfill (`web/src/{worker,pool,polyfill}.js`)

- `pool.js`: `DecoderPool` — one persistent `threads`-variant worker when `crossOriginIsolated`
  (it already parallelises one decode across `hardwareConcurrency` via rayon; a second one would
  just contend for the same cores), else a round-robin pool of up to 4 `simd`-variant workers so
  independent images decode concurrently.
- `worker.js`: picks `simd`/`threads` by `crossOriginIsolated`, fetches model bundles through the
  Cache API (`m<id>_common.zjb` + `m<id>_<op>.zjb`), decodes, transfers the RGBA buffer back.
  **Two real bugs found running this in an actual browser for the first time** (both fixed, both
  the kind of thing that only shows up under real concurrent load, not a single manual decode
  call):
  - `self.onmessage` was assigned only after the top-level `await` (wasm init +
    `initThreadPool`). A `message` event dispatched to a global scope with no listener registered
    at dispatch time is dropped, not queued — so any decode request posted before the worker
    finished loading (exactly what `polyfill.js`'s `scan()` does: it starts decoding every
    matching `<img>` immediately, without waiting for the pool to report ready) was silently
    lost. Every affected decode hung forever: no result, no error. Fix: attach the listener
    first, buffer into an internal queue, drain it once ready.
  - `mod.decode()` is not reentrant. `self.onmessage` being `async` let a second `'decode'`
    message start (at the first `await` inside `ensureModels`) before the first one's
    `mod.decode()` call returned; two calls in flight at once on the `threads` package deadlocks
    rayon (workers left `Atomics.wait`-ing on state a still-in-flight sibling call owns). The
    same queue/drain fix above also serializes this — one worker never runs two decodes at once.
  - The `threads` wasm build itself never linked before this session:
    `-Ctarget-feature=+atomics,+bulk-memory,...` and `--max-memory=...` alone still produce a
    NON-shared `WebAssembly.Memory`; `wasm-bindgen-rayon`'s own worker spawner
    (`workerHelpers.no-bundler.js`, `worker.postMessage({memory, ...})`) then throws `Failed to
    execute 'postMessage' on 'Worker': #<Memory> could not be cloned` the instant it tries to
    start a second worker. Needed `-Clink-arg=--shared-memory -Clink-arg=--import-memory`, and
    separately `-Clink-arg=--export=__wasm_init_tls` (+ `__tls_size`/`__tls_align`/`__tls_base`)
    or `wasm-bindgen`'s thread-prep pass fails with "failed to find `__wasm_init_tls`". All in
    `web/scripts/build-wasm.sh`'s `threads` case now, with inline comments.
  - `modelsBaseUrl` passed through as a relative string resolved against the WORKER's own script
    location (`.../src/`), not the page that constructed the pool — fixed in `pool.js` by
    resolving it to an absolute URL against `location.href` at construction time.
- `polyfill.js`: finds `img[src$=".jai"]`, `img[data-src$=".jai"]`, `img[srcset*=".jai"]` /
  `data-srcset`, and `picture > source[type="image/jpeg-ai"]`; MutationObserver for content added
  later. `loading="lazy"` is honoured via `IntersectionObserver` (decode deferred until near
  viewport) rather than dropped. Two render modes: default **`canvas`** (replace the `<img>` with
  a `<canvas>`, `putImageData` directly — zero extra encode step; `alt` becomes
  `role="img"`/`aria-label`, `width`/`height`/`id`/`class`/`style` copied over) and opt-in
  **`img`** (`data-jai-render="img"`: keep the `<img>`, `canvas.toBlob('image/png')` +
  `URL.createObjectURL`, native `<img>` semantics preserved but pays a PNG encode). No eval, no
  inline script (every fixture/demo script is an external file specifically so this holds under
  a CSP with no `'unsafe-inline'`).
- **Extension / MIME type**: no official one exists. Checked 2026-09-17/18: ISO/IEC 6048-1:2025
  is JPEG AI's actual standard number (not "6048-1" as a draft — it published in 2025); IANA's
  `image/*` media-type registry (`iana.org/assignments/media-types/image`) has no `jpeg-ai`
  entry; the reference software's docs/code name no extension either. `.jai` / `image/jpeg-ai`
  are this project's placeholder, defined as exactly two exported constants
  (`EXTENSION`/`MIME` in `polyfill.js`) — change those two lines if/when one is registered.

### 2. Playwright suite (`web/tests/*.spec.ts`), 3 static servers (`web/scripts/serve.mjs`)

`playwright.config.ts` starts all three (three consecutive ports derived from the checkout's
absolute path, in the required 3000-3999 range)
via `webServer: [...]` before any test:
- **`plain`**: no headers. Proves the `simd`-package fallback.
- **`isolated`**: COOP `same-origin` + COEP `require-corp`. Proves the `threads` package (the
  worker-bug-hunting above happened running tests against this profile).
- **`strict`**: isolated headers + `Content-Security-Policy: default-src 'self'; script-src
  'self' 'wasm-unsafe-eval'; worker-src 'self' blob:; img-src 'self'; style-src 'self'
  'unsafe-inline'; connect-src 'self'; base-uri 'none'`. `worker-src` needs `blob:` — not this
  project's code, `wasm-bindgen-rayon`'s own bundler-less child-worker spawn (it fetches its own
  script as a blob and starts a `Worker` from the object URL, to dodge a cross-origin import-path
  restriction); without it `initThreadPool` hangs forever waiting on child workers a bare CSP
  silently blocked. `img-src` deliberately has no `blob:`/`data:`: proves the `canvas` render
  mode needs neither, and that the opt-in `img` mode is BLOCKED under this policy (its own test).

84 tests across 4 projects — chromium, firefox, **webkit**, and `chromium-webgpu` (a second
Chromium launch with WebGPU flags; see §7). All installed and ran clean — no "if it installs"
fallback needed this run, `npx playwright install chromium firefox webkit` succeeded after
`sudo npx playwright install-deps` for `libmanette-0.2-0`. **84 passed, 0 skipped in ~3 min**
(`npx playwright test`). Browser-specific tests are gated by `testIgnore`/`testMatch` in
`playwright.config.ts`, not by runtime `test.skip` — a gated-out test is excluded from the run
entirely, so anything listed as running must pass.

Port note: the port base is derived deterministically from the checkout's absolute path, so a
sibling jj workspace's servers can never collide with this one's; `JAI_PORT_BASE=<n>` shifts
all three when the derivation itself collides. `reuseExistingServer` can still adopt a stale
server of THIS workspace squatting on the derived port — cache-swap.spec.ts' manifest-identity
assertion guards against exactly that (seen live: a foreign server on the isolated port made
"isolated/strict" tests exercise a build that wasn't this tree's).

- `wasm-decode.spec.ts`: plain->simd, isolated->threads, strict->still decodes; every decode
  pixel-compared (in-browser, via the browser's own PNG decoder — `createImageBitmap` + canvas,
  `tests/fixtures/decode.js`) against the reference decoder's PNG for the same stream: **80 of
  3,000,000 samples differ, all by exactly 1** (well inside the documented "< 1/5000, all off by
  at most 1" bound — the ratio here is ~1/37,500).
- `polyfill.spec.ts`: direct `src`, `srcset` candidate selection, `picture>source`, the `img`
  render mode, `loading=lazy` deferral until scroll-into-view — all pass.
- `strict-csp.spec.ts`: default mode clean (zero CSP violations reported via
  `securitypolicyviolation`), `img` mode's `blob:` load is blocked and reported as an `img-src`
  violation, exactly as designed.
- `scheduling.spec.ts`: the decode-scheduling contract — bounded concurrency
  (`maxInflight <= min(navigator.hardwareConcurrency - 1, 4)` on the simd pool, exactly 1 on the
  threads pool), viewport-gated demo decodes with pre-sized placeholders, first-image bound,
  and "per-image decode under 8x load stays <= 2x a solo decode". Appends to
  `benchmarks/wasm_demo_scheduling_<date>.tsv`; chromium-only. `JAI_BASE_PLAIN` /
  `JAI_BASE_ISOLATED` env vars point it at another site tree (used for the pre-fix comparison).
- `webgpu-decode.spec.ts` (added 2026-09-18): `gpu=on` / `software` / `off` package selection
  and canvas presentation, GPU-path parity reported separately — details in §7.

- `benchmark.spec.ts`: decodes all 16 demo streams per browser, appends timings to
  `benchmarks/wasm_decode_<date>.tsv` + a `.meta` (commit/host/command). `just web-test` or the
  Pages CI workflow run it; `npm test` in `web/` runs the whole suite.
- `benchmark-gpu.spec.ts` (added 2026-09-18): same corpus through the GPU pool (`gpu=on`, or
  `software` under `WEBGPU_ADAPTER=swiftshader`) on the `chromium-webgpu` project plus
  `gpu=off` rows on the threads and simd packages, decode and present timed separately, appends
  to `benchmarks/wasm_decode_<date>_gpu.tsv` + `.meta` (adapter name and launch flags).

Measured decode times (isolated server, `threads` package — Ryzen 9 9950X3D, shared box):
**~95-130 ms** median per ~1 MP image in chromium/firefox, ~130 ms in webkit — roughly half of
the 2026-09-18 baseline. The wins: the rayon pool defaults to `min(hardwareConcurrency, 16)`
child workers (a 32-child pool regressed past the 16-core knee: 162 vs 132 ms mean —
`benchmarks/wasm_threads_2026-09-18.tsv`), and the conv kernel's per-row border scratch is
`thread_local` instead of two allocations per output row. The `run_row` micro-kernel itself is
at its bit-identity floor (unfused `mul`+`add`, 16 v128 registers) — see
`benchmarks/wasm_profile_2026-09-18.md` for the profile and the rejected experiments.
Plain server (`simd` package, single-threaded): **~730-870 ms**. Full 192-row table:
`benchmarks/wasm_decode_2026-09-18.tsv`.

### 3. Demo (`web/demo/`) + `demo-assets-v1` release

8 images from imazen-26 (`car`, `mountain`, `dumplings`, `artwork`, `brochure`, `patent-scan`,
`line-plot`, `clipart` — one per a diverse set of corpus categories: photo, landscape, food,
CC0 artwork reproduction, born-digital document, bilevel scan, synthetic line art, AI-generated
flat graphic), downscaled to ~1 MP with `web/scripts/prep-demo-images` (a standalone Rust binary:
`zenpng` decode/encode + `zenresize` Lanczos resample — Imazen tooling only), encoded at two
rates (target 0.25 and 0.75 bpp) with the **reference** encoder
(`jpeg-ai-reference-software`'s `python -m src.reco.coders.encoder`). Full provenance and
licences: `web/demo/IMAGES.md`. None of the 40 binary assets (16 streams, 8 model-bundle halves,
16 reference PNGs) are in git — published as prerelease
[`demo-assets-v1`](https://github.com/imazen/zenjpegai/releases/tag/demo-assets-v1);
`web/scripts/fetch-demo-assets.mjs` downloads them (needs `gh` authenticated against the repo —
it's private today) into `web/.demo-assets/` (gitignored), and `web/scripts/build-site.mjs`
assembles the servable tree at `web/dist/site/` (also gitignored) that both Playwright and the
Pages workflow serve.

`web/demo/demo.js`: per-image card; the low-rate variant decodes once the card is within one
viewport height of the viewport (`IntersectionObserver`, `rootMargin: '100% 0px'`, in
visibility order — a synchronous rect check starts already-near cards without waiting for the
first observer callback), and clicking the other rate decodes it with queue-jumping priority.
Each card's canvas is pre-sized from the manifest so the layout never shifts while a decode is
queued (`data-state`: pending/queued/decoding/done/error). Timing shows fetch, queue wait,
model-load, and decode ms, plus build variant (`simd`/`threads`), SIMD tier, and a WebGPU
feature-detection note (see the "Not done" table). `web/demo/sw-coi.js` +
`coi-loader.js`: an own-written (not vendored) from-scratch implementation of the
`coi-serviceworker` technique — a service worker that adds COOP/COEP to every same-origin
response so a static host that can't send those headers (GitHub Pages) still gets
`crossOriginIsolated === true` after one reload, enabling the `threads` package there. The
promotion wait is bounded (1.5 s) so a stalled or blocked service-worker install can't hang
the top-level await in `demo.js`. Demo-only
by design: wired into `demo.js`, **not** into `src/polyfill.js` — a general-purpose polyfill
embedded on someone else's page has no business unilaterally installing a service worker that
rewrites every response on their origin. `DecoderPool` re-checks `crossOriginIsolated` itself
regardless, so the non-isolated (`simd`) path always works with or without the shim.

### 4. wasm crate (`wasm/`, `zenjpegai-wasm`)

`addModels(bundle)`, `hasModels(modelId, op)`, `info(stream)`, `decode(stream) -> {width,
height, rgba}`, `buildMode()`, `simdTier()`, `releaseBuffers()`, `initThreadPool` in the threads
build. The `gpu` feature adds `initGpu(mode)`, `gpuStatus()`, `present(stream, offscreenCanvas)`,
`disableGpu()`, and makes `decode` async (see §7). `web/scripts/build-wasm.sh [simd] [threads]
[webgpu]` (threads genuinely works, see the worker-bug section above).

| package | after wasm-bindgen | after `wasm-opt -O3` | gzip -9 | brotli -q11 |
| --- | ---: | ---: | ---: | ---: |
| `pkg-simd` | 434,161 | **362,789** | 150,436 | 124,606 |
| `pkg-threads` | 715,950 | 482,103 | 181,659 | 146,787 |
| `pkg-webgpu` (later rebuild, §7) | 910,067 | 651,502 | 265,613 | 213,817 |

(`pkg-simd` was 389,178 bytes post-`wasm-opt` before this session's size audit — see next
section — a 6.8% cut.)

### 5. Size audit (`benchmarks/wasm_size_2026-09-18.md`) — full detail there, summary here

- Installed `twiggy` 0.8.0 and `cargo-bloat` 0.12.1 (`cargo install`, `~/.cargo/bin`).
- **Verified clean**: `zenpixels-convert` and `zencodec` were never dependencies of this crate;
  `zenpng`/`rayon`/`imgref`/`rgb` were already behind the `cli`/`parallel` features, both off for
  the `simd` wasm build (`cargo tree -p zenjpegai-wasm --target wasm32-unknown-unknown` — none of
  the four appear; `rayon` appears only with `--features zenjpegai-wasm/threads`, correctly).
  Re-verified after another agent's `zencodec` core-crate feature landed on `main` — still off
  for wasm, because `wasm/Cargo.toml` never opts into it.
- **Found and fixed**: `src/weights/mod.rs` always linked a ~450-line data-only pickle
  interpreter (`src/weights/{pickle,zip}.rs`) needed only for reading raw upstream `.pth` files —
  wasm only ever reads packed `ZJM1`/`ZJB1` bundles, but the branch that picks between them was a
  runtime check on the file's magic bytes, so the linker couldn't drop the dead path. New
  additive `pth` feature (default ON, so native/CLI is unaffected; off for wasm because
  `wasm/Cargo.toml` doesn't list it) gates it. **6.8% smaller** shipped `pkg-simd`
  (389,178 -> 362,789 bytes after `wasm-opt`), 5.3% smaller gzipped. All 13 test binaries still
  pass with `pth` on (the default); `cargo fmt --check` / `cargo clippy -D warnings` clean.
- **Measured and rejected**: `opt-level = "z"` (a new, inert, documented
  `[profile.wasm-release-z]` in `Cargo.toml` for repeatable comparison) is 14.6% smaller after
  `wasm-opt -O3` but **2.4x slower to decode** (1651ms vs 690ms median-of-8, node, one stream).
  `wasm-release` stays at `opt-level = 3`.
- **Measured, no change**: `wasm-opt -Oz` vs `-O3` (opt-level 3 held fixed) — 0.3% smaller,
  decode time within measurement noise on this shared box. Kept `-O3`.
- `panic = "abort"` was already the default; no `std::fmt`/`Debug` formatting found in any hot
  decode-path function (only in `Error` construction, unavoidable and off the success path).
- `cargo-bloat --crates` itself never completed against this `-Z build-std` +
  `wasm32-unknown-unknown` + `crate-type = ["cdylib", "rlib"]` combination ("only 'bin', 'dylib'
  and 'cdylib' crate types are supported") — a tool limitation, documented, not worked around;
  `twiggy top`/`dominators` covered the same ground (both tools' output is in the size-audit doc).

### 6. Colour

Canvas output is 8-bit sRGB RGBA today (`wasm/src/lib.rs::decode` truncates any >8-bit source via
`>> shift`); the decoder's `to_rgb_planes` produces BT.709/sRGB YCbCr already, so this is
colour-correct for every stream in the demo corpus and requires no conversion to hand to a plain
`ctx.getContext('2d')` canvas. **HDR / wide-gamut canvas output is out of scope for this
session** — but measured, not assumed, per the user's follow-up:

`web/tests/colorspace-probe.spec.ts` asks each installed browser directly (`canvas.getContext('2d',
{colorSpace:'display-p3'})`, `new ImageData(data, w, h, {colorSpace:'display-p3'})`,
`OffscreenCanvas`+`display-p3`) rather than trusting a compat table:

| browser (installed version) | `getContext('2d',{colorSpace})` | `ImageData` `colorSpace` | `OffscreenCanvas` P3 |
| --- | --- | --- | --- |
| Chromium 153.0.8010.12 | `display-p3` | `display-p3` | `display-p3` |
| Firefox 155.0 | falls back to `srgb` (no error, silently ignored) | **no `colorSpace` property exists on `ImageData` at all** | unavailable |
| WebKit 26.6 (Safari engine) | `display-p3` | `display-p3` | **unavailable** (main-thread canvas supports it, `OffscreenCanvas` does not) |

Cross-checked against MDN's browser-compat-data (`api/ImageData.json`, `colorSpace`): Chrome/Edge
92+, Safari 15.2+, **Firefox `version_added: false`** — matches the live probe exactly. So: P3
canvas output is a real option in Chromium- and WebKit-based browsers today, but **Firefox has no
wide-gamut canvas path at all** (not a version gap — never implemented) and no realistic wasm
build could paper over that. Note the polyfill's canvas mode now DOES transfer an
`OffscreenCanvas` to the worker (`pool.decodeToCanvas`, §7) — safe only because output is
sRGB-only today; a future P3 path must check `getContext('2d')` colour space support on the
OffscreenCanvas (WebKit's gap above would silently reduce fidelity there) before handing one a
`display-p3` context.

**The hook, not yet built**: `Headers::rendering` (`src/header.rs`) already carries the stream's
CICP (colour primaries / transfer / matrix), so a stream that signals wide-gamut primaries is
knowable without guessing. The place to add a P3 path is `wasm/src/lib.rs::decode`'s output
struct — add an explicit `colorSpace: 'srgb' | 'display-p3'` field derived from that CICP data
(never inferred, never silently converted), and have `polyfill.js`'s canvas-mode branch pass
`{colorSpace}` to `getContext`/`ImageData` when it's `'display-p3'` AND the browser actually
returned that colour space from `getContextAttributes()` (Firefox always needs the sRGB path
regardless of what the stream says). Not built this session — no stream in the demo corpus or
reference-vector set carries non-sRGB CICP to develop and test it against.

### 7. WebGPU synthesis path (`pkg-webgpu`, `gpu` cargo feature) — 2026-09-18

Third wasm package (`web/scripts/build-wasm.sh webgpu`): same SIMD CPU engine as `pkg-simd`
plus `zenjpegai-gpu` (wgpu 30 web backend) behind the `gpu` cargo feature. `pkg-simd` and
`pkg-threads` are unchanged.

- **Package selection** (`web/src/worker.js`, driven by `?gpu=` on the worker URL from
  `DecoderPool`'s `gpu` option): the worker probes `navigator.gpu.requestAdapter()` in JS first
  and reports `adapterProbe` (vendor / architecture / `isFallbackAdapter`) in its `ready`
  message — the wgpu-side adapter name comes back empty under Chrome's Dawn mapping, so tests
  and the demo read the probe. `on` downloads `pkg-webgpu` only when a **non-software** adapter
  answers (a software rasteriser is slower than this decoder's own CPU engine, so software-only
  browsers skip the ~214 KB brotli download entirely). `initGpu(mode)` inside wasm then builds
  a `GpuContext`; any failure discards the package and loads `simd`/`threads` as before.
  `software` / `force-software` accept software adapters (test/debug; `force-software` also
  sets WebGPU's `forceFallbackAdapter`). `off` never touches `pkg-webgpu`.
- **`auto` (the default) is the CPU engine, measured**: on an RTX 2080 (2026-09-18, Chromium
  153, Dawn/Vulkan) the WebGPU path's wall-clock `decode_ms` ran 178-302 ms across the ~1 MP
  demo streams vs 158-287 ms for the `threads` CPU engine — at parity or slower on every
  stream, because the per-call upload/dispatch/readback latency (~150 ms) dwarfs the 20-130 ms
  of device time at these sizes. The native sweep (`benchmarks/gpu_decode_2026-09-17_rtx2080`)
  shows the GPU pulling ahead only past ~0.25-1 MP of synthesis work per picture; the demo
  streams sit right at that boundary. So `auto` does not pay the pkg-webgpu download; `?gpu=on`
  opts in (and on a non-isolated page where only `pkg-simd` is available the GPU does win,
  ~4-8x — `on` is worth it there).
- **Decode**: `decode()` is async in the webgpu build. It runs `GpuDecoder` (CPU entropy +
  latent stages, GPU synthesis); any `GpuError` or adapter problem falls back to the CPU engine
  inside the same call and reports `path: 'cpu'` + `gpuError` in timings. Non-GPU packages keep
  the synchronous `decode`.
- **No-readback present**: `pool.decodeToCanvas(stream, canvas)` transfers an
  `OffscreenCanvas` to the worker once (cached on the element as `__jaiCanvasId`/`__jaiWorker`;
  repeat calls address it by id — re-transferring a neutered canvas throws). The worker calls
  `present()`, which hands the canvas to wgpu's `SurfaceTarget::OffscreenCanvas` and blits the
  decoder's
  `rgba8unorm` texture without any CPU readback (`presented: 'gpu'`). Non-presentable pictures
  or any present failure decode on CPU and `putImageData` on a 2d context (`presented: '2d'`).
  `verify` asks the worker to also return a PNG of the canvas for tests.
- **Trap recovery** is now defence in depth, not load-bearing: the `create_buffer_init` /
  `mappedAtCreation` trap was fixed at the root in `gpu/` (unmapped `create_buffer` + chunked
  `queue.write_buffer` — see `gpu/README.md` "Known limitations"). The handler stays because a
  Rust panic in this build is `panic=abort`, which escapes the wasm-bindgen promise as an
  uncaught `RuntimeError`: `self.onerror` catches it, calls `disableGpu()`, fails the in-flight
  message with `trapped: true`, and `pool.js` retries the call once on the CPU engine.
- **Playwright**: `chromium-webgpu` project takes `WEBGPU_ADAPTER=hardware|swiftshader`.
  `hardware` adds `--use-angle=vulkan --ignore-gpu-blocklist` and the specs **fail loudly** if
  `adapterProbe.isFallbackAdapter !== false` or the decode did not take `path: 'gpu'`;
  `swiftshader` adds `--use-webgpu-adapter=swiftshader` for the software path;
  unset keeps the base flags. `webgpu-decode.spec.ts` covers `on`, `software`, `off`, and
  canvas presentation. Hardware run (this host's RTX 2080): all decodes on `path: 'gpu'`,
  `presented: 'gpu'`, 195 of 3,000,000 samples differ from the reference PNG (all by 1 — same
  bound as the CPU tests; GPU output is not bit-identical to CPU output).
- **Benchmarks**: `tests/benchmark-gpu.spec.ts` writes `benchmarks/wasm_decode_<date>_gpu.tsv`
  (+ `.meta` with adapter name / launch flags) — GPU rows plus `off (threads)` / `off (simd)`
  CPU rows for the crossover above. SwiftShader (`WEBGPU_ADAPTER=swiftshader`): bpp25 streams
  run the GPU path (~2 s per decode); at model-2 (bpp75) load Dawn's SwiftShader device is lost
  ("async map a buffer" device error — a tighter software-adapter ceiling, requests are already
  clamped to adapter limits) and remaining rows are honest per-call CPU fallbacks.
- **Package sizes** appended to `benchmarks/wasm_size_2026-09-18.md` §7: `pkg-webgpu` is
  651,502 bytes after `wasm-opt` (+74% over `pkg-simd`), 213,817 brotli.

## Next steps, in order

1. Enable GitHub Pages (human decision — see "Not done" table) and watch `.github/workflows/pages.yml`
   run for real; the workflow itself has never executed (this session only ran the equivalent
   steps locally).
2. ~~Re-run §7's benchmark on a hardware adapter + fix the `create_buffer_init` trap~~ — done
   2026-09-18 (§7): RTX 2080 through Dawn/Vulkan, all decodes on the GPU path, trap removed at
   the root. Open leftovers: Dawn's SwiftShader device loss at model-2 load, and `auto` stays
   on the CPU engine because the GPU does not beat it at demo sizes — revisit the default if a
   fast-path closes the ~150 ms per-call upload/readback gap.
3. Speed: the `Wasm128` micro-kernel is still slower than native AVX-512 on one thread (measured
   2026-09-17: 0.35s vs 0.093s). Untried: relaxed-simd is ruled out by the numeric policy, a wider
   register block for 1x1 convolutions and a `node --cpu-prof` profile are not.
4. A stream with non-sRGB CICP in the reference-vector set, to build and test the P3 hook above
   against real data instead of leaving it as a documented no-op.

## Demo asset hygiene

`manifest.json` in the `demo-assets-v1` release is published verbatim on the site. Its
`sourceFile` values must stay redacted to `<id>_..._<WxH>.<ext>` (like `demo/IMAGES.md`): the
full imazen-26 filenames encode where and when a photo was taken and on which device. Checked
and scrubbed 2026-09-18; re-check whenever the release is regenerated
(`web/scripts/update-demo-manifest.mjs` refuses to write a manifest with an unredacted
`sourceFile`).

**Cache-safety** (2026-09-18 incident: a swapped stream kept its file name and the demo kept
rendering the stale bytes): the manifest carries `file` + `sha256` per variant and a `models`
map of every bundle's digest; `demo.js` fetches `streams/<file>?v=<sha256>` and `manifest.json`
with `cache: 'no-cache'`; `worker.js` keys its Cache API entries on `?v=` digests
(`zenjpegai-models-v2`, the unversioned `v1` namespace is deleted on load). Regenerating the
manifest re-stamps every digest, so a swapped asset is a different URL everywhere it is cached.
Regression test: `web/tests/cache-swap.spec.ts`.
