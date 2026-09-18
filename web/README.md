# zenjpegai in the browser — status and next steps

Goal (user, 2026-09-17): wasm fully optimised including wgpu and workers, a fallback for strict
mode (no cross-origin isolation), a polyfill so `<img>` JPEG AI pictures decode in-browser fast,
proper Playwright testing, a demo pushed to GitHub Pages from CI, demo images from imazen-26.

Updated 2026-09-18. What follows is exact: missing things first.

## Not done / open

| # | Deliverable | State |
| --- | --- | --- |
| 6 | GitHub Pages | **not enabled** — `gh api repos/imazen/zenjpegai/pages` -> 404, repo is `private` (`gh api repos/imazen/zenjpegai -q '{private,visibility}'`). `.github/workflows/pages.yml` is ready and correctly configured but its `deploy` job will fail at `actions/deploy-pages` until a human does exactly this: **Settings -> Pages -> Build and deployment -> Source: GitHub Actions.** Once enabled, this makes PUBLIC: the 16 demo `.jai` streams, the 8 packed model-bundle halves (`m{0,1,2,3}_{common,bop}.zjb`, BSD-licensed upstream weights — notice text embedded in every bundle and shown on the demo page + `upstream-notices/LICENSE`), the reference-decoder PNGs used as the test oracle, and this wasm build of the AGPL/commercial-licensed crate. Nothing else. |
| 6 | GPU (`gpu/`) wiring into the web build | **not started, and correctly so per the brief's own fallback clause**: `gpu/README.md` does not exist on `main` (checked 2026-09-18: `find gpu -type f` lists no README), so there is no documented usable async API to wire in yet. `gpu/src/decoder.rs` exists (native wgpu, not built for `wasm32-unknown-unknown` by anything in this session) — the demo (`web/demo/demo.js`) feature-detects `navigator.gpu` and reports whether WebGPU is available in the browser, but always decodes via the CPU/wasm path. Wire a real WebGPU path once `gpu/README.md` lands. |
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

`playwright.config.ts` starts all three (ports 3031/3032/3033, in the required 3000-3999 range)
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

33 tests, chromium + firefox + **webkit** (all three installed and ran clean — no "if it
installs" fallback needed this run, `npx playwright install chromium firefox webkit` succeeded
after `sudo npx playwright install-deps` for `libmanette-0.2-0`), **33 passed in ~48s**
(`npx playwright test`):

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
- `benchmark.spec.ts`: decodes all 16 demo streams per browser, appends timings to
  `benchmarks/wasm_decode_<date>.tsv` + a `.meta` (commit/host/command). `just web-test` or the
  Pages CI workflow run it; `npm test` in `web/` runs the whole suite.

Measured decode times (isolated server, `threads` package, this session — Ryzen 9 9950X3D,
shared box, chromium): **160-360 ms** per ~1 MP image including first-time model-bundle fetch;
plain server (`simd` package, single-threaded): **~800-870 ms**. Firefox/WebKit numbers are
close (all three engines run the exact same wasm bytes) — see
`benchmarks/wasm_decode_2026-09-18.tsv` for the full 48-row table (16 streams x 3 browsers).

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

`web/demo/demo.js`: per-image card, decode-on-load for the low-rate variant, click the other rate
to decode it too; shows fetch/model/decode timing, build variant (`simd`/`threads`), SIMD tier,
and a WebGPU feature-detection note (see the "Not done" table). `web/demo/sw-coi.js` +
`coi-loader.js`: an own-written (not vendored) from-scratch implementation of the
`coi-serviceworker` technique — a service worker that adds COOP/COEP to every same-origin
response so a static host that can't send those headers (GitHub Pages) still gets
`crossOriginIsolated === true` after one reload, enabling the `threads` package there. Demo-only
by design: wired into `demo.js`, **not** into `src/polyfill.js` — a general-purpose polyfill
embedded on someone else's page has no business unilaterally installing a service worker that
rewrites every response on their origin. `DecoderPool` re-checks `crossOriginIsolated` itself
regardless, so the non-isolated (`simd`) path always works with or without the shim.

### 4. wasm crate (`wasm/`, `zenjpegai-wasm`)

`addModels(bundle)`, `hasModels(modelId, op)`, `info(stream)`, `decode(stream) -> {width,
height, rgba}`, `buildMode()`, `simdTier()`, `releaseBuffers()`, `initThreadPool` in the threads
build. `web/scripts/build-wasm.sh [simd] [threads]` (now genuinely both, see the worker-bug
section above for what `threads` needed to actually link).

| package | after wasm-bindgen | after `wasm-opt -O3` | gzip -9 | brotli -q11 |
| --- | ---: | ---: | ---: | ---: |
| `pkg-simd` | 434,161 | **362,789** | 150,436 | 124,606 |
| `pkg-threads` | 715,950 | 482,103 | 181,659 | 146,787 |

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
build could paper over that. This is also why the polyfill never uses `OffscreenCanvas` for its
render path (WebKit's gap above would silently reduce colour fidelity there).

**The hook, not yet built**: `Headers::rendering` (`src/header.rs`) already carries the stream's
CICP (colour primaries / transfer / matrix), so a stream that signals wide-gamut primaries is
knowable without guessing. The place to add a P3 path is `wasm/src/lib.rs::decode`'s output
struct — add an explicit `colorSpace: 'srgb' | 'display-p3'` field derived from that CICP data
(never inferred, never silently converted), and have `polyfill.js`'s canvas-mode branch pass
`{colorSpace}` to `getContext`/`ImageData` when it's `'display-p3'` AND the browser actually
returned that colour space from `getContextAttributes()` (Firefox always needs the sRGB path
regardless of what the stream says). Not built this session — no stream in the demo corpus or
reference-vector set carries non-sRGB CICP to develop and test it against.

## Next steps, in order

1. Enable GitHub Pages (human decision — see "Not done" table) and watch `.github/workflows/pages.yml`
   run for real; the workflow itself has never executed (this session only ran the equivalent
   steps locally).
2. Once `gpu/README.md` describes a usable async API, wire a WebGPU decode path into
   `web/demo/demo.js` behind the existing `navigator.gpu` feature detection.
3. Speed: the `Wasm128` micro-kernel is still slower than native AVX-512 on one thread (measured
   2026-09-17: 0.35s vs 0.093s). Untried: relaxed-simd is ruled out by the numeric policy, a wider
   register block for 1x1 convolutions and a `node --cpu-prof` profile are not.
4. A stream with non-sRGB CICP in the reference-vector set, to build and test the P3 hook above
   against real data instead of leaving it as a documented no-op.

## Demo asset hygiene

`manifest.json` in the `demo-assets-v1` release is published verbatim on the site. Its
`sourceFile` values must stay redacted to `<id>_..._<WxH>.<ext>` (like `demo/IMAGES.md`): the
full imazen-26 filenames encode where and when a photo was taken and on which device. Checked
and scrubbed 2026-09-18; re-check whenever the release is regenerated.
