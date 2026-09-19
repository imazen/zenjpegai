// Model-bundle prefetch (src/prefetch.js, src/stream-info.js) and the rayon child-worker
// request collapse (scripts/pack-rayon-child.mjs), 2026-11:
//
//  - `probeStreamInfo` reads `model_id` + the first operating point out of the codestream PIH
//    in plain JS — the polyfill's prefetch is only correct if this bit-level probe agrees
//    with the header fields the encoder recorded in manifest.json (asserted for all 16
//    streams; also checked in Node during development, this is the shipped-code check).
//  - The demo page prefetches every needed `common`+`op` bundle into the SAME Cache API
//    namespace/keys the worker's `ensureModels` reads (`zenjpegai-models-v2`,
//    `<url>?v=<sha>`), so the first decode's model wait is a cache hit — asserted by key.
//  - The polyfill does the same for the visible `<img>` set (unversioned keys when no
//    `bundleVersions` option is given, as in the fixture page).
//  - pack-rayon-child.mjs collapses wasm-bindgen-rayon's ~3 module requests PER rayon child
//    into one `rayon-child.js` fetch per pool: `zenjpegai.js` exactly once, zero
//    `workerHelpers` requests, `rayon-child.js` exactly once, and a small total JS count
//    (was ~50 for a 16-thread pool). Playwright's context routing covers dedicated-worker
//    module requests in Chromium; on engines where it doesn't, the count is observed as 0
//    and the strong assertions are skipped (the pool's own init is covered by
//    wasm-decode/threads specs everywhere).
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(__dirname, '..', '.demo-assets', 'manifest.json'), 'utf8'));
const BASE_PLAIN = process.env.JAI_BASE_PLAIN ?? `http://127.0.0.1:${PORTS.plain}`;
const BASE_ISOLATED = process.env.JAI_BASE_ISOLATED ?? `http://127.0.0.1:${PORTS.isolated}`;
const MODEL_CACHE = 'zenjpegai-models-v2';

const DONE_COUNT_SRC = `(() => [...document.querySelectorAll('.card')].filter((card) => {
  if (card.dataset.state) return card.dataset.state === 'done';
  const c = card.querySelector('canvas');
  return c && c.width > 300;
}).length)()`;

const cacheKeys = (name) =>
  caches.open(name).then((c) => c.keys()).then((keys) => keys.map((r) => r.url));

test('stream-info probe matches manifest modelId/operatingPoint for every stream', async ({ page }) => {
  await page.goto(`${BASE_PLAIN}/decode.html?gpu=off`);
  const rows = await page.evaluate(async () => {
    const { probeStreamInfo } = await import('./src/stream-info.js');
    const mf = await fetch('manifest.json', { cache: 'no-cache' }).then((r) => r.json());
    const out = [];
    for (const img of mf.images) {
      for (const v of img.variants) {
        const bytes = await fetch(`streams/${v.file}${v.sha256 ? `?v=${v.sha256}` : ''}`).then((r) => r.arrayBuffer());
        out.push({ file: v.file, got: probeStreamInfo(bytes), want: { modelId: v.modelId, op: v.operatingPoint } });
      }
    }
    return out;
  });
  for (const r of rows) expect(r.got, r.file).toEqual(r.want);
});

test('demo page prefetches the visible models into the worker cache', async ({ page }) => {
  await page.goto(`${BASE_ISOLATED}/index.html?gpu=off`);
  await page.waitForFunction(`${DONE_COUNT_SRC} >= 1`, { timeout: 30_000 });
  const keys = await page.evaluate(cacheKeys, MODEL_CACHE);
  // The first card's model pair was prefetched (and its bytes transferred to the worker);
  // the cache entry is what remains for later decodes of the same model.
  const first = manifest.images[0].variants[0];
  for (const name of [`m${first.modelId}_common.zjb`, `m${first.modelId}_${first.operatingPoint}.zjb`]) {
    const sha = manifest.models[name]?.sha256;
    expect(
      keys.some((k) => k.includes(`models/${name}`) && (!sha || k.includes(`?v=${sha}`))),
      `${name} in ${MODEL_CACHE}`,
    ).toBe(true);
  }
});

test('polyfill prefetches the visible img models into the worker cache', async ({ page }) => {
  await page.goto(`${BASE_ISOLATED}/polyfill.html`);
  await page.waitForSelector('canvas#direct', { timeout: 30_000 });
  const keys = await page.evaluate(cacheKeys, MODEL_CACHE);
  // The fixture page passes no bundleVersions, so the key is the bare models/ URL — the
  // same key the worker's ensureModels uses under that configuration.
  expect(
    keys.some((k) => /models\/m\d+_(common|sop|bop|hop)\.zjb$/.test(k)),
    `a model bundle in ${MODEL_CACHE}: ${JSON.stringify(keys)}`,
  ).toBe(true);
});

test('rayon pool: <=2 package JS requests, no per-child fetches', async ({ page }) => {
  const requests: string[] = [];
  await page.context().route('**/*', (route) => {
    requests.push(route.request().url());
    return route.continue();
  });
  await page.goto(`${BASE_ISOLATED}/index.html?gpu=off`);
  await page.waitForFunction(`${DONE_COUNT_SRC} >= 1`, { timeout: 30_000 });
  const glue = requests.filter((u) => /zenjpegai\.js(\?|$)/.test(u));
  const snippet = requests.filter((u) => u.includes('workerHelpers'));
  const child = requests.filter((u) => u.includes('rayon-child.js'));
  const js = requests.filter((u) => /\.m?js(\?|$)/.test(new URL(u).pathname));
  if (glue.length + child.length > 0) {
    // Worker traffic was observable: assert the full collapse. Was: 17x zenjpegai.js +
    // 33x workerHelpers snippet for a 16-thread pool.
    expect(glue.length).toBe(1);
    expect(child.length).toBe(1);
    expect(snippet.length).toBe(0);
    // All page+worker JS together: demo.js, coi-loader.js, pool.js, prefetch.js,
    // stream-info.js (polyfill is not imported by the demo), worker.js, zenjpegai.js,
    // rayon-child.js — single digits, not ~50.
    expect(js.length).toBeLessThanOrEqual(10);
  } else {
    // Routing doesn't see dedicated-worker traffic on this engine — the decode above still
    // proves the pool initialised; the request assertions ran on engines that can see it.
    test.info().annotations.push({ type: 'note', description: 'worker requests not observable via context.route here' });
  }
});
