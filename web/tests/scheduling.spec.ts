// Decode scheduling (user report 2026-09-18: "with all images decoding at once each takes
// about a second"). The pool must throttle: at most min(navigator.hardwareConcurrency - 1, N)
// concurrent single-thread (simd) workers behind ONE shared queue, exactly one decode in
// flight on the threads build (it already spans all cores), and the demo/polyfill must
// decode in visibility order, only once an image is within one viewport height, with a
// per-image placeholder while it waits.
//
// Measured here, in chromium (numbers appended to benchmarks/wasm_demo_scheduling_<date>.tsv
// by the last test):
//   - solo_ms: one stream decoded alone on the simd path (the bound baseline)
//   - first_ms: demo page load -> first card's pixels on screen (wall clock, Node side —
//     the coi-loader's service-worker promotion reloads the page once, so in-page
//     performance.now() would reset mid-measurement)
//   - total_ms: demo page load -> every card decoded (a scroll pass is driven so
//     viewport-gated decodes are all reached)
//   - per-image timings.decode on a page that decodes the same stream 8 times at once:
//     every one must stay <= 2x solo_ms (the throttle's whole point is that queueing,
//     not CPU oversubscription, is what serialises them)
//
// JAI_BASE_PLAIN / JAI_BASE_ISOLATED override the server URLs — used to point the same
// measurements at a site tree built from the pre-fix revision for the before/after report.
import { test, expect } from '@playwright/test';
import { readFileSync, existsSync, appendFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(__dirname, '..', '.demo-assets', 'manifest.json'), 'utf8'));
const N_IMAGES = manifest.images.length;
const MANIFEST_DIMS = manifest.images.map((i) => [i.variants[0].width, i.variants[0].height]);
const STREAM = 'streams/mountain_bpp25.jai'; // ~1 MP photo, model 1
const BASE_PLAIN = process.env.JAI_BASE_PLAIN ?? `http://127.0.0.1:${PORTS.plain}`;
const BASE_ISOLATED = process.env.JAI_BASE_ISOLATED ?? `http://127.0.0.1:${PORTS.isolated}`;

// Bound for "first image decoded" on the demo page (chromium): pool spawn + wasm init +
// (on the plain profile) one coi-loader service-worker reload + first model-bundle
// fetch/parse + one decode. 15 s leaves generous margin on a shared box without making
// the test useless.
const FIRST_IMAGE_BOUND_MS = 15_000;

const metrics: Record<string, number> = {};

// A card counts as decoded when the new markup reports data-state="done", or — on the
// pre-fix page (no data-state) — when its canvas has grown past the 300x150 default.
// Passed to waitForFunction as a string so it is self-contained in the page.
const DONE_COUNT_SRC = `(() => [...document.querySelectorAll('.card')].filter((card) => {
  if (card.dataset.state) return card.dataset.state === 'done';
  const c = card.querySelector('canvas');
  return c && c.width > 300;
}).length)()`;

async function soloDecodeMs(page): Promise<number> {
  await page.goto(`${BASE_PLAIN}/decode.html`);
  await page.evaluate(() => window.__ready);
  const r = await page.evaluate(async (stream) => {
    const bytes = await fetch(stream).then((res) => res.arrayBuffer());
    const out = await window.__pool.decode(bytes);
    return out.timings;
  }, STREAM);
  return r.decode;
}

// Scroll the page to the bottom in viewport-sized steps so every viewport-gated decode is
// triggered, then back to the top.
async function scrollThrough(page) {
  await page.evaluate(async () => {
    const step = window.innerHeight;
    for (let y = 0; y <= document.body.scrollHeight; y += step) {
      window.scrollTo(0, y);
      await new Promise((r) => setTimeout(r, 60));
    }
    window.scrollTo(0, 0);
  });
}

test.describe('decode scheduling', () => {
  test.skip(({ browserName }) => browserName !== 'chromium', 'scheduling numbers are chromium-only');

  test('demo page: first image lands fast, everything decodes', async ({ page }) => {
    test.setTimeout(120_000);
    const t0 = Date.now();
    await page.goto(`${BASE_PLAIN}/index.html`);

    // Placeholder contract (post-fix markup only): every card's canvas is pre-sized to the
    // decoded picture's box before any decode finishes, so the layout never shifts and the
    // slot is visibly reserved while the decode is queued.
    const hasState = await page.evaluate(() => Boolean(document.querySelector('.card')?.dataset.state));
    if (hasState) {
      const dims = await page.evaluate(
        () => [...document.querySelectorAll('.card canvas')].map((c) => [c.width, c.height]),
      );
      expect(dims.length).toBe(N_IMAGES);
      dims.forEach(([w, h], i) => expect([w, h]).toEqual(MANIFEST_DIMS[i]));
    }

    await page.waitForFunction(`${DONE_COUNT_SRC} >= 1`, { timeout: 60_000, polling: 100 });
    const tFirst = Date.now() - t0;
    metrics.first_ms = tFirst;
    console.log(`[metric] first_image_ms=${tFirst}`);
    expect(tFirst).toBeLessThan(FIRST_IMAGE_BOUND_MS);

    await scrollThrough(page);
    await page.waitForFunction(`${DONE_COUNT_SRC} >= ${N_IMAGES}`, { timeout: 90_000, polling: 150 });
    const tTotal = Date.now() - t0;
    metrics.total_ms = tTotal;
    console.log(`[metric] total_ms=${tTotal}`);

    // Pool self-report: concurrency never exceeded the simd cap (min(4, hwc - 1)).
    const stats = await page.evaluate(() => window.__poolStats?.() ?? null);
    if (stats) {
      const hwc = await page.evaluate(() => navigator.hardwareConcurrency);
      console.log(`[metric] max_inflight=${stats.maxInflight} hwc=${hwc} isolated=${stats.isolated}`);
      expect(stats.maxInflight).toBeLessThanOrEqual(stats.size);
      expect(stats.maxInflight).toBeGreaterThan(0);
    }
  });

  test('demo page without service workers stays on the simd pool', async ({ browser }) => {
    test.setTimeout(120_000);
    // Aborting the sw-coi.js fetch makes serviceWorker.register() reject, so coi-loader
    // gives up and the page stays non-isolated — the non-isolated multi-worker simd path
    // end to end. (Playwright's serviceWorkers:'block' instead leaves registration pending
    // forever, which would freeze the page on the top-level await.)
    const ctx = await browser.newContext();
    const page = await ctx.newPage();
    await page.route('**/sw-coi.js', (route) => route.abort());
    try {
      const t0 = Date.now();
      await page.goto(`${BASE_PLAIN}/index.html`);
      await page.waitForFunction(`${DONE_COUNT_SRC} >= 1`, { timeout: 60_000, polling: 100 });
      metrics.first_ms_simd = Date.now() - t0;
      console.log(`[metric] first_image_simd_ms=${metrics.first_ms_simd}`);

      // Pool self-report (post-fix only; the pre-fix pool exposes no stats hook). The simd
      // pool leaves one hardware thread for the page's main thread: min(4, hwc - 1).
      const hwc = await page.evaluate(() => navigator.hardwareConcurrency);
      const stats0 = await page.evaluate(() => window.__poolStats?.() ?? null);
      if (stats0) {
        expect(stats0.isolated).toBe(false);
        expect(stats0.size).toBe(Math.max(1, Math.min(4, hwc - 1)));
      }

      await scrollThrough(page);
      await page.waitForFunction(`${DONE_COUNT_SRC} >= ${N_IMAGES}`, { timeout: 90_000, polling: 150 });
      metrics.total_ms_simd = Date.now() - t0;
      console.log(`[metric] total_simd_ms=${metrics.total_ms_simd}`);

      const stats = await page.evaluate(() => window.__poolStats?.() ?? null);
      if (stats) {
        console.log(`[metric] simd_path max_inflight=${stats.maxInflight} size=${stats.size} hwc=${hwc}`);
        expect(stats.maxInflight).toBeLessThanOrEqual(stats.size);
        expect(stats.maxInflight).toBeGreaterThan(1); // proves real concurrency, not serialisation
        expect(stats.completed).toBe(N_IMAGES);
      }
    } finally {
      await ctx.close();
    }
  });

  test('threads build runs exactly one decode at a time', async ({ page }) => {
    test.setTimeout(120_000);
    await page.goto(`${BASE_ISOLATED}/index.html`);
    await page.waitForFunction(`${DONE_COUNT_SRC} >= 1`, { timeout: 60_000, polling: 100 });
    const stats = await page.evaluate(() => window.__poolStats?.() ?? null);
    if (stats) expect(stats.maxInflight).toBe(1);
    const workers = await page.evaluate(() => window.__pool?.workers?.length ?? -1);
    if (workers >= 0) expect(workers).toBe(1);
  });

  test('per-image decode under load stays within 2x of a solo decode (simd path)', async ({ page }) => {
    test.setTimeout(120_000);
    const solo = await soloDecodeMs(page);
    metrics.solo_ms = solo;
    console.log(`[metric] solo_decode_ms=${solo.toFixed(0)}`);

    await page.goto(`${BASE_PLAIN}/many.html`);
    const decodes: number[] = await page.evaluate(async (stream) => {
      const out: number[] = [];
      window.addEventListener('jpegaidecoded', (e) => out.push(e.detail.timings.decode));
      // 8 copies of the same stream through the polyfill: all queued at once.
      for (let i = 0; i < 8; i++) {
        const img = document.createElement('img');
        img.src = stream;
        img.alt = `copy ${i}`;
        document.body.append(img);
      }
      const t0 = performance.now();
      while (out.length < 8) {
        if (performance.now() - t0 > 60_000) break;
        await new Promise((r) => setTimeout(r, 50));
      }
      return out;
    }, STREAM);
    expect(decodes.length).toBe(8);
    const worst = Math.max(...decodes);
    console.log(`[metric] loaded_decode_max_ms=${worst.toFixed(0)} solo=${solo.toFixed(0)} all=${decodes.map((d) => d.toFixed(0)).join(',')}`);
    expect(worst).toBeLessThanOrEqual(solo * 2);
  });

  test('append measurements to benchmarks/wasm_demo_scheduling_<date>.tsv', async ({}, testInfo) => {
    const date = new Date().toISOString().slice(0, 10);
    const dir = join(__dirname, '..', '..', 'benchmarks');
    const tsv = join(dir, `wasm_demo_scheduling_${date}.tsv`);
    if (!existsSync(tsv)) writeFileSync(tsv, 'date\tmetric\tms\n');
    const rows = Object.entries(metrics).map(([k, v]) => `${date}\t${k}\t${v.toFixed(1)}`);
    appendFileSync(tsv, rows.join('\n') + '\n');
    await testInfo.attach('scheduling-metrics', { body: JSON.stringify(metrics), contentType: 'application/json' });
  });
});
