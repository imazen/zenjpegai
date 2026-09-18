// Core parity + isolation-switch test: the plain server (no COOP/COEP) must fall back to the
// `simd` package; the isolated server (COOP+COEP) must pick `threads` — the first time the
// `threads` package has run in a real browser (see web/README.md "Not done" before this).
// Every decode is pixel-compared to the reference decoder's PNG for that stream (Playwright's
// own browser decodes the PNG; see tests/fixtures/decode.html), within the documented wasm
// policy: differing samples < 1/5000, all off by at most 1 (PORTING.md "WebAssembly numeric
// policy" / CONTRIBUTING.md "How parity is proven").
import { test, expect } from '@playwright/test';
import { PORTS } from '../playwright.config';

const STREAM = 'streams/line-plot_bpp25.jai';
const PNG = '_native/line-plot_bpp25.native.png';
const MAX_DIFFERING_RATIO = 1 / 5000;

async function decodeOnce(page, port: number) {
  await page.goto(`http://127.0.0.1:${port}/decode.html`);
  await page.evaluate(() => window.__ready);
  return page.evaluate(
    ({ stream, png }) => window.__decodeAndCompare(stream, png),
    { stream: STREAM, png: PNG },
  );
}

test('plain server (no isolation) falls back to the simd package', async ({ page }) => {
  const r = await decodeOnce(page, PORTS.plain);
  expect(r.timings.variant).toBe('simd');
  expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
  expect(r.max).toBeLessThanOrEqual(1);
  console.log(`[plain/simd] ${r.differing}/${r.samples} differ (max ${r.max}), decode ${r.timings.decode.toFixed(1)}ms`);
});

test('isolated server (COOP+COEP) uses the threads package', async ({ page }) => {
  const r = await decodeOnce(page, PORTS.isolated);
  expect(r.timings.variant).toBe('threads');
  expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
  expect(r.max).toBeLessThanOrEqual(1);
  console.log(`[isolated/threads] ${r.differing}/${r.samples} differ (max ${r.max}), decode ${r.timings.decode.toFixed(1)}ms`);
});

test('strict CSP server also decodes correctly (canvas render path needs no blob:/data:)', async ({ page }) => {
  const r = await decodeOnce(page, PORTS.strict);
  expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
  expect(r.max).toBeLessThanOrEqual(1);
});
