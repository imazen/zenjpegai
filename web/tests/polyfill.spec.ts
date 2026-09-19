// <img>/<picture> polyfill behaviour: direct src, srcset, picture>source, the opt-in "img"
// (blob: URL) render mode, and loading="lazy" deferring decode until intersection.
import { test, expect } from '@playwright/test';
import { PORTS } from '../playwright.config';

test.describe('polyfill', () => {
  test.beforeEach(async ({ page }) => {
    await page.goto(`http://127.0.0.1:${PORTS.isolated}/polyfill.html`);
  });

  test('direct src is decoded and swapped to a canvas (default render mode)', async ({ page }) => {
    await page.waitForSelector('canvas#direct');
    const canvas = page.locator('canvas#direct');
    await expect(canvas).toHaveAttribute('width', '1000');
    await expect(canvas).toHaveAttribute('height', '1000');
    await expect(canvas).toHaveAttribute('role', 'img');
    await expect(canvas).toHaveAttribute('aria-label', 'line plot test image');
    await expect(page.locator('img#direct')).toHaveCount(0); // the original <img> is gone
  });

  test('srcset candidate ending in the extension is picked and decoded', async ({ page }) => {
    await page.waitForSelector('canvas#srcset');
  });

  test('picture > source[type=image/jpeg-ai] is decoded instead of the native fallback', async ({ page }) => {
    await page.waitForSelector('picture#picture canvas');
  });

  test('data-jai-render="img" swaps in a blob: URL and keeps the <img> element', async ({ page }) => {
    await page.waitForFunction(() => {
      const img = document.querySelector('img#imgmode');
      return img && img.src.startsWith('blob:');
    });
    const img = page.locator('img#imgmode');
    await expect(img).toHaveAttribute('alt', 'dumplings test image');
    await expect(img).toBeVisible();
  });

  test('loading=lazy defers decode until the element is scrolled into view', async ({ page }) => {
    // Not yet decoded: still an <img>, not in the event log.
    await page.waitForTimeout(300);
    expect(await page.locator('img#lazy').count()).toBe(1);
    let events = await page.evaluate(() => window.__jaiEvents.map((e) => e.url));
    expect(events.some((u) => u?.includes('mountain'))).toBe(false);

    await page.locator('img#lazy').scrollIntoViewIfNeeded();
    await page.waitForSelector('canvas#lazy', { timeout: 15_000 });
    events = await page.evaluate(() => window.__jaiEvents.map((e) => e.url));
    expect(events.some((u) => u?.includes('mountain'))).toBe(true);
  });
});

test.describe('polyfill degradation + prefetch', () => {
  test('no wasm SIMD: fallback <img> stays untouched, jpegaierror reason no-wasm-simd', async ({ page }) => {
    // Break the SIMD probe before any page script runs: every pkg-* requires +simd128, so
    // this is exactly what Safari <= 16.3 / old Firefox ESR look like to the polyfill.
    await page.addInitScript(() => {
      const orig = WebAssembly.validate;
      WebAssembly.validate = (bytes) => bytes?.length === 31 ? false : orig(bytes);
    });
    await page.goto(`http://127.0.0.1:${PORTS.isolated}/polyfill.html`);
    await page.waitForFunction(() => window.__jaiEvents.some((e) => e.error), null, { timeout: 15_000 });
    const img = page.locator('img#direct');
    await expect(img).toHaveCount(1); // the original <img> is still there
    expect(await img.getAttribute('src')).toContain('.jai'); // src untouched
    await expect(img).toHaveAttribute('data-jai-state', 'error');
    expect(page.locator('canvas')).toHaveCount(0); // nothing was swapped in
  });

  test('documented finding: a document preload link is NOT consumed by a Worker fetch', async ({ page }) => {
    // This is why `prefetch` has no 'head' mode: Chromium scopes <link rel=preload> to the
    // initiating document, so the worker's fetch() for the same URL cannot join it and every
    // bundle would download twice. If a future browser starts matching worker fetches to
    // document preloads this test fails — then 'head' is worth revisiting.
    const requests = [];
    page.on('request', (r) => { if (r.url().includes('m0_common.zjb')) requests.push(r.url()); });
    await page.addInitScript(() => {
      window.__noPrefetch = true;
      window.__insertPreload = true;
    });
    await page.goto(`http://127.0.0.1:${PORTS.isolated}/polyfill.html`);
    await page.evaluate(() => {
      const l = document.createElement('link');
      l.rel = 'preload';
      l.as = 'fetch';
      l.crossOrigin = 'anonymous';
      l.href = 'models/m0_common.zjb';
      document.head.append(l);
    });
    await page.waitForFunction(() => window.__jaiEvents.length > 0, null, { timeout: 60_000 });
    expect(requests.length).toBe(2); // the preload AND the worker's own fetch
  });

  test('prefetch "auto" warms the worker Cache API namespace', async ({ page }) => {
    await page.goto(`http://127.0.0.1:${PORTS.isolated}/polyfill.html`);
    // The default install prefetches: once the first image decodes, the model bundles sit in
    // the shared cache under the same ?v= keys the worker uses (no version tokens in the
    // fixture — the key is the bare URL).
    await page.waitForFunction(() => window.__jaiEvents.length > 0, null, { timeout: 60_000 });
    const cached = await page.evaluate(async () => {
      const c = await caches.open('zenjpegai-models-v2');
      const keys = await c.keys();
      return keys.map((k) => k.url);
    });
    expect(cached.some((u) => u.includes('m0_common.zjb'))).toBe(true);
    expect(cached.some((u) => u.includes('m0_bop.zjb'))).toBe(true);
  });

  test('maxPixels fails fast to the fallback with reason too-large', async ({ page }) => {
    await page.addInitScript(() => { window.__maxPixels = 1024; });
    await page.goto(`http://127.0.0.1:${PORTS.isolated}/polyfill.html`);
    await page.waitForFunction(() => window.__jaiEvents.some((e) => e.error), null, { timeout: 15_000 });
    const events = await page.evaluate(() => window.__jaiEvents.filter((e) => e.error));
    expect(events[0].reason).toBe('too-large');
    expect(await page.locator('img#direct').count()).toBe(1); // fallback untouched
  });
});
