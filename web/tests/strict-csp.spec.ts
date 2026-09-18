// Proves the default (canvas) polyfill path needs nothing beyond `script-src 'self'
// 'wasm-unsafe-eval'; worker-src 'self'; connect-src 'self'; img-src 'self'` (see
// scripts/serve.mjs "strict" profile) — no eval, no inline script, no blob:/data:. Also proves
// the opt-in "img" (blob: URL) render mode is BLOCKED under that same policy, which is the
// documented reason "canvas" is the default.
import { test, expect } from '@playwright/test';
import { PORTS } from '../playwright.config';

test('default render mode decodes cleanly under the strict CSP, no violations', async ({ page }) => {
  // The fixture page also has a data-jai-render="img" element (#imgmode), whose blob: load is
  // SUPPOSED to violate img-src — that is the next test below. Filter both violation sources
  // (the page's own listener and Playwright's console feed) to "not img-src" so this test only
  // asserts about the canvas-mode elements it actually checks.
  const violations: string[] = [];
  page.on('console', (msg) => {
    if (msg.type() === 'error' && /Content Security Policy/i.test(msg.text()) && !msg.text().includes('img-src')) {
      violations.push(msg.text());
    }
  });
  await page.goto(`http://127.0.0.1:${PORTS.strict}/polyfill.html`);
  await page.waitForSelector('canvas#direct');
  await page.waitForSelector('picture#picture canvas');
  const reported = await page.evaluate(() => window.__cspViolations);
  expect(reported.filter((d: string) => !d.startsWith('img-src'))).toEqual([]);
  expect(violations).toEqual([]);
});

test('the "img" (blob: URL) render mode is blocked by img-src under the strict CSP', async ({ page }) => {
  await page.goto(`http://127.0.0.1:${PORTS.strict}/polyfill.html`);
  // decodeAndSwap still runs (fetch/decode aren't blocked — only the blob: image load is), so
  // wait for the jpegaidecoded event instead of a DOM change that will never happen.
  await page.waitForFunction(
    () => window.__jaiEvents.some((e) => e.url?.includes('dumplings')),
    { timeout: 15_000 },
  );
  await page.waitForFunction(() => window.__cspViolations.some((d: string) => d.startsWith('img-src')), { timeout: 5_000 });
  // The element is still an <img> (never swapped to canvas) and never got a usable image.
  expect(await page.locator('img#imgmode').count()).toBe(1);
  const naturalWidth = await page.locator('img#imgmode').evaluate((el: HTMLImageElement) => el.naturalWidth);
  expect(naturalWidth).toBe(0);
});
