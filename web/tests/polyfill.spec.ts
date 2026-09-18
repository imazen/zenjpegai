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
