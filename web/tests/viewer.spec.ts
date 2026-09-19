// Full-window viewer (web/demo/viewer.js): readout honesty, device-pixel geometry, chrome,
// and the compare path's "decode the other rate exactly once" contract.
//
// Geometry contract (the spec the demo implements): the viewer canvas's backing store is
// sized in DEVICE pixels, canvas.width = round(cssW * devicePixelRatio), and the readout's
// `zoom` is image-pixels-per-CSS-pixel (imgW / cssW) as a percent — so at "1:1 device" zoom
// every image pixel lands on exactly one device pixel (r = 1) and
// canvas.width == imageWidth * dpr / zoom holds exactly at both dpr 1 and 2.
//
// Pixel sourcing: opening the viewer on an already-decoded card must NOT re-decode — the
// presented canvas is captured into the staging canvas. (A browser that can't sample a
// transferred canvas falls back to one pool.decode; the strict decode-count assertions
// below are made across the compare click, which is engine-independent.)
//
// Runs in every project: on chromium/firefox/webkit the plain server uses the simd pool;
// on chromium-webgpu (WEBGPU_ADAPTER=hardware) the same page takes the GPU-present path,
// exercising drawImage capture of a WebGPU-blit canvas.
import { test, expect } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(__dirname, '..', '.demo-assets', 'manifest.json'), 'utf8'));
const IMG0 = manifest.images[0];
const [W, H] = [IMG0.variants[0].width, IMG0.variants[0].height];
const BASE = `http://127.0.0.1:${PORTS.plain}`;

const READOUT_RE =
  /1 image px = ([\d.]+) device px · dppx ([\d.]+) · zoom (\d+)% · (\d+)×(\d+) image · (\d+)×(\d+) CSS px · (\d+)×(\d+) device px/;

async function firstCardDone(page) {
  await page.waitForFunction(
    `document.querySelector('.card')?.dataset.state === 'done'`,
    { timeout: 60_000, polling: 100 },
  );
}

async function poolIdle(page) {
  await page.waitForFunction(
    `(() => { const s = window.__poolStats?.(); return s && s.inflight === 0 && s.queued === 0; })()`,
    { timeout: 90_000, polling: 150 },
  );
}

async function openViewer(page) {
  await page.locator('.card canvas').first().click();
  const viewer = page.locator('#jai-viewer');
  await expect(viewer).toBeVisible();
  // First acquire: wait until the readout carries real numbers.
  const readout = page.locator('#jai-viewer-readout');
  await expect(readout).toContainText('device px', { timeout: 30_000 });
  return { viewer, readout };
}

for (const dpr of [1, 2]) {
  test(`viewer readout and 1:1 device-pixel geometry at dppx ${dpr}`, async ({ browser, browserName }) => {
    // deviceScaleFactor is a Chromium-only context option; Firefox/WebKit run the dppx-1
    // case at their native scale (which still exercises capture/fallback per engine).
    test.skip(dpr !== 1 && browserName !== 'chromium', 'deviceScaleFactor is chromium-only');
    test.setTimeout(120_000);
    const ctx = await browser.newContext({ deviceScaleFactor: dpr });
    const page = await ctx.newPage();
    try {
      await page.goto(`${BASE}/index.html`);
      await firstCardDone(page);
      const { viewer, readout } = await openViewer(page);

      // Default mode is fit for the ~1 MP demo images (they don't fit 1:1-device at these
      // viewport sizes); the exact default doesn't matter — the explicit 1:1 mode does.
      let m = (await readout.textContent()).match(READOUT_RE);
      expect(m, `readout shape: ${await readout.textContent()}`).toBeTruthy();
      expect(Number(m[2])).toBe(dpr); // dppx
      expect([Number(m[4]), Number(m[5])]).toEqual([W, H]); // image dims
      // The contract, stated two ways (both within readout integer rounding):
      //   canvas.width == cssW * dpr                       (backing store in device px)
      //   zoom% == 100 * imgW * dpr / canvas.width         (== imgW*dpr/zoom by definition)
      const canvasW = await page.locator('.jai-viewer-canvas').evaluate((c: HTMLCanvasElement) => c.width);
      expect(Math.abs(canvasW - Number(m[6]) * dpr)).toBeLessThanOrEqual(dpr + 1);
      expect(Math.abs((W * dpr * 100) / canvasW - Number(m[3]))).toBeLessThanOrEqual(1);

      // 1:1 device mode: r == 1 exactly, zoom == dpr * 100 %, canvas.width == image width.
      await viewer.getByRole('button', { name: '1:1' }).click();
      m = (await readout.textContent()).match(READOUT_RE);
      expect(m).toBeTruthy();
      expect(Number(m[1]), '1 image px = r device px').toBe(1);
      expect(Number(m[3]), 'zoom %').toBe(dpr * 100);
      expect(Number(m[8])).toBe(W); // device px wide
      expect(Number(m[9])).toBe(H);
      const view = page.locator('.jai-viewer-canvas');
      expect(await view.evaluate((c: HTMLCanvasElement) => c.width)).toBe(W);
      expect(await view.evaluate((c: HTMLCanvasElement) => c.height)).toBe(H);
      // And literally: canvas.width == image width * dpr / zoom.
      const zf = Number(m[3]) / 100;
      expect(await view.evaluate((c: HTMLCanvasElement) => c.width)).toBeCloseTo((W * dpr) / zf, 3);
    } finally {
      await ctx.close();
    }
  });
}

test('viewer closes on Esc, outside click, and browser Back', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto(`${BASE}/index.html`);
  await firstCardDone(page);

  const { viewer } = await openViewer(page);
  await page.keyboard.press('Escape');
  await expect(viewer).toBeHidden();

  // Reopen, close by clicking the pane outside the canvas.
  await openViewer(page);
  const box = await page.locator('.jai-viewer-pane').boundingBox();
  await page.mouse.click(box.x + 10, box.y + box.height - 10); // bottom-left corner, off the canvas
  await expect(viewer).toBeHidden();

  // Reopen, close with the browser Back button (history state).
  await openViewer(page);
  await Promise.all([page.waitForFunction(`document.getElementById('jai-viewer').hidden`), page.goBack()]);
  await expect(viewer).toBeHidden();
});

test('compare decodes the other rate exactly once, timings land on the card', async ({ page }) => {
  test.setTimeout(120_000);
  await page.goto(`${BASE}/index.html`);
  await firstCardDone(page);
  await poolIdle(page); // quiesce auto-decodes so completed deltas are attributable
  const before = await page.evaluate(() => window.__poolStats().completed);

  const { viewer } = await openViewer(page);
  // Opening must not itself decode (the presented canvas is reused); allow one extra only
  // for engines that can't sample a transferred canvas (the spec's fallback branch).
  const afterOpen = await page.evaluate(() => window.__poolStats().completed);
  expect(afterOpen - before, 'decodes spent on open').toBeLessThanOrEqual(1);

  // The "compare" control: rate chips in the viewer bar. Clicking the other rate drives the
  // card's own decode path — exactly one pool job, timings shown on the card.
  const chip = viewer.locator('.jai-viewer-compare button').nth(1);
  await expect(chip).toHaveText('0.75');
  await poolIdle(page);
  const c0 = await page.evaluate(() => window.__poolStats().completed);
  await chip.click();
  await page.waitForFunction(
    `window.__poolStats().completed >= ${c0 + 1}`,
    { timeout: 60_000, polling: 100 },
  );
  await poolIdle(page);
  const c1 = await page.evaluate(() => window.__poolStats().completed);
  expect(c1, 'exactly one decode for the other rate').toBe(c0 + 1);

  // The card shows the new rate: its second rate button is the active one and the card
  // reports done. (Timing text format is demotiming's; the state contract is ours.)
  const card = page.locator('.card').first();
  await expect(card).toHaveAttribute('data-state', 'done');
  const pressed = await card.locator('button[aria-pressed="true"]').count();
  expect(pressed).toBeGreaterThan(0);

  // Viewer still open, showing 0.75; readout still sane.
  await expect(page.locator('#jai-viewer-readout')).toContainText('device px');
  await expect(viewer.locator('.jai-viewer-title')).toContainText('0.75');
});
