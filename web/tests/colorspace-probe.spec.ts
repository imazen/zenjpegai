// Not a correctness test (nothing here can fail): a probe that prints what THIS installed
// browser actually returns for wide-gamut (Display P3) canvas support, so the numbers in
// web/README.md "Colour" are measured, not copied from MDN's compat tables for a browser
// version we didn't run. See web/README.md for the decision this feeds (sRGB output today,
// P3 as a documented future hook keyed on the stream's CICP, never silent gamut conversion).
import { test } from '@playwright/test';
import { PORTS } from '../playwright.config';

test('probe: display-p3 canvas/ImageData support', async ({ page, browserName, browser }) => {
  await page.goto(`http://127.0.0.1:${PORTS.plain}/decode.html?gpu=off`);
  const result = await page.evaluate(() => {
    const out: Record<string, string> = {};
    try {
      const c = document.createElement('canvas');
      const ctx = c.getContext('2d', { colorSpace: 'display-p3' } as any);
      out.contextColorSpace = (ctx as any)?.getContextAttributes?.()?.colorSpace ?? 'unavailable';
    } catch (e) {
      out.contextColorSpace = `error: ${e}`;
    }
    try {
      const id = new ImageData(new Uint8ClampedArray(4), 1, 1, { colorSpace: 'display-p3' } as any);
      out.imageDataColorSpace = (id as any).colorSpace ?? 'no-colorSpace-property';
    } catch (e) {
      out.imageDataColorSpace = `error: ${e}`;
    }
    try {
      const oc = new OffscreenCanvas(1, 1);
      const octx = oc.getContext('2d', { colorSpace: 'display-p3' } as any);
      out.offscreenContextColorSpace = (octx as any)?.getContextAttributes?.()?.colorSpace ?? 'unavailable';
    } catch (e) {
      out.offscreenContextColorSpace = `error: ${e}`;
    }
    return out;
  });
  console.log(`[colorspace-probe] ${browserName} ${browser.version()}:`, JSON.stringify(result));
});
