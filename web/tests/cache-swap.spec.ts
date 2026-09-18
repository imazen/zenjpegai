// Cache-safety of the demo's asset URLs — regression test for the 2026-09-18 incident: a
// stream swapped in the `demo-assets-v1` release kept its file name (`<slug>_bpp<NN>.jai`),
// and browsers/Cache-API entries keyed on the URL kept serving the OLD bytes, so the live
// demo rendered the previous image. Now manifest.json carries each variant's sha256, demo.js
// fetches `streams/<file>?v=<sha256>` and manifest.json itself with `cache: 'no-cache'`, and
// worker.js keys the Cache API on `?v=` bundle digests.
//
// The test reproduces the incident on the real demo page: it overwrites one stream's bytes
// on disk UNDER THE SAME FILE NAME (the mountain stream's bytes land in `car_bpp25.jai`),
// updates that variant's digest in the served manifest, then reloads the page in the SAME
// browser context — nothing about storage is cleared; the HTTP cache and Cache API both
// persist across the reload. The card must render the new content, not the cached old one.
// It also asserts the mechanism: the stream request carries the new `?v=` digest and the
// model-bundle requests carry theirs (the Cache API would otherwise pin stale bundles).
//
// dist/site is mutated and restored in `finally`; it is a build artifact (build-site.mjs).
// JAI_BASE_PLAIN overrides the server URL — needed when another workspace's playwright run
// left servers on the default ports (reuseExistingServer): point it at your own
// `node scripts/serve.mjs dist/site plain <port>` instead.
import { test, expect } from '@playwright/test';
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const SITE = join(__dirname, '..', 'dist', 'site');
const ASSETS = join(__dirname, '..', '.demo-assets');
const BASE = process.env.JAI_BASE_PLAIN ?? `http://127.0.0.1:${PORTS.plain}`;

const CAR = 'car_bpp25.jai'; // first card in manifest order, decodes automatically
const DONOR = 'mountain_bpp25.jai'; // same 1155x866 canvas size, visibly different picture

// The placeholder canvas is pre-sized, so "pixels on screen" is only knowable through the
// card's data-state: hash only after the card reports done.
const cardDone = (slug: string) => `(() => {
  const c = [...document.querySelectorAll('.card')].find(
    (x) => x.querySelector('h3')?.textContent === '${slug}');
  return c?.dataset.state === 'done';
})()`;

// FNV-1a over a card's canvas pixels (-1 if the card/canvas isn't there). Draws through a
// scratch 2d canvas so it also works on a canvas whose control was transferred offscreen.
const CARD_HASH = `((slug) => {
  const card = [...document.querySelectorAll('.card')].find(
    (c) => c.querySelector('h3')?.textContent === slug);
  const canvas = card?.querySelector('canvas');
  if (!canvas || !canvas.width) return -1;
  const t = document.createElement('canvas');
  t.width = canvas.width; t.height = canvas.height;
  t.getContext('2d').drawImage(canvas, 0, 0);
  const d = t.getContext('2d').getImageData(0, 0, t.width, t.height).data;
  let h = 0x811c9dc5;
  for (let i = 0; i < d.length; i++) { h ^= d[i]; h = Math.imul(h, 0x01000193) >>> 0; }
  return h;
})`;
// `data-state=done` is set when the worker's decodeToCanvas resolves, but a transferred
// OffscreenCanvas only updates the element's displayed bitmap on the next compositor commit —
// hashing immediately can read the still-blank placeholder (its all-zero FNV-1a is 2645745317).
// A double rAF waits out one frame before reading.
const cardHash = (page, slug: string) => page.evaluate(`(async () => {
  await new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));
  return (${CARD_HASH})(${JSON.stringify(slug)});
})()`);

test('stream swapped under the same file name renders new content without clearing storage', async ({ page }) => {
  test.setTimeout(180_000);
  const streamRequests: string[] = [];
  const bundleRequests: string[] = [];
  const manifestRequests: string[] = [];
  page.on('request', (req) => {
    const u = req.url();
    if (u.includes('/streams/')) streamRequests.push(u);
    else if (u.includes('/models/')) bundleRequests.push(u);
    else if (u.endsWith('/manifest.json')) manifestRequests.push(u);
  });

  const manifestPath = join(SITE, 'manifest.json');
  const carPath = join(SITE, 'streams', CAR);
  const origManifest = readFileSync(manifestPath, 'utf8');
  const manifest = JSON.parse(origManifest);
  expect(manifest.images[0].slug).toBe('car');
  const carV0 = manifest.images[0].variants[0];
  const oldSha: string = carV0.sha256;
  expect(oldSha).toMatch(/^[0-9a-f]{64}$/);
  const donorSha = createHash('sha256').update(readFileSync(join(ASSETS, DONOR))).digest('hex');
  const donorBytes = readFileSync(join(ASSETS, DONOR)).byteLength;

  // The page must be served from THIS workspace's dist/site — a reused server on the default
  // port may belong to a sibling workspace's tree, which this test would corrupt/misread.
  const served = await page.request.get(`${BASE}/manifest.json`).then((r) => r.text());
  if (JSON.parse(served)?.images?.[0]?.variants?.[0]?.sha256 !== oldSha) {
    test.skip(true, `${BASE} serves a different manifest than dist/site — set JAI_BASE_PLAIN to a server rooted here`);
  }

  try {
    // --- visit 1: populate the HTTP cache and Cache API with the original assets ---
    await page.goto(`${BASE}/index.html`);
    await page.waitForFunction(cardDone('car'), { timeout: 90_000, polling: 100 });
    const hashBefore = await cardHash(page, 'car');
    expect(hashBefore).not.toBe(-1);

    // The stream fetch was content-addressed through the manifest digest, and the model
    // bundles carried theirs (the Cache API key is content-versioned).
    expect(streamRequests.some((u) => u.includes(`/${CAR}?v=${oldSha}`))).toBe(true);
    expect(bundleRequests.length).toBeGreaterThan(0);
    for (const u of bundleRequests) {
      const name = u.split('/models/')[1].split('?')[0];
      expect(u).toContain(`?v=${manifest.models[name]?.sha256}`);
    }

    // --- the incident: swap the bytes under the same file name, re-stamp the manifest ---
    copyFileSync(join(ASSETS, DONOR), carPath);
    carV0.sha256 = donorSha;
    carV0.bytes = donorBytes;
    writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

    // --- visit 2: reload, same context, nothing cleared ---
    await page.reload();
    await page.waitForFunction(cardDone('car'), { timeout: 90_000, polling: 100 });
    const hashAfter = await cardHash(page, 'car');

    // manifest.json was fetched again (cache: 'no-cache'), the stream went out under the
    // NEW digest — a URL nothing in this context's caches can answer with stale bytes.
    expect(manifestRequests.length).toBeGreaterThanOrEqual(2);
    expect(streamRequests.some((u) => u.includes(`/${CAR}?v=${donorSha}`))).toBe(true);
    expect(hashAfter).not.toBe(-1);
    expect(hashAfter).not.toBe(hashBefore);

    // Strongest check: the car card now shows exactly the donor picture (identical stream
    // bytes, deterministic decode). Force the mountain card's decode via its rate button —
    // it may sit below the viewport margin on short viewports.
    await page.locator('.card', { has: page.locator('h3', { hasText: 'mountain' }) }).locator('button').first().click();
    await page.waitForFunction(cardDone('mountain'), { timeout: 90_000, polling: 100 });
    expect(await cardHash(page, 'mountain')).toBe(hashAfter);
  } finally {
    copyFileSync(join(ASSETS, CAR), carPath);
    writeFileSync(manifestPath, origManifest);
  }
});
