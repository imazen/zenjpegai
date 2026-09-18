import { chromium } from '@playwright/test';
import { copyFileSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { join } from 'node:path';

const SITE = 'dist/site';
const ASSETS = '.demo-assets';
const BASE = 'http://127.0.0.1:3031';
const CAR = 'car_bpp25.jai';
const DONOR = 'mountain_bpp25.jai';

const manifestPath = join(SITE, 'manifest.json');
const carPath = join(SITE, 'streams', CAR);
const origManifest = readFileSync(manifestPath, 'utf8');
const manifest = JSON.parse(origManifest);
const donorSha = createHash('sha256').update(readFileSync(join(ASSETS, DONOR))).digest('hex');

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (m) => { if (m.type() === 'error') console.log('[page-err]', m.text()); });
const HASH = `((slug) => {
  const card = [...document.querySelectorAll('.card')].find((c) => c.querySelector('h3')?.textContent === slug);
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
const INFO = `((slug) => {
  const card = [...document.querySelectorAll('.card')].find((c) => c.querySelector('h3')?.textContent === slug);
  const canvas = card?.querySelector('canvas');
  return { state: card?.dataset.state, w: canvas?.width, h: canvas?.height,
    timing: card?.querySelector('.timing')?.textContent };
})`;
const done = (slug) => `(() => { const c = [...document.querySelectorAll('.card')].find((x) => x.querySelector('h3')?.textContent === '${slug}'); return c?.dataset.state === 'done'; })()`;

try {
  await page.goto(`${BASE}/index.html`);
  await page.waitForFunction(done('car'), { timeout: 60000 });
  console.log('v1 car', await page.evaluate(`(${HASH})("car")`), JSON.stringify(await page.evaluate(`(${INFO})("car")`)));

  copyFileSync(join(ASSETS, DONOR), carPath);
  manifest.images[0].variants[0].sha256 = donorSha;
  writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + '\n');

  await page.reload();
  await page.waitForFunction(done('car'), { timeout: 60000 });
  console.log('v2 car', await page.evaluate(`(${HASH})("car")`), JSON.stringify(await page.evaluate(`(${INFO})("car")`)));

  await page.locator('.card', { has: page.locator('h3', { hasText: 'mountain' }) }).locator('button').first().click();
  await page.waitForFunction(done('mountain'), { timeout: 60000 });
  console.log('v2 mountain', await page.evaluate(`(${HASH})("mountain")`), JSON.stringify(await page.evaluate(`(${INFO})("mountain")`)));
} finally {
  copyFileSync(join(ASSETS, CAR), carPath);
  writeFileSync(manifestPath, origManifest);
  await browser.close();
}
