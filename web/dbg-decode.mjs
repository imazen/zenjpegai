import { chromium } from '@playwright/test';
const browser = await chromium.launch();
const page = await browser.newPage();
page.on('console', (m) => console.log('[page]', m.type(), m.text().slice(0, 200)));
page.on('pageerror', (e) => console.log('[pageerror]', String(e).slice(0, 300)));
await page.goto('http://127.0.0.1:3032/decode.html?gpu=off');
const ready = await page.evaluate(() => window.__ready);
console.log('ready:', JSON.stringify(ready));
const r = await page.evaluate(async () => {
  const bytes = await fetch('streams/line-plot_bpp25.jai').then((r) => r.arrayBuffer());
  const out = await window.__pool.decode(bytes);
  return { w: out.width, h: out.height, timings: out.timings };
});
console.log('decode:', JSON.stringify(r.timings));
await browser.close();
