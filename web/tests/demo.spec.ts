// Demo page timing UX: the page-level startup panel (absolute ms since navigation start),
// the per-card ttr breakdown (the image's own costs only), model-bundle provenance rows, and
// the reduced-quality (progressive maxChannels) toggle semantics.
//
// Arithmetic checked here: `ttr ≈ stream + wait + decode + present` (a few ms of postMessage
// overhead sits outside the split) and `wait == waitQueue + waitRuntime + waitModel`; a card
// that waited for a model download reports the same ms as that model's row in the panel.
import { test, expect } from '@playwright/test';
import { PORTS } from '../playwright.config';

const DEMO = `http://127.0.0.1:${PORTS.isolated}/index.html?gpu=off`;

async function firstDoneCard(page) {
  await page.waitForSelector('.card[data-state="done"] .timing', { timeout: 60_000 });
  await page.waitForFunction(() => [...document.querySelectorAll('.card')].some((c) => c.__timings), null, { timeout: 60_000 });
  return page.evaluate(() => {
    const card = [...document.querySelectorAll('.card')].find((c) => c.__timings);
    return {
      line: card.querySelector('.timing').textContent,
      facts: card.querySelector('.facts').textContent,
      badges: [...card.querySelectorAll('.badge')].map((b) => b.textContent),
      t: card.__timings,
    };
  });
}

test.describe('demo startup panel + per-card timings', () => {
  test('startup timeline shows milestones with navigation-relative ms', async ({ page }) => {
    await page.goto(DEMO);
    await page.waitForFunction(
      () => document.querySelector('#startup-totals')?.textContent.includes('runtime ready'),
      null, { timeout: 30_000 },
    );
    const rows = await page.$$eval('#startup-timeline div', (els) => els.map((e) => e.textContent));
    const text = rows.join('\n');
    expect(text).toContain('worker spawned');
    expect(text).toMatch(/wasm fetched: [\d.]+ MB in \d+ms · (network|cache)/);
    expect(text).toMatch(/wasm compiled \+ instantiated/);
    expect(text).toMatch(/thread pool ready \(\d+ threads/); // isolated server -> threads package
    expect(text).toContain('runtime ready');
    // Every row is prefixed with an absolute ms-since-navigation-start stamp.
    for (const r of rows) expect(r).toMatch(/^\d+ ms/);
  });

  test('model rows carry bytes, ms and network|cache-storage provenance', async ({ page }) => {
    await page.goto(DEMO);
    await page.waitForSelector('#startup-models .mrow', { timeout: 60_000 });
    const rows = await page.$$eval('#startup-models .mrow', (els) => els.map((e) => e.textContent));
    expect(rows.length).toBeGreaterThanOrEqual(1);
    for (const r of rows) {
      expect(r).toMatch(/^\d+ ms/);
      expect(r).toMatch(/m\d+ common\+\w+/);
      expect(r).toMatch(/[\d.]+ MB/);
      expect(r).toMatch(/\d+ms/);
      // 'prefetch' = bytes the page prefetched and transferred into the worker (the demo
      // prefetches at t≈0); 'network'/'cache-storage' when the worker fetched them itself.
      expect(r).toMatch(/network|cache-storage|cache|prefetch/);
    }
    const totals = await page.textContent('#startup-totals');
    expect(totals).toMatch(/runtime ready at \d+ ms/);
    expect(totals).toMatch(/first model ready at \d+ ms/);
  });

  test('per-card line: ttr adds up and wait splits into runtime/model/queue', async ({ page }) => {
    await page.goto(DEMO);
    const card = await firstDoneCard(page);
    const t = card.t;
    // Required schema fields exist.
    for (const f of ['ttr', 'stream', 'streamBytes', 'wait', 'waitQueue', 'waitRuntime', 'waitModel', 'decode', 'present', 'path', 'variant', 'tier']) {
      expect(t[f], `timings.${f}`).not.toBeUndefined();
    }
    expect(t.streamBytes).toBeGreaterThan(0);
    // wait is exactly its three components.
    expect(t.wait).toBeCloseTo(t.waitQueue + t.waitRuntime + t.waitModel, 5);
    // ttr ≈ stream + wait + decode + present (postMessage transfer sits outside the split).
    const split = t.stream + t.wait + t.decode + t.present;
    expect(Math.abs(t.ttr - split)).toBeLessThan(15);
    // First card waited for its model download: wait.model equals a model row's ms.
    if (t.waitModel > 0.5) {
      expect(t.modelRows.length).toBeGreaterThanOrEqual(1);
      const fetched = t.modelRows.reduce((a, r) => a + Math.max(r.ms, 0), 0);
      // Parallel fetch: total wait >= the slowest single bundle, bounded by their sum.
      expect(t.waitModel).toBeGreaterThanOrEqual(Math.max(...t.modelRows.map((r) => r.ms)) - 1);
      expect(t.waitModel).toBeLessThanOrEqual(fetched + 50);
      for (const r of t.modelRows) expect(r.source).toMatch(/network|cache|prefetch/);
    }
    // Card line renders the new labels.
    expect(card.line).toMatch(/ttr \d+ms/);
    expect(card.line).toMatch(/stream \d+ms \([\d.]+ KB\)/);
    expect(card.line).toMatch(/wait \d+ms/);
    expect(card.line).toMatch(/decode \d+ms \(cpu \d+ms \+ cpu \d+ms\)/); // gpu=off -> CPU path
    expect(card.line).toMatch(/present \d+ms/);
    expect(card.line).toContain('threads · cpu');
    // CPU-path stage split: cpu (entropy+latent) + cpuSynth (synthesis+output) ≈ decode.
    expect(t.cpu).toBeGreaterThan(0);
    expect(t.cpuSynth).toBeGreaterThan(0);
    expect(t.stats.total).toBeGreaterThan(0);
    // Image facts on the card: dims, model, operating point, chroma, bit depth, bpp.
    const factsMatch = card.facts.match(/(\d+)×(\d+) · (m\d+) (\w+) · (\d:\d:\d) · (\d+)-bit · ([\d.]+) KB → ([\d.]+) bpp/);
    expect(factsMatch, card.facts).toBeTruthy();
    expect(card.badges).toContain(factsMatch[3]); // the decoded rate's model badge
    // Panel totals include first-image-rendered once a card completes.
    const totals = await page.textContent('#startup-totals');
    expect(totals).toMatch(/first image rendered at \d+ ms \(ttr\)/);
    // The CPU/GPU split bar is visible and labelled.
    expect(await page.textContent('#splitbar')).toMatch(/cpu/);
  });

  test('second load reports cache-storage for model bundles', async ({ page }, testInfo) => {
    // `prefetch=off` keeps the page-side prefetch from handing the worker the bytes, so the
    // worker's own fetchBundle path is what the rows report — 'network' or 'cache' (the
    // build-time <link rel=preload> can serve it from the preload map/HTTP cache), never
    // 'prefetch'.
    await page.goto(`${DEMO}&prefetch=off`);
    await page.waitForSelector('#startup-models .mrow', { timeout: 60_000 });
    const first = await page.$$eval('#startup-models .mrow', (els) => els.map((e) => e.textContent).join(''));
    expect(first).toMatch(/network|cache/);
    expect(first).not.toContain('prefetch');
    // Same browser context: the Cache API namespace persists across the reload.
    await page.reload();
    await page.waitForSelector('#startup-models .mrow', { timeout: 60_000 });
    const second = await page.$$eval('#startup-models .mrow', (els) => els.map((e) => e.textContent).join(''));
    // WebKit's dedicated-worker Cache API does not retain entries across a reload (the
    // worker's cache.match misses and it re-fetches — 'network' is then the honest answer).
    if (testInfo.project.name === 'webkit') expect(second).toMatch(/cache-storage|network/);
    else expect(second).toContain('cache-storage');
  });

  test('reduced quality (maxChannels y64,uv32) decodes, differs, and skips residual work', async ({ page }) => {
    test.setTimeout(120_000);
    // The single-threaded simd package keeps stage timings least noisy.
    await page.goto(`http://127.0.0.1:${PORTS.plain}/index.html?gpu=off`);
    await page.waitForFunction(() => window.__pool, null, { timeout: 30_000 });
    const r = await page.evaluate(async () => {
      const bytes = await fetch('streams/car_bpp25.jai').then((x) => x.arrayBuffer());
      const med = (xs) => xs.slice().sort((a, b) => a - b)[Math.floor(xs.length / 2)];
      const fullDs = [], partDs = [], fullRes = [], partRes = [];
      let full, part;
      // Interleave so thermal/scheduling drift hits both arms equally. The cap truncates the
      // residual-symbol ANS decode (`num_decode_chs` — same as the reference); latent nets and
      // synthesis still run at full channel count, so the robust work-reduction signal is the
      // residual stage itself — the whole-decode delta is only a few % and inside noise.
      // `entropy_residual` = the residual ANS decode + scatter — the only stage the cap
      // truncates (measured ~1-3 ms on the demo streams at Date.now()'s 1 ms granularity).
      const residual = (t) => (t.stats.entropyResidual?.[0] ?? 0) + (t.stats.entropyResidual?.[1] ?? 0);
      for (let i = 0; i < 7; i++) {
        full = await window.__pool.decode(bytes, {});
        part = await window.__pool.decode(bytes, { maxChannels: [64, 32] });
        fullDs.push(full.timings.decode);
        partDs.push(part.timings.decode);
        fullRes.push(residual(full.timings));
        partRes.push(residual(part.timings));
      }
      let diff = 0;
      const a = full.rgba;
      const b = part.rgba;
      for (let i = 0; i < a.length; i += 13) if (a[i] !== b[i]) diff++;
      return {
        fullMs: med(fullDs), partMs: med(partDs),
        fullRes: med(fullRes), partRes: med(partRes),
        diff, samples: Math.ceil(a.length / 13),
      };
    });
    expect(r.diff).toBeGreaterThan(0); // visibly different picture
    // The skipped work is honestly small: `num_decode_chs` truncates only the residual ANS
    // stage (~1-3 ms of a ~1 s simd decode — the latent networks and synthesis still run
    // full-width, matching the reference). Assert the residual stage does not GROW rather
    // than a ratio the timer's 1 ms granularity can't support.
    expect(r.partRes).toBeLessThanOrEqual(r.fullRes + 1);
    // And the reduced decode is not slower overall (the saving is a few % — inside run-to-run
    // noise — so this is a "no slower" bound, not a flaky strict-less-than).
    expect(r.partMs).toBeLessThanOrEqual(r.fullMs * 1.05);
    console.log(`[reduced] residual ${r.fullRes.toFixed(0)}ms -> ${r.partRes.toFixed(0)}ms; decode median ${r.fullMs.toFixed(0)} -> ${r.partMs.toFixed(0)}ms (${r.diff}/${r.samples} samples differ)`);
  });
});
