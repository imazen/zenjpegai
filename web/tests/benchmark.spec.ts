// Decodes every demo-assets stream once per browser (isolated server: exercises the `threads`
// package; the earlier plain-server case in wasm-decode.spec.ts covers `simd`) and appends
// fetch/models/decode/render timings to benchmarks/wasm_decode_<date>.tsv, one row per
// (browser, stream). Committed per CLAUDE.md "Command Output" (>60s runs, comparative results).
import { test } from '@playwright/test';
import { readFileSync, existsSync, appendFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(__dirname, '..', '.demo-assets', 'manifest.json'), 'utf8'));
const date = new Date().toISOString().slice(0, 10);
const tsvPath = join(__dirname, '..', '..', 'benchmarks', `wasm_decode_${date}.tsv`);
const metaPath = tsvPath.replace(/\.tsv$/, '.meta');
const HEADER = ['browser', 'browser_version', 'os', 'slug', 'bpp', 'model_id', 'width', 'height', 'stream_bytes', 'variant', 'tier', 'fetch_ms', 'models_ms', 'decode_ms', 'total_ms'].join('\t');

test('decode every demo stream and record timings', async ({ page, browserName, browser }, testInfo) => {
  test.setTimeout(180_000);
  await page.goto(`http://127.0.0.1:${PORTS.isolated}/decode.html?gpu=off`);
  await page.evaluate(() => window.__ready);
  const version = browser.version();
  const os = process.platform;

  if (!existsSync(tsvPath)) writeFileSync(tsvPath, HEADER + '\n');
  if (!existsSync(metaPath)) {
    const commit = safeExec(`jj log -r @ --no-graph -T "commit_id.short()"`) || safeExec('git rev-parse --short HEAD') || 'unknown';
    writeFileSync(
      metaPath,
      `commit\t${commit}\nhost\t${safeExec('hostname') || 'unknown'}\ncommand\tnpx playwright test tests/benchmark.spec.ts\ndate\t${date}\n`,
    );
  }

  const rows: string[] = [];
  for (const img of manifest.images) {
    for (const v of img.variants) {
      // Same content-addressed URL the demo uses (`?v=<sha256>` keeps the fetch honest if a
      // stream file was swapped under its name); fall back to the name convention.
      const file = v.file || `${img.slug}_bpp${String(Math.round(v.bpp * 100)).padStart(2, '0')}.jai`;
      const stream = `streams/${file}${v.sha256 ? `?v=${v.sha256}` : ''}`;
      const r = await page.evaluate(
        async ({ stream }) => {
          const f0 = performance.now();
          const bytes = await fetch(stream).then((res) => res.arrayBuffer());
          const fetch_ms = performance.now() - f0;
          const streamBytes = bytes.byteLength; // pool.decode() transfers `bytes`, detaching it
          const out = await window.__pool.decode(bytes);
          return { width: out.width, height: out.height, timings: out.timings, bytes: streamBytes, fetch_ms };
        },
        { stream },
      );
      rows.push(
        [browserName, version, os, img.slug, v.bpp, v.modelId, r.width, r.height, r.bytes, r.timings.variant, r.timings.tier, r.fetch_ms.toFixed(2), r.timings.models.toFixed(2), r.timings.decode.toFixed(2), (r.fetch_ms + r.timings.total).toFixed(2)].join('\t'),
      );
    }
  }
  appendFileSync(tsvPath, rows.join('\n') + '\n');
  await testInfo.attach('wasm-decode-rows', { body: rows.join('\n'), contentType: 'text/tab-separated-values' });
});

function safeExec(cmd: string): string | null {
  try {
    return execSync(cmd, { cwd: join(__dirname, '..', '..'), stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return null;
  }
}
