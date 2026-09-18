// Rayon-pool thread scaling of the `threads` wasm package (isolated server): decodes every
// demo stream once per pool size in {1, 2, 4, 8, 16, navigator.hardwareConcurrency} and
// appends decode_ms to benchmarks/wasm_threads_<date>.tsv. Chromium only — the wasm bytes are
// identical across engines (see wasm_decode_* rows) and a shared box makes cross-engine
// scaling runs prohibitively long. A single-threaded `simd`-package row per stream (plain
// server) is included as the no-isolation baseline.
//
// Each pool size gets a fresh DecoderPool: initThreadPool can only size a worker's rayon pool
// once, so the pool is terminated and respawned per configuration. `threads` is plumbed
// through DecoderPool -> `?threads=N` on the worker URL -> initThreadPool(N).
import { test } from '@playwright/test';
import { readFileSync, existsSync, appendFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(__dirname, '..', '.demo-assets', 'manifest.json'), 'utf8'));
const date = new Date().toISOString().slice(0, 10);
const tsvPath = join(__dirname, '..', '..', 'benchmarks', `wasm_threads_${date}.tsv`);
const metaPath = tsvPath.replace(/\.tsv$/, '.meta');
const HEADER = ['browser', 'browser_version', 'os', 'variant', 'threads', 'slug', 'bpp', 'model_id', 'width', 'height', 'stream_bytes', 'decode_ms'].join('\t');

// One decode of each manifest stream on a fresh pool sized to `threads` rayon workers.
const DECODE_ALL = `async (threads) => {
  const { DecoderPool } = await import('/src/pool.js');
  const pool = new DecoderPool({ modelsBaseUrl: 'models/', threads });
  await pool.ready();
  const out = [];
  try {
    for (const img of ${JSON.stringify(manifest)}.images) {
      for (const v of img.variants) {
        const bpp2 = String(Math.round(v.bpp * 100)).padStart(2, '0');
        const bytes = await fetch(\`streams/\${img.slug}_bpp\${bpp2}.jai\`).then((r) => r.arrayBuffer());
        const r = await pool.decode(bytes);
        out.push({ slug: img.slug, bpp: v.bpp, modelId: v.modelId, width: r.width, height: r.height, bytes: bytes.byteLength, decode_ms: r.timings.decode });
      }
    }
  } finally {
    pool.terminate();
  }
  return out;
}`;

test('rayon thread scaling on the threads package', async ({ page, browserName, browser }, testInfo) => {
  test.skip(browserName !== 'chromium', 'thread-scaling numbers are chromium-only');
  test.setTimeout(600_000);
  await page.goto(`http://127.0.0.1:${PORTS.isolated}/decode.html`);
  await page.evaluate(() => window.__ready);
  const version = browser.version();
  const os = process.platform;
  const hwc = await page.evaluate(() => navigator.hardwareConcurrency || 4);
  const isolated = await page.evaluate(() => crossOriginIsolated);
  if (!isolated) throw new Error('isolated server is not cross-origin isolated');

  if (!existsSync(tsvPath)) writeFileSync(tsvPath, HEADER + '\n');
  if (!existsSync(metaPath)) {
    const commit = safeExec(`jj log -r @ --no-graph -T "commit_id.short()"`) || safeExec('git rev-parse --short HEAD') || 'unknown';
    writeFileSync(
      metaPath,
      `commit\t${commit}\nhost\t${safeExec('hostname') || 'unknown'}\ncommand\tnpx playwright test tests/threads.spec.ts\ndate\t${date}\nhardware_concurrency\t${hwc}\n`,
    );
  }

  const decodeAll = new Function(`return (${DECODE_ALL})`)() as never;

  const rows: string[] = [];
  const counts = [...new Set([1, 2, 4, 8, 16, hwc])].filter((n) => n <= 256).sort((a, b) => a - b);
  for (const threads of counts) {
    const results = (await page.evaluate(decodeAll, threads)) as Array<Record<string, number | string>>;
    for (const r of results) {
      rows.push(
        [browserName, version, os, 'threads', threads, r.slug, r.bpp, r.modelId, r.width, r.height, r.bytes, Number(r.decode_ms).toFixed(2)].join('\t'),
      );
    }
  }

  // Non-isolated baseline: the single-threaded `simd` package on the plain server.
  const plain = await browser.newPage();
  try {
    await plain.goto(`http://127.0.0.1:${PORTS.plain}/decode.html`);
    await plain.evaluate(() => window.__ready);
    const results = (await plain.evaluate(decodeAll, undefined)) as Array<Record<string, number | string>>;
    for (const r of results) {
      rows.push(
        [browserName, version, os, 'simd', 1, r.slug, r.bpp, r.modelId, r.width, r.height, r.bytes, Number(r.decode_ms).toFixed(2)].join('\t'),
      );
    }
  } finally {
    await plain.close();
  }

  appendFileSync(tsvPath, rows.join('\n') + '\n');
  await testInfo.attach('wasm-threads-rows', { body: rows.join('\n'), contentType: 'text/tab-separated-values' });
});

function safeExec(cmd: string): string | null {
  try {
    return execSync(cmd, { cwd: join(__dirname, '..', '..'), stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return null;
  }
}
