// GPU-path benchmark: decodes every demo-assets stream through `decode` AND `decodeToCanvas`
// in the pool's current gpu mode (`?gpu=` — 'software' when WEBGPU_ADAPTER=swiftshader is used
// on a GPU-less host, 'auto' otherwise) and appends one row per (call, stream) to
// benchmarks/wasm_decode_<date>_gpu.tsv — a separate file from wasm_decode_<date>.tsv because
// GPU output is not bit-identical to CPU output and its timings must never be conflated.
// The `path`/`presented`/`adapter`/`software` columns say what actually ran; a row where
// path=cpu is a fallback, not GPU throughput. Only runs in the chromium-webgpu project.
import { test } from '@playwright/test';
import { readFileSync, existsSync, appendFileSync, writeFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PORTS } from '../playwright.config';

const __dirname = dirname(fileURLToPath(import.meta.url));
const manifest = JSON.parse(readFileSync(join(__dirname, '..', '.demo-assets', 'manifest.json'), 'utf8'));
const date = new Date().toISOString().slice(0, 10);
const tsvPath = join(__dirname, '..', '..', 'benchmarks', `wasm_decode_${date}_gpu.tsv`);
const metaPath = tsvPath.replace(/\.tsv$/, '.meta');
const HEADER = ['browser', 'browser_version', 'os', 'slug', 'bpp', 'model_id', 'width', 'height', 'stream_bytes', 'variant', 'tier', 'gpu_mode', 'call', 'path', 'presented', 'adapter', 'software_adapter', 'gpu_ns', 'gpu_error', 'fetch_ms', 'models_ms', 'decode_ms', 'total_ms'].join('\t');

// Restricted to the chromium-webgpu project in playwright.config.ts (testIgnore) rather than
// skipped at run time — the row columns only make sense from the WebGPU-enabled launch.
test('decode every demo stream on the GPU path and record timings', async ({ page, browserName, browser }, testInfo) => {
  test.setTimeout(300_000);
  const gpuMode = process.env.WEBGPU_ADAPTER ? 'software' : 'auto';
  await page.goto(`http://127.0.0.1:${PORTS.isolated}/decode.html?gpu=${gpuMode}`);
  const ready = await page.evaluate(() => window.__ready);
  const info = ready.find((r) => r.ok) || {};
  const adapter = (info.gpu && (info.gpu.adapter || info.gpu.backend)) || info.gpuError || 'none';
  const software = info.gpu ? String(!!info.gpu.software) : '';
  testInfo.annotations.push({ type: 'adapter', description: String(adapter) });
  testInfo.annotations.push({ type: 'gpu-mode', description: gpuMode });
  const version = browser.version();
  const os = process.platform;

  if (!existsSync(tsvPath)) writeFileSync(tsvPath, HEADER + '\n');
  if (!existsSync(metaPath)) {
    const commit = safeExec(`jj log -r @ --no-graph -T "commit_id.short()"`) || safeExec('git rev-parse --short HEAD') || 'unknown';
    writeFileSync(
      metaPath,
      `commit\t${commit}\nhost\t${safeExec('hostname') || 'unknown'}\ncommand\tWEBGPU_ADAPTER=${process.env.WEBGPU_ADAPTER || ''} npx playwright test tests/benchmark-gpu.spec.ts --project chromium-webgpu\nlaunch_args\t--enable-unsafe-webgpu --enable-features=Vulkan ${process.env.WEBGPU_ADAPTER ? `--use-webgpu-adapter=${process.env.WEBGPU_ADAPTER}` : ''}\nadapter\t${adapter}\ndate\t${date}\n`,
    );
  }

  const rows: string[] = [];
  for (const img of manifest.images) {
    for (const v of img.variants) {
      const bpp2 = String(Math.round(v.bpp * 100)).padStart(2, '0');
      const stream = `streams/${img.slug}_bpp${bpp2}.jai`;
      for (const call of ['decode', 'present']) {
        const r = await page.evaluate(
          async ({ stream, call }) => {
            const f0 = performance.now();
            const bytes = await fetch(stream).then((res) => res.arrayBuffer());
            const fetch_ms = performance.now() - f0;
            const streamBytes = bytes.byteLength; // the calls below transfer `bytes`
            let out;
            if (call === 'present') {
              const canvas = new OffscreenCanvas(2, 2);
              out = await window.__pool.decodeToCanvas(bytes, canvas);
            } else {
              out = await window.__pool.decode(bytes);
            }
            return { width: out.width, height: out.height, presented: out.presented || '', timings: out.timings, bytes: streamBytes, fetch_ms };
          },
          { stream, call },
        );
        const t = r.timings;
        const gpuError = String(t.gpuError || '').replace(/\s+/g, ' ').trim();
        rows.push(
          [browserName, version, os, img.slug, v.bpp, v.modelId, r.width, r.height, r.bytes, t.variant, t.tier, gpuMode, call, t.path, r.presented, adapter, software, t.gpu && t.gpu.gpuNs != null ? t.gpu.gpuNs : '', gpuError, r.fetch_ms.toFixed(2), t.models.toFixed(2), t.decode.toFixed(2), (r.fetch_ms + t.total).toFixed(2)].join('\t'),
        );
      }
    }
  }
  appendFileSync(tsvPath, rows.join('\n') + '\n');
  await testInfo.attach('wasm-decode-gpu-rows', { body: rows.join('\n'), contentType: 'text/tab-separated-values' });
});

function safeExec(cmd: string): string | null {
  try {
    return execSync(cmd, { cwd: join(__dirname, '..', '..'), stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return null;
  }
}
