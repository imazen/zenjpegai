// GPU-path benchmark: decodes every demo-assets stream through `decode` AND `decodeToCanvas`
// in the pool's current gpu mode (`?gpu=` — 'software' when WEBGPU_ADAPTER=swiftshader is used
// on a GPU-less host, 'auto' otherwise) and appends one row per (call, stream) to
// benchmarks/wasm_decode_<date>_gpu.tsv — a separate file from wasm_decode_<date>.tsv because
// GPU output is not bit-identical to CPU output and its timings must never be conflated.
// The `path`/`presented`/`adapter`/`software` columns say what actually ran; a row where
// path=cpu is a fallback, not GPU throughput. Only runs in the chromium-webgpu project.
import { test, expect } from '@playwright/test';
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
  const wantAdapter = process.env.WEBGPU_ADAPTER || ''; // 'hardware' | 'swiftshader' | ''
  // 'on' opts the worker into the hardware adapter (auto is the CPU default — see worker.js);
  // on a GPU-less host 'on' still produces honest rows with path=cpu.
  const gpuMode = wantAdapter === 'swiftshader' ? 'software' : 'on';
  await page.goto(`http://127.0.0.1:${PORTS.isolated}/decode.html?gpu=${gpuMode}`);
  const ready = await page.evaluate(() => window.__ready);
  const info = ready.find((r) => r.ok) || {};
  // A hardware run must not silently downgrade: fail, don't record a software row as hardware.
  if (wantAdapter === 'hardware') {
    expect(
      info.gpu?.ok && info.gpu.software === false && info.adapterProbe?.isFallbackAdapter === false,
      `WEBGPU_ADAPTER=hardware but no hardware WebGPU adapter: ${JSON.stringify({ gpu: info.gpu, gpuError: info.gpuError, adapterProbe: info.adapterProbe })}`,
    ).toBeTruthy();
  }
  const probe = info.adapterProbe;
  const adapter =
    (probe && [probe.description, probe.vendor, probe.architecture].filter(Boolean).join('/')) ||
    (info.gpu && (info.gpu.adapter || info.gpu.backend)) ||
    info.gpuError ||
    'none';
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
      `commit\t${commit}\nhost\t${safeExec('hostname') || 'unknown'}\ncommand\tWEBGPU_ADAPTER=${process.env.WEBGPU_ADAPTER || ''} npx playwright test tests/benchmark-gpu.spec.ts --project chromium-webgpu\nlaunch_args\t--enable-unsafe-webgpu --enable-features=Vulkan --ignore-gpu-blocklist ${wantAdapter === 'hardware' ? '--use-angle=vulkan' : wantAdapter === 'swiftshader' ? '--use-webgpu-adapter=swiftshader' : ''}\nadapter\t${adapter}\nadapter_fallback\t${probe ? String(probe.isFallbackAdapter) : ''}\ndate\t${date}\n`,
    );
  }

  const rows: string[] = [];
  for (const img of manifest.images) {
    for (const v of img.variants) {
      const stream = streamUrl(img, v);
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

  // CPU comparison rows in the same file: `?gpu=off` on the isolated server loads the
  // `threads` package and on the plain server the `simd` package, so the GPU-vs-CPU crossover
  // for `auto` mode is decided from one artifact rather than two files with different schemas.
  for (const [port, cpuTag] of [[PORTS.isolated, 'threads'], [PORTS.plain, 'simd']]) {
    await page.goto(`http://127.0.0.1:${port}/decode.html?gpu=off`);
    await page.evaluate(() => window.__ready);
    for (const img of manifest.images) {
      for (const v of img.variants) {
        const stream = streamUrl(img, v);
        const r = await page.evaluate(async (stream) => {
          const f0 = performance.now();
          const bytes = await fetch(stream).then((res) => res.arrayBuffer());
          const fetch_ms = performance.now() - f0;
          const out = await window.__pool.decode(bytes);
          return { width: out.width, height: out.height, timings: out.timings, bytes: bytes.byteLength, fetch_ms };
        }, stream);
        const t = r.timings;
        rows.push(
          [browserName, version, os, img.slug, v.bpp, v.modelId, r.width, r.height, r.bytes, t.variant, t.tier, `off (${cpuTag})`, 'decode', t.path, '', 'cpu', '', '', '', r.fetch_ms.toFixed(2), t.models.toFixed(2), t.decode.toFixed(2), (r.fetch_ms + t.total).toFixed(2)].join('\t'),
        );
      }
    }
  }
  appendFileSync(tsvPath, rows.join('\n') + '\n');
  await testInfo.attach('wasm-decode-gpu-rows', { body: rows.join('\n'), contentType: 'text/tab-separated-values' });
});

// Same content-addressed URL the demo and benchmark.spec.ts build: `?v=<sha256>` keeps the
// fetch honest if a stream file was swapped under its name; the name convention is the
// fallback for manifests from before the `file`/`sha256` fields.
function streamUrl(img, v) {
  const file = v.file || `${img.slug}_bpp${String(Math.round(v.bpp * 100)).padStart(2, '0')}.jai`;
  return `streams/${file}${v.sha256 ? `?v=${v.sha256}` : ''}`;
}

function safeExec(cmd: string): string | null {
  try {
    return execSync(cmd, { cwd: join(__dirname, '..', '..'), stdio: ['ignore', 'pipe', 'ignore'] }).toString().trim();
  } catch {
    return null;
  }
}
