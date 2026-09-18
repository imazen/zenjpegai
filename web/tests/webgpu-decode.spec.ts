// WebGPU synthesis path coverage. `decode.html?gpu=<mode>` picks the pool's adapter policy:
//   auto            default — the webgpu package on a non-software adapter (measured faster
//                   than both CPU packages on RTX 2080/Dawn, web/README §7), CPU otherwise
//   on              GPU opt-in — a webgpu package on a non-software adapter, else the CPU packages
//   software        accept software adapters too (Dawn SwiftShader on GPU-less hosts)
//   off             never load pkg-webgpu
// `path` in each decode's timings says what actually ran ('gpu' = WebGPU synthesis, 'cpu' =
// the CPU engine after a GpuError fallback or because no usable adapter existed); the worker
// `ready` message's `gpu`/`gpuError` fields say why. Every decode is pixel-compared to the
// reference PNG with the same bound as wasm-decode.spec.ts (differing < 1/5000, max <= 1) —
// GPU output is not bit-identical to CPU but must stay inside it. GPU-path rows are logged
// and annotated separately so reports never conflate them with CPU numbers.
import { test, expect } from '@playwright/test';
import { PORTS } from '../playwright.config';

const STREAM = 'streams/line-plot_bpp25.jai';
const PNG = '_native/line-plot_bpp25.native.png';
const MAX_DIFFERING_RATIO = 1 / 5000;

async function setup(page, port: number, gpuMode: string) {
  await page.goto(`http://127.0.0.1:${port}/decode.html?gpu=${gpuMode}`);
  const ready = await page.evaluate(() => window.__ready);
  return ready.find((r) => r.ok) || ready[0];
}

async function decodeOnce(page, port: number, gpuMode: string) {
  const info = await setup(page, port, gpuMode);
  const r = await page.evaluate(
    ({ stream, png }) => window.__decodeAndCompare(stream, png),
    { stream: STREAM, png: PNG },
  );
  return { info, r };
}

function annotate(testInfo, label: string, value: unknown) {
  testInfo.annotations.push({ type: label, description: String(value) });
}

test('gpu=on: decode parity on whichever path the adapter allows', async ({ page }, testInfo) => {
  const { info, r } = await decodeOnce(page, PORTS.isolated, 'on');
  annotate(testInfo, 'decode-path', r.timings.path);
  annotate(testInfo, 'adapter', info.gpu ? `${info.gpu.adapter} (${info.gpu.backend})` : info.gpuError || 'none');
  // The adapter policy only applies to the WebGPU-enabled launch — the other projects
  // never get `--enable-unsafe-webgpu`/`navigator.gpu`, so `hardware` is a requirement
  // there and meaningless here (playwright.config.ts documents this scoping).
  if (process.env.WEBGPU_ADAPTER === 'hardware' && testInfo.project.name === 'chromium-webgpu') {
    // A hardware run must not silently downgrade to a fallback adapter or a CPU path.
    expect(
      info.gpu?.ok && info.gpu.software === false && info.adapterProbe?.isFallbackAdapter === false,
      `WEBGPU_ADAPTER=hardware but no hardware WebGPU adapter: ${JSON.stringify({ gpu: info.gpu, gpuError: info.gpuError, adapterProbe: info.adapterProbe })}`,
    ).toBeTruthy();
    expect(r.timings.path, `expected the GPU path, got cpu: ${r.timings.gpuError || ''}`).toBe('gpu');
  }
  expect(r.timings.path === 'gpu' || r.timings.gpuError || info.gpuError).toBeTruthy();
  expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
  expect(r.max).toBeLessThanOrEqual(1);
  console.log(
    `[on/${r.timings.path}] adapter=${info.gpu ? info.gpu.adapter : 'none'} probe=${JSON.stringify(info.adapterProbe)} ${r.differing}/${r.samples} differ (max ${r.max}), decode ${r.timings.decode.toFixed(1)}ms`,
  );
});

test('gpu=software: GPU code path through a software adapter (or reported CPU fallback)', async ({ page }, testInfo) => {
  const { info, r } = await decodeOnce(page, PORTS.isolated, 'software');
  annotate(testInfo, 'decode-path', r.timings.path);
  annotate(testInfo, 'adapter', info.gpu ? `${info.gpu.adapter} (${info.gpu.backend})` : info.gpuError || 'none');
  // 'gpu' here can be a software adapter — honest reporting: the path ran the GPU pipeline,
  // `software` marks that the adapter was not hardware.
  if (r.timings.path === 'gpu') {
    expect(info.gpu && info.gpu.ok).toBeTruthy();
    console.log(
      `[software/gpu${info.gpu.software ? ' (software adapter)' : ''}] ${r.differing}/${r.samples} differ (max ${r.max}), decode ${r.timings.decode.toFixed(1)}ms`,
    );
  } else {
    expect(info.gpuError || r.timings.gpuError).toBeTruthy();
    console.log(`[software/cpu-fallback] ${info.gpuError || r.timings.gpuError}`);
  }
  expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
  expect(r.max).toBeLessThanOrEqual(1);
});

test('gpu=off: never loads the webgpu package', async ({ page }) => {
  const { info, r } = await decodeOnce(page, PORTS.isolated, 'off');
  expect(info.variant).not.toBe('webgpu');
  expect(r.timings.path).toBe('cpu');
  expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
  expect(r.max).toBeLessThanOrEqual(1);
});

test('canvas presentation: GPU blit or 2d fallback, pixels verified', async ({ page }, testInfo) => {
  const info = await setup(page, PORTS.isolated, 'on');
  const r = await page.evaluate(
    ({ stream, png }) => window.__presentAndCompare(stream, png),
    { stream: STREAM, png: PNG },
  );
  annotate(testInfo, 'presented', r.presented);
  annotate(testInfo, 'decode-path', r.timings.path);
  annotate(testInfo, 'adapter', info.gpu ? `${info.gpu.adapter} (${info.gpu.backend})` : info.gpuError || 'none');
  console.log(`[present/${r.presented}/${r.timings.path}] decode ${r.timings.decode.toFixed(1)}ms`);
  if (r.png) {
    expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
    expect(r.max).toBeLessThanOrEqual(1);
  } else {
    // convertToBlob may legitimately refuse on a canvas whose frame already presented; a
    // 2d canvas must always verify, so absent PNG + presented '2d' is a failure.
    expect(r.presented).toBe('gpu');
  }
});

test('canvas presentation on the software GPU path exercises the WebGPU blit', async ({ page }, testInfo) => {
  const info = await setup(page, PORTS.isolated, 'software');
  const r = await page.evaluate(
    ({ stream, png }) => window.__presentAndCompare(stream, png),
    { stream: STREAM, png: PNG },
  );
  annotate(testInfo, 'presented', r.presented);
  annotate(testInfo, 'adapter', info.gpu ? `${info.gpu.adapter} (${info.gpu.backend})` : info.gpuError || 'none');
  console.log(`[present-software/${r.presented}/${r.timings.path}] decode ${r.timings.decode.toFixed(1)}ms`);
  if (r.timings.path === 'gpu') {
    // The stream is 8-bit 4:4:4 BT.709 with no post-filters: it must present on the GPU.
    expect(r.presented).toBe('gpu');
  }
  if (r.png) {
    expect(r.differing / r.samples).toBeLessThan(MAX_DIFFERING_RATIO);
    expect(r.max).toBeLessThanOrEqual(1);
  } else {
    expect(r.presented).toBe('gpu');
  }
});
