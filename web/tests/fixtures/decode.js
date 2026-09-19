import { DecoderPool } from './src/pool.js';

// `?gpu=` selects the pool's adapter policy (auto|on|software|force-software|off); the tests
// use it to force the software-adapter GPU path on hosts without hardware WebGPU.
const gpuMode = new URLSearchParams(location.search).get('gpu') || 'auto';
// `?maxBytes=` gives the pool a byte-based in-flight cap (scheduling tests).
const maxBytes = Number(new URLSearchParams(location.search).get('maxBytes')) || undefined;
const pool = new DecoderPool({ modelsBaseUrl: 'models/', gpu: gpuMode, maxBytesInFlight: maxBytes });
window.__pool = pool;
window.__ready = pool.ready();

// In-page pixel-parity check against a browser-decodable reference PNG: avoids serialising a
// multi-megabyte RGBA array back to the Node test process, and needs no custom PNG decoder (the
// browser has one). Ignores alpha (the wasm decoder always emits 255).
window.__decodeAndCompare = async (streamUrl, pngUrl) => {
  const bytes = await fetch(streamUrl).then((r) => r.arrayBuffer());
  const t0 = performance.now();
  const { width, height, rgba, timings } = await pool.decode(bytes);
  const total_ms = performance.now() - t0;

  const ref = await pngPixels(pngUrl);
  const { differing, samples, max } = compare(rgba, ref);
  return { width, height, differing, samples, max, timings, total_ms, refWidth: ref.width, refHeight: ref.height };
};

// Same parity check through the no-readback presentation path: `decodeToCanvas` draws onto a
// transferred OffscreenCanvas; `verify` makes the worker post back a PNG of what it drew, which
// is compared against the reference PNG here. `presented` reports 'gpu' (blit) or '2d'.
window.__presentAndCompare = async (streamUrl, pngUrl) => {
  const bytes = await fetch(streamUrl).then((r) => r.arrayBuffer());
  const t0 = performance.now();
  const canvas = new OffscreenCanvas(2, 2);
  const r = await pool.decodeToCanvas(bytes, canvas, { verify: true });
  const total_ms = performance.now() - t0;
  if (!r.png) return { presented: r.presented, timings: r.timings, total_ms, png: false };
  const got = await pngPixels(new Uint8Array(r.png));
  const ref = await pngPixels(pngUrl);
  const { differing, samples, max } = compare(got.pixels, ref);
  return {
    presented: r.presented, timings: r.timings, total_ms, png: true,
    width: got.width, height: got.height, differing, samples, max,
    refWidth: ref.width, refHeight: ref.height,
  };
};

async function pngPixels(src) {
  const blob = src instanceof Uint8Array ? new Blob([src], { type: 'image/png' }) : await fetch(src).then((r) => r.blob());
  const bitmap = await createImageBitmap(blob);
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  return { pixels: ctx.getImageData(0, 0, bitmap.width, bitmap.height).data, width: bitmap.width, height: bitmap.height };
}

function compare(a, ref) {
  if (a.length !== ref.pixels.length) {
    throw new Error(`pixel count differs: ${a.length} decoded vs ${ref.pixels.length} reference`);
  }
  let differing = 0, samples = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    if (i % 4 === 3) continue; // alpha
    samples++;
    const d = Math.abs(a[i] - ref.pixels[i]);
    if (d > 0) { differing++; max = Math.max(max, d); }
  }
  return { differing, samples, max };
}
