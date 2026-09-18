import { DecoderPool } from './src/pool.js';

const pool = new DecoderPool({ modelsBaseUrl: 'models/' });
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

  const bitmap = await createImageBitmap(await fetch(pngUrl).then((r) => r.blob()));
  const c = new OffscreenCanvas(bitmap.width, bitmap.height);
  const ctx = c.getContext('2d');
  ctx.drawImage(bitmap, 0, 0);
  const ref = ctx.getImageData(0, 0, bitmap.width, bitmap.height).data;

  let differing = 0, samples = 0, max = 0;
  for (let i = 0; i < rgba.length; i++) {
    if (i % 4 === 3) continue; // alpha
    samples++;
    const d = Math.abs(rgba[i] - ref[i]);
    if (d > 0) { differing++; max = Math.max(max, d); }
  }
  return { width, height, differing, samples, max, timings, total_ms, refWidth: bitmap.width, refHeight: bitmap.height };
};
