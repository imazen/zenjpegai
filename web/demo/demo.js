import { DecoderPool } from './src/pool.js';
import { ensureCrossOriginIsolated } from './coi-loader.js';

// Best-effort: on a host that can't send COOP/COEP (GitHub Pages), this registers a service
// worker that adds them and reloads once. If it can't (or the reload is already in flight),
// DecoderPool below re-checks crossOriginIsolated itself and falls back to the simd package —
// this call is a speed optimisation, never a correctness dependency.
await ensureCrossOriginIsolated();

const $status = document.getElementById('status');
const $gallery = document.getElementById('gallery');
const $gpuNote = document.getElementById('gpu-note');
const $noticeBox = document.getElementById('license-notice');
const $noticeText = document.getElementById('license-text');

function statusRow(label, value) {
  const dt = document.createElement('dt');
  dt.textContent = label;
  const dd = document.createElement('dd');
  dd.style.margin = '0';
  dd.textContent = value;
  const wrap = document.createElement('div');
  wrap.append(dt, dd);
  $status.append(wrap);
  return dd;
}

const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
statusRow('cross-origin isolated', String(isolated));
const variantCell = statusRow('build', '…');
const tierCell = statusRow('SIMD tier', '…');
const workersCell = statusRow('workers', '…');

// WebGPU: feature-detected only. `gpu/` (zenjpegai-gpu, wgpu-based synthesis backend) has no
// README describing a usable async API as of this build and is not wired into the wasm/web
// target — see PORTING.md "Work queue" item 2. This just reports what the browser offers.
if ('gpu' in navigator) {
  navigator.gpu.requestAdapter().then((adapter) => {
    $gpuNote.textContent = adapter
      ? `WebGPU is available in this browser (adapter: ${adapter.info?.description || 'unnamed'}), but zenjpegai has no WebGPU decode path yet — decoding above runs on the CPU (wasm) path. See PORTING.md.`
      : 'navigator.gpu exists but no adapter was returned; CPU (wasm) decode only.';
  }, () => { $gpuNote.textContent = 'WebGPU requestAdapter failed; CPU (wasm) decode only.'; });
} else {
  $gpuNote.textContent = 'This browser has no WebGPU (navigator.gpu); CPU (wasm) decode only.';
}

fetch('upstream-notices/LICENSE').then((r) => r.ok ? r.text() : null).then((text) => {
  if (!text) return;
  $noticeText.textContent = 'JPEG AI reference software model weights (BSD): ' + text.split('\n').slice(0, 4).join(' ');
  $noticeBox.hidden = false;
});

const manifest = await fetch('manifest.json').then((r) => r.json());
const pool = new DecoderPool({ modelsBaseUrl: 'models/' });
const ready = await pool.ready();
const ok = ready.filter((r) => r.ok);
variantCell.textContent = ok[0]?.variant || 'failed';
tierCell.textContent = ok[0]?.tier || 'n/a';
workersCell.textContent = `${ok.length}/${ready.length}`;
if (!ok.length) {
  const warn = document.createElement('div');
  warn.style.color = '#c33';
  warn.textContent = 'no worker loaded wasm — see console';
  $status.append(warn);
}

for (const img of manifest.images) {
  const card = document.createElement('div');
  card.className = 'card';
  const canvas = document.createElement('canvas');
  const h3 = document.createElement('h3');
  h3.textContent = img.slug;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${img.category} — ${img.license}`;
  const rates = document.createElement('div');
  rates.className = 'rates';
  const timing = document.createElement('div');
  timing.className = 'timing';

  const buttons = img.variants.map((v) => {
    const b = document.createElement('button');
    b.textContent = `${v.bpp} bpp (${(v.bytes / 1024).toFixed(1)} KB)`;
    b.addEventListener('click', () => decodeVariant(img.slug, v, canvas, timing, buttons, b));
    rates.append(b);
    return b;
  });

  card.append(canvas, h3, meta, rates, timing);
  $gallery.append(card);

  // Decode the first (lowest-bpp) variant automatically so the gallery isn't empty on load.
  decodeVariant(img.slug, img.variants[0], canvas, timing, buttons, buttons[0]);
}

async function decodeVariant(slug, variant, canvas, timing, buttons, active) {
  for (const b of buttons) b.setAttribute('aria-pressed', String(b === active));
  timing.textContent = 'fetching…';
  const bpp2 = String(Math.round(variant.bpp * 100)).padStart(2, '0');
  const t0 = performance.now();
  const bytes = await fetch(`streams/${slug}_bpp${bpp2}.jai`).then((r) => r.arrayBuffer());
  const t1 = performance.now();
  timing.textContent = 'decoding…';
  try {
    const { width, height, rgba, timings } = await pool.decode(bytes);
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
    const total = performance.now() - t0;
    timing.textContent = `fetch ${(t1 - t0).toFixed(0)}ms · models ${timings.models.toFixed(0)}ms · decode ${timings.decode.toFixed(0)}ms · total ${total.toFixed(0)}ms · ${timings.variant}/${timings.tier}`;
  } catch (err) {
    timing.textContent = `error: ${err.message}`;
    console.error(err);
  }
}
