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
// Test/debug hooks: the Playwright scheduling spec reads these.
window.__pool = pool;
window.__poolStats = () => pool.stats();
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

// Decode scheduling: a card's first (lowest-rate) variant decodes only once the card is within
// one viewport height of the viewport, in the order the cards get there; while a decode waits
// in the pool's queue the card shows its final-sized placeholder (the checkerboard canvas is
// sized from the manifest up front, so the layout never shifts). A click on the other rate is
// user-initiated and jumps the queue.
const URGENT = -1_000_000_000;

const nearViewport = 'IntersectionObserver' in globalThis
  ? new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (!e.isIntersecting) continue;
        nearViewport.unobserve(e.target);
        e.target.__start?.();
      }
    }, { rootMargin: '100% 0px' })
  : null;

for (const img of manifest.images) {
  const card = document.createElement('div');
  card.className = 'card';
  card.dataset.state = 'pending';
  const canvas = document.createElement('canvas');
  // Placeholder: reserve the decoded picture's box before any decode starts.
  canvas.width = img.variants[0].width;
  canvas.height = img.variants[0].height;
  const h3 = document.createElement('h3');
  h3.textContent = img.slug;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = `${img.category} — ${img.license}`;
  const rates = document.createElement('div');
  rates.className = 'rates';
  const timing = document.createElement('div');
  timing.className = 'timing';
  timing.textContent = 'waiting for viewport…';

  const buttons = img.variants.map((v) => {
    const b = document.createElement('button');
    b.textContent = `${v.bpp} bpp (${(v.bytes / 1024).toFixed(1)} KB)`;
    b.addEventListener('click', () => decodeVariant(img.slug, v, canvas, timing, buttons, b, URGENT));
    rates.append(b);
    return b;
  });

  card.append(canvas, h3, meta, rates, timing);
  $gallery.append(card);

  // Decode the first (lowest-bpp) variant automatically once the card is near the viewport.
  let started = false;
  card.__start = () => {
    if (started) return;
    started = true;
    decodeVariant(img.slug, img.variants[0], canvas, timing, buttons, buttons[0], 0);
  };
  if (nearViewport) {
    nearViewport.observe(card);
    // The observer's first callback lands a rendering step late; cards already inside the
    // one-viewport-height margin start now instead of waiting for it.
    const vh = window.innerHeight || 1024;
    const r = card.getBoundingClientRect();
    if (r.bottom > -vh && r.top < 2 * vh) {
      nearViewport.unobserve(card);
      card.__start();
    }
  } else {
    card.__start();
  }
}

async function decodeVariant(slug, variant, canvas, timing, buttons, active, priority = 0) {
  const card = canvas.closest('.card');
  for (const b of buttons) b.setAttribute('aria-pressed', String(b === active));
  timing.textContent = 'queued…';
  if (card) card.dataset.state = 'queued';
  const bpp2 = String(Math.round(variant.bpp * 100)).padStart(2, '0');
  const t0 = performance.now();
  const bytes = await fetch(`streams/${slug}_bpp${bpp2}.jai`).then((r) => r.arrayBuffer());
  const t1 = performance.now();
  try {
    const { width, height, rgba, timings } = await pool.decode(bytes, {
      priority,
      onDispatch: () => {
        timing.textContent = 'decoding…';
        if (card) card.dataset.state = 'decoding';
      },
    });
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
    if (card) card.dataset.state = 'done';
    const total = performance.now() - t0;
    timing.textContent = `fetch ${(t1 - t0).toFixed(0)}ms · queued ${timings.queued.toFixed(0)}ms · models ${timings.models.toFixed(0)}ms · decode ${timings.decode.toFixed(0)}ms · total ${total.toFixed(0)}ms · ${timings.variant}/${timings.tier}`;
  } catch (err) {
    if (card) card.dataset.state = 'error';
    timing.textContent = `error: ${err.message}`;
    console.error(err);
  }
}
