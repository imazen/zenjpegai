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

// WebGPU: `DecoderPool` (default `gpu: 'auto'`) loads the pkg-webgpu build when this browser
// offers a non-software adapter; the note below is filled in once the pool reports ready —
// until then it only says what the browser advertises.
if ('gpu' in navigator) {
  navigator.gpu.requestAdapter().then((adapter) => {
    $gpuNote.textContent = adapter
      ? `WebGPU adapter: ${adapter.info?.description || adapter.info?.device || 'unnamed'} — checking it below…`
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

// `no-cache`: the manifest is the version authority — every asset URL below derives a `?v=`
// token from it, so it must always be revalidated, never served stale. (Without validators
// this is just an unconditional fetch; with ETag/Last-Modified it is a conditional one.)
const manifest = await fetch('manifest.json', { cache: 'no-cache' }).then((r) => r.json());
// Model-bundle digests -> the worker's Cache API keys (see worker.js): a bundle replaced
// under the same file name still gets a fresh cache entry.
const bundleVersions = {};
for (const [name, info] of Object.entries(manifest.models || {})) bundleVersions[name] = info.sha256;
// `?gpu=off|auto|software|force-software` overrides the pool's adapter policy (default auto).
const gpuMode = new URLSearchParams(location.search).get('gpu') || 'auto';
const pool = new DecoderPool({ modelsBaseUrl: 'models/', bundleVersions, gpu: gpuMode });
// Test/debug hooks: the Playwright scheduling spec reads these.
window.__pool = pool;
window.__poolStats = () => pool.stats();
const ready = await pool.ready();
const ok = ready.filter((r) => r.ok);
variantCell.textContent = ok[0]?.variant || 'failed';
tierCell.textContent = ok[0]?.tier || 'n/a';
workersCell.textContent = `${ok.length}/${ready.length}`;
const gpuReady = ok.find((r) => r.gpu && r.gpu.ok);
if (gpuReady) {
  $gpuNote.textContent = `WebGPU synthesis active (${gpuReady.gpu.adapter}, ${gpuReady.gpu.backend}${gpuReady.gpu.software ? ', software adapter' : ''}); GPU-presentable pictures draw without a CPU readback.`;
} else if (gpuMode !== 'off') {
  const why = ok.find((r) => r.gpuError)?.gpuError;
  if (why) $gpuNote.textContent = `CPU (wasm) decode: ${why}`;
}
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
  // Content-addressed URL: a stream file keeps its name across asset swaps, so the digest
  // from the manifest goes in the query — swapped content is a different URL and can never
  // be served from a stale HTTP-cache/Cache-API entry. The `variant.file`/`sha256` fields
  // are absent in manifests from before this scheme; fall back to the name convention.
  const file = variant.file || `${slug}_bpp${String(Math.round(variant.bpp * 100)).padStart(2, '0')}.jai`;
  const url = `streams/${file}${variant.sha256 ? `?v=${variant.sha256}` : ''}`;
  const t0 = performance.now();
  const bytes = await fetch(url).then((r) => r.arrayBuffer());
  const t1 = performance.now();
  try {
    // decodeToCanvas keeps presentation on the worker: 'gpu' means the RGBA texture was
    // blitted straight onto the canvas surface, '2d' means decoded pixels were putImageData'd.
    const r = await pool.decodeToCanvas(bytes, canvas, {
      priority,
      onDispatch: () => {
        timing.textContent = 'decoding…';
        if (card) card.dataset.state = 'decoding';
      },
    });
    // The canvas's control is transferred offscreen: the IDL width/height setters would throw,
    // so set the display size through the content attributes.
    canvas.setAttribute('width', String(r.width));
    canvas.setAttribute('height', String(r.height));
    if (card) card.dataset.state = 'done';
    const total = performance.now() - t0;
    const t = r.timings;
    const gpuNs = t.gpu && t.gpu.gpuNs != null ? ` · gpu ${(t.gpu.gpuNs / 1e6).toFixed(0)}ms` : '';
    const fell = t.gpuError ? ` · fallback: ${t.gpuError}` : '';
    timing.textContent = `fetch ${(t1 - t0).toFixed(0)}ms · queued ${(t.queued || 0).toFixed(0)}ms · models ${t.models.toFixed(0)}ms · decode ${t.decode.toFixed(0)}ms · total ${total.toFixed(0)}ms · ${t.variant}/${t.tier} · ${t.path}+${r.presented}${gpuNs}${fell}`;
  } catch (err) {
    if (card) card.dataset.state = 'error';
    timing.textContent = `error: ${err.message}`;
    console.error(err);
  }
}
