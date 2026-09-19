import { DecoderPool } from './src/pool.js';
import { ensureCrossOriginIsolated } from './coi-loader.js';
import { prefetchModelPairs, takeModelBuffers, modelPairNames } from './src/prefetch.js';
import { installDemoViewer } from './viewer.js';

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

// WebGPU: `DecoderPool`'s default `gpu: 'auto'` uses the GPU when a hardware adapter exists —
// measured faster than the CPU packages on an RTX 2080 (web/README §7); `?gpu=off` forces CPU.
// The note below is filled in once the pool reports ready — until then it only says what the
// browser advertises.
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

// Prefetch, while the wasm module is still downloading, the `common`+`op` bundles of the
// model the FIRST card needs — then, once that pair is in, every other card's auto-decoded
// variant model in manifest (= scroll) order, and finally the click-only second-rate
// models — into the same Cache API keys the worker's `ensureModels` checks
// (`src/prefetch.js`). The staggering matters: browsers run ~6 connections per origin, so
// issuing all bundles at once would split the first card's ~16 MB across a third of the
// link instead of all of it (the build-time <link preload> already starts this pair at
// parse). This runs only on the post-reload (isolated) load on GitHub Pages: the coi reload
// already happened in `ensureCrossOriginIsolated` above, so nothing prefetches on the
// instance that is about to be thrown away.
const pairOf = (v) => ({ modelId: v.modelId, op: v.operatingPoint });
const firstVariant = manifest.images[0]?.variants?.[0];
const prefetched = prefetchModelPairs('models/', firstVariant ? [pairOf(firstVariant)] : [], bundleVersions);
Promise.allSettled([...prefetched.values()]).then(() => {
  prefetchModelPairs('models/', [
    ...manifest.images.slice(1).map((img) => img.variants[0]),
    ...manifest.images.flatMap((img) => img.variants.slice(1)),
  ].map(pairOf), bundleVersions)
    .forEach((p, name) => {
      if (!prefetched.has(name)) prefetched.set(name, p);
    });
});

// `?gpu=off|auto|on|software|force-software` overrides the pool's adapter policy (default auto).
const gpuMode = new URLSearchParams(location.search).get('gpu') || 'auto';
const pool = new DecoderPool({ modelsBaseUrl: 'models/', bundleVersions, gpu: gpuMode });
// Test/debug hooks: the Playwright specs read these.
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

// Content-addressed stream URL: a stream file keeps its name across asset swaps, so the
// digest from the manifest goes in the query — swapped content is a different URL and can
// never be served from a stale HTTP-cache/Cache-API entry. The `variant.file`/`sha256`
// fields are absent in manifests from before this scheme; fall back to the name convention.
function streamUrl(slug, variant) {
  const file = variant.file || `${slug}_bpp${String(Math.round(variant.bpp * 100)).padStart(2, '0')}.jai`;
  return `streams/${file}${variant.sha256 ? `?v=${variant.sha256}` : ''}`;
}

const viewerCards = [];
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

  // Click-to-open viewer wiring (see viewer.js): `activeVariant` reports the variant the
  // card canvas currently shows (set inside decodeVariant once the present lands) so the
  // viewer can capture it instead of re-decoding; `presentVariant` runs the same queue-jump
  // decode a rate-button click would, so its timings land on the card as usual.
  viewerCards.push({
    el: card,
    canvas,
    img,
    title: img.slug,
    activeVariant: () => canvas.__variantIdx ?? -1,
    presentVariant: (idx) => decodeVariant(img.slug, img.variants[idx], canvas, timing, buttons, buttons[idx], URGENT),
  });

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
  const url = streamUrl(slug, variant);
  const t0 = performance.now();
  const bytes = await fetch(url).then((r) => r.arrayBuffer());
  const t1 = performance.now();
  try {
    // Hand the worker this stream's prefetched model bundles by transfer: they were fetched
    // during init, so this skips even the Cache API read. `takeModelBuffers` consumes them —
    // each later decode of the same model hits the cache entry prefetch also wrote.
    const modelBuffers = variant.modelId != null && variant.operatingPoint
      ? await takeModelBuffers(prefetched, modelPairNames(variant.modelId, variant.operatingPoint))
      : null;
    // decodeToCanvas keeps presentation on the worker: 'gpu' means the RGBA texture was
    // blitted straight onto the canvas surface, '2d' means decoded pixels were putImageData'd.
    const r = await pool.decodeToCanvas(bytes, canvas, {
      priority,
      modelBuffers,
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
    // The canvas now shows this variant — the viewer's `activeVariant` reads it.
    canvas.__variantIdx = buttons.indexOf(active);
    const total = performance.now() - t0;
    const t = r.timings;
    const gpuNs = t.gpu && t.gpu.gpuNs != null ? ` · gpu ${(t.gpu.gpuNs / 1e6).toFixed(0)}ms` : '';
    const fell = t.gpuError ? ` · fallback: ${t.gpuError}` : '';
    timing.textContent = `fetch ${(t1 - t0).toFixed(0)}ms · queued ${(t.queued || 0).toFixed(0)}ms · models ${t.models.toFixed(0)}ms · decode ${t.decode.toFixed(0)}ms · total ${total.toFixed(0)}ms · ${t.variant}/${t.tier} · ${t.path}+${r.presented}${gpuNs}${fell}`;
    return r;
  } catch (err) {
    if (card) card.dataset.state = 'error';
    timing.textContent = `error: ${err.message}`;
    console.error(err);
    return null;
  }
}

installDemoViewer({
  cards: viewerCards,
  // Fallback for browsers that can't sample a canvas whose control was transferred
  // offscreen: one re-decode through the pool, feeding it this variant's prefetched model
  // buffers exactly like the card's decode path does.
  decode: async (img, variant, bytes, opts) =>
    pool.decode(bytes, {
      ...opts,
      modelBuffers:
        variant.modelId != null && variant.operatingPoint
          ? await takeModelBuffers(prefetched, modelPairNames(variant.modelId, variant.operatingPoint))
          : null,
    }),
  fetchStream: (img, variant) => fetch(streamUrl(img.slug, variant)).then((r) => r.arrayBuffer()),
  // Reference-decoder PNGs ship as _native/<stem>.native.png (see build-site.mjs).
  nativeUrl: (img, variant) =>
    `_native/${(variant.file || `${img.slug}_bpp${String(Math.round(variant.bpp * 100)).padStart(2, '0')}.jai`).replace(/\.jai$/, '.native.png')}`,
});
