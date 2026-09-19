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
const $factsLine = document.getElementById('startup-facts');
const $timeline = document.getElementById('startup-timeline');
const $models = document.getElementById('startup-models');
const $totals = document.getElementById('startup-totals');
const $splitbar = document.getElementById('splitbar');
const $ctlNote = document.getElementById('ctl-note');

const ms = (v) => `${v.toFixed(0)}ms`;
const kb = (v) => `${(v / 1024).toFixed(1)} KB`;
const mb = (v) => `${(v / 1048576).toFixed(1)} MB`;

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

// --- User controls --------------------------------------------------------------
// gpu/threads persist in the URL (a shared link reproduces the mode); quality is a live
// toggle — it only changes what the next decode asks for (full vs progressive y64,uv32).
const params = new URLSearchParams(location.search);
const gpuMode = params.get('gpu') || 'auto';
const $ctlGpu = document.getElementById('ctl-gpu');
const $ctlThreads = document.getElementById('ctl-threads');
const $ctlQuality = document.getElementById('ctl-quality');
$ctlGpu.value = ['auto', 'on', 'off'].includes(gpuMode) ? gpuMode : 'auto';
// `?threads=` semantics on the worker side: any value overrides the rayon pool size; the demo
// treats "off" as 1 (a single-threaded pool ≈ off) and absence as the default pool.
const threadsOff = params.get('threads') === '1' || params.get('threads') === 'off';
$ctlThreads.value = threadsOff ? 'off' : 'on';
$ctlGpu.addEventListener('change', () => {
  params.set('gpu', $ctlGpu.value);
  location.search = params.toString();
});
$ctlThreads.addEventListener('change', () => {
  if ($ctlThreads.value === 'off') params.set('threads', '1');
  else params.delete('threads');
  location.search = params.toString();
});
document.getElementById('ctl-clearcache').addEventListener('click', async (ev) => {
  const n = await caches.delete('zenjpegai-models-v2').catch(() => false);
  ev.target.textContent = n ? 'model cache cleared' : 'no model cache';
  setTimeout(() => { ev.target.textContent = 'clear model cache'; }, 2000);
});
let quality = 'full';
$ctlQuality.addEventListener('change', () => { quality = $ctlQuality.value; });

// --- Status + startup panel ------------------------------------------------------
const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
statusRow('cross-origin isolated', String(isolated));
const variantCell = statusRow('build', '…');
const tierCell = statusRow('SIMD tier', '…');
const workersCell = statusRow('workers', '…');

// Package-choice matrix, one line — which pkg-* each capability combination selects.
$factsLine.textContent =
  'package: isolated+WebGPU → webgpu-threads · isolated → threads · WebGPU only → webgpu · ' +
  'otherwise → simd · no wasm SIMD → no package (page fallback stays)';

const MILESTONE_LABEL = {
  'worker-spawned': () => 'worker spawned',
  'worker-start': () => 'worker script running',
  glue: (e) => `worker JS loaded (${ms(e.ms)})`,
  wasm: (e) => `wasm fetched: ${mb(e.bytes)} in ${ms(e.ms)} · ${e.source}`,
  instantiate: (e) => `wasm compiled + instantiated (${ms(e.ms)})`,
  threads: (e) => `thread pool ready (${e.count} threads, ${ms(e.ms)})`,
  'gpu-probe': (e) => `GPU adapter probed (${ms(e.ms)})`,
  'gpu-init': (e) => (e.ok ? `GPU device ready (${ms(e.ms)})` : `GPU init failed (${ms(e.ms)})`),
  ready: (e) => `runtime ready · pkg-${e.variant}${e.tier ? ` · ${e.tier}` : ''}`,
};

// Live startup timeline: one row per milestone, absolute ms since navigation start.
const seen = new Set();
function timelineRow(e) {
  const label = MILESTONE_LABEL[e.name]?.(e);
  if (!label) return;
  // One 'ready' row per worker is enough (the demo pool has 1 on isolated pages; the simd
  // pool's workers are identical) — same for the repetitive per-worker startup events.
  if (e.name !== 'model' && seen.has(e.name)) return;
  seen.add(e.name);
  const row = document.createElement('div');
  const dt = document.createElement('dt');
  dt.textContent = `${e.atMs.toFixed(0)} ms`;
  const dd = document.createElement('dd');
  dd.textContent = label;
  row.append(dt, dd);
  $timeline.append(row);
}

// Model rows: "m<N> common+bop  16.2 MB  412 ms  network|cache-storage" — one per bundle the
// worker actually loaded (a Cache API hit reports ~0 ms, which is the honest answer to
// "why don't I see the download"). A card that waited for that bundle reports the same ms as
// its `wait: model` part.
function modelRow(e) {
  const row = document.createElement('div');
  row.className = 'mrow';
  const when = document.createElement('span');
  when.textContent = `${e.atMs.toFixed(0)} ms`;
  const name = document.createElement('span');
  name.textContent = e.file;
  const size = document.createElement('span');
  size.textContent = mb(e.bytes);
  const dur = document.createElement('span');
  dur.textContent = ms(e.ms);
  const src = document.createElement('span');
  src.className = 'msrc';
  src.textContent = e.source;
  row.append(when, name, size, dur, src);
  $models.append(row);
}

const totals = { readyAt: null, firstModelAt: null, firstImageAt: null };
function renderTotals() {
  const parts = [];
  if (totals.readyAt != null) parts.push(`runtime ready at ${totals.readyAt.toFixed(0)} ms`);
  if (totals.firstModelAt != null) parts.push(`first model ready at ${totals.firstModelAt.toFixed(0)} ms`);
  if (totals.firstImageAt != null) parts.push(`first image rendered at ${totals.firstImageAt.toFixed(0)} ms (ttr)`);
  $totals.textContent = parts.join(' · ');
}

function onPoolEvent(e) {
  if (e.name === 'model') {
    modelRow(e);
    if (totals.firstModelAt == null) {
      totals.firstModelAt = e.atMs + e.ms;
      renderTotals();
    }
    return;
  }
  timelineRow(e);
  if (e.name === 'ready' && totals.readyAt == null) {
    totals.readyAt = e.atMs;
    renderTotals();
  }
}

// CPU/GPU/transfer split bar for the last completed decode.
function renderSplit(t) {
  let cpu;
  let gpu;
  let xfer;
  if (t.gpuMs != null || (t.gpu && t.gpu.gpuNs != null)) {
    cpu = t.cpu || 0;
    gpu = t.gpuMs || 0;
    xfer = t.xfer || 0;
  } else if (t.cpu != null) {
    cpu = t.cpu;
    gpu = 0;
    xfer = t.cpuSynth || 0; // CPU synthesis+output — the "would-be-GPU" part
  } else {
    return;
  }
  const total = cpu + gpu + xfer || 1;
  const segs = $splitbar.querySelectorAll('.seg');
  segs[0].style.width = `${(cpu / total) * 100}%`;
  segs[1].style.width = `${(gpu / total) * 100}%`;
  segs[2].style.width = `${(xfer / total) * 100}%`;
  const label = t.gpuMs != null
    ? `cpu ${ms(cpu)} · gpu ${ms(gpu)} · xfer ${ms(xfer)}`
    : `cpu(entropy+latent) ${ms(cpu)} · cpu(synthesis+output) ${ms(xfer)}`;
  $splitbar.querySelector('.seglabel').textContent = label;
  $splitbar.hidden = false;
}

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
// `?prefetch=off` disables the page-side model prefetch entirely: the worker's `ensureModels`
// then fetches each bundle itself — the way to observe a cold model download in the panel
// (first load 'network', reload 'cache-storage') instead of the 'prefetch' provenance the
// transferred bytes report.
const prefetchOff = params.get('prefetch') === 'off';
const pairOf = (v) => ({ modelId: v.modelId, op: v.operatingPoint });
const firstVariant = manifest.images[0]?.variants?.[0];
const prefetched = prefetchOff
  ? new Map()
  : prefetchModelPairs('models/', firstVariant ? [pairOf(firstVariant)] : [], bundleVersions);
if (!prefetchOff) Promise.allSettled([...prefetched.values()]).then(() => {
  prefetchModelPairs('models/', [
    ...manifest.images.slice(1).map((img) => img.variants[0]),
    ...manifest.images.flatMap((img) => img.variants.slice(1)),
  ].map(pairOf), bundleVersions)
    .forEach((p, name) => {
      if (!prefetched.has(name)) prefetched.set(name, p);
    });
});

// `gpuMode` comes from `?gpu=` (see the controls block above).
const pool = new DecoderPool({ modelsBaseUrl: 'models/', bundleVersions, gpu: gpuMode, onEvent: onPoolEvent });
// Test/debug hooks: the Playwright specs read these.
window.__pool = pool;
window.__poolStats = () => pool.stats();
const ready = await pool.ready();
const ok = ready.filter((r) => r.ok);
const variant = ok[0]?.variant || 'failed';
variantCell.textContent = `pkg-${variant}`;
tierCell.textContent = ok[0]?.tier || 'n/a';
workersCell.textContent = `${ok.length}/${ready.length}`;
// Why this package: the one line that lets a visitor map their browser onto the matrix above.
{
  let why;
  if (!ok.length) {
    why = ready[0]?.reason === 'no-wasm-simd' ? 'no WebAssembly SIMD128 in this browser' : 'failed to load';
  } else if (variant.startsWith('webgpu')) {
    why = `gpu=${gpuMode}, WebGPU adapter present${isolated ? ' + cross-origin isolated' : ''}`;
  } else if (variant === 'threads') {
    const g = ok.find((r) => r.gpuError)?.gpuError;
    why = gpuMode === 'off' ? 'gpu=off' : `${g || 'no hardware WebGPU adapter'} + cross-origin isolated`;
  } else {
    const g = ok.find((r) => r.gpuError)?.gpuError;
    why = gpuMode === 'off' ? 'gpu=off + not cross-origin isolated' : `${g ? `${g} + ` : ''}not cross-origin isolated`;
  }
  variantCell.textContent = `pkg-${variant} because ${why}`;
}
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
if (!isolated) $ctlNote.textContent = 'threads need a cross-origin isolated page';
else if (threadsOff) $ctlNote.textContent = 'threads=1 via ?threads=1';

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
  const facts = document.createElement('div');
  facts.className = 'facts';
  const rates = document.createElement('div');
  rates.className = 'rates';
  const timing = document.createElement('div');
  timing.className = 'timing';
  timing.textContent = 'waiting for viewport…';

  const buttons = img.variants.map((v) => {
    const b = document.createElement('button');
    b.textContent = `${v.bpp} bpp (${kb(v.bytes)})`;
    // Which model this rate decodes with — the manifest's modelId, shown next to the button.
    const badge = document.createElement('span');
    badge.className = 'badge';
    badge.textContent = `m${v.modelId}`;
    b.append(badge);
    b.addEventListener('click', () => decodeVariant(img.slug, v, canvas, timing, facts, buttons, b, URGENT));
    rates.append(b);
    return b;
  });
  const v0 = img.variants[0];
  facts.textContent = `${v0.width}×${v0.height} · m${v0.modelId} ${v0.operatingPoint} · ${kb(v0.bytes)} → ${v0.bpp} bpp`;

  card.append(canvas, h3, meta, facts, rates, timing);
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
    presentVariant: (idx) => decodeVariant(img.slug, img.variants[idx], canvas, timing, facts, buttons, buttons[idx], URGENT),
  });

  // Decode the first (lowest-bpp) variant automatically once the card is near the viewport.
  let started = false;
  card.__start = () => {
    if (started) return;
    started = true;
    decodeVariant(img.slug, img.variants[0], canvas, timing, facts, buttons, buttons[0], 0);
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

// A hoverable per-stage breakdown for the `decode` figure (title attribute): the wasm
// DecodeStats on the CPU path, the GPU Timing fields on the WebGPU path.
function decodeTitle(t) {
  if (t.stats) {
    const s = t.stats;
    const sum = (a) => (a ? a[0] + a[1] : 0);
    return [
      `entropy ${ms(s.headers + s.models + sum(s.entropyZ) + s.entropyQmap + sum(s.entropyBody))}`,
      `latent ${ms(sum(s.latent) + sum(s.latentHyper) + sum(s.latentMcm) + sum(s.lsbs))}`,
      `synthesis ${ms(s.synthLuma + s.synthesis + s.chroma)}`,
      `output ${ms(s.filters + s.output)}`,
      `total ${ms(s.total)}`,
    ].join(' · ');
  }
  if (t.gpu) {
    const g = t.gpu;
    return [
      `entropy ${ms(g.entropyMs)}`,
      `latent ${ms(g.latentMs)}`,
      `gpu ${t.gpuMs != null ? ms(t.gpuMs) : 'n/a'}`,
      `xfer ${ms(t.xfer || 0)}`,
    ].join(' · ');
  }
  return '';
}

async function decodeVariant(slug, variant, canvas, timing, facts, buttons, active, priority = 0) {
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
      streamMs: t1 - t0,
      maxChannels: quality === 'reduced' ? [64, 32] : undefined,
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
    const t = r.timings;
    card.__timings = t; // Playwright reads this
    if (totals.firstImageAt == null) {
      totals.firstImageAt = performance.now();
      renderTotals();
    }
    renderSplit(t);
    // Image facts, refreshed with the header fields the wasm `info` reported.
    const inf = t.info;
    if (inf) {
      facts.textContent =
        `${inf.width}×${inf.height} · m${inf.modelId} ${inf.operatingPoint} · ` +
        `${inf.chromaFormat} · ${inf.bitDepth}-bit · ${kb(t.streamBytes ?? variant.bytes)} → ${variant.bpp} bpp` +
        `${inf.postFilters ? ' · post-filters' : ''}`;
    }
    // The per-card line: only this image's own costs. `wait` splits into the parts that were
    // this card's fault vs the page's: runtime (wasm/GPU startup), model (bundle download —
    // matches a row in the panel above), queue (behind other decodes).
    const parts = [`ttr ${ms(performance.now() - t0)}`];
    if (t.stream != null) parts.push(`stream ${ms(t.stream)} (${kb(t.streamBytes)})`);
    const waits = [];
    if (t.waitRuntime > 0.5) waits.push(`runtime ${ms(t.waitRuntime)}`);
    if (t.waitModel > 0.5) waits.push(`model ${ms(t.waitModel)}`);
    if (t.waitQueue > 0.5) waits.push(`queue ${ms(t.waitQueue)}`);
    parts.push(`wait ${ms(t.wait)}${waits.length ? ` (${waits.join(' · ')})` : ''}`);
    let dec = `decode ${ms(t.decode)}`;
    if (t.gpuMs != null) dec += ` (cpu ${ms(t.cpu)} + gpu ${ms(t.gpuMs)} + xfer ${ms(t.xfer)})`;
    else if (t.cpu != null) dec += ` (cpu ${ms(t.cpu)} + cpu ${ms(t.cpuSynth)})`;
    parts.push(dec);
    parts.push(`present ${ms(t.present)}`);
    parts.push(`${t.variant} · ${t.path}${r.presented === 'gpu' ? '+presented' : ''}`);
    if (quality === 'reduced') parts.push('reduced(y64,uv32)');
    const fell = t.gpuError ? ` · fallback: ${t.gpuError}` : '';
    // Tracked-heap report of the decode (wasm export `memory` — see MemoryReport in the crate
    // docs) and the module's linear-memory size; both are null on older packages.
    const mib = (b) => `${(b / 1048576).toFixed(0)} MiB`;
    if (t.memory) parts.push(`peak ${mib(t.memory.trackedPeakBytes)} · pool ${mib(t.memory.poolBytes)}`);
    if (t.wasmBytes) parts.push(`wasm ${mib(t.wasmBytes)}`);
    timing.textContent = parts.join(' · ') + fell;
    const title = decodeTitle(t);
    if (title) timing.title = title;
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
