// Decode worker: one module Worker owns one wasm module instance.
//
// Package selection: the `?gpu=` query param on the worker URL (set by `pool.js`) chooses
//   off             -> `threads` (cross-origin isolated) or `simd`, exactly the historical choice
//   auto (default)  -> same CPU packages: measured on an RTX 2080 (2026-09-18, demo streams
//                      ~1 MP) the WebGPU path's wall-clock decode does not beat the threads
//                      engine, so `auto` does not pay the pkg-webgpu download (see web/README §7)
//   on              -> try `pkg-webgpu` + `initGpu(0)`; a non-software adapter keeps that
//                      package, anything else falls back to the CPU packages
//   software        -> `pkg-webgpu` + `initGpu(1)`: software adapters accepted (test/debug mode —
//                      the GPU code path runs even on a GPU-less host, slower than the CPU engine)
//   force-software  -> `pkg-webgpu` + `initGpu(2)`: additionally passes WebGPU's
//                      forceFallbackAdapter, so the software adapter is used even next to a GPU
// A failed `initGpu` discards the webgpu module and loads the CPU package: its decode() falls
// back to the CPU engine internally anyway, but the threads package's CPU engine is faster.
//
// Message protocol (posted from `pool.js`):
//   {type: 'decode', id, stream: ArrayBuffer, modelsBaseUrl, bundleVersions?}
//     -> {type: 'result', id, width, height, rgba: ArrayBuffer, timings} (rgba.buffer transferred)
//     -> {type: 'error', id, message}
//   {type: 'present', id, stream: ArrayBuffer, canvas: OffscreenCanvas, modelsBaseUrl, verify}
//     -> {type: 'result', id, width, height, presented: 'gpu'|'2d', png?, timings}
//     Draws onto the transferred canvas: the webgpu package blits the decoder's RGBA texture
//     without a CPU readback (`presented: 'gpu'`); anything else decodes and putImageData's on
//     a 2d context (`presented: '2d'`). `verify` also posts back a PNG of the canvas for tests.
//   {type: 'releaseBuffers'}   (no reply; see zj.releaseBuffers doc)
// On startup, before any decode message: {type: 'ready', variant, tier, gpu, gpuError}, or
// {type: 'ready-error', message} if wasm itself failed to load/instantiate (e.g. this browser
// has no wasm SIMD128 and the build requires it — see web/README.md caniuse note). `gpu` is the
// initGpu() result ({ok, adapter, backend, software}) or null; `gpuError` says why it is null.
// `adapterProbe` is the JS-side `GPUAdapter.info` from the pre-download probe ({vendor,
// architecture, device, description, isFallbackAdapter}) — the wgpu-side `gpu.adapter` only
// carries `description`, which Chrome leaves empty for hardware adapters.

const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
const cpuVariant = isolated ? 'threads' : 'simd';
const gpuMode = new URLSearchParams(self.location.search).get('gpu') || 'auto';
const gpuSoftwareMode = gpuMode === 'force-software' ? 2 : gpuMode === 'software' ? 1 : 0;
const cpuPkgBase = new URL(`../dist/pkg-${cpuVariant}/`, import.meta.url);

// `?threads=N` on the worker URL overrides the rayon pool size (thread-scaling benchmarks,
// tests/threads.spec.ts). The default caps at 16 child workers: the calling thread runs rayon
// jobs too, so N children mean N+1 busy threads, and on a 32-hwc host (16 physical cores)
// measured decode saturates at ~16 children and regresses at 31-32
// (benchmarks/wasm_threads_2026-09-18.tsv: 132 ms mean at 16 vs 162 ms at 32).
function rayonThreads() {
  const q = new URL(self.location.href).searchParams.get('threads');
  const n = q ? Number.parseInt(q, 10) : NaN;
  return Number.isFinite(n) && n > 0
    ? Math.min(n, 256)
    : Math.min(Math.max(1, navigator.hardwareConcurrency || 4), 16);
}

// Cache namespace for the model bundles. Keys inside it carry the bundle's own `?v=` version
// token (see fetchBundle), so a bundle whose bytes change under the same file name gets a new
// key. v1 keys had no version and could pin a stale bundle forever; the bump abandons them and
// the delete below reclaims the space.
const CACHE_NAME = 'zenjpegai-models-v2';
const STALE_CACHES = ['zenjpegai-models-v1'];
if (typeof caches !== 'undefined') for (const name of STALE_CACHES) caches.delete(name).catch(() => {});

async function fetchBundle(url, version) {
  // Cache API, not HTTP cache alone: guarantees the (multi-MB) bundle is fetched at most once
  // per browser profile regardless of HTTP cache eviction, and lets the demo/tests assert a
  // cache hit on the second image that reuses a model. The version token is part of the key
  // (and of the fetch URL — harmless to a static host, which ignores the query): content
  // swapped under the same file name is a different key, never a stale hit.
  const key = version ? `${url}?v=${version}` : url;
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
  } catch {
    // Cache API unavailable (some private-browsing modes); fall back to a plain fetch.
  }
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetch(key);
  if (!res.ok) throw new Error(`model bundle fetch failed: ${key} (${res.status})`);
  if (cache) {
    try {
      await cache.put(key, res.clone());
    } catch {
      // ignore cache write failures (quota, opaque response, etc.)
    }
  }
  return new Uint8Array(await res.arrayBuffer());
}

/** `models/m<id>_common.zjb` + `models/m<id>_<op>.zjb`, relative to `modelsBaseUrl`. */
async function ensureModels(mod, modelsBaseUrl, modelId, op, bundleVersions) {
  if (mod.hasModels(modelId, op)) return;
  const base = modelsBaseUrl.endsWith('/') ? modelsBaseUrl : `${modelsBaseUrl}/`;
  const common = `m${modelId}_common.zjb`;
  mod.addModels(await fetchBundle(`${base}${common}`, bundleVersions?.[common]));
  if (mod.hasModels(modelId, op)) return;
  const part = `m${modelId}_${op}.zjb`;
  mod.addModels(await fetchBundle(`${base}${part}`, bundleVersions?.[part]));
}

// `self.onmessage` MUST be assigned before any `await` below, in the same synchronous turn the
// worker script starts running. A `message` event dispatched to a global scope that has no
// listener registered AT DISPATCH TIME is simply dropped — it is NOT queued and replayed once a
// listener shows up later. `pool.js` does not wait for the 'ready' handshake before calling
// `decode()` (a page's own `installJpegAiPolyfill()` call does not either, deliberately — it
// starts every matching image decoding as soon as it finds it), so any decode messages posted
// while wasm was still loading were being silently discarded: every one hung forever with no
// result and no error (caught running the polyfill against 4 images on one page for the first
// time in a real browser). Fix: attach the listener first, buffer into `queue` until `ready`
// resolves, then drain it — same fix incidentally gives the strict per-worker FIFO already
// needed for correctness (`mod.decode()` is not reentrant; two calls in flight at once on the
// `threads` package deadlocks rayon, its workers left Atomics.wait-ing on state a still-in-flight
// sibling call owns), so there is no separate `chain` variable — `queue` IS the serialization.
let mod = null;
let variant = cpuVariant;
let gpuInfo = null;
let gpuError = null;
let adapterProbe = null;
const queue = [];
let draining = false;
// In-flight call and its trap-detector: a Rust panic in the wasm build is panic=abort, so it
// does NOT reject `mod.decode()`'s promise — the trap escapes wasm-bindgen's executor as an
// uncaught RuntimeError and lands on `self.onerror` below while the promise stays pending
// forever. `trapSignal` is how `drain` unsticks itself; `currentId` is which message died.
let currentId = null;
let trapSignal = null;
// One-line note merged into the NEXT result's timings.gpuError so the retried call reports
// that the GPU path trapped and CPU ran instead.
let trapNote = null;
// OffscreenCanvases transferred here by 'present' messages, keyed by pool-assigned id: a
// transferred canvas is neutered for the sender, so repeat presents reference it by `canvasId`.
const canvases = new Map();

self.onerror = (ev) => {
  const message = String((ev && ev.message) || ev);
  // GPU state may be corrupt (a RefCell borrow or a half-built plan can outlive the trap):
  // drop it so the next call takes the CPU engine.
  try { if (mod && typeof mod.disableGpu === 'function') mod.disableGpu(); } catch { /* already poisoned */ }
  trapNote = `wasm trap in GPU path: ${message}`;
  self.postMessage({ type: 'error', id: currentId, message: trapNote, trapped: true });
  currentId = null;
  if (trapSignal) trapSignal();
};

const ready = (async () => {
  try {
    if (gpuMode !== 'off') {
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        // Ask for the adapter in JS first: it costs one requestAdapter and lets 'auto' skip the
        // whole pkg-webgpu download when the only adapter is software (a CPU rasteriser is
        // slower than this decoder's own CPU engine, so that adapter is never worth it).
        const probe = await navigator.gpu.requestAdapter().catch(() => null);
        const pi = probe && probe.info;
        adapterProbe = pi
          ? {
              vendor: pi.vendor || '',
              architecture: pi.architecture || '',
              device: pi.device || '',
              description: pi.description || '',
              isFallbackAdapter: !!pi.isFallbackAdapter,
            }
          : null;
        const fallbackOnly = !!adapterProbe?.isFallbackAdapter;
        const wantGpu = gpuMode === 'on' || gpuSoftwareMode >= 1;
        if (probe && wantGpu && (gpuSoftwareMode >= 1 || !fallbackOnly)) {
          try {
            const m = await import(new URL('../dist/pkg-webgpu/zenjpegai.js', import.meta.url).href);
            await m.default(); // fetches `zenjpegai_bg.wasm` next to zenjpegai.js
            gpuInfo = await m.initGpu(gpuSoftwareMode);
            mod = m;
            variant = 'webgpu';
          } catch (err) {
            gpuError = `no usable GPU context: ${String((err && err.message) || err)}`;
          }
        } else if (!probe) {
          gpuError = 'requestAdapter returned null';
        } else if (!wantGpu) {
          gpuError = 'gpu=auto keeps the CPU engine (see web/README §7); ?gpu=on opts in';
        } else {
          gpuError = `only a software WebGPU adapter (${probe.info?.description || 'fallback'})`;
        }
      } else {
        gpuError = 'navigator.gpu is absent';
      }
    }
    if (!mod) {
      mod = await import(`${cpuPkgBase}zenjpegai.js`);
      // Object form: wasm-bindgen's positional `init(module_or_path, memory)` signature is
      // deprecated and warns. The wasm file defaults to `zenjpegai_bg.wasm` next to the glue.
      await mod.default({ module_or_path: new URL('zenjpegai_bg.wasm', cpuPkgBase).href });
      if (cpuVariant === 'threads') {
        await mod.initThreadPool(rayonThreads());
      }
    }
    self.postMessage({ type: 'ready', variant, tier: mod.simdTier(), gpu: gpuInfo, gpuError, adapterProbe });
  } catch (err) {
    mod = null;
    self.postMessage({ type: 'ready-error', variant, message: String((err && err.message) || err) });
  }
})();

async function drain() {
  if (draining) return;
  draining = true;
  await ready; // resolves either way; `mod` is null on failure
  while (queue.length) {
    const msg = queue.shift();
    if (!mod) {
      if (msg.type === 'decode' || msg.type === 'present') {
        self.postMessage({ type: 'error', id: msg.id, message: 'wasm module failed to load' });
      }
      continue;
    }
    if (msg.type === 'releaseBuffers') mod.releaseBuffers();
    else if (msg.type === 'decode' || msg.type === 'present') {
      // Race the call against the trap signal: if the wasm traps, `onerror` above fails the
      // message and `trapSignal` lets this loop continue instead of hanging on a promise that
      // will never settle.
      currentId = msg.id;
      const trapped = new Promise((resolve) => { trapSignal = resolve; });
      const call = msg.type === 'decode' ? decodeOne(msg) : presentOne(msg);
      await Promise.race([call, trapped]);
      trapSignal = null;
      currentId = null;
    }
  }
  draining = false;
}

self.onmessage = (ev) => {
  queue.push(ev.data);
  drain();
};

function timings(t0, t1, t2, img) {
  const gpuError = [(img && img.gpuError) || null, trapNote].filter(Boolean).join('; ') || null;
  trapNote = null;
  return {
    total: t2 - t0,
    models: t1 - t0,
    decode: t2 - t1,
    variant,
    tier: mod.simdTier(),
    path: (img && img.path) || 'cpu',
    gpuError,
    gpu: (img && img.gpu) || null,
  };
}

async function decodeOne(msg) {
  const { id, modelsBaseUrl } = msg;
  try {
    const bytes = new Uint8Array(msg.stream);
    const t0 = performance.now();
    const head = mod.info(bytes);
    await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint, msg.bundleVersions);
    const t1 = performance.now();
    const img = await mod.decode(bytes); // a Promise in the webgpu package, sync elsewhere
    const t2 = performance.now();
    self.postMessage(
      {
        type: 'result',
        id,
        width: img.width,
        height: img.height,
        rgba: img.rgba.buffer,
        timings: timings(t0, t1, t2, img),
      },
      [img.rgba.buffer],
    );
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String((err && err.message) || err) });
  }
}

async function presentOne(msg) {
  const { id, modelsBaseUrl, canvasId } = msg;
  if (msg.canvas) canvases.set(canvasId, msg.canvas);
  const canvas = canvases.get(canvasId);
  if (!canvas) {
    self.postMessage({ type: 'error', id, message: `unknown canvasId ${canvasId}` });
    return;
  }
  try {
    const bytes = new Uint8Array(msg.stream);
    const t0 = performance.now();
    const head = mod.info(bytes);
    await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint, msg.bundleVersions);
    const t1 = performance.now();
    let presented = '2d';
    let img = null;
    let width;
    let height;
    let presentError = null;
    if (typeof mod.present === 'function') {
      try {
        const r = await mod.present(bytes, canvas);
        presented = 'gpu';
        width = r.width;
        height = r.height;
        img = r;
      } catch (err) {
        // Not GPU-presentable (post-filters, subsampled chroma, 10 bit) or the surface failed:
        // decode below and draw through the canvas's 2d context instead.
        presentError = String((err && err.message) || err);
      }
    } else {
      presentError = 'this package has no GPU presentation';
    }
    if (presented !== 'gpu') {
      img = await mod.decode(bytes);
      width = img.width;
      height = img.height;
      if (img.gpuError) {
        img.gpuError = `${presentError ? `${presentError}; ` : ''}${img.gpuError}`;
      } else {
        img.gpuError = presentError;
      }
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('canvas is already bound to a WebGPU surface');
      ctx.putImageData(new ImageData(img.rgba, width, height), 0, 0);
    }
    const t2 = performance.now();
    let png = null;
    if (msg.verify) {
      try {
        png = await (await canvas.convertToBlob({ type: 'image/png' })).arrayBuffer();
      } catch {
        png = null; // a canvas whose frame already presented may refuse; the test reads `presented`
      }
    }
    const tm = timings(t0, t1, t2, img);
    if (presented === 'gpu') tm.path = 'gpu';
    self.postMessage(
      { type: 'result', id, width, height, presented, png, timings: tm },
      png ? [png] : [],
    );
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String((err && err.message) || err) });
  }
}
