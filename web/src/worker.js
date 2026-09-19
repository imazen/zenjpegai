// Decode worker: one module Worker owns one wasm module instance.
//
// Package selection: the `?gpu=` query param on the worker URL (set by `pool.js`) chooses
//   off             -> `threads` (cross-origin isolated) or `simd`, exactly the historical choice
//   auto (default)  -> the webgpu package when `navigator.gpu` offers a NON-software adapter,
//                      else the CPU packages: measured on an RTX 2080 through Dawn/Vulkan
//                      (2026-09-18, demo streams ~1 MP, benchmarks/wasm_gpu_phases_2026-09-18.tsv)
//                      warm GPU decode ~52-65 ms vs ~88-104 ms on `threads` and ~632-773 ms on
//                      `simd` — the GPU wins on isolated AND non-isolated pages on this
//                      hardware class. A software adapter alone keeps the CPU packages (a CPU
//                      rasteriser is slower than this decoder's own engine).
//   on              -> try the webgpu package + `initGpu(0)`; a non-software adapter keeps that
//                      package, anything else falls back to the CPU packages
//   software        -> webgpu package + `initGpu(1)`: software adapters accepted (test/debug
//                      mode — the GPU code path runs even on a GPU-less host, slower than the
//                      CPU engine)
//   force-software  -> webgpu package + `initGpu(2)`: additionally passes WebGPU's
//                      forceFallbackAdapter, so the software adapter is used even next to a GPU
// The webgpu package is `pkg-webgpu-threads` on a cross-origin isolated page (its CPU
// entropy/latent stages then run on the rayon pool — the single-threaded latent stage is the
// webgpu build's dominant cost, ~120 ms of a ~250 ms 1 MP decode on RTX 2080/Dawn,
// benchmarks/wasm_gpu_phases_2026-09-18.tsv) and `pkg-webgpu` elsewhere.
// A failed `initGpu` discards the webgpu module and loads the CPU package: its decode() falls
// back to the CPU engine internally anyway, but the threads package's CPU engine is faster.
//
// Message protocol (posted from `pool.js`):
//   {type: 'decode', id, stream: ArrayBuffer, modelsBaseUrl, bundleVersions?, modelBuffers?}
//     -> {type: 'result', id, width, height, rgba: ArrayBuffer, timings} (rgba.buffer transferred)
//     -> {type: 'error', id, message}
//   {type: 'present', id, stream: ArrayBuffer, canvas: OffscreenCanvas, modelsBaseUrl, verify, modelBuffers?}
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

import { prefetchBundle } from './prefetch.js';

const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
const cpuVariant = isolated ? 'threads' : 'simd';
const gpuMode = new URLSearchParams(self.location.search).get('gpu') || 'auto';
const gpuSoftwareMode = gpuMode === 'force-software' ? 2 : gpuMode === 'software' ? 1 : 0;

// The four built packages (`web/dist/pkg-<name>/`). `threads` marks the rayon builds: their
// CPU entropy/latent stages run on a worker pool (requires cross-origin isolation) and the
// package exports `initThreadPool`.
const PACKAGES = {
  'webgpu-threads': { threads: true },
  webgpu: { threads: false },
  threads: { threads: true },
  simd: { threads: false },
};

// Instantiate `pkg-<name>` and start its rayon pool if it has one.
async function loadPackage(name) {
  const base = new URL(`../dist/pkg-${name}/`, import.meta.url);
  const m = await import(new URL('zenjpegai.js', base).href);
  // Object form: wasm-bindgen's positional `init(module_or_path, memory)` signature is
  // deprecated and warns. The wasm file defaults to `zenjpegai_bg.wasm` next to the glue.
  await m.default({ module_or_path: new URL('zenjpegai_bg.wasm', base).href });
  if (PACKAGES[name].threads) await m.initThreadPool(rayonThreads());
  variant = name;
  return m;
}

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

// Cache namespace for the model bundles lives in `prefetch.js` (`MODEL_CACHE_NAME`). Keys
// inside it carry the bundle's own `?v=` version token, so a bundle whose bytes change under
// the same file name gets a new key. v1 keys had no version and could pin a stale bundle
// forever; the v2 bump abandons them and the delete below reclaims the space. The page-side
// prefetch writes entries under these same keys so `ensureModels` below hits them.
const STALE_CACHES = ['zenjpegai-models-v1'];
if (typeof caches !== 'undefined') for (const name of STALE_CACHES) caches.delete(name).catch(() => {});

/** `models/m<id>_common.zjb` + `models/m<id>_<op>.zjb`, relative to `modelsBaseUrl`.
 * The two halves are fetched in parallel (they are independent files; `addModels` still
 * runs common-then-part). `buffers` is the optional `{fileName: ArrayBuffer}` the page
 * transferred in with the job (`pool.js` `modelBuffers`): bundles the page prefetched —
 * used directly, skipping even the Cache API read; anything missing falls back to
 * `prefetchBundle`, which itself usually hits the prefetched cache entry. */
async function ensureModels(mod, modelsBaseUrl, modelId, op, bundleVersions, buffers) {
  if (mod.hasModels(modelId, op)) return;
  const base = modelsBaseUrl.endsWith('/') ? modelsBaseUrl : `${modelsBaseUrl}/`;
  const common = `m${modelId}_common.zjb`;
  const part = `m${modelId}_${op}.zjb`;
  const get = async (name) => {
    const buf = buffers && buffers[name];
    return buf ? new Uint8Array(buf) : new Uint8Array(await prefetchBundle(`${base}${name}`, bundleVersions?.[name]));
  };
  const [c, p] = await Promise.all([get(common), get(part)]);
  mod.addModels(c);
  if (mod.hasModels(modelId, op)) return;
  mod.addModels(p);
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
        // auto opts into the GPU only for a hardware adapter: measured faster than both CPU
        // packages on RTX 2080/Dawn (see the header); software-only adapters stay on CPU.
        const wantGpu = gpuMode === 'on' || gpuSoftwareMode >= 1 || gpuMode === 'auto';
        if (probe && wantGpu && (gpuSoftwareMode >= 1 || !fallbackOnly)) {
          try {
            // Isolated pages get the rayon build (parallel CPU entropy/latent stages);
            // `webgpu` is the non-isolated build, and also the fallback if a deployment
            // was assembled without `webgpu-threads`.
            const candidates = isolated ? ['webgpu-threads', 'webgpu'] : ['webgpu'];
            for (const name of candidates) {
              try {
                mod = await loadPackage(name);
                break;
              } catch { /* package not built into this site; try the next */ }
            }
            if (!mod) throw new Error(`none of ${candidates.join(', ')} is served`);
            gpuInfo = await mod.initGpu(gpuSoftwareMode);
          } catch (err) {
            mod = null; // a failed initGpu discards the webgpu module; the CPU package loads below
            gpuError = `no usable GPU context: ${String((err && err.message) || err)}`;
          }
        } else if (!probe) {
          gpuError = 'requestAdapter returned null';
        } else if (!wantGpu) {
          gpuError = 'gpu=off keeps the CPU engine; ?gpu=on or auto opts in';
        } else {
          gpuError = `only a software WebGPU adapter (${probe.info?.description || 'fallback'})`;
        }
      } else {
        gpuError = 'navigator.gpu is absent';
      }
    }
    if (!mod) {
      mod = await loadPackage(cpuVariant);
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
    await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint, msg.bundleVersions, msg.modelBuffers);
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
    await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint, msg.bundleVersions, msg.modelBuffers);
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
