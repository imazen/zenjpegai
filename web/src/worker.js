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
//   {type: 'decode', id, stream: ArrayBuffer, modelsBaseUrl, bundleVersions?, modelBuffers?,
//    maxChannels?, maxPixels?}
//     -> {type: 'result', id, width, height, rgba: ArrayBuffer, timings} (rgba.buffer transferred)
//     -> {type: 'error', id, message, reason?}
//   {type: 'present', id, stream: ArrayBuffer, canvas: OffscreenCanvas, modelsBaseUrl, verify,
//    canvasId, modelBuffers?, maxChannels?, maxPixels?}
//     -> {type: 'result', id, width, height, presented: 'gpu'|'2d', png?, timings}
//     Draws onto the transferred canvas: the webgpu package blits the decoder's RGBA texture
//     without a CPU readback (`presented: 'gpu'`); anything else decodes and putImageData's on
//     a 2d context (`presented: '2d'`). `verify` also posts back a PNG of the canvas for tests.
//   `maxChannels: [y, uv]` caps the latent channels decoded per component (the reference's
//   num_decode_chs progressive decode — a coarser picture for less work; 0/omitted = all).
//   `maxPixels` fails the decode with {reason:'too-large'} when width*height exceeds it —
//   checked from the header before any model fetch or decode work.
//   {type: 'releaseBuffers'}   (no reply; see zj.releaseBuffers doc)
// On startup, before any decode message: {type: 'ready', variant, tier, gpu, gpuError}, or
// {type: 'ready-error', message, reason?} if wasm itself failed to load/instantiate (e.g. this
// browser has no wasm SIMD128 and the build requires it — reason 'no-wasm-simd'; see
// web/README.md caniuse note). `gpu` is the initGpu() result ({ok, adapter, backend, software})
// or null; `gpuError` says why it is null. `adapterProbe` is the JS-side `GPUAdapter.info` from
// the pre-download probe ({vendor, architecture, device, description, isFallbackAdapter}) — the
// wgpu-side `gpu.adapter` only carries `description`, which Chrome leaves empty for hardware
// adapters.
//
// Startup milestones stream out as {type:'milestone', name, at, ...detail} messages — `at` is
// this worker's performance.now() (epoch ≈ when the pool constructed the Worker, so the page
// maps milestones onto the navigation timeline as pool-spawn-time + at). The page panel shows
// them as the one-time startup timeline; per-decode timings stay on the result messages.

import { MODEL_CACHE_NAME } from './prefetch.js';

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

// A minimal simd128 module (func () -> v128 { i32.const 0; i8x16.splat; i8x16.popcnt }) — the
// wasm-feature-detect probe. Every pkg-* build requires +simd128; a browser that rejects this
// (Safari <= 16.3, old Firefox ESR) cannot run any of them.
const SIMD_PROBE = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253,
  15, 253, 98, 11,
]);
const hasSimd = (() => {
  try {
    return typeof WebAssembly !== 'undefined' && WebAssembly.validate(SIMD_PROBE);
  } catch {
    return false;
  }
})();

// Milliseconds since this worker started running (its performance.now() epoch ≈ the pool's
// `new Worker()` call). Milestones ride to the page as live startup-timeline rows.
function milestone(name, detail) {
  self.postMessage({ type: 'milestone', name, at: performance.now(), ...(detail || {}) });
}
milestone('worker-start');

// Whether `url`'s last fetch was served without network bytes (HTTP cache / preload / SW) —
// from the resource-timing entry: transferSize 0 with a body means the bytes were already
// here. Absent an entry (timing disabled) report 'network', the conservative guess.
function fetchSource(url) {
  try {
    const entries = performance.getEntriesByName(url, 'resource');
    const e = entries[entries.length - 1];
    if (e && e.transferSize === 0 && e.decodedBodySize > 0) return 'cache';
    if (e && e.transferSize === 0 && e.encodedBodySize === 0) return 'cache';
  } catch { /* resource timing unavailable */ }
  return 'network';
}

// Instantiate `pkg-<name>` and start its rayon pool if it has one. The .wasm is fetched here
// (not inside wasm-bindgen's init) so the page panel can time fetch vs compile+instantiate
// separately and say how many bytes arrived from where.
async function loadPackage(name) {
  const base = new URL(`../dist/pkg-${name}/`, import.meta.url);
  const glueUrl = new URL('zenjpegai.js', base).href;
  const wasmUrl = new URL('zenjpegai_bg.wasm', base).href;
  let t = performance.now();
  const m = await import(glueUrl);
  milestone('glue', { ms: performance.now() - t });
  t = performance.now();
  const res = await fetch(wasmUrl);
  if (!res.ok) throw new Error(`wasm fetch failed: ${wasmUrl} (${res.status})`);
  const wasmBytes = await res.arrayBuffer();
  milestone('wasm', { ms: performance.now() - t, bytes: wasmBytes.byteLength, source: fetchSource(wasmUrl) });
  t = performance.now();
  // Object form: wasm-bindgen's positional `init(module_or_path, memory)` signature is
  // deprecated and warns. Passing the fetched bytes makes init a pure compile+instantiate.
  const init = await m.default({ module_or_path: wasmBytes });
  wasmMem = init.memory || null;
  milestone('instantiate', { ms: performance.now() - t });
  if (PACKAGES[name].threads) {
    const n = rayonThreads();
    t = performance.now();
    await m.initThreadPool(n);
    milestone('threads', { ms: performance.now() - t, count: n });
  }
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

// In-flight bundle fetches, keyed by cache key: two decodes of the same model can race only
// before the model is in wasm memory; this keeps that race to one network fetch per worker.
const pendingBundles = new Map();

// Fetch one bundle -> {file, bytes, ms, source, data}: 'cache-storage' for a Cache API hit,
// else 'network' (a `fetch` — which itself may land in HTTP cache/preload, see fetchSource).
// Every bundle load also feeds a 'model' milestone so the page panel lists each bundle with
// its provenance — Cache API hits show up as ~0 ms, answering "why don't I see the download".
async function fetchBundle(url, version, file) {
  const key = version ? `${url}?v=${version}` : url;
  const pending = pendingBundles.get(key);
  if (pending) return pending;
  const run = (async () => {
    const t = performance.now();
    let bytes;
    let source = 'network';
    let cache = null;
    try {
      cache = await caches.open(MODEL_CACHE_NAME);
    } catch {
      // Cache API unavailable (some private-browsing modes); fall back to a plain fetch.
    }
    if (cache) {
      const hit = await cache.match(key);
      if (hit) {
        bytes = new Uint8Array(await hit.arrayBuffer());
        source = 'cache-storage';
      }
    }
    if (!bytes) {
      const res = await fetch(key);
      if (!res.ok) throw new Error(`model bundle fetch failed: ${key} (${res.status})`);
      if (cache) {
        try {
          await cache.put(key, res.clone());
        } catch {
          // ignore cache write failures (quota, opaque response, etc.)
        }
      }
      bytes = new Uint8Array(await res.arrayBuffer());
      source = fetchSource(key);
    }
    return { file, bytes: bytes.byteLength, ms: performance.now() - t, source, data: bytes };
  })().finally(() => pendingBundles.delete(key));
  pendingBundles.set(key, run);
  return run;
}

/** `models/m<id>_common.zjb` + `models/m<id>_<op>.zjb`, relative to `modelsBaseUrl`. The two
 * bundles fetch in parallel (they are independent files; `addModels` still runs
 * common-then-part). `buffers` is the optional `{fileName: ArrayBuffer}` the page transferred
 * in with the job (`pool.js` `modelBuffers`): bundles the page prefetched — used directly,
 * skipping even the Cache API read (source 'prefetch'); anything missing falls back to
 * `fetchBundle`, which itself usually hits the prefetched cache entry. Resolves to the
 * per-bundle rows loaded by this call ([] when the model was already in wasm memory — the
 * card's model wait is then honestly 0). One 'model' milestone per load goes to the page
 * panel ("m<N> common+<op> — MB — ms — source"). */
async function ensureModels(mod, modelsBaseUrl, modelId, op, bundleVersions, buffers) {
  if (mod.hasModels(modelId, op)) return [];
  const base = modelsBaseUrl.endsWith('/') ? modelsBaseUrl : `${modelsBaseUrl}/`;
  const files = [`m${modelId}_common.zjb`, `m${modelId}_${op}.zjb`];
  const t = performance.now();
  const got = await Promise.all(
    files.map(async (f) => {
      const buf = buffers && buffers[f];
      // Page-prefetched bytes transferred with the job: no fetch at all on this worker.
      if (buf) {
        return { file: f, bytes: buf.byteLength, ms: 0, source: 'prefetch', data: new Uint8Array(buf) };
      }
      return fetchBundle(`${base}${f}`, bundleVersions?.[f], f);
    }),
  );
  mod.addModels(got[0].data);
  if (!mod.hasModels(modelId, op)) mod.addModels(got[1].data);
  const rows = got.map(({ file, bytes, ms, source }) => ({ file, bytes, ms, source }));
  milestone('model', {
    file: `m${modelId} common+${op}`,
    bytes: rows.reduce((a, r) => a + r.bytes, 0),
    ms: performance.now() - t,
    source: rows.some((r) => r.source === 'network') ? 'network' : rows[0]?.source || 'cache-storage',
  });
  return rows;
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
let wasmMem = null; // the module's WebAssembly.Memory, captured at init (its byteLength grows)
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
    if (!hasSimd) {
      // Every package is built with +simd128; fail here rather than at instantiate so the
      // pool/polyfill can keep the page's fallback content and report 'no-wasm-simd'.
      self.postMessage({
        type: 'ready-error',
        variant: cpuVariant,
        reason: 'no-wasm-simd',
        message: 'this browser has no WebAssembly SIMD128 support',
      });
      return;
    }
    if (gpuMode !== 'off') {
      if (typeof navigator !== 'undefined' && navigator.gpu) {
        // Ask for the adapter in JS first: it costs one requestAdapter and lets 'auto' skip the
        // whole pkg-webgpu download when the only adapter is software (a CPU rasteriser is
        // slower than this decoder's own CPU engine, so that adapter is never worth it).
        const tProbe = performance.now();
        const probe = await navigator.gpu.requestAdapter().catch(() => null);
        milestone('gpu-probe', { ms: performance.now() - tProbe });
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
            const tGpu = performance.now();
            gpuInfo = await mod.initGpu(gpuSoftwareMode);
            milestone('gpu-init', { ms: performance.now() - tGpu, ok: !!(gpuInfo && gpuInfo.ok) });
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
    milestone('ready', { variant, tier: mod.simdTier() });
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
    else if (msg.type === 'estimate') {
      // Header-parse estimate for the pool's maxBytesInFlight admission: cheap (no models,
      // no decode) and cannot trap the GPU path, so it needs no trap race.
      try {
        const est = mod.estimateMemory(new Uint8Array(msg.stream));
        self.postMessage({ type: 'estimate-result', id: msg.id, liveBytes: est.liveBytes, peakBytes: est.peakBytes });
      } catch (err) {
        // A stream whose headers won't parse fails its decode later with the real error;
        // a zero estimate just lets it be scheduled meanwhile.
        self.postMessage({ type: 'estimate-result', id: msg.id, liveBytes: 0, peakBytes: 0, error: String((err && err.message) || err) });
      }
    }
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
  // Stamped on arrival: drain() reports `waitRuntime` — the time the message sat in this
  // worker's queue (wasm still initialising, or a prior in-flight call — the pool only sends
  // to idle workers, so the startup wait dominates).
  ev.data.tRecv = performance.now();
  queue.push(ev.data);
  drain();
};

// The CPU/GPU phase split for one decode, ms. GPU path (img.gpu is the wasm `Timing`):
//   cpu  = headers + common + weights + entropy + latent  (host CPU stages — always CPU)
//   gpu  = gpuNs                                          (device time: synthesis + RGBA convert)
//   xfer = upload + plan + submit + convert + wait + readback/finish  (host<->device choreography)
// CPU path (img.stats is DecodeStats): cpu = headers+models+chains (entropy+latent),
// cpuSynth = synthLuma+synthesis+chroma+filters+output (synthesis+output on CPU).
// web/README.md "which stages run where" documents why the split lands this way.
function phaseSplit(img) {
  const g = img && img.gpu;
  if (g) {
    return {
      cpu: g.headersMs + g.commonMs + g.weightsMs + g.entropyMs + g.latentMs,
      gpuMs: g.gpuNs != null ? g.gpuNs / 1e6 : null,
      xfer:
        g.uploadMs + g.planMs + g.submitMs + g.convertMs + g.waitMs + g.readbackMs + g.finishMs,
      cpuSynth: null,
    };
  }
  const s = img && img.stats;
  if (s) {
    return {
      cpu: s.headers + s.models + s.chains,
      gpuMs: null,
      xfer: 0,
      cpuSynth: s.synthLuma + s.synthesis + s.chroma + s.filters + s.output,
    };
  }
  return { cpu: null, gpuMs: null, xfer: null, cpuSynth: null };
}

function timings(t0, t1, t2, presentMs, img, modelRows, waitRuntime, head) {
  const gpuError = [(img && img.gpuError) || null, trapNote].filter(Boolean).join('; ') || null;
  trapNote = null;
  const split = phaseSplit(img);
  return {
    total: t2 - t0 + (presentMs || 0),
    models: t1 - t0,
    waitRuntime: waitRuntime || 0,
    decode: t2 - t1 - (presentMs || 0),
    present: presentMs || 0,
    cpu: split.cpu,
    cpuSynth: split.cpuSynth,
    gpuMs: split.gpuMs,
    xfer: split.xfer,
    stats: (img && img.stats) || null,
    modelRows,
    info: head
      ? {
          width: head.width,
          height: head.height,
          bitDepth: head.bitDepth,
          modelId: head.modelId,
          operatingPoint: head.operatingPoint,
          chromaFormat: head.chromaFormat,
          postFilters: head.postFilters,
        }
      : null,
    variant,
    tier: mod.simdTier(),
    path: (img && img.path) || 'cpu',
    gpuError,
    gpu: (img && img.gpu) || null,
    // Per-call tracked-heap report from the wasm decode (`{trackedPeakBytes, trackedLiveBytes,
    // poolBytes}`; see MemoryReport in the crate docs) and this module's wasm linear-memory
    // size — the two numbers a page-level memory panel needs.
    memory: (img && img.memory) || null,
    wasmBytes: (wasmMem && wasmMem.buffer.byteLength) || null,
  };
}

/** Fails the message fast when the header-declared size exceeds the caller's pixel budget. */
function checkPixels(msg, head) {
  const max = msg.maxPixels;
  if (max && head.width * head.height > max) {
    const err = new Error(
      `picture ${head.width}x${head.height} exceeds maxPixels ${max} (${head.width * head.height} px)`,
    );
    err.reason = 'too-large';
    throw err;
  }
}

async function decodeOne(msg) {
  const { id, modelsBaseUrl } = msg;
  try {
    const bytes = new Uint8Array(msg.stream);
    const t0 = performance.now();
    const head = mod.info(bytes);
    checkPixels(msg, head);
    const modelRows = await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint, msg.bundleVersions, msg.modelBuffers);
    const t1 = performance.now();
    const mc = msg.maxChannels;
    const img = mc && typeof mod.decodePartial === 'function'
      ? await mod.decodePartial(bytes, mc[0] | 0, mc[1] | 0)
      : await mod.decode(bytes); // a Promise in the webgpu package, sync elsewhere
    const t2 = performance.now();
    self.postMessage(
      {
        type: 'result',
        id,
        width: img.width,
        height: img.height,
        rgba: img.rgba.buffer,
        timings: timings(t0, t1, t2, 0, img, modelRows, t0 - (msg.tRecv || t0), head),
      },
      [img.rgba.buffer],
    );
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String((err && err.message) || err), reason: err.reason || null });
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
    checkPixels(msg, head);
    const modelRows = await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint, msg.bundleVersions, msg.modelBuffers);
    const t1 = performance.now();
    const mc = msg.maxChannels;
    const canPartial = typeof mod.presentPartial === 'function';
    let presented = '2d';
    let img = null;
    let width;
    let height;
    let presentError = null;
    // The present sub-cost inside the call below, ms — surfaced as timings.present so a card
    // can separate "decode" from "getting pixels onto the canvas".
    let presentMs = 0;
    if (typeof mod.present === 'function') {
      try {
        const r = mc && canPartial
          ? await mod.presentPartial(bytes, canvas, mc[0] | 0, mc[1] | 0)
          : await mod.present(bytes, canvas);
        presented = 'gpu';
        width = r.width;
        height = r.height;
        img = r;
        // The wasm-side surface configure + blit + present; the rest of the call was decode.
        presentMs = (r.gpu && r.gpu.presentMs) || 0;
      } catch (err) {
        // Not GPU-presentable (post-filters, subsampled chroma, 10 bit) or the surface failed:
        // decode below and draw through the canvas's 2d context instead.
        presentError = String((err && err.message) || err);
      }
    } else {
      presentError = 'this package has no GPU presentation';
    }
    if (presented !== 'gpu') {
      img = mc && typeof mod.decodePartial === 'function'
        ? await mod.decodePartial(bytes, mc[0] | 0, mc[1] | 0)
        : await mod.decode(bytes);
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
      const tP = performance.now();
      ctx.putImageData(new ImageData(img.rgba, width, height), 0, 0);
      presentMs = performance.now() - tP;
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
    const tm = timings(t0, t1, t2, presentMs, img, modelRows, t0 - (msg.tRecv || t0), head);
    if (presented === 'gpu') tm.path = 'gpu';
    self.postMessage(
      { type: 'result', id, width, height, presented, png, timings: tm },
      png ? [png] : [],
    );
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String((err && err.message) || err), reason: err.reason || null });
  }
}
