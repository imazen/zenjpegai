// Decode worker: one module Worker owns one wasm module instance.
//
// Picks the `threads` package (rayon over Web Workers via `wasm-bindgen-rayon`, needs a
// SharedArrayBuffer) iff this worker's global scope is cross-origin isolated; otherwise the
// single-threaded `simd` package, which needs no isolation and runs on any page. Both packages
// decode to bit-identical pixels (`PORTING.md` "WebAssembly numeric policy") — the switch only
// changes throughput, never output.
//
// Message protocol (posted from `pool.js`):
//   {type: 'decode', id, stream: ArrayBuffer, modelsBaseUrl}
//     -> {type: 'result', id, width, height, rgba: ArrayBuffer, timings} (rgba.buffer transferred)
//     -> {type: 'error', id, message}
//   {type: 'releaseBuffers'}   (no reply; see zj.releaseBuffers doc)
// On startup, before any decode message: {type: 'ready', variant, tier}, or
// {type: 'ready-error', message} if wasm itself failed to load/instantiate (e.g. this browser
// has no wasm SIMD128 and the build requires it — see web/README.md caniuse note).

const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
const variant = isolated ? 'threads' : 'simd';
const pkgBase = new URL(`../dist/pkg-${variant}/`, import.meta.url);

const CACHE_NAME = 'zenjpegai-models-v1';

async function fetchBundle(url) {
  // Cache API, not HTTP cache alone: guarantees the (multi-MB) bundle is fetched at most once
  // per browser profile regardless of HTTP cache eviction, and lets the demo/tests assert a
  // cache hit on the second image that reuses a model.
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
  } catch {
    // Cache API unavailable (some private-browsing modes); fall back to a plain fetch.
  }
  if (cache) {
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`model bundle fetch failed: ${url} (${res.status})`);
  if (cache) {
    try {
      await cache.put(url, res.clone());
    } catch {
      // ignore cache write failures (quota, opaque response, etc.)
    }
  }
  return new Uint8Array(await res.arrayBuffer());
}

/** `models/m<id>_common.zjb` + `models/m<id>_<op>.zjb`, relative to `modelsBaseUrl`. */
async function ensureModels(mod, modelsBaseUrl, modelId, op) {
  if (mod.hasModels(modelId, op)) return;
  const base = modelsBaseUrl.endsWith('/') ? modelsBaseUrl : `${modelsBaseUrl}/`;
  mod.addModels(await fetchBundle(`${base}m${modelId}_common.zjb`));
  if (mod.hasModels(modelId, op)) return;
  mod.addModels(await fetchBundle(`${base}m${modelId}_${op}.zjb`));
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
const queue = [];
let draining = false;

const ready = (async () => {
  try {
    mod = await import(`${pkgBase}zenjpegai.js`);
    await mod.default(); // fetches `zenjpegai_bg.wasm` next to zenjpegai.js
    if (variant === 'threads') {
      await mod.initThreadPool(Math.max(1, navigator.hardwareConcurrency || 4));
    }
    self.postMessage({ type: 'ready', variant, tier: mod.simdTier() });
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
      if (msg.type === 'decode') self.postMessage({ type: 'error', id: msg.id, message: 'wasm module failed to load' });
      continue;
    }
    if (msg.type === 'releaseBuffers') mod.releaseBuffers();
    else if (msg.type === 'decode') await decodeOne(msg);
  }
  draining = false;
}

self.onmessage = (ev) => {
  queue.push(ev.data);
  drain();
};

async function decodeOne(msg) {
  const { id, modelsBaseUrl } = msg;
  try {
    const bytes = new Uint8Array(msg.stream);
    const t0 = performance.now();
    const head = mod.info(bytes);
    await ensureModels(mod, modelsBaseUrl, head.modelId, head.operatingPoint);
    const t1 = performance.now();
    const img = mod.decode(bytes);
    const t2 = performance.now();
    self.postMessage(
      {
        type: 'result',
        id,
        width: img.width,
        height: img.height,
        rgba: img.rgba.buffer,
        timings: { total: t2 - t0, models: t1 - t0, decode: t2 - t1, variant, tier: mod.simdTier() },
      },
      [img.rgba.buffer],
    );
  } catch (err) {
    self.postMessage({ type: 'error', id, message: String((err && err.message) || err) });
  }
}
