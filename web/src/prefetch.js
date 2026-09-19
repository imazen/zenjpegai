// Model-bundle prefetch, shared by the demo page (demo.js) and the polyfill (polyfill.js).
//
// Why: a decode's model bundles (`m<id>_common.zjb` + `m<id>_<op>.zjb`, ~12+4 MB per model)
// used to be fetched by the worker's `ensureModels` one at a time and only after wasm + GPU
// init had finished — the slowest link in the cold-load chain. Prefetching starts the same
// fetches at t≈0, in parallel, while the wasm module is still downloading.
//
// Cache discipline: bundles go into the same Cache API namespace and under the same key the
// worker uses (`worker.js` -> fetchBundle), so the worker's `ensureModels` hits the entry this
// file wrote instead of fetching again. For the FIRST decode the caller can additionally hand
// the fetched ArrayBuffer to the worker by transfer (`DecoderPool`'s `modelBuffers` option) —
// faster than a second Cache API read; the cache copy still serves every later decode.
export const MODEL_CACHE_NAME = 'zenjpegai-models-v2';

// `<url>?v=<version>` — the same key shape `worker.js` derives from `bundleVersions`
// (manifest.json's per-bundle sha256). A bundle swapped under the same file name is a
// different key everywhere it is cached.
export function bundleKey(url, version) {
  return version ? `${url}?v=${version}` : url;
}

/** `models/m<id>_common.zjb` + `models/m<id>_<op>.zjb`, in load order. */
export function modelPairNames(modelId, op) {
  return [`m${modelId}_common.zjb`, `m${modelId}_${op}.zjb`];
}

function bundleUrl(modelsBaseUrl, name) {
  const base = modelsBaseUrl.endsWith('/') ? modelsBaseUrl : `${modelsBaseUrl}/`;
  return new URL(base + name, typeof location !== 'undefined' ? location.href : undefined).href;
}

// Fetch one bundle with the same cache-first semantics as `worker.js`'s fetchBundle:
// Cache API hit -> bytes, else network -> populate the cache -> bytes. Resolves to an
// ArrayBuffer so the caller can transfer it into the worker (it is ALSO already in the
// cache for every later decode).
export async function prefetchBundle(url, version) {
  const key = bundleKey(url, version);
  let cache = null;
  try {
    cache = typeof caches !== 'undefined' ? await caches.open(MODEL_CACHE_NAME) : null;
  } catch {
    // Cache API unavailable (some private-browsing modes); a plain fetch still warms the
    // HTTP cache and returns usable bytes.
  }
  if (cache) {
    const hit = await cache.match(key);
    if (hit) return hit.arrayBuffer();
  }
  // `credentials: 'include'`: the site is same-origin and static, so this sends exactly what
  // the default 'same-origin' would — and it is the credentials mode the preload links
  // (`<link rel="preload" as="fetch" crossorigin="use-credentials">`, stamped by
  // build-site.mjs) must see to be consumed instead of double-fetched. Measured in
  // Chromium: anonymous `crossorigin` + `credentials: 'omit'` does NOT match.
  const res = await fetch(key, { credentials: 'include' });
  if (!res.ok) throw new Error(`model bundle fetch failed: ${key} (${res.status})`);
  // Read the bytes BEFORE `cache.put`: putting `res.clone()` tees the body stream, and a
  // failed cache write can take the source down with it — `res.arrayBuffer()` then rejects
  // and the caller loses a bundle that downloaded fine (observed for the 12 MB
  // m1_common.zjb: the worker had to fetch it a third time).
  const buf = await res.arrayBuffer();
  if (cache) {
    try {
      await cache.put(key, new Response(buf.slice(0), { headers: res.headers }));
    } catch {
      // ignore cache write failures (quota, etc.) — the bytes are what matters
    }
  }
  return buf;
}

// Start fetching, in parallel, the `common`+`op` bundle of every (modelId, op) pair in
// `pairs` (list order decides request issue order — put the bundles needed first at the
// front). Returns a Map<fileName, Promise<ArrayBuffer>>: `await` an entry to hand the bytes
// to a worker (`modelBuffers`), `delete` it once consumed — a buffer can be transferred
// only once.
export function prefetchModelPairs(modelsBaseUrl, pairs, bundleVersions) {
  const out = new Map();
  for (const { modelId, op } of pairs) {
    if (modelId == null || !op) continue;
    for (const name of modelPairNames(modelId, op)) {
      if (!out.has(name)) {
        out.set(name, prefetchBundle(bundleUrl(modelsBaseUrl, name), bundleVersions?.[name]));
      }
    }
  }
  return out;
}

// Consume `names` from a prefetch map into a `{name: ArrayBuffer}` object for
// `DecoderPool`'s `modelBuffers` option. Entries are deleted so each prefetched buffer is
// transferred at most once (postMessage transfer detaches it); a rejected prefetch entry is
// dropped silently — the worker then fetches the bundle itself, exactly as before.
export async function takeModelBuffers(prefetched, names) {
  const out = {};
  if (!prefetched) return out;
  for (const name of names) {
    const p = prefetched.get(name);
    if (!p) continue;
    prefetched.delete(name);
    const buf = await p.catch(() => null);
    if (buf) out[name] = buf;
  }
  return out;
}
