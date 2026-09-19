// <img> / <picture><source> polyfill for JPEG AI (ISO/IEC 6048-1:2025) pictures: finds tags
// pointing at a JPEG AI codestream, decodes it in a worker pool (`pool.js`), and swaps the
// pixels in as fast as the platform allows.
//
// File extension / MIME type: as of this standard's 2025 publication there is no IANA media-type
// registration and no extension listed in the reference software (checked 2026-09-17: no
// `image/jpeg-ai` entry at iana.org/assignments/media-types/image, nothing in
// jpeg-ai-reference-software's docs or code). `.jai` / `image/jpeg-ai` below are this project's
// placeholder — change exactly these two constants if/when a real one is registered; nothing
// else in this file encodes the choice.
//
// CSP: no eval, no inline script, no Function() — safe under `script-src 'self'
// 'wasm-unsafe-eval'`. Needs `worker-src 'self'` (or default-src covering it) for the module
// Worker, and `img-src ... blob:` only if a page opts a picture into render mode "img" (see
// below); the default "canvas" mode needs neither `blob:` nor `data:`.
import { DecoderPool, hasWasmSimd } from './pool.js';
import { prefetchModelPairs, takeModelBuffers, modelPairNames } from './prefetch.js';
import { probeStreamInfo } from './stream-info.js';

export const EXTENSION = '.jai';
export const MIME = 'image/jpeg-ai';

// The header probe lives in `stream-info.js` (`probeStreamInfo`) — it reads `modelId` and
// the operating point straight off the codestream bytes so the model bundles a decode will
// need can be prefetched without wasm. `prefetch.js` (`prefetchModelPairs`) fetches those
// bundles into the worker's Cache API namespace.

const ATTR_HANDLED = 'data-jai-handled';
const SELECTOR = [
  `img[src$="${EXTENSION}"]`,
  `img[data-src$="${EXTENSION}"]`,
  `img[srcset*="${EXTENSION}"]`,
  `img[data-srcset*="${EXTENSION}"]`,
  'picture > source[type="image/jpeg-ai"]',
].join(', ');

/**
 * @typedef {Object} PolyfillOptions
 * @property {string} modelsBaseUrl - passed straight to `DecoderPool`.
 * @property {URL|string} [workerUrl]
 * @property {number} [maxWorkers]
 * @property {'canvas'|'img'} [renderMode] - default render strategy; a matched element can
 *   override it per-instance with `data-jai-render="canvas"` or `"img"` (see README for the
 *   measured tradeoff: canvas skips a PNG encode/decode round trip, img keeps native `<img>`
 *   semantics — alt text, "Save Image As", srcset re-evaluation on resize).
 * @property {Document|ShadowRoot} [root] - defaults to `document`.
 * @property {Object<string,string>} [bundleVersions] - passed to `DecoderPool`: model-bundle
 *   file name -> version token, appended to bundle URLs as `?v=` (see pool.js/worker.js).
 * @property {'auto'|'on'|'software'|'force-software'|'off'} [gpu] - passed to `DecoderPool`:
 *   'auto' (default) stays on the CPU engine (the GPU path does not beat it at typical sizes);
 *   'on' opts into WebGPU, which also presents through it without a CPU readback where the
 *   browser and the stream allow it.
 * @property {number[]} [maxChannels] - `[y, uv]` latent-channel caps forwarded to every decode
 *   (progressive/num_decode_chs decode — a coarser, faster picture; omit for full quality).
 * @property {number} [maxPixels] - reject pictures whose header declares more pixels, before
 *   any model fetch or decode work; the failure is a `jpegaierror` with reason 'too-large'
 *   and the <img>/<picture> fallback stays untouched.
 * @property {'auto'|false} [prefetch] - model-bundle prefetching. 'auto' (default) probes each
 *   matched stream's header and warms the worker's Cache API namespace
 *   (`zenjpegai-models-v2`) with the `m<N>_common`/`m<N>_<op>` bundles those pictures need —
 *   for near-viewport pictures this starts at handle time (a small Range fetch) so it overlaps
 *   the worker's wasm load; every decode also hands the fetched bytes to its worker by
 *   transfer. false disables prefetching.
 *   (`<link rel="preload">` was tried and dropped: a Worker's fetch() cannot consume the
 *   document's preload map, so every bundle double-downloaded — see web/README.md. A page with
 *   a known model set can warm the same cache keys directly at install time via
 *   `prefetch.js`/`prefetchModelPairs`.)
 */

// Scheduling (see pool.js's contract): every decode enters the pool's shared queue ordered by
// `priority`, and the priority here is visibility order. `visRank` records the order in which
// images first came within one viewport height of the viewport — by a synchronous rect check
// at handle time (the observer's first callback lands a frame later, after the first jobs are
// already dispatched) or by the observer once the user scrolls. A queued job's function
// priority is re-evaluated at every dispatch, so an eager image that scrolls into view while
// still waiting is promoted ahead of offscreen work. `loading="lazy"` images are only enqueued
// at all once they intersect (same one-viewport-height margin); everything else enqueues at
// scan time, ordered by that rank. `data-jai-state` tracks pending -> queued -> decoding ->
// (attribute removed on swap) / "error" so a page can style the queued placeholder.
const FAR_PRIORITY = 1_000_000_000;

/** @param {PolyfillOptions} options */
export function installJpegAiPolyfill(options) {
  const opts = { renderMode: 'canvas', root: document, prefetch: 'auto', ...options };
  const pool = new DecoderPool({ modelsBaseUrl: opts.modelsBaseUrl, maxWorkers: opts.maxWorkers, workerUrl: opts.workerUrl, bundleVersions: opts.bundleVersions, gpu: opts.gpu });
  // img -> Map<fileName, Promise<ArrayBuffer>> from prefetchModelPairs: the in-flight
  // bundle fetches a decodeAndSwap can hand to its worker by transfer (`modelBuffers`).
  const prefetches = new WeakMap();

  const visRank = new WeakMap();
  let rankSeq = 0;
  let domSeq = 0;
  const lazyRun = new Map(); // img -> run(), for images waiting on intersection
  const io =
    'IntersectionObserver' in globalThis
      ? new IntersectionObserver(
          (entries) => {
            for (const e of entries) {
              if (!e.isIntersecting) continue;
              const img = e.target;
              io.unobserve(img);
              if (!visRank.has(img)) visRank.set(img, rankSeq++);
              const run = lazyRun.get(img);
              if (run) {
                lazyRun.delete(img);
                run();
              }
            }
          },
          // One viewport height of lookahead in the scroll axis.
          { rootMargin: '100% 0px' },
        )
      : null;

  const scan = () => {
    for (const el of opts.root.querySelectorAll(SELECTOR)) handle(el, pool, opts, { visRank, nextRank: () => rankSeq++, nextDom: () => domSeq++, io, lazyRun, prefetches });
  };
  scan();

  const observer = new MutationObserver((mutations) => {
    const sched = { visRank, nextRank: () => rankSeq++, nextDom: () => domSeq++, io, lazyRun, prefetches };
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        if (node.matches?.(SELECTOR)) handle(node, pool, opts, sched);
        node.querySelectorAll?.(SELECTOR).forEach((el) => handle(el, pool, opts, sched));
      }
      if (m.type === 'attributes' && m.target.nodeType === Node.ELEMENT_NODE && m.target.matches?.(SELECTOR)) {
        handle(m.target, pool, opts, sched);
      }
    }
  });
  observer.observe(opts.root === document ? document.documentElement : opts.root, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'srcset', 'data-src', 'data-srcset'],
  });

  return { pool, observer, stop: () => { observer.disconnect(); io?.disconnect(); } };
}

// True when the element's box is within one viewport height of the viewport, in either
// direction — the synchronous twin of the observer's '100% 0px' margin, used to rank images
// that are already near-visible when handle() runs (before the first observer callback).
function nearViewport(el) {
  const vh = globalThis.innerHeight || 1024;
  const r = el.getBoundingClientRect();
  return r.bottom > -vh && r.top < 2 * vh;
}

// Probe one stream's header and start fetching the model bundles it will need
// (`prefetch.js` writes the same Cache API keys the worker reads). The Range fetch asks for
// the PIH only; servers that ignore Range return the whole stream (which the decode fetch
// then hits in HTTP cache — still a win). The returned prefetch map is stashed on the
// picture so `decodeAndSwap` can hand the bytes to its worker by transfer.
// Best-effort: any failure just skips prefetch — the worker's `ensureModels` is the fallback.
async function prefetchModels(img, url, pool, opts, prefetches) {
  try {
    const res = await fetch(url, { headers: { Range: 'bytes=0-4095' } });
    if (!res.ok) return;
    const head = probeStreamInfo(await res.arrayBuffer());
    if (!head) return;
    prefetches.set(img, prefetchModelPairs(pool.modelsBaseUrl, [head], opts.bundleVersions));
  } catch { /* see above */ }
}

function handle(el, pool, opts, sched) {
  const img = el.tagName === 'SOURCE' ? el.closest('picture')?.querySelector('img') : el;
  if (!img || img.hasAttribute(ATTR_HANDLED)) return;
  const url = resolveUrl(el, img);
  if (!url) return;
  img.setAttribute(ATTR_HANDLED, '1');
  img.setAttribute('data-jai-state', 'pending');
  const domIdx = sched.nextDom();
  if (sched.io) {
    sched.io.observe(img);
    if (nearViewport(img) && !sched.visRank.has(img)) sched.visRank.set(img, sched.nextRank());
  }
  // Prefetch while the worker is still loading wasm — the window is seconds wide, so even a
  // warm that starts here lands well before ensureModels needs the bundles. Only for images
  // that will decode soon (eager, or already near the viewport); deep-lazy images' models
  // are fetched on demand.
  if (opts.prefetch && !pool.noSimd && (img.loading !== 'lazy' || nearViewport(img))) {
    prefetchModels(img, url, pool, opts, sched.prefetches);
  }

  const run = () => {
    if (pool.noSimd) {
      // No wasm SIMD in this browser: no package can run — keep the <img>/<picture>
      // fallback exactly as authored and report why.
      img.setAttribute('data-jai-state', 'error');
      img.dispatchEvent(new CustomEvent('jpegaierror', { bubbles: true, detail: { error: new Error('no WebAssembly SIMD128'), reason: 'no-wasm-simd', url } }));
      return;
    }
    img.setAttribute('data-jai-state', 'queued');
    decodeAndSwap(img, url, pool, opts, {
      priority: () => sched.visRank.get(img) ?? FAR_PRIORITY + domIdx,
      onDispatch: () => img.setAttribute('data-jai-state', 'decoding'),
    }, sched.prefetches.get(img)).catch((err) => {
      img.setAttribute('data-jai-state', 'error');
      img.dispatchEvent(new CustomEvent('jpegaierror', { bubbles: true, detail: { error: err, reason: err.reason || null, url } }));
    });
  };

  if (img.loading === 'lazy' && sched.io) {
    sched.lazyRun.set(img, run);
  } else {
    // A microtask, not a direct call: install-time failures (no-wasm-simd, too-large) then
    // reach listeners a page attaches right after installJpegAiPolyfill returns.
    queueMicrotask(run);
  }
}

/** Pick the URL to fetch: `src`/`data-src` directly, else the first `.jai` candidate in a
 * `srcset`/`data-srcset` (simple `w`-descriptor closest-above-`clientWidth*devicePixelRatio`
 * selection; falls back to the last listed candidate for bare-URL or `x`-descriptor lists). */
function resolveUrl(el, img) {
  if (el.tagName === 'SOURCE') {
    const srcset = el.getAttribute('srcset');
    return srcset ? pickFromSrcset(srcset, img) : null;
  }
  const direct = el.getAttribute('src') || el.getAttribute('data-src');
  if (direct && direct.endsWith(EXTENSION)) return direct;
  const srcset = el.getAttribute('srcset') || el.getAttribute('data-srcset');
  return srcset ? pickFromSrcset(srcset, img) : null;
}

function pickFromSrcset(srcset, img) {
  const candidates = srcset
    .split(',')
    .map((s) => s.trim().split(/\s+/))
    .filter(([url]) => url && url.endsWith(EXTENSION))
    .map(([url, descriptor]) => ({ url, w: descriptor?.endsWith('w') ? parseInt(descriptor, 10) : null }));
  if (!candidates.length) return null;
  const target = (img.clientWidth || img.width || 0) * (globalThis.devicePixelRatio || 1);
  const withWidth = candidates.filter((c) => c.w != null).sort((a, b) => a.w - b.w);
  if (target && withWidth.length) {
    const fit = withWidth.find((c) => c.w >= target);
    return (fit || withWidth[withWidth.length - 1]).url;
  }
  return candidates[candidates.length - 1].url;
}

async function decodeAndSwap(img, url, pool, opts, decodeOpts = {}, prefetched = null) {
  const t0 = performance.now();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const bytes = await res.arrayBuffer();
  const t1 = performance.now();
  // The worker would read these same two header fields (`modelId`, operating point) after
  // wasm init and only then start the multi-MB model fetch — prefetching here, straight off
  // the stream bytes, overlaps that download with the queue ahead of this job (`prefetch.js`
  // writes the same Cache API keys `ensureModels` checks). The fetched pair is then handed
  // to the worker by transfer — awaited BEFORE enqueueing so the worker never starts a
  // second, racing fetch for the same bundle. `prefetched` is the map handle() started when
  // this picture first came near the viewport; a stream that does not probe (or with
  // prefetch disabled) gets no modelBuffers and the worker fetches exactly as before.
  const head = probeStreamInfo(bytes);
  const modelBuffers = head && opts.prefetch
    ? await takeModelBuffers(
        prefetched ?? prefetchModelPairs(pool.modelsBaseUrl, [head], opts.bundleVersions),
        modelPairNames(head.modelId, head.op),
      )
    : null;
  if (opts.maxChannels && !decodeOpts.maxChannels) decodeOpts.maxChannels = opts.maxChannels;
  if (opts.maxPixels && !decodeOpts.maxPixels) decodeOpts.maxPixels = opts.maxPixels;
  const mode = img.getAttribute('data-jai-render') || opts.renderMode;
  const alt = img.getAttribute('alt') ?? '';
  // Canvas mode can draw entirely inside the worker (`pool.decodeToCanvas`): the GPU path
  // blits without a CPU readback and the CPU path still skips the rgba transfer to this thread.
  // Needs transferControlToOffscreen (Safari >= 16.4); anything older takes the classic path.
  // Both branches take the same queue options — a worker-drawn canvas still goes through the
  // pool's priority queue.
  const probe = document.createElement('canvas');
  const canWorkerDraw = mode === 'canvas' && typeof probe.transferControlToOffscreen === 'function';
  let width;
  let height;
  let rgba = null;
  let presented = '2d';
  let timings;
  let canvas = null;
  if (canWorkerDraw) {
    canvas = probe;
    const r = await pool.decodeToCanvas(bytes, canvas, { ...decodeOpts, modelBuffers });
    ({ width, height, presented, timings } = r);
  } else {
    const r = await pool.decode(bytes, { ...decodeOpts, modelBuffers });
    ({ width, height, rgba, timings } = r);
  }
  img.removeAttribute('data-jai-state');
  const t2 = performance.now();

  if (mode === 'img') {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
    const blob = await new Promise((resolve, reject) =>
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('canvas.toBlob failed'))), 'image/png'),
    );
    const objectUrl = URL.createObjectURL(blob);
    const prevSrc = img.src;
    img.addEventListener('load', () => URL.revokeObjectURL(objectUrl), { once: true });
    img.src = objectUrl;
    img.removeAttribute('srcset');
    if (!img.hasAttribute('width')) img.width = width;
    if (!img.hasAttribute('height')) img.height = height;
    void prevSrc;
  } else {
    if (!canvas) canvas = document.createElement('canvas');
    if (canvas.__jaiCanvasId != null) {
      // Control is transferred to a worker: the IDL width/height setters throw InvalidStateError
      // now; the content attributes still set the element's display size.
      canvas.setAttribute('width', String(width));
      canvas.setAttribute('height', String(height));
    } else {
      canvas.width = width;
      canvas.height = height;
    }
    for (const attr of ['id', 'class', 'style', 'title', 'lang']) {
      const v = img.getAttribute(attr);
      if (v != null) canvas.setAttribute(attr, v);
    }
    if (img.hasAttribute('width')) canvas.setAttribute('width', img.getAttribute('width'));
    if (img.hasAttribute('height')) canvas.setAttribute('height', img.getAttribute('height'));
    canvas.setAttribute('role', 'img');
    if (alt) canvas.setAttribute('aria-label', alt);
    else canvas.setAttribute('aria-hidden', 'true');
    for (const { name, value } of Array.from(img.attributes)) {
      if (name.startsWith('data-jai') || ['src', 'srcset', 'data-src', 'data-srcset', 'alt', 'loading'].includes(name)) continue;
      canvas.setAttribute(name, value);
    }
    if (rgba) canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
    img.replaceWith(canvas);
  }
  const t3 = performance.now();
  (img.ownerDocument.defaultView || globalThis).dispatchEvent(
    new CustomEvent('jpegaidecoded', {
      detail: { url, width, height, mode, presented, bytes: bytes.byteLength, timings: { ...timings, fetch: t1 - t0, decodeCall: t2 - t1, render: t3 - t2, total: t3 - t0 } },
    }),
  );
}
