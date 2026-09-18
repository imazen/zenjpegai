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
import { DecoderPool } from './pool.js';

export const EXTENSION = '.jai';
export const MIME = 'image/jpeg-ai';

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
 */

/** @param {PolyfillOptions} options */
export function installJpegAiPolyfill(options) {
  const opts = { renderMode: 'canvas', root: document, ...options };
  const pool = new DecoderPool({ modelsBaseUrl: opts.modelsBaseUrl, maxWorkers: opts.maxWorkers, workerUrl: opts.workerUrl });

  const scan = () => {
    for (const el of opts.root.querySelectorAll(SELECTOR)) handle(el, pool, opts);
  };
  scan();

  const observer = new MutationObserver((mutations) => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType !== Node.ELEMENT_NODE) continue;
        if (node.matches?.(SELECTOR)) handle(node, pool, opts);
        node.querySelectorAll?.(SELECTOR).forEach((el) => handle(el, pool, opts));
      }
      if (m.type === 'attributes' && m.target.nodeType === Node.ELEMENT_NODE && m.target.matches?.(SELECTOR)) {
        handle(m.target, pool, opts);
      }
    }
  });
  observer.observe(opts.root === document ? document.documentElement : opts.root, {
    childList: true,
    subtree: true,
    attributes: true,
    attributeFilter: ['src', 'srcset', 'data-src', 'data-srcset'],
  });

  return { pool, observer, stop: () => observer.disconnect() };
}

function handle(el, pool, opts) {
  const img = el.tagName === 'SOURCE' ? el.closest('picture')?.querySelector('img') : el;
  if (!img || img.hasAttribute(ATTR_HANDLED)) return;
  const url = resolveUrl(el, img);
  if (!url) return;
  img.setAttribute(ATTR_HANDLED, '1');

  const run = () => decodeAndSwap(img, url, pool, opts).catch((err) => {
    img.dispatchEvent(new CustomEvent('jpegaierror', { detail: { error: err, url } }));
  });

  if (img.loading === 'lazy' && 'IntersectionObserver' in globalThis) {
    const io = new IntersectionObserver((entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          io.disconnect();
          run();
        }
      }
    }, { rootMargin: '200px' });
    io.observe(img);
  } else {
    run();
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

async function decodeAndSwap(img, url, pool, opts) {
  const t0 = performance.now();
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url} failed: ${res.status}`);
  const bytes = await res.arrayBuffer();
  const t1 = performance.now();
  const { width, height, rgba, timings } = await pool.decode(bytes);
  const t2 = performance.now();
  const mode = img.getAttribute('data-jai-render') || opts.renderMode;
  const alt = img.getAttribute('alt') ?? '';

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
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
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
    canvas.getContext('2d').putImageData(new ImageData(rgba, width, height), 0, 0);
    img.replaceWith(canvas);
  }
  const t3 = performance.now();
  (img.ownerDocument.defaultView || globalThis).dispatchEvent(
    new CustomEvent('jpegaidecoded', {
      detail: { url, width, height, mode, bytes: bytes.byteLength, timings: { ...timings, fetch: t1 - t0, decodeCall: t2 - t1, render: t3 - t2, total: t3 - t0 } },
    }),
  );
}
