// Registers sw-coi.js and reloads the page once so the reload's own navigation request is
// intercepted by the now-active service worker (a service worker cannot retroactively rewrite
// the response that loaded the CURRENT document — only future requests, starting with the next
// navigation). No-op if already isolated, if service workers are unavailable, or if isolation
// still can't be achieved after one attempt (third-party iframe embeds without
// `allow="cross-origin-isolated"`, service workers disabled, etc.) — guarded by a
// sessionStorage flag so a page that genuinely cannot become isolated reloads once, not forever.
//
// Demo-only (see web/demo/demo.js): a general-purpose polyfill embedded on someone else's page
// has no business unilaterally installing a service worker that rewrites every response on
// their origin — that is a page-owner decision, not something src/polyfill.js should impose as
// a side effect of decoding one image. Sites that want threaded wasm can wire this same
// technique in themselves (or simply send the headers, which is the real fix).
const RELOAD_GUARD = 'zenjpegai-coi-reload-attempted';

export async function ensureCrossOriginIsolated() {
  if (typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated) return true;
  if (!('serviceWorker' in navigator)) return false;
  try {
    await navigator.serviceWorker.register('sw-coi.js');
    await navigator.serviceWorker.ready;
    if (navigator.serviceWorker.controller) {
      sessionStorage.removeItem(RELOAD_GUARD);
      // Controlled but still not isolated (e.g. embedded without permission): give up quietly.
      return typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
    }
    if (sessionStorage.getItem(RELOAD_GUARD) === '1') return false;
    sessionStorage.setItem(RELOAD_GUARD, '1');
    location.reload();
    return new Promise(() => {}); // page is reloading; this instance stops here
  } catch {
    return false;
  }
}
