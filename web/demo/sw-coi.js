// Minimal cross-origin-isolation (COI) shim for static hosts that cannot set
// Cross-Origin-Opener-Policy / Cross-Origin-Embedder-Policy response headers — GitHub Pages
// being the one this demo actually needs it for. The technique was first popularized by
// gzuidhof/coi-serviceworker (MIT); this is our own from-scratch implementation of it, kept to
// the ~20 lines the idea needs. A service worker intercepts every same-origin fetch (navigation
// included) and re-wraps the response with COOP/COEP added, so the page that loads under it
// becomes cross-origin isolated without the origin ever sending those headers itself.
//
// Demo-only: see coi-loader.js for why this is not wired into src/polyfill.js.
self.addEventListener('install', () => self.skipWaiting());
self.addEventListener('activate', (event) => event.waitUntil(self.clients.claim()));

self.addEventListener('fetch', (event) => {
  if (event.request.cache === 'only-if-cached' && event.request.mode !== 'same-origin') return;
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.status === 0) return response; // opaque cross-origin response: pass through
        const headers = new Headers(response.headers);
        headers.set('Cross-Origin-Embedder-Policy', 'require-corp');
        headers.set('Cross-Origin-Opener-Policy', 'same-origin');
        return new Response(response.body, { status: response.status, statusText: response.statusText, headers });
      })
      .catch((err) => new Response(String(err && err.message || err), { status: 500 })),
  );
});
