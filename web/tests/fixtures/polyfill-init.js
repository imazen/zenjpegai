import { installJpegAiPolyfill } from './src/polyfill.js';

// Test switches (set by addInitScript before this module runs):
//   window.__noPrefetch -> install with prefetch:false (default 'auto')
//   window.__maxPixels  -> install with maxPixels:<n>
window.__jaiEvents = [];
window.addEventListener('jpegaidecoded', (e) => window.__jaiEvents.push(e.detail));
window.addEventListener('jpegaierror', (e) => window.__jaiEvents.push({ error: String(e.detail.error), reason: e.detail.reason || null }));
window.__jai = installJpegAiPolyfill({
  modelsBaseUrl: 'models/',
  prefetch: window.__noPrefetch ? false : 'auto',
  maxPixels: window.__maxPixels || undefined,
});
