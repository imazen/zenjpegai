import { installJpegAiPolyfill } from './src/polyfill.js';

window.__jai = installJpegAiPolyfill({ modelsBaseUrl: 'models/' });
window.__jaiEvents = [];
window.addEventListener('jpegaidecoded', (e) => window.__jaiEvents.push(e.detail));
window.addEventListener('jpegaierror', (e) => window.__jaiEvents.push({ error: String(e.detail.error) }));
