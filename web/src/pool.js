// Main-thread (or other worker) orchestrator for `worker.js`.
//
// One persistent worker when the page is cross-origin isolated: the `threads` package already
// parallelises one decode across `navigator.hardwareConcurrency` via rayon, so a second threaded
// worker would just contend for the same cores. Otherwise a small round-robin pool of
// single-threaded `simd` workers, so independent images decode concurrently instead of queueing
// behind one thread.
export class DecoderPool {
  /**
   * @param {string} modelsBaseUrl - directory holding `m<id>_common.zjb` / `m<id>_<op>.zjb`.
   * @param {number} [maxWorkers] - cap on the simd-variant pool size (ignored when isolated).
   * @param {URL|string} [workerUrl] - override for `worker.js` (tests point this at a fixture).
   */
  constructor({ modelsBaseUrl, maxWorkers = 4, workerUrl } = {}) {
    if (!modelsBaseUrl) throw new Error('DecoderPool requires modelsBaseUrl');
    // Resolved to an absolute URL against the PAGE's location: workers have their own base URL
    // (the worker script's own location), so a relative modelsBaseUrl passed through as-is would
    // resolve against `worker.js`'s directory instead of the page that constructed this pool.
    this.modelsBaseUrl = new URL(modelsBaseUrl, typeof location !== 'undefined' ? location.href : undefined).href;
    this.isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
    this.size = this.isolated ? 1 : Math.max(1, Math.min(maxWorkers, (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4));
    this.workers = [];
    this.readyInfo = [];
    this._next = 0;
    this._pending = new Map();
    this._seq = 0;
    const url = workerUrl || new URL('./worker.js', import.meta.url);
    for (let i = 0; i < this.size; i++) this._spawn(url);
  }

  _spawn(url) {
    const w = new Worker(url, { type: 'module' });
    const ready = new Promise((resolve) => {
      const onFirst = (ev) => {
        const msg = ev.data;
        if (msg.type === 'ready') {
          w.removeEventListener('message', onFirst);
          resolve({ ok: true, variant: msg.variant, tier: msg.tier });
        } else if (msg.type === 'ready-error') {
          w.removeEventListener('message', onFirst);
          resolve({ ok: false, variant: msg.variant, error: msg.message });
        }
      };
      w.addEventListener('message', onFirst);
    });
    w.addEventListener('message', (ev) => {
      const msg = ev.data;
      if (msg.type !== 'result' && msg.type !== 'error') return;
      const p = this._pending.get(msg.id);
      if (!p) return;
      this._pending.delete(msg.id);
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else p.resolve(msg);
    });
    w.addEventListener('error', (ev) => {
      // A worker-level error (e.g. the module script itself failed to parse) doesn't carry an
      // `id`; reject everything still pending on this worker so callers don't hang forever.
      for (const [id, p] of this._pending) {
        p.reject(new Error(`worker error: ${ev.message || ev}`));
        this._pending.delete(id);
      }
    });
    this.workers.push(w);
    this.readyInfo.push(ready);
  }

  /** Resolves once every worker in the pool has reported ready (or failed to load wasm). */
  async ready() {
    return Promise.all(this.readyInfo);
  }

  /**
   * Decode one JPEG AI codestream. `stream` may be an ArrayBuffer (transferred, so do not reuse
   * it afterwards) or any typed-array view (copied once so the caller's buffer is untouched).
   * @returns {Promise<{width:number, height:number, rgba:Uint8ClampedArray, timings:object}>}
   */
  decode(stream) {
    const buf = ArrayBuffer.isView(stream)
      ? stream.buffer.slice(stream.byteOffset, stream.byteOffset + stream.byteLength)
      : stream;
    const worker = this.workers[this._next];
    this._next = (this._next + 1) % this.workers.length;
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._pending.set(id, { resolve, reject });
      worker.postMessage({ type: 'decode', id, stream: buf, modelsBaseUrl: this.modelsBaseUrl }, [buf]);
    }).then((msg) => ({
      width: msg.width,
      height: msg.height,
      rgba: new Uint8ClampedArray(msg.rgba),
      timings: msg.timings,
    }));
  }

  /** Drop each worker's feature-map buffers (see `zj.releaseBuffers`); call on idle/hidden. */
  releaseBuffers() {
    for (const w of this.workers) w.postMessage({ type: 'releaseBuffers' });
  }

  terminate() {
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
    this.readyInfo.length = 0;
    for (const [, p] of this._pending) p.reject(new Error('pool terminated'));
    this._pending.clear();
  }
}
