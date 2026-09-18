// Main-thread (or other worker) orchestrator for `worker.js`.
//
// Scheduling contract (2026-09-18, after "every image on the demo decodes at once and each
// takes about a second"):
// - cross-origin isolated page: ONE persistent `threads`-variant worker — the rayon build
//   already spreads one decode across `navigator.hardwareConcurrency`, so exactly one decode
//   is in flight at a time and everything else waits in the queue.
// - otherwise: up to `min(navigator.hardwareConcurrency - 1, maxWorkers)` single-threaded
//   `simd` workers — leaving one hardware thread for the page's main thread and the browser,
//   because at exactly `hardwareConcurrency` busy decode workers a 4-core box pushed each
//   decode past 2x its solo time (measured on a 4-vCPU GitHub runner: 3815 ms vs 1776 ms
//   solo). Never more decodes in flight than workers.
// - ALL decodes go through one shared queue here; a job is posted to a worker only when that
//   worker is idle (the earlier round-robin assigned jobs to workers immediately, so a job
//   could sit behind a busy worker's backlog while a sibling idled, and nothing could be
//   re-ordered). Queue order: `priority` ascending, then arrival order. `priority` may be a
//   function — it is re-evaluated on every dispatch, so a caller can bubble a queued job up
//   when e.g. its image scrolls into view (see polyfill.js).
export class DecoderPool {
  /**
   * @param {string} modelsBaseUrl - directory holding `m<id>_common.zjb` / `m<id>_<op>.zjb`.
   * @param {number} [maxWorkers] - cap N on the simd-variant pool size (ignored when isolated).
   * @param {URL|string} [workerUrl] - override for `worker.js` (tests point this at a fixture).
   * @param {Object<string,string>} [bundleVersions] - file name -> version token (the demo
   *   passes each bundle's sha256 from manifest.json). Appended to bundle URLs as `?v=` so a
   *   bundle swapped under the same file name can't be served stale from the Cache API.
   */
  constructor({ modelsBaseUrl, maxWorkers = 4, workerUrl, bundleVersions } = {}) {
    if (!modelsBaseUrl) throw new Error('DecoderPool requires modelsBaseUrl');
    // Resolved to an absolute URL against the PAGE's location: workers have their own base URL
    // (the worker script's own location), so a relative modelsBaseUrl passed through as-is would
    // resolve against `worker.js`'s directory instead of the page that constructed this pool.
    this.modelsBaseUrl = new URL(modelsBaseUrl, typeof location !== 'undefined' ? location.href : undefined).href;
    this.bundleVersions = bundleVersions || null;
    this.isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated === true;
    const hwc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
    this.size = this.isolated ? 1 : Math.max(1, Math.min(maxWorkers, hwc - 1));
    this.workers = [];
    this.readyInfo = [];
    this._idle = []; // workers with no in-flight decode, in spawn order
    this._queue = []; // jobs waiting for a worker: {id, buf, priority, seq, resolve, reject, onDispatch, enqueuedAt}
    this._pending = new Map(); // id -> {resolve, reject, worker}
    this._seq = 0;
    this._maxInflight = 0;
    this._completed = 0;
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
      this._completed++;
      if (msg.type === 'error') p.reject(new Error(msg.message));
      else {
        msg.queued = p.queuedMs;
        p.resolve(msg);
      }
      this._idle.push(w);
      this._dispatch();
    });
    w.addEventListener('error', (ev) => {
      // A worker-level error (e.g. the module script itself failed to parse) doesn't carry an
      // `id`; reject this worker's in-flight job so the caller doesn't hang forever, then take
      // the worker out of rotation.
      for (const [id, p] of this._pending) {
        if (p.worker !== w) continue;
        p.reject(new Error(`worker error: ${ev.message || ev}`));
        this._pending.delete(id);
      }
      const i = this._idle.indexOf(w);
      if (i >= 0) this._idle.splice(i, 1);
      // If that was the last worker able to run jobs, nothing queued can ever start —
      // reject the backlog instead of letting those callers hang forever.
      if (!this._idle.length && !this._pending.size) {
        for (const job of this._queue.splice(0)) job.reject(new Error('no live decode workers'));
      }
    });
    this.workers.push(w);
    this._idle.push(w);
    this.readyInfo.push(ready);
  }

  /** Resolves once every worker in the pool has reported ready (or failed to load wasm). */
  async ready() {
    return Promise.all(this.readyInfo);
  }

  /**
   * Decode one JPEG AI codestream. `stream` may be an ArrayBuffer (transferred, so do not reuse
   * it afterwards) or any typed-array view (copied once so the caller's buffer is untouched).
   * @param {object} [opts]
   * @param {number|function():number} [opts.priority] - smaller runs sooner; a function is
   *   re-evaluated each time the queue is dispatched, so a queued job can be promoted after
   *   the fact (the polyfill uses this for images that scroll into view while queued).
   * @param {function()} [opts.onDispatch] - called once the job leaves the queue and is posted
   *   to a worker (drives "queued -> decoding" placeholder states).
   * @returns {Promise<{width:number, height:number, rgba:Uint8ClampedArray, timings:object}>}
   *   `timings.queued` is the ms the job spent waiting for a worker.
   */
  decode(stream, opts = {}) {
    const buf = ArrayBuffer.isView(stream)
      ? stream.buffer.slice(stream.byteOffset, stream.byteOffset + stream.byteLength)
      : stream;
    const id = ++this._seq;
    return new Promise((resolve, reject) => {
      this._queue.push({
        id,
        buf,
        priority: opts.priority ?? 0,
        seq: id,
        resolve,
        reject,
        onDispatch: opts.onDispatch,
        enqueuedAt: performance.now(),
      });
      this._dispatch();
    }).then((msg) => ({
      width: msg.width,
      height: msg.height,
      rgba: new Uint8ClampedArray(msg.rgba),
      timings: { ...msg.timings, queued: msg.queued },
    }));
  }

  _dispatch() {
    while (this._idle.length && this._queue.length) {
      let best = 0;
      for (let i = 1; i < this._queue.length; i++) {
        if (compareJobs(this._queue[i], this._queue[best]) < 0) best = i;
      }
      const job = this._queue.splice(best, 1)[0];
      const worker = this._idle.shift();
      this._pending.set(job.id, {
        resolve: job.resolve,
        reject: job.reject,
        worker,
        queuedMs: performance.now() - job.enqueuedAt,
      });
      this._maxInflight = Math.max(this._maxInflight, this._pending.size);
      job.onDispatch?.();
      worker.postMessage(
        { type: 'decode', id: job.id, stream: job.buf, modelsBaseUrl: this.modelsBaseUrl, bundleVersions: this.bundleVersions },
        [job.buf],
      );
    }
  }

  /** Live scheduling counters (tests and the demo's status readout). */
  stats() {
    return {
      size: this.size,
      isolated: this.isolated,
      inflight: this._pending.size,
      maxInflight: this._maxInflight,
      queued: this._queue.length,
      completed: this._completed,
    };
  }

  /** Drop each worker's feature-map buffers (see `zj.releaseBuffers`); call on idle/hidden. */
  releaseBuffers() {
    for (const w of this.workers) w.postMessage({ type: 'releaseBuffers' });
  }

  terminate() {
    for (const w of this.workers) w.terminate();
    this.workers.length = 0;
    this.readyInfo.length = 0;
    this._idle.length = 0;
    for (const [, p] of this._pending) p.reject(new Error('pool terminated'));
    this._pending.clear();
    for (const job of this._queue.splice(0)) job.reject(new Error('pool terminated'));
  }
}

// Queue order: priority ascending (a function priority is re-evaluated here on every pass, so
// a queued job can be promoted after the fact), then arrival order.
function compareJobs(a, b) {
  const pa = typeof a.priority === 'function' ? a.priority() : a.priority;
  const pb = typeof b.priority === 'function' ? b.priority() : b.priority;
  return pa - pb || a.seq - b.seq;
}
