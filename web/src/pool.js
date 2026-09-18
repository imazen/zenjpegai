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
//   when e.g. its image scrolls into view (see polyfill.js). A job may also be `pinned` to one
//   worker (`decodeToCanvas` repeat presents — a transferred OffscreenCanvas can only be
//   drawn on by the worker that owns it); pinned jobs wait for their worker specifically.
// - The `gpu` constructor option is appended to the worker URL as `?gpu=` and selects whether
//   the worker loads `pkg-webgpu` (see worker.js's header for the mode table).
export class DecoderPool {
  /**
   * @param {string} modelsBaseUrl - directory holding `m<id>_common.zjb` / `m<id>_<op>.zjb`.
   * @param {number} [maxWorkers] - cap N on the simd-variant pool size (ignored when isolated).
   * @param {URL|string} [workerUrl] - override for `worker.js` (tests point this at a fixture).
   * @param {Object<string,string>} [bundleVersions] - file name -> version token (the demo
   *   passes each bundle's sha256 from manifest.json). Appended to bundle URLs as `?v=` so a
   *   bundle swapped under the same file name can't be served stale from the Cache API.
   * @param {'auto'|'software'|'force-software'|'off'} [gpu] - WebGPU synthesis: 'auto' (default)
   *   uses the `pkg-webgpu` package when `navigator.gpu` yields a non-software adapter and falls
   *   back to the CPU packages otherwise; 'software' additionally accepts software adapters
   *   (exercises the GPU code path on a GPU-less host); 'force-software' always takes the
   *   software adapter; 'off' never loads `pkg-webgpu`.
   * @param {number} [threads] - rayon pool size of each `threads`-variant worker (isolated
   *   pages only; `worker.js` reads it off `?threads=` on its own URL). Default:
   *   `min(navigator.hardwareConcurrency, 16)` — measured optimum on a 32-hwc host
   *   (benchmarks/wasm_threads_2026-09-18.tsv). Benchmarks/tests use it; production callers
   *   should not.
   */
  constructor({ modelsBaseUrl, maxWorkers = 4, workerUrl, bundleVersions, gpu = 'auto', threads } = {}) {
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
    this._queue = []; // jobs waiting for a worker: {id, msg, canvas?, pinned?, priority, seq, resolve, reject, onDispatch, enqueuedAt}
    this._pending = new Map(); // id -> {resolve, reject, worker, msg, queuedMs}
    this._seq = 0;
    this._canvasSeq = 0;
    this._maxInflight = 0;
    this._completed = 0;
    this.threads = threads ?? null;
    const url = new URL(workerUrl || new URL('./worker.js', import.meta.url), typeof location !== 'undefined' ? location.href : undefined);
    url.searchParams.set('gpu', gpu);
    if (threads) {
      url.searchParams.set('threads', String(Math.max(1, Math.floor(threads))));
    }
    for (let i = 0; i < this.size; i++) this._spawn(url);
  }

  _spawn(url) {
    const w = new Worker(url, { type: 'module' });
    const ready = new Promise((resolve) => {
      const onFirst = (ev) => {
        const msg = ev.data;
        if (msg.type === 'ready') {
          w.removeEventListener('message', onFirst);
          resolve({ ok: true, variant: msg.variant, tier: msg.tier, gpu: msg.gpu || null, gpuError: msg.gpuError || null });
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
      if (msg.type === 'error' && msg.trapped && !p.msg._retried) {
        // The GPU path panicked inside wasm (panic=abort turns it into a trap that escapes the
        // decode promise). The worker survived and disabled its GPU context; retry the call —
        // it now runs on the CPU engine and its timings.gpuError records the trap. The worker
        // stays out of `_idle` until the retry settles.
        p.msg._retried = true;
        this._send(p.worker, p.msg);
        return;
      }
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
      // A worker-level error doesn't carry an `id`; reject this worker's in-flight job so the
      // caller doesn't hang forever — EXCEPT one already re-queued by the `trapped` path above
      // (the worker posts its `trapped` message before this event reaches us, so its retry is
      // in flight and its own result will settle it) — then take the worker out of rotation.
      for (const [id, p] of this._pending) {
        if (p.worker !== w || p.msg._retried) continue;
        p.reject(new Error(`worker error: ${ev.message || ev}`));
        this._pending.delete(id);
      }
      // Queued jobs pinned to this worker (canvas presents) can never run now — the canvas
      // lived inside it — so fail them loudly instead of leaving them queued forever.
      const stuck = this._queue.filter((j) => j.pinned === w);
      this._queue = this._queue.filter((j) => j.pinned !== w);
      for (const job of stuck) job.reject(new Error(`worker error: ${ev.message || ev}`));
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

  _send(worker, msg) {
    // The stream is NOT transferred (structured-clone copies it — a few hundred KB, nothing
    // next to a multi-ms decode): the copy lets a trapped GPU call be retried on the CPU
    // engine, and keeps the caller's buffer valid either way. Canvases MUST be transferred —
    // an OffscreenCanvas can only be cloned into a worker by transfer — and the worker keeps
    // it registered by canvasId so a retry needs no canvas.
    const transfer = [];
    if (msg.canvas) transfer.push(msg.canvas);
    worker.postMessage(msg, transfer);
    // The canvas is gone from this thread now (transfer neuters it); the worker registered it
    // under `canvasId`, so a retry sends the id alone. Clear it so a retry can't try to
    // re-transfer a detached object (that throws DataCloneError).
    msg.canvas = null;
  }

  _dispatch() {
    while (this._idle.length && this._queue.length) {
      // Best (idle worker, job) pair: a pinned job can only go to its owner worker, so the
      // best queued job may not be runnable on the first idle worker — scan them all.
      let bi = -1;
      let bj = -1;
      for (let wi = 0; wi < this._idle.length; wi++) {
        const worker = this._idle[wi];
        for (let i = 0; i < this._queue.length; i++) {
          const job = this._queue[i];
          if (job.pinned && job.pinned !== worker) continue;
          if (bj < 0 || compareJobs(job, this._queue[bj]) < 0) {
            bi = wi;
            bj = i;
          }
        }
      }
      if (bj < 0) break; // everything left is pinned to workers that are busy right now
      const worker = this._idle.splice(bi, 1)[0];
      const job = this._queue.splice(bj, 1)[0];
      if (job.canvas) {
        // First present for this canvas: whichever worker took the job owns it from now on
        // (a transferred OffscreenCanvas can only be used by the worker holding it).
        job.canvas.__jaiWorker = worker;
      }
      this._pending.set(job.id, {
        resolve: job.resolve,
        reject: job.reject,
        worker,
        msg: job.msg,
        queuedMs: performance.now() - job.enqueuedAt,
      });
      this._maxInflight = Math.max(this._maxInflight, this._pending.size);
      job.onDispatch?.();
      this._send(worker, job.msg);
    }
  }

  /** Live scheduling counters (tests and the demo's status readout). */
  stats() {
    return {
      size: this.size,
      threads: this.threads,
      isolated: this.isolated,
      inflight: this._pending.size,
      maxInflight: this._maxInflight,
      queued: this._queue.length,
      completed: this._completed,
    };
  }

  /**
   * Decode one JPEG AI codestream. `stream` may be an ArrayBuffer or any typed-array view; the
   * caller's buffer stays valid (it is cloned into the worker, not transferred — the copy also
   * lets a trapped GPU call be retried on the CPU engine).
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
    const msg = { type: 'decode', id, stream: buf, modelsBaseUrl: this.modelsBaseUrl, bundleVersions: this.bundleVersions };
    return new Promise((resolve, reject) => {
      this._queue.push({
        id,
        msg,
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

  /**
   * Decode `stream` and draw it on `canvas` — an `HTMLCanvasElement` (its control is transferred
   * offscreen inside the first call and cached on the element, so repeat calls reuse it) or a
   * detached `OffscreenCanvas` (transferred; neutered afterwards). On the `webgpu` package with
   * a GPU-presentable picture this is a pure GPU path — the RGBA texture is blitted onto the
   * canvas surface, pixels never touch CPU memory (`presented: 'gpu'`); otherwise the worker
   * decodes and `putImageData`s on a 2d context (`presented: '2d'`). The first present for a
   * canvas queues like a decode and binds the canvas to whichever worker runs it; repeats are
   * pinned to that worker. Calls on the same canvas are chained — presents to one surface are
   * ordered anyway, and the owner worker is only known once the first job is dispatched.
   * Takes the same `priority` / `onDispatch` options as `decode()`, plus `verify` to also
   * return a PNG of the canvas (tests only).
   * @returns {Promise<{width:number, height:number, presented:'gpu'|'2d', timings:object, png?:ArrayBuffer}>}
   */
  decodeToCanvas(stream, canvas, opts = {}) {
    const run = (canvas.__jaiPresentChain || Promise.resolve()).then(() => this._present(stream, canvas, opts));
    canvas.__jaiPresentChain = run.catch(() => {});
    return run;
  }

  _present(stream, canvas, { verify = false, priority, onDispatch } = {}) {
    const buf = ArrayBuffer.isView(stream)
      ? stream.buffer.slice(stream.byteOffset, stream.byteOffset + stream.byteLength)
      : stream;
    const id = ++this._seq;
    let canvasId;
    let off = null;
    let pinned = null;
    if (canvas.__jaiCanvasId != null) {
      // Already handed to a worker: that worker owns it (a transferred OffscreenCanvas cannot
      // be posted again), so this job is pinned to it and references the canvas by id.
      canvasId = canvas.__jaiCanvasId;
      pinned = canvas.__jaiWorker;
    } else {
      off = typeof OffscreenCanvas !== 'undefined' && canvas instanceof OffscreenCanvas
        ? canvas
        : canvas.transferControlToOffscreen();
      canvasId = ++this._canvasSeq;
      canvas.__jaiCanvasId = canvasId;
      // `__jaiWorker` is assigned in _dispatch once a worker takes the job.
    }
    const msg = { type: 'present', id, stream: buf, canvas: off, canvasId, modelsBaseUrl: this.modelsBaseUrl, bundleVersions: this.bundleVersions, verify };
    return new Promise((resolve, reject) => {
      this._queue.push({
        id,
        msg,
        canvas: off ? canvas : null,
        pinned,
        priority: priority ?? 0,
        seq: id,
        resolve,
        reject,
        onDispatch,
        enqueuedAt: performance.now(),
      });
      this._dispatch();
    }).then(
      (msg) => ({
        width: msg.width,
        height: msg.height,
        presented: msg.presented,
        png: msg.png || null,
        timings: { ...msg.timings, queued: msg.queued },
      }),
      (err) => {
        // The job was rejected before any worker took it (pool terminated, backlog drained):
        // no worker owns the canvas, so drop the marker. The element is still neutered —
        // transferControlToOffscreen() neuters on call — but a fresh call fails loudly at
        // transfer time rather than silently pinning to a worker that does not exist.
        if (!canvas.__jaiWorker) delete canvas.__jaiCanvasId;
        throw err;
      },
    );
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
