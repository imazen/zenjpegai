// Full-window viewer for decoded demo cards.
//
// installDemoViewer(opts) wires a click-to-open overlay onto the gallery. The overlay shows
// the decoded picture at a chosen zoom with a readout that states the sampling honestly:
// `1 image px = <r> device px` is the primary number — at "1:1 device" zoom r is exactly 1,
// the only way to inspect codec artefacts without resampling.
//
// Zoom model: `zoom` is image pixels per CSS pixel (imgW / cssW), shown as a percent. The
// canvas backing store is sized in DEVICE pixels: canvas.width = round(cssW * dpr) =
// imgW * dpr / zoom. Consequences: on a dpr-1 screen zoom 100% is natural size; on a dpr-2
// screen the 1:1-device view reads "zoom 200%" (two image pixels per CSS pixel). fit/fill
// compute cssW from the pane; wheel/pinch set cssW freely.
//
// Pixel sourcing (memory rule: at most ONE full-resolution RGBA copy per open viewer):
// - "ours" states reuse the card's already-presented canvas via drawImage — the worker-side
//   OffscreenCanvas (GPU blit or 2d putImageData) is the source, no re-decode. A decoded
//   picture is fully opaque, so a capture that comes back all-alpha-0 means this browser
//   can't sample a transferred canvas; we then fall back to pool.decode() once.
// - the other rate is decoded on demand through the card's own rate-button path
//   (queue-jump priority, timings shown in the card), then captured the same way.
// - "ref" states draw `_native/<variant>.native.png` (the reference decoder's PNG, shipped
//   only where the demo assets include it).
// The single staging canvas holds the current state at natural size; it is dropped on close.
//
// Chrome/accessibility: role=dialog + aria-modal, focus trap, Esc / click-outside / browser
// Back (history state) to close, arrow keys move between cards keeping zoom+pan, 'c' flips
// the compare rate, 'r' toggles the reference PNG, 'f'/'1'/'0' set zoom modes.

export function installDemoViewer({
  cards, // [{el, canvas, img, title, activeVariant() -> idx|-1, presentVariant(idx) -> Promise<result>}]
  decode, // (img, variant, bytes, opts) -> Promise<{width,height,rgba}> — pool.decode fallback when a presented canvas can't be sampled
  fetchStream, // (img, variant) -> Promise<ArrayBuffer> (demo's ?v= URL builder)
  nativeUrl, // (img, variant) -> string — _native/<stem>.native.png (or null if never shipped)
}) {
  const state = {
    open: false,
    cardIdx: -1,
    variantIdx: 0,
    source: 'ours', // 'ours' | 'ref'
    mode: 'fit', // 'fit' | 'device' | 'fill' | 'free'
    cssW: 0,
    cssH: 0,
    panX: 0,
    panY: 0, // CSS px offset of the canvas centre from the pane centre
    acquireSeq: 0,
    staging: document.createElement('canvas'),
    stagingFor: '', // "<cardIdx>:<variantIdx>:<source>" the staging currently holds
    refOk: new Map(), // variantIdx -> bool: does _native/<variant>.native.png exist
    lastFocus: null,
    pushed: false, // we own a history entry
  };
  state.staging.setAttribute('aria-hidden', 'true');

  // ---- DOM (built once, lazily) ------------------------------------------------------
  let root = null;
  let pane = null;
  let view = null; // the zoomable canvas
  let viewCtx = null;
  let readout = null;
  let status = null;
  let titleEl = null;
  let chips = {}; // source/state buttons: {v0, v1, ref}
  let zoomBtns = {};

  function buildDom() {
    root = document.createElement('div');
    root.className = 'jai-viewer';
    root.id = 'jai-viewer';
    root.setAttribute('role', 'dialog');
    root.setAttribute('aria-modal', 'true');
    root.tabIndex = -1;
    root.hidden = true;
    // The author `display: flex` on .jai-viewer beats the UA `[hidden]` rule, so the
    // closed overlay must be hidden imperatively too — otherwise it still intercepts
    // every click on the page.
    root.style.display = 'none';

    const bar = document.createElement('div');
    bar.className = 'jai-viewer-bar';
    titleEl = document.createElement('div');
    titleEl.className = 'jai-viewer-title';
    titleEl.id = 'jai-viewer-title';
    readout = document.createElement('div');
    readout.className = 'jai-viewer-readout';
    readout.id = 'jai-viewer-readout';
    readout.setAttribute('aria-live', 'polite');

    const zoom = document.createElement('div');
    zoom.className = 'jai-viewer-zoom';
    for (const [mode, label, hint] of [
      ['fit', 'fit', 'whole picture visible'],
      ['device', '1:1', 'one image pixel per device pixel'],
      ['fill', 'fill', 'fill the window'],
    ]) {
      const b = document.createElement('button');
      b.type = 'button';
      b.textContent = label;
      b.title = hint;
      b.addEventListener('click', () => setMode(mode));
      zoom.append(b);
      zoomBtns[mode] = b;
    }

    const compare = document.createElement('div');
    compare.className = 'jai-viewer-compare';
    const nVariants = Math.max(0, ...cards.map((c) => c.img.variants.length));
    for (let i = 0; i < Math.min(2, nVariants); i++) {
      const b = document.createElement('button');
      b.type = 'button';
      b.addEventListener('click', () => setVariant(i, 'ours'));
      compare.append(b);
      chips[`v${i}`] = b;
    }
    const refb = document.createElement('button');
    refb.type = 'button';
    refb.textContent = 'ref';
    refb.title = 'reference decoder PNG (_native/)';
    refb.hidden = true;
    refb.addEventListener('click', () => setVariant(state.variantIdx, state.source === 'ref' ? 'ours' : 'ref'));
    compare.append(refb);
    chips.ref = refb;

    const close = document.createElement('button');
    close.type = 'button';
    close.className = 'jai-viewer-close';
    close.textContent = '×';
    close.setAttribute('aria-label', 'close viewer');
    close.addEventListener('click', () => closeViewer());

    bar.append(titleEl, readout, zoom, compare, close);

    pane = document.createElement('div');
    pane.className = 'jai-viewer-pane';
    view = document.createElement('canvas');
    view.className = 'jai-viewer-canvas';
    view.setAttribute('role', 'img');
    pane.append(view);
    status = document.createElement('div');
    status.className = 'jai-viewer-status';
    status.hidden = true;
    pane.append(status);
    const hint = document.createElement('div');
    hint.className = 'jai-viewer-hint';
    hint.textContent = 'drag to pan · wheel/pinch to zoom · ←→ other images · c compare · r reference · Esc close';
    pane.append(hint);

    root.append(bar, pane);
    document.body.append(root);
    wireInput();
  }

  // ---- state helpers -------------------------------------------------------------------
  const card = () => cards[state.cardIdx];
  const variant = () => card().img.variants[state.variantIdx];
  const refKey = () => `${state.cardIdx}:${state.variantIdx}`;
  const imgW = () => stagingNaturalW();
  const imgH = () => stagingNaturalH();
  function stagingNaturalW() {
    return state.staging.width || variant().width;
  }
  function stagingNaturalH() {
    return state.staging.height || variant().height;
  }

  // HEAD-probe the reference PNG for the current card+variant once; a miss hides the chip.
  function probeRef() {
    if (state.refOk.get(refKey()) !== undefined) return;
    const url = nativeUrl(card().img, variant());
    if (!url) {
      state.refOk.set(refKey(), false);
      return;
    }
    fetch(url, { method: 'HEAD', cache: 'force-cache' })
      .then((r) => {
        state.refOk.set(refKey(), r.ok);
        if (state.open) updateChrome();
      })
      .catch(() => {});
  }

  // ---- acquire: fill the staging canvas for the current state --------------------------
  async function acquire() {
    const seq = ++state.acquireSeq;
    probeRef();
    const want = `${state.cardIdx}:${state.variantIdx}:${state.source}`;
    if (state.stagingFor === want) return;
    const v = variant();
    const c = card();
    status.textContent = 'loading…';
    status.hidden = false;
    try {
      if (state.source === 'ref') {
        const img = await loadImage(nativeUrl(c.img, v));
        if (seq !== state.acquireSeq) return;
        blit(img, img.naturalWidth || v.width, img.naturalHeight || v.height);
      } else {
        // The card's canvas already holds this variant's presented frame — capture it
        // instead of re-decoding. Only when the card shows a different variant (or none
        // yet) do we run the same path as the card's rate buttons: a queue-jump decode
        // whose timings land on the card, drawn onto that same canvas.
        if (c.activeVariant() !== state.variantIdx) {
          const r = await c.presentVariant(state.variantIdx);
          if (seq !== state.acquireSeq) return;
          if (!r) throw new Error('variant decode failed');
        }
        if (!capturePresented(c.canvas)) {
          // This browser can't sample a transferred canvas — pay one re-decode (same
          // queue-jump priority the card's rate buttons use).
          const bytes = await fetchStream(c.img, v);
          if (seq !== state.acquireSeq) return;
          const r = await decode(c.img, v, bytes, { priority: -1_000_000_000 });
          if (seq !== state.acquireSeq) return;
          const sctx = state.staging.getContext('2d');
          state.staging.width = r.width;
          state.staging.height = r.height;
          sctx.putImageData(new ImageData(r.rgba, r.width, r.height), 0, 0);
        }
      }
      state.stagingFor = want;
      status.hidden = true;
      render();
      updateChrome();
    } catch (err) {
      if (seq !== state.acquireSeq) return;
      status.textContent = `failed: ${err.message || err}`;
      if (state.source === 'ref') {
        // Reference PNG not shipped for this variant — drop back to our decode and hide ref.
        state.refOk.set(refKey(), false);
        updateChrome();
        setVariant(state.variantIdx, 'ours');
      }
    }
  }

  function blit(src, w, h) {
    state.staging.width = w;
    state.staging.height = h;
    const sctx = state.staging.getContext('2d');
    sctx.clearRect(0, 0, w, h);
    sctx.drawImage(src, 0, 0);
  }

  // drawImage on a canvas whose control was transferred offscreen: works in Chromium (the
  // placeholder canvas samples the presented frame). Verify: decoded output is opaque, so an
  // all-transparent capture means this engine can't do it -> caller falls back to re-decode.
  function capturePresented(canvas) {
    const w = canvas.width;
    const h = canvas.height;
    if (!w || !h) return false;
    state.staging.width = w;
    state.staging.height = h;
    const sctx = state.staging.getContext('2d');
    sctx.clearRect(0, 0, w, h);
    try {
      sctx.drawImage(canvas, 0, 0);
    } catch {
      return false;
    }
    for (const [fx, fy] of [
      [0.5, 0.5],
      [0.25, 0.25],
      [0.75, 0.75],
      [0.1, 0.9],
      [0.9, 0.1],
    ]) {
      const d = sctx.getImageData(Math.min(w - 1, (w * fx) | 0), Math.min(h - 1, (h * fy) | 0), 1, 1).data;
      if (d[3] !== 0) return true;
    }
    return false;
  }

  function loadImage(url) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`no reference PNG at ${url}`));
      img.src = url;
    });
  }

  // ---- render: staging -> view canvas, sized in device pixels ---------------------------
  function paneSize() {
    return { w: pane.clientWidth || innerWidth, h: pane.clientHeight || innerHeight };
  }

  function computeCssSize() {
    const dpr = window.devicePixelRatio || 1;
    const { w, h } = paneSize();
    const iw = imgW();
    const ih = imgH();
    if (state.mode === 'device') return { cssW: iw / dpr, cssH: ih / dpr };
    const fit = Math.min(w / iw, h / ih);
    const fill = Math.max(w / iw, h / ih);
    const s = state.mode === 'fill' ? fill : fit;
    if (state.mode === 'free') return { cssW: state.cssW, cssH: state.cssH };
    return { cssW: iw * s, cssH: ih * s };
  }

  function clampPan() {
    // Keep at least 96 CSS px of the picture inside the pane on each axis.
    const { w, h } = paneSize();
    const keep = Math.min(96, Math.min(state.cssW, state.cssH) / 2);
    const maxX = (state.cssW + w) / 2 - keep;
    const maxY = (state.cssH + h) / 2 - keep;
    state.panX = Math.max(-maxX, Math.min(maxX, state.panX));
    state.panY = Math.max(-maxY, Math.min(maxY, state.panY));
  }

  function render() {
    if (!state.open || !state.staging.width) return;
    const dpr = window.devicePixelRatio || 1;
    const { cssW, cssH } = computeCssSize();
    state.cssW = cssW;
    state.cssH = cssH;
    clampPan();
    // Canvas backing in DEVICE pixels: canvas.width = cssW * dpr = imgW * dpr / zoom.
    const dw = Math.max(1, Math.round(cssW * dpr));
    const dh = Math.max(1, Math.round(cssH * dpr));
    if (view.width !== dw) view.width = dw;
    if (view.height !== dh) view.height = dh;
    view.style.width = `${cssW}px`;
    view.style.height = `${cssH}px`;
    // pixelated only at >= 2:1 magnification (cssW >= 2*imgW); bilinear below.
    const magnified = cssW >= 2 * imgW() - 0.5;
    view.style.imageRendering = magnified ? 'pixelated' : 'auto';
    if (!viewCtx) viewCtx = view.getContext('2d');
    viewCtx.imageSmoothingEnabled = !magnified;
    viewCtx.clearRect(0, 0, dw, dh);
    viewCtx.drawImage(state.staging, 0, 0, dw, dh);
    // Canvas centre sits at pane centre + pan (the canvas is absolutely positioned at 0,0).
    const { w: pw, h: ph } = paneSize();
    view.style.transform = `translate(${(pw - cssW) / 2 + state.panX}px, ${(ph - cssH) / 2 + state.panY}px)`;

    // Readout. r = device px per image px; zoom = imgW*dpr/canvas.width (image px per CSS px).
    const r = dw / imgW();
    const zoom = (imgW() * dpr) / dw;
    readout.textContent =
      `1 image px = ${trim(r)} device px · dppx ${trim(dpr)} · zoom ${Math.round(zoom * 100)}% · ` +
      `${imgW()}×${imgH()} image · ${Math.round(cssW)}×${Math.round(cssH)} CSS px · ${dw}×${dh} device px`;
  }

  function trim(n) {
    return Number(n.toFixed(3)).toString();
  }

  // ---- chrome (title, chips, mode buttons, ref availability) -----------------------------
  function updateChrome() {
    const c = card();
    const v = variant();
    titleEl.textContent = `${c.title} — ${state.source === 'ref' ? 'reference ' : ''}${v.bpp} bpp`;
    view.setAttribute(
      'aria-label',
      `${c.title}: JPEG AI ${state.source === 'ref' ? 'reference decoder' : 'browser decode'} at ${v.bpp} bpp, ${v.width}×${v.height}`,
    );
    for (const [m, b] of Object.entries(zoomBtns)) b.setAttribute('aria-pressed', String(m === state.mode));
    for (let i = 0; i < 2; i++) {
      const b = chips[`v${i}`];
      if (!b) continue;
      const vv = c.img.variants[i];
      b.hidden = !vv;
      if (!vv) continue;
      b.textContent = `${vv.bpp}`;
      b.title = `our decode, ${vv.bpp} bpp`;
      b.setAttribute('aria-pressed', String(state.source === 'ours' && state.variantIdx === i));
    }
    chips.ref.hidden = state.refOk.get(refKey()) === false || !nativeUrl(c.img, v);
    chips.ref.textContent = `ref ${v.bpp}`;
    chips.ref.setAttribute('aria-pressed', String(state.source === 'ref'));
  }

  // ---- modes / states --------------------------------------------------------------------
  function setMode(mode) {
    state.mode = mode;
    if (mode !== 'free') {
      state.panX = 0;
      state.panY = 0;
    }
    render();
    updateChrome();
  }

  function setVariant(idx, source) {
    if (idx !== state.variantIdx) state.source = 'ours';
    if (source) state.source = source;
    state.variantIdx = idx;
    acquire();
  }

  function nextCard(d) {
    const n = cards.length;
    state.cardIdx = (state.cardIdx + d + n) % n;
    state.variantIdx = Math.max(0, card().activeVariant());
    state.source = 'ours';
    // Keep zoom + pan across cards (same-mode sizes differ per image).
    state.stagingFor = '';
    acquire();
  }

  // ---- open / close ----------------------------------------------------------------------
  async function openViewer(idx) {
    if (!root) buildDom();
    state.cardIdx = idx;
    const c = card();
    state.variantIdx = Math.max(0, c.activeVariant());
    state.source = 'ours';
    state.stagingFor = '';
    state.panX = 0;
    state.panY = 0;
    // Default: 1:1 device pixels when the picture fits at that scale, else fit.
    const dpr = window.devicePixelRatio || 1;
    const { w, h } = { w: innerWidth, h: innerHeight - 96 };
    const v = variant();
    state.mode = v.width <= w * dpr && v.height <= h * dpr ? 'device' : 'fit';
    state.lastFocus = document.activeElement;
    root.hidden = false;
    root.style.display = '';
    document.documentElement.classList.add('jai-viewer-open');
    state.open = true;
    // History: Back closes the viewer instead of leaving the page.
    if (!state.pushed) {
      try {
        history.pushState({ jaiViewer: true }, '');
        state.pushed = true;
      } catch {
        /* opaque-origin contexts may refuse pushState */
      }
    }
    (root.querySelector('.jai-viewer-close') || root).focus();
    acquire();
  }

  function closeViewer(fromPop = false) {
    if (!state.open) return;
    state.open = false;
    state.acquireSeq++; // cancel in-flight acquires
    root.hidden = true;
    root.style.display = 'none';
    document.documentElement.classList.remove('jai-viewer-open');
    // Release the one retained full-resolution copy.
    state.staging.width = 0;
    state.staging.height = 0;
    state.stagingFor = '';
    state.lastFocus?.focus?.();
    state.lastFocus = null;
    if (!fromPop && state.pushed && history.state?.jaiViewer) {
      state.pushed = false;
      history.back();
    } else {
      state.pushed = false;
    }
  }

  // ---- input ------------------------------------------------------------------------------
  function wireInput() {
    // Click/tap outside the canvas and bar closes — unless the click just ended a drag pan.
    let dragged = false;
    pane.addEventListener('click', (e) => {
      if (dragged) {
        dragged = false;
        return;
      }
      if (e.target === pane || e.target.classList.contains('jai-viewer-hint')) closeViewer();
    });
    root.addEventListener('click', (e) => {
      if (e.target === root) closeViewer();
    });

    // Wheel zoom anchored at the cursor.
    pane.addEventListener(
      'wheel',
      (e) => {
        e.preventDefault();
        const factor = Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015));
        zoomAt(e.clientX, e.clientY, factor);
      },
      { passive: false },
    );

    // Drag pan + two-pointer pinch.
    const pointers = new Map();
    let pinch0 = null;
    let dragMoved = 0;
    pane.addEventListener('pointerdown', (e) => {
      pane.setPointerCapture(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      dragMoved = 0;
      if (pointers.size === 2) {
        const [a, b] = [...pointers.values()];
        const rect = pane.getBoundingClientRect();
        pinch0 = {
          d: Math.hypot(a.x - b.x, a.y - b.y),
          cssW: state.cssW,
          mid: { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top },
        };
      }
    });
    pane.addEventListener('pointermove', (e) => {
      if (!pointers.has(e.pointerId)) return;
      const prev = pointers.get(e.pointerId);
      pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pointers.size === 2 && pinch0) {
        const [a, b] = [...pointers.values()];
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (pinch0.d > 0 && d > 0 && state.cssW) {
          const rect = pane.getBoundingClientRect();
          const mid = { x: (a.x + b.x) / 2 - rect.left, y: (a.y + b.y) / 2 - rect.top };
          state.panX += mid.x - pinch0.mid.x;
          state.panY += mid.y - pinch0.mid.y;
          pinch0.mid = mid;
          state.cssW = pinch0.cssW * (d / pinch0.d);
          state.cssH = state.cssW * (imgH() / imgW());
          state.mode = 'free';
          clampPan();
          render();
          updateChrome();
        }
        dragged = true;
        return;
      }
      state.mode = 'free';
      state.panX += e.clientX - prev.x;
      state.panY += e.clientY - prev.y;
      dragMoved += Math.abs(e.clientX - prev.x) + Math.abs(e.clientY - prev.y);
      if (dragMoved > 4) dragged = true;
      clampPan();
      render();
    });
    const drop = (e) => {
      pointers.delete(e.pointerId);
      if (pointers.size < 2) pinch0 = null;
    };
    pane.addEventListener('pointerup', drop);
    pane.addEventListener('pointercancel', drop);

    document.addEventListener('keydown', onKey);
    window.addEventListener('popstate', onPop);
    window.addEventListener('resize', () => {
      if (state.open) render();
    });
    watchDpr();
  }

  function zoomAt(clientX, clientY, factor) {
    if (!state.cssW || !state.staging.width) return;
    const rect = pane.getBoundingClientRect();
    const { w, h } = paneSize();
    // Image point under the cursor stays under the cursor.
    const cx = clientX - rect.left - w / 2 - state.panX;
    const cy = clientY - rect.top - h / 2 - state.panY;
    const scale = state.cssW / imgW();
    const imgX = cx / scale;
    const imgY = cy / scale;
    const dpr = window.devicePixelRatio || 1;
    let newW = state.cssW * factor;
    // Bounds: at least 16 device px on the long side, at most a 16384-px backing store.
    newW = Math.max(Math.min(imgW(), imgH()) * 0.02, Math.min(16384 / dpr, newW));
    const newScale = newW / imgW();
    state.cssW = newW;
    state.cssH = newW * (imgH() / imgW());
    state.panX = clientX - rect.left - w / 2 - imgX * newScale;
    state.panY = clientY - rect.top - h / 2 - imgY * newScale;
    state.mode = 'free';
    clampPan();
    render();
    updateChrome();
  }

  // dpr changes (moving the window across screens, browser zoom): re-render + update readout.
  function watchDpr() {
    const mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
    mq.addEventListener('change', () => {
      if (state.open) render();
      watchDpr();
    }, { once: true });
  }

  function onKey(e) {
    if (!state.open) return;
    if (e.key === 'Tab') {
      trapFocus(e);
      return;
    }
    switch (e.key) {
      case 'Escape':
        e.preventDefault();
        closeViewer();
        break;
      case 'ArrowLeft':
      case 'ArrowUp':
        e.preventDefault();
        nextCard(-1);
        break;
      case 'ArrowRight':
      case 'ArrowDown':
        e.preventDefault();
        nextCard(1);
        break;
      case 'c':
      case 'C': {
        const other = state.variantIdx === 0 && card().img.variants.length > 1 ? 1 : 0;
        setVariant(other, 'ours');
        break;
      }
      case 'r':
      case 'R':
        if (!chips.ref.hidden) setVariant(state.variantIdx, state.source === 'ref' ? 'ours' : 'ref');
        break;
      case 'f':
      case 'F':
        setMode('fit');
        break;
      case '1':
        setMode('device');
        break;
      case '0':
        setMode('fit');
        break;
      default:
    }
  }

  function trapFocus(e) {
    const focusables = [...root.querySelectorAll('button:not([hidden]), [tabindex]:not([tabindex="-1"])')].filter(
      (el) => !el.hidden && el.getClientRects().length > 0,
    );
    if (!focusables.length) return;
    const first = focusables[0];
    const last = focusables[focusables.length - 1];
    const active = document.activeElement;
    if (e.shiftKey && (active === first || !root.contains(active))) {
      e.preventDefault();
      last.focus();
    } else if (!e.shiftKey && (active === last || !root.contains(active))) {
      e.preventDefault();
      first.focus();
    }
  }

  function onPop() {
    if (state.open) closeViewer(true);
  }

  // ---- wire card clicks ---------------------------------------------------------------------
  cards.forEach((c, i) => {
    c.el.addEventListener('click', (e) => {
      if (e.target.closest('button, a')) return;
      openViewer(i);
    });
    c.el.classList.add('jai-card-clickable');
  });

  return { open: openViewer, close: closeViewer, state };
}
