// Minimal PNG reader for the parity scripts and the browser tests: 8-bit gray / RGB / RGBA,
// non-interlaced. Returns { width, height, channels, data: Uint8Array } (rows top to bottom).
import { inflateSync } from 'node:zlib';

export function decodePng(buf) {
  const sig = [137, 80, 78, 71, 13, 10, 26, 10];
  if (!sig.every((b, i) => buf[i] === b)) throw new Error('not a PNG');
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  let pos = 8;
  let width = 0, height = 0, channels = 0;
  const idat = [];
  while (pos < buf.length) {
    const len = view.getUint32(pos);
    const type = String.fromCharCode(...buf.subarray(pos + 4, pos + 8));
    const body = buf.subarray(pos + 8, pos + 8 + len);
    if (type === 'IHDR') {
      width = view.getUint32(pos + 8);
      height = view.getUint32(pos + 12);
      const [depth, colour, , , interlace] = body.subarray(8);
      channels = { 0: 1, 2: 3, 6: 4 }[colour];
      if (depth !== 8 || !channels || interlace) throw new Error('unsupported PNG flavour');
    } else if (type === 'IDAT') idat.push(body);
    else if (type === 'IEND') break;
    pos += 12 + len;
  }
  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const data = new Uint8Array(stride * height);
  const paeth = (a, b, c) => {
    const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
    return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
  };
  for (let y = 0; y < height; y++) {
    const f = raw[y * (stride + 1)];
    const src = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const out = data.subarray(y * stride, (y + 1) * stride);
    const up = y ? data.subarray((y - 1) * stride, y * stride) : null;
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? out[x - channels] : 0;
      const b = up ? up[x] : 0;
      const c = up && x >= channels ? up[x - channels] : 0;
      const pred = f === 0 ? 0 : f === 1 ? a : f === 2 ? b : f === 3 ? (a + b) >> 1 : paeth(a, b, c);
      out[x] = (src[x] + pred) & 255;
    }
  }
  return { width, height, channels, data };
}

/// Histogram of absolute differences over the colour samples (alpha ignored).
export function diffSamples(a, b) {
  if (a.width !== b.width || a.height !== b.height) throw new Error('size mismatch');
  const hist = new Map();
  let total = 0, differing = 0, max = 0;
  const n = a.width * a.height;
  const nc = Math.min(a.channels, b.channels, 3);
  for (let i = 0; i < n; i++) {
    for (let c = 0; c < nc; c++) {
      const d = Math.abs(a.data[i * a.channels + c] - b.data[i * b.channels + c]);
      total++;
      if (d) {
        differing++;
        hist.set(d, (hist.get(d) || 0) + 1);
        if (d > max) max = d;
      }
    }
  }
  return { total, differing, max, hist: Object.fromEntries([...hist].sort((x, y) => x[0] - y[0])) };
}
