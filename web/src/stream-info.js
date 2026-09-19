// Reads just the two PIH fields `ensureModels` needs — `model_id` and the first
// `synthesis_transform` (the operating point) — straight from the codestream bytes, without
// wasm. Lets the polyfill prefetch the right model bundles while the worker is still loading
// its module (the demo gets the same fields from manifest.json and never needs this).
//
// Port of the container framing (`src/container.rs`: `SOC | PIH | size ue(v) | payload`) and
// of the LEADING fields of `src/header.rs::PictureHeader::parse`, up to `model_id` — nothing
// past it is read. Bit order is MSB-first (`src/bitio.rs`), `bounded(v)` is
// `ceil(log2(max+1))` bits. Any malformed input returns null rather than throwing: prefetch
// is a best-effort optimisation and the worker's `ensureModels` is always the fallback.
// Cross-checked against `info()` (the wasm parse) for every demo stream by
// tests/prefetch.spec.ts.

const SOC = 0xff80;
const PIH = 0xff82;
const OP_NAMES = ['sop', 'bop', 'hop'];

class BitReader {
  constructor(u8, pos = 0) {
    this.d = u8;
    this.pos = pos; // bits
  }
  bits(n) {
    let v = 0;
    for (let left = n; left > 0; ) {
      if (this.pos >= this.d.length * 8) throw new Error('eof');
      const byte = this.d[this.pos >> 3];
      const bitOff = this.pos & 7;
      const avail = 8 - bitOff;
      const take = Math.min(avail, left);
      v = (v << take) | ((byte >> (avail - take)) & ((1 << take) - 1));
      this.pos += take;
      left -= take;
    }
    return v >>> 0;
  }
  // Unsigned Exp-Golomb, order 0 (`read_ue`).
  ue() {
    let k = 0;
    while (!this.bits(1)) {
      if (++k > 32) throw new Error('exp-golomb prefix too long');
    }
    return (1 << k) + this.bits(k) - 1;
  }
  // `read_bounded`: ceil(log2(max + 1)) bits == bits needed to represent max.
  bounded(max) {
    return this.bits(max === 0 ? 0 : 32 - Math.clz32(max));
  }
  get bytePos() {
    return Math.ceil(this.pos / 8);
  }
}

const u16be = (d, i) => (d[i] << 8) | d[i + 1];

/**
 * @param {ArrayBuffer|Uint8Array} bytes - a JPEG AI codestream.
 * @returns {{modelId:number, op:string}|null} the model index and first operating point,
 *   or null if the stream does not parse far enough (never throws).
 */
export function probeStreamInfo(bytes) {
  try {
    const d = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    if (d.length < 8 || u16be(d, 0) !== SOC || u16be(d, 2) !== PIH) return null;
    // `size ue(v)` is byte-aligned at offset 4; the PIH payload starts at the byte boundary
    // after it (`pos += r.bytes_consumed()` in container.rs).
    const hdr = new BitReader(d, 32);
    hdr.ue(); // PIH size — its value is not needed, only its length
    const r = new BitReader(d.subarray(hdr.bytePos));

    r.bits(4); // stream_profile_idc
    r.bits(4); // decoder_profile_id
    const nTransforms = r.bits(4) + 1;
    const op = r.bits(4); // first synthesis_transform = the operating point decode uses
    for (let i = 1; i < nTransforms; i++) r.bits(4);
    r.bits(8); // level_idc
    r.bounded(0xffff); // img_width_minus64
    r.bounded(0xffff); // img_height_minus64
    r.bits(6); // diff_display_width
    r.bits(6); // diff_display_height
    r.bounded(4); // bit_depth_idc (BIT_DEPTHS has 5 entries)
    // s_ver/s_hor are `read_bits(1) + 1`: bit 0 -> factor 1, and ONLY then are
    // c_ver/c_hor coded at all (a bit of their own); bit 1 -> factor 2, inferred.
    const sVerMinus1 = r.bits(1);
    const sHorMinus1 = r.bits(1);
    if (!sVerMinus1) r.bits(1); // c_ver coded only when s_ver == 1
    if (!sHorMinus1) r.bits(1); // c_hor coded only when s_hor == 1
    const colourTransform = r.bounded(2);
    if (colourTransform === 2) r.bits(96); // custom: 9x8 matrix + 3x8 offset
    else if (colourTransform > 2) return null;
    const modelId = r.bounded(15); // MultiToolsEngine(max_models_count=16)
    const opName = OP_NAMES[op];
    if (modelId > 15 || !opName) return null;
    return { modelId, op: opName };
  } catch {
    return null;
  }
}
