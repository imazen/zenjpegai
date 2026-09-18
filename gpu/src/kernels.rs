//! WGSL kernel sources.
//!
//! # Data layout
//!
//! Feature maps live in storage buffers as `array<vec4<f32>>` in **HWC4** order: element
//! `(y * W + x) * C4 + c4` holds channels `4 * c4 .. 4 * c4 + 3` of pixel `(x, y)`, with
//! `C4 = ceil(C / 4)` and zero in the lanes past `C`. All channels of a pixel are contiguous, so
//! the inner loop of a convolution walks memory linearly, and one `mat4x4 * vec4` product does
//! 16 multiply-adds (4 input channels into 4 output channels).
//!
//! Convolution weights are pre-packed as `mat4x4` in exactly the order the shader walks them:
//! `[out group of OB blocks][ky][kx][in block][block within the out group]`. One invocation
//! computes `OB` output blocks (up to 16 output channels) of one pixel, so every input load is
//! shared by `OB` matrix products (register tiling; it quarters the memory traffic of the 3x3
//! and 1x1 layers that dominate the synthesis transforms).
//!
//! Kernel geometry (kernel size, stride, `OB`, fused activations) is baked into the source as
//! constants, so the tap loops unroll; sizes that vary per tile come from a uniform block of 16
//! `u32`.

/// Workgroup edge of the 2-D kernels (8 x 8 = 64 invocations; every WebGPU device allows 256).
pub const WG: u32 = 8;
/// Workgroup size of the 1-D (pointwise / reduction) kernels.
pub const WG1: u32 = 64;

/// Uniform block: 16 named `u32`.
fn params(names: &[&str]) -> String {
    assert!(names.len() <= 16);
    let mut s = String::from("struct P {\n");
    for i in 0..16 {
        match names.get(i) {
            Some(n) => s.push_str(&format!("  {n}: u32,\n")),
            None => s.push_str(&format!("  pad{i}: u32,\n")),
        }
    }
    s.push_str("}\n@group(0) @binding(0) var<uniform> p: P;\n");
    s
}

/// Fused activation applied to a convolution's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Act {
    None,
    Relu,
}

/// A second operand combined with the convolution's result as it is stored, saving the read,
/// write and dispatch of a separate pointwise pass. The operand is indexed exactly like `dst`,
/// and the arithmetic is the same as the standalone [`Pointwise`] operation, so fusing does not
/// change a single rounding.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Res {
    /// `dst = acc + res` (residual block).
    Add,
    /// `dst = res * (1 + acc)` (ResAU gate).
    Gate,
}

/// Compile-time geometry of a convolution kernel.
#[derive(Clone, Copy, Debug)]
pub struct ConvVariant {
    pub kh: u32,
    pub kw: u32,
    pub stride: u32,
    /// Output blocks (of 4 channels) per invocation: 1, 2 or 4.
    pub ob: u32,
    /// Workgroup edge. A wider workgroup shares each broadcast weight block between more
    /// invocations, which pays off exactly when the weights are the bulk of the traffic (many
    /// input blocks) and costs edge lanes when they are not; [`crate::plan`] picks it per layer.
    pub wg: u32,
    pub act: Act,
    /// Apply ReLU6 to every input sample as it is loaded (the ResAU gate's first step).
    pub pre_relu6: bool,
    /// Fused second operand (applied after `act`).
    pub res: Option<Res>,
}

impl ConvVariant {
    pub fn key(&self) -> String {
        format!(
            "conv_k{}x{}_s{}_ob{}_wg{}_{:?}_{}_{:?}",
            self.kh, self.kw, self.stride, self.ob, self.wg, self.act, self.pre_relu6, self.res
        )
    }
}

fn acc_decl(ob: u32, bias_index: &str) -> String {
    (0..ob)
        .map(|i| format!("  var a{i} = bias[{bias_index} + {i}u];\n"))
        .collect()
}

fn acc_step(ob: u32) -> String {
    (0..ob)
        .map(|i| format!("        a{i} += wt[w + {i}u] * v;\n"))
        .collect()
}

fn acc_store(ob: u32, act: Act, res: Option<Res>) -> String {
    (0..ob)
        .map(|i| {
            let v = match act {
                Act::None => format!("a{i}"),
                Act::Relu => format!("max(a{i}, vec4<f32>(0.0))"),
            };
            match res {
                None => format!("  dst[o + {i}u] = {v};\n"),
                Some(Res::Add) => format!("  dst[o + {i}u] = {v} + res[o + {i}u];\n"),
                Some(Res::Gate) => {
                    format!("  dst[o + {i}u] = res[o + {i}u] * (vec4<f32>(1.0) + {v});\n")
                }
            }
        })
        .collect()
}

const CONV_BINDINGS: &str = "
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> wt: array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> bias: array<vec4<f32>>;
";

/// Direct convolution, zero padding by bounds check, any group count whose groups are whole
/// blocks. One invocation computes `ob` output blocks of one pixel.
///
/// Register tiling over *pixels* (2x2 and 2x1 outputs per invocation, which amortises the
/// `4 * ob` weight loads of every tap over that many products) was implemented and measured on
/// an RTX 2080: it helps SOP (1024 x 1024 synthesis 8.48 -> 7.23 ms device) and loses on the
/// operating points that matter more (BOP 22.4 -> 26.3 ms, HOP 555 -> 711 ms), so the kernel
/// computes one pixel per invocation. The loss tracks register pressure — `ob * px * py`
/// accumulators — not the load count, so the lever to try next is fewer channel blocks per
/// invocation together with a pixel tile, not a larger tile.
/// Params: `in_h in_w in_c4 out_h out_w out_c4 pad_y pad_x icg4 ocg4`.
pub fn conv(v: &ConvVariant) -> String {
    let load = if v.pre_relu6 {
        "clamp(src[sb + ic], vec4<f32>(0.0), vec4<f32>(6.0))"
    } else {
        "src[sb + ic]"
    };
    let res_binding = match v.res {
        None => "",
        Some(_) => "@group(0) @binding(5) var<storage, read> res: array<vec4<f32>>;\n",
    };
    format!(
        "{params}{CONV_BINDINGS}{res_binding}
const KH: u32 = {kh}u; const KW: u32 = {kw}u; const STRIDE: u32 = {stride}u; const OB: u32 = {ob}u;
@compute @workgroup_size({wg}, {wg}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.out_w || gid.y >= p.out_h) {{ return; }}
  let og = gid.z;
  let oc4 = og * OB;
  let icb = (oc4 / p.ocg4) * p.icg4;
{decl}  for (var ky = 0u; ky < KH; ky++) {{
    let iy = i32(gid.y * STRIDE + ky) - i32(p.pad_y);
    if (iy < 0 || iy >= i32(p.in_h)) {{ continue; }}
    for (var kx = 0u; kx < KW; kx++) {{
      let ix = i32(gid.x * STRIDE + kx) - i32(p.pad_x);
      if (ix < 0 || ix >= i32(p.in_w)) {{ continue; }}
      let sb = (u32(iy) * p.in_w + u32(ix)) * p.in_c4 + icb;
      let wb = ((og * KH + ky) * KW + kx) * p.icg4 * OB;
      for (var ic = 0u; ic < p.icg4; ic++) {{
        let v = {load};
        let w = wb + ic * OB;
{step}      }}
    }}
  }}
  let o = (gid.y * p.out_w + gid.x) * p.out_c4 + oc4;
{store}}}
",
        params = params(&[
            "in_h", "in_w", "in_c4", "out_h", "out_w", "out_c4", "pad_y", "pad_x", "icg4", "ocg4"
        ]),
        kh = v.kh,
        kw = v.kw,
        stride = v.stride,
        ob = v.ob,
        wg = v.wg,
        decl = acc_decl(v.ob, "oc4"),
        step = acc_step(v.ob),
        store = acc_store(v.ob, v.act, v.res),
    )
}

/// Transposed convolution (no groups): output `(oy, ox)` gathers the taps with
/// `oy + pad - ky` divisible by the stride. Only those taps are visited — `ky` starts at
/// `(oy + pad) mod stride` and steps by `stride`, which is exactly the set the divisibility
/// test used to select, in the same ascending order, so the sum is bit-identical to testing all
/// `K * K` of them. At stride 2 that is 4 taps of 16 for `K = 4` and at most 4 of 9 for `K = 3`.
/// Params: `in_h in_w in_c4 out_h out_w out_c4 pad`.
pub fn conv_transpose(k: u32, stride: u32, ob: u32) -> String {
    format!(
        "{params}{CONV_BINDINGS}
const K: u32 = {k}u; const STRIDE: u32 = {stride}u; const OB: u32 = {ob}u;
@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.out_w || gid.y >= p.out_h) {{ return; }}
  let og = gid.z;
  let oc4 = og * OB;
  let by = gid.y + p.pad;
  let bx = gid.x + p.pad;
{decl}  for (var ky = by % STRIDE; ky < K; ky += STRIDE) {{
    if (ky > by) {{ break; }}
    let iy = (by - ky) / STRIDE;
    if (iy >= p.in_h) {{ continue; }}
    for (var kx = bx % STRIDE; kx < K; kx += STRIDE) {{
      if (kx > bx) {{ break; }}
      let ix = (bx - kx) / STRIDE;
      if (ix >= p.in_w) {{ continue; }}
      let sb = (iy * p.in_w + ix) * p.in_c4;
      let wb = ((og * K + ky) * K + kx) * p.in_c4 * OB;
      for (var ic = 0u; ic < p.in_c4; ic++) {{
        let v = src[sb + ic];
        let w = wb + ic * OB;
{step}      }}
    }}
  }}
  let o = (gid.y * p.out_w + gid.x) * p.out_c4 + oc4;
{store}}}
",
        params = params(&["in_h", "in_w", "in_c4", "out_h", "out_w", "out_c4", "pad"]),
        decl = acc_decl(ob, "oc4"),
        step = acc_step(ob),
        store = acc_store(ob, Act::None, None),
    )
}

/// Depthwise 3x3, stride 1, padding 1, no bias. Weights: `vec4` per `(block, ky, kx)`.
/// Params: `h w c4`.
///
/// One channel block per workgroup, so the `(WG + 2)^2` input halo the workgroup needs is
/// 1600 bytes of workgroup memory. Staging it there turns the nine reads per output pixel into
/// `(WG + 2)^2 / WG^2` (1.56 at `WG = 8`) — this kernel does one multiply-add per 16 bytes read,
/// so it is bound by nothing else. Out-of-picture halo samples are stored as zero and their taps
/// still added, which is bit-identical to skipping them: adding an exact `w * 0.0` changes no
/// finite accumulator.
pub fn depthwise3x3() -> String {
    let halo = WG + 2;
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> wt: array<vec4<f32>>;
const HALO: u32 = {halo}u;
var<workgroup> tile: array<vec4<f32>, {halo_sq}u>;
@compute @workgroup_size({WG}, {WG}, 1)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {{
  let c = gid.z;
  let x0 = i32(wid.x * {WG}u) - 1;
  let y0 = i32(wid.y * {WG}u) - 1;
  for (var i = lid.y * {WG}u + lid.x; i < HALO * HALO; i += {WG}u * {WG}u) {{
    let sy = y0 + i32(i / HALO);
    let sx = x0 + i32(i % HALO);
    var v = vec4<f32>(0.0);
    if (sy >= 0 && sy < i32(p.h) && sx >= 0 && sx < i32(p.w)) {{
      v = src[(u32(sy) * p.w + u32(sx)) * p.c4 + c];
    }}
    tile[i] = v;
  }}
  workgroupBarrier();
  if (gid.x >= p.w || gid.y >= p.h) {{ return; }}
  var acc = vec4<f32>(0.0);
  for (var ky = 0u; ky < 3u; ky++) {{
    for (var kx = 0u; kx < 3u; kx++) {{
      acc += wt[c * 9u + ky * 3u + kx] * tile[(lid.y + ky) * HALO + lid.x + kx];
    }}
  }}
  dst[(gid.y * p.w + gid.x) * p.c4 + c] = acc;
}}
",
        params = params(&["h", "w", "c4"]),
        halo_sq = halo * halo,
    )
}

/// Windowed channel copy: `dst[:, y, x][dc4 ..] = src[:, y + sy, x + sx][sc4off .. + n4]` for
/// the `h x w` window. Does crops, latent tile windows, channel slices and concatenation.
/// Params: `h w n4 src_w src_c4 sx sy src_c4off dst_w dst_c4 dst_c4off`.
pub fn copy_channels() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.w || gid.y >= p.h) {{ return; }}
  let s = ((gid.y + p.sy) * p.src_w + gid.x + p.sx) * p.src_c4 + p.src_c4off;
  let d = (gid.y * p.dst_w + gid.x) * p.dst_c4 + p.dst_c4off;
  for (var c = 0u; c < p.n4; c++) {{ dst[d + c] = src[s + c]; }}
}}
",
        params = params(&[
            "h",
            "w",
            "n4",
            "src_w",
            "src_c4",
            "sx",
            "sy",
            "src_c4off",
            "dst_w",
            "dst_c4",
            "dst_c4off"
        ]),
    )
}

/// In-place pointwise operations over a whole feature map.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pointwise {
    /// `a = max(a, 0)`
    Relu,
    /// `a += b`
    Add,
    /// `a = a * (1 + b)` (ResAU gate; two operations like the reference, not an FMA on purpose,
    /// though a driver may still fuse it)
    Gate,
    /// `a = c + a * sigmoid(b)` (CAB)
    SigmoidMulAdd,
}

impl Pointwise {
    pub fn inputs(self) -> usize {
        match self {
            Self::Relu => 0,
            Self::Add | Self::Gate => 1,
            Self::SigmoidMulAdd => 2,
        }
    }
}

/// Params: `n row` (`n` elements, `row` = elements per dispatch row).
pub fn pointwise(op: Pointwise) -> String {
    let (extra, expr) = match op {
        Pointwise::Relu => ("", "max(a[i], vec4<f32>(0.0))"),
        Pointwise::Add => (
            "@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;\n",
            "a[i] + b[i]",
        ),
        Pointwise::Gate => (
            "@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;\n",
            "a[i] * (vec4<f32>(1.0) + b[i])",
        ),
        Pointwise::SigmoidMulAdd => (
            "@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;\n@group(0) @binding(3) var<storage, read> c: array<vec4<f32>>;\n",
            "c[i] + a[i] * (vec4<f32>(1.0) / (vec4<f32>(1.0) + exp(-b[i])))",
        ),
    };
    format!(
        "{params}
@group(0) @binding(1) var<storage, read_write> a: array<vec4<f32>>;
{extra}@compute @workgroup_size({WG1}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = gid.y * p.row + gid.x;
  if (gid.x >= p.row || i >= p.n) {{ return; }}
  a[i] = {expr};
}}
",
        params = params(&["n", "row"]),
    )
}

/// `dst[dst_off ..][..n4] = elu(a[a_off ..]) * b[b_off ..]` per pixel, in blocks (the gated
/// feed-forward of the transformer block; `a` and `b` may be the same map).
/// Params: `n row n4 a_c4 a_off b_c4 b_off dst_c4 dst_off` with `n` pixels.
pub fn elu_gate() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size({WG1}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = gid.y * p.row + gid.x;
  if (gid.x >= p.row || i >= p.n) {{ return; }}
  for (var c = 0u; c < p.n4; c++) {{
    let x = a[i * p.a_c4 + p.a_off + c];
    let e = select(exp(min(x, vec4<f32>(0.0))) - vec4<f32>(1.0), x, x > vec4<f32>(0.0));
    dst[i * p.dst_c4 + p.dst_off + c] = e * b[i * p.b_c4 + p.b_off + c];
  }}
}}
",
        params = params(&[
            "n", "row", "n4", "a_c4", "a_off", "b_c4", "b_off", "dst_c4", "dst_off"
        ]),
    )
}

/// PixelShuffle(r) between HWC4 maps. Params: `out_h out_w out_c out_c4 in_w in_c4`.
pub fn pixel_shuffle(r: u32) -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec4<f32>>;
const R: u32 = {r}u;
fn fetch(px: u32, ch: u32) -> f32 {{ return src[px * p.in_c4 + ch / 4u][ch % 4u]; }}
@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.out_w || gid.y >= p.out_h) {{ return; }}
  let px = (gid.y / R) * p.in_w + gid.x / R;
  let sub = (gid.y % R) * R + gid.x % R;
  let c0 = gid.z * 4u;
  var v = vec4<f32>(0.0);
  for (var l = 0u; l < 4u; l++) {{
    if (c0 + l < p.out_c) {{ v[l] = fetch(px, (c0 + l) * R * R + sub); }}
  }}
  dst[(gid.y * p.out_w + gid.x) * p.out_c4 + gid.z] = v;
}}
",
        params = params(&["out_h", "out_w", "out_c", "out_c4", "in_w", "in_c4"]),
    )
}

/// Last step of a synthesis transform: PixelShuffle(r) of the tile's final feature map,
/// `x * 128 + 127.5` clamped to `[0, 255]`, written for the tile's *core* only, straight into
/// the picture-sized planar `f32` buffer. Invocation `(x, y, z)` is core pixel `(x, y)` of
/// output plane `z`. Params: `core_w core_h core_x core_y off_x off_y in_w in_c4 pic_w pic_h
/// plane0`.
pub fn emit(r: u32) -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> rec: array<f32>;
const R: u32 = {r}u;
@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.core_w || gid.y >= p.core_h) {{ return; }}
  let tx = gid.x + p.off_x;
  let ty = gid.y + p.off_y;
  let ch = gid.z * R * R + (ty % R) * R + tx % R;
  let v = src[((ty / R) * p.in_w + tx / R) * p.in_c4 + ch / 4u][ch % 4u];
  let o = ((p.plane0 + gid.z) * p.pic_h + gid.y + p.core_y) * p.pic_w + gid.x + p.core_x;
  rec[o] = clamp(v * 128.0 + 127.5, 0.0, 255.0);
}}
",
        params = params(&[
            "core_w", "core_h", "core_x", "core_y", "off_x", "off_y", "in_w", "in_c4", "pic_w",
            "pic_h", "plane0"
        ]),
    )
}

/// Layer norm over the channels of each pixel (`eps = 1e-5`). `wb` holds the weight
/// blocks followed by the bias blocks. Params: `n row c c4`.
pub fn layer_norm() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> wb: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size({WG1}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = gid.y * p.row + gid.x;
  if (gid.x >= p.row || i >= p.n) {{ return; }}
  let base = i * p.c4;
  // The lanes past `c` are zero, so they do not disturb the sums.
  var s = vec4<f32>(0.0);
  for (var c = 0u; c < p.c4; c++) {{ s += a[base + c]; }}
  let mean = (s.x + s.y + s.z + s.w) / f32(p.c);
  var q = vec4<f32>(0.0);
  for (var c = 0u; c < p.c4; c++) {{
    var d = a[base + c] - vec4<f32>(mean);
    let live = min(p.c - c * 4u, 4u);
    for (var l = live; l < 4u; l++) {{ d[l] = 0.0; }}
    q += d * d;
  }}
  let inv = 1.0 / sqrt((q.x + q.y + q.z + q.w) / f32(p.c) + 1e-5);
  for (var c = 0u; c < p.c4; c++) {{
    dst[base + c] = (a[base + c] - vec4<f32>(mean)) * inv * wb[c] + wb[p.c4 + c];
  }}
}}
",
        params = params(&["n", "row", "c", "c4"]),
    )
}

/// Pixels per first-level reduction chunk.
pub const CHUNK: u32 = 256;

/// First level of the attention Gram matrix: for block pair `(i4, j4)` of a head and one chunk
/// of `CHUNK` pixels, `M[col j][row i] = sum q_i * k_j`. With `qoff == koff` and only the
/// diagonal used it also yields the squared norms. Invocation `(x, y)` = (chunk, pair index),
/// pair index = `(head * hc4 + i4) * hc4 + j4`.
/// Params: `n chunks pairs src_c4 qoff koff hc4`.
pub fn gram_chunks() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> part: array<mat4x4<f32>>;
@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.chunks || gid.y >= p.pairs) {{ return; }}
  let j4 = gid.y % p.hc4;
  let i4 = (gid.y / p.hc4) % p.hc4;
  let head = gid.y / (p.hc4 * p.hc4);
  let qi = p.qoff + head * p.hc4 + i4;
  let kj = p.koff + head * p.hc4 + j4;
  let first = gid.x * {CHUNK}u;
  let last = min(first + {CHUNK}u, p.n);
  var c0 = vec4<f32>(0.0); var c1 = vec4<f32>(0.0); var c2 = vec4<f32>(0.0); var c3 = vec4<f32>(0.0);
  for (var px = first; px < last; px++) {{
    let q = src[px * p.src_c4 + qi];
    let k = src[px * p.src_c4 + kj];
    c0 += q * k.x; c1 += q * k.y; c2 += q * k.z; c3 += q * k.w;
  }}
  part[gid.y * p.chunks + gid.x] = mat4x4<f32>(c0, c1, c2, c3);
}}
",
        params = params(&["n", "chunks", "pairs", "src_c4", "qoff", "koff", "hc4"]),
    )
}

/// Second level: sum each pair's chunk results, 16 at a time then the group sums (pairwise-ish
/// order keeps the float error far below the first level's). Params: `chunks pairs`.
pub fn gram_reduce() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> part: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read_write> gram: array<mat4x4<f32>>;
@compute @workgroup_size({WG1}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.pairs) {{ return; }}
  let z = vec4<f32>(0.0);
  var total = mat4x4<f32>(z, z, z, z);
  for (var g = 0u; g < p.chunks; g += 16u) {{
    var s = mat4x4<f32>(z, z, z, z);
    for (var c = g; c < min(g + 16u, p.chunks); c++) {{ s += part[gid.x * p.chunks + c]; }}
    total += s;
  }}
  gram[gid.x] = total;
}}
",
        params = params(&["chunks", "pairs"]),
    )
}

/// Attention matrix: `softmax_j(q_i . k_j / (|q_i| |k_j|) * temperature[head])`, written as the
/// `mat4x4` blocks [`attention_apply`] multiplies with. One invocation per row `i` (all heads).
/// `qq` / `kk` are Gram results whose diagonals are the squared norms. Params: `rows hc hc4`.
pub fn attention_softmax() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> qk: array<mat4x4<f32>>;
@group(0) @binding(2) var<storage, read> qq: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read> kk: array<mat4x4<f32>>;
@group(0) @binding(4) var<storage, read> temperature: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> attn: array<f32>;
fn diag(head: u32, ch: u32, is_q: bool) -> f32 {{
  let b = ch / 4u; let l = ch % 4u;
  let m = (head * p.hc4 + b) * p.hc4 + b;
  if (is_q) {{ return qq[m][l][l]; }}
  return kk[m][l][l];
}}
fn logit(head: u32, i: u32, j: u32, nq: f32, t: f32) -> f32 {{
  let m = (head * p.hc4 + i / 4u) * p.hc4 + j / 4u;
  let nk = max(sqrt(diag(head, j, false)), 1e-12);
  return qk[m][j % 4u][i % 4u] / (nq * nk) * t;
}}
@compute @workgroup_size({WG1}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.rows) {{ return; }}
  let head = gid.x / p.hc;
  let i = gid.x % p.hc;
  let t = temperature[0][head];
  let nq = max(sqrt(diag(head, i, true)), 1e-12);
  var top = logit(head, i, 0u, nq, t);
  for (var j = 1u; j < p.hc; j++) {{ top = max(top, logit(head, i, j, nq, t)); }}
  var sum = 0.0;
  for (var j = 0u; j < p.hc; j++) {{ sum += exp(logit(head, i, j, nq, t) - top); }}
  for (var j = 0u; j < p.hc; j++) {{
    let m = (head * p.hc4 + i / 4u) * p.hc4 + j / 4u;
    attn[m * 16u + (j % 4u) * 4u + i % 4u] = exp(logit(head, i, j, nq, t) - top) / sum;
  }}
}}
",
        params = params(&["rows", "hc", "hc4"]),
    )
}

/// `out = attn @ v` per pixel. Invocation `(x, y)` = (pixel in row, output block).
/// Params: `n row src_c4 voff hc4 out_c4`.
pub fn attention_apply() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> src: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> attn: array<mat4x4<f32>>;
@group(0) @binding(3) var<storage, read_write> dst: array<vec4<f32>>;
@compute @workgroup_size({WG1}, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  let i = gid.y * p.row + gid.x;
  if (gid.x >= p.row || i >= p.n) {{ return; }}
  let o4 = gid.z;
  let head = o4 / p.hc4;
  let i4 = o4 % p.hc4;
  var acc = vec4<f32>(0.0);
  for (var j4 = 0u; j4 < p.hc4; j4++) {{
    acc += attn[(head * p.hc4 + i4) * p.hc4 + j4] * src[i * p.src_c4 + p.voff + head * p.hc4 + j4];
  }}
  dst[i * p.out_c4 + o4] = acc;
}}
",
        params = params(&["n", "row", "src_c4", "voff", "hc4", "out_c4"]),
    )
}

/// Planar YUV `[3, h, w]` in `[0, 255]` (4:4:4) to an `rgba8unorm` storage texture, BT.709, with
/// the same operation order as the CPU output stage. For presentation without a readback.
/// Params: `w h pic_w pic_h`.
pub fn yuv_to_rgba() -> String {
    format!(
        "{params}
@group(0) @binding(1) var<storage, read> rec: array<f32>;
@group(0) @binding(2) var tex: texture_storage_2d<rgba8unorm, write>;
const KRY: f32 = 1.5748; const KBY: f32 = 1.8556;
const GU: f32 = {gu:?}; const GV: f32 = {gv:?};
// The CPU output stage clips to [0, 255] and then rounds half to even (`ImageIO.write_png`).
// Rounding here instead of leaving it to the fixed-function `rgba8unorm` conversion is what
// makes the texture match it: the hardware's float-to-unorm rounding is not the same on every
// driver (llvmpipe agreed with the CPU stage, NVIDIA's differs on ~3% of samples by one step),
// and an already-integral k/255 converts back to k everywhere.
fn unit(x: f32) -> f32 {{ return round(clamp((x * 255.0) / 255.0, 0.0, 1.0) * 255.0) / 255.0; }}
@compute @workgroup_size({WG}, {WG}, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {{
  if (gid.x >= p.w || gid.y >= p.h) {{ return; }}
  let plane = p.pic_w * p.pic_h;
  let i = gid.y * p.pic_w + gid.x;
  let y = rec[i] / 255.0;
  let u = rec[plane + i] / 255.0 - 0.5;
  let v = rec[2u * plane + i] / 255.0 - 0.5;
  let rgb = vec3<f32>(unit(y + KRY * v), unit(y - GU * u - GV * v), unit(y + KBY * u));
  textureStore(tex, vec2<i32>(gid.xy), vec4<f32>(rgb, 1.0));
}}
",
        params = params(&["w", "h", "pic_w", "pic_h"]),
        gu = (0.0722f64 * 1.8556 / 0.7152) as f32,
        gv = (0.2126f64 * 1.5748 / 0.7152) as f32,
    )
}
