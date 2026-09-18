//! Recording and replaying a network as a fixed list of compute dispatches.
//!
//! A [`Graph`] records operations on virtual tensors. [`Graph::finish`] computes each tensor's
//! lifetime, maps tensors onto a small set of pooled buffers (a buffer is reused as soon as its
//! tensor is dead), writes all uniform blocks into one buffer and creates the bind groups. The
//! resulting [`Plan`] replays with no allocation, no bind-group creation and no readback: one
//! compute pass, one command buffer.

use std::sync::{Arc, Mutex};

use crate::context::GpuContext;
use crate::error::{GpuError, Result};
use crate::kernels::{self, Act, ConvVariant, Pointwise, Res, WG, WG1};
use crate::layers::{GpuConv, GpuConvTranspose, GpuDepthwise, GpuLayerNorm};

/// A virtual feature map in HWC4 layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct T {
    id: usize,
    pub c: usize,
    pub h: usize,
    pub w: usize,
}

impl T {
    pub fn c4(&self) -> usize {
        self.c.div_ceil(4)
    }
    /// Elements (`vec4`) of the map.
    pub fn len4(&self) -> usize {
        self.h * self.w * self.c4()
    }
}

/// A retained buffer may exceed what the current plan needs by this factor before
/// [`Pool::fit`] replaces it with a right-sized one.
pub(crate) const RETAIN_FACTOR: u64 = 4;
/// ... or by this many bytes, whichever is larger (keeps small-picture jitter from
/// reallocating on every decode).
pub(crate) const RETAIN_SLACK: u64 = 64 << 20;

/// Pooled activation buffers shared by all plans of one context user.
///
/// Bounded like the CPU pool (`zenjpegai::nn::fast::release_buffers`): a plan whose needs
/// are far below what the pool retains shrinks the slots it uses to their required size and
/// drops the rest, so a small decode after a large picture does not keep the large
/// picture's buffers alive (on an RTX 2080 one 4096 x 4096 picture parks ~0.5 GiB). The
/// retained pool stays under `RETAIN_FACTOR * need + RETAIN_SLACK` of the most recent plan.
#[derive(Default)]
pub struct Pool {
    slots: Vec<wgpu::Buffer>,
    /// Bumped whenever a slot had to be replaced (cached plans then hold stale, still valid,
    /// buffers and should be dropped to release them).
    pub generation: u64,
}

impl Pool {
    pub fn bytes(&self) -> u64 {
        self.slots.iter().map(|b| b.size()).sum()
    }

    /// Drop every slot; the next [`Graph::finish`] refills the pool.
    pub fn release(&mut self) {
        if !self.slots.is_empty() {
            self.slots.clear();
            self.generation += 1;
        }
    }

    /// One buffer per `slot_bytes` entry, taken from the pool where a slot already fits and
    /// allocated where it does not. See the struct docs for the eviction rule.
    fn fit(&mut self, dev: &wgpu::Device, slot_bytes: &[u64]) -> Vec<wgpu::Buffer> {
        let need: u64 = slot_bytes.iter().sum();
        // `need == 0` marks a graph that uses no activation tensors at all (the RGBA
        // presentation tail): it must not flush the pool.
        let shrink = need > 0
            && self.bytes()
                > need
                    .saturating_mul(RETAIN_FACTOR)
                    .saturating_add(RETAIN_SLACK);
        let mut grew = false;
        for (s, &bytes) in slot_bytes.iter().enumerate() {
            let replace = match self.slots.get(s) {
                None => true,
                Some(b) => b.size() < bytes || (shrink && b.size() > bytes),
            };
            if !replace {
                continue;
            }
            let b = dev.create_buffer(&wgpu::BufferDescriptor {
                label: Some("zenjpegai activation slot"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            if s < self.slots.len() {
                self.slots[s] = b;
                grew = true;
            } else {
                self.slots.push(b);
            }
        }
        if shrink && self.slots.len() > slot_bytes.len() {
            self.slots.truncate(slot_bytes.len());
            grew = true;
        }
        if grew {
            self.generation += 1;
        }
        self.slots[..slot_bytes.len()].to_vec()
    }
}

enum Bind {
    Tensor(usize),
    Buffer(wgpu::Buffer),
    Texture(wgpu::TextureView),
}

struct Step {
    label: String,
    pipeline: wgpu::ComputePipeline,
    params: [u32; 16],
    binds: Vec<Bind>,
    dispatch: [u32; 3],
}

struct Info {
    bytes: u64,
    last: usize,
    pinned: bool,
    external: Option<wgpu::Buffer>,
}

/// Network recorder.
pub struct Graph<'a> {
    ctx: &'a GpuContext,
    steps: Vec<Step>,
    tensors: Vec<Info>,
}

fn grid2(w: usize, h: usize) -> (u32, u32) {
    ((w as u32).div_ceil(WG), (h as u32).div_ceil(WG))
}

/// 1-D kernels over `n` items: `(row, dispatch)` with `row` items per dispatch row.
fn grid1(n: usize, z: u32) -> (u32, [u32; 3]) {
    let groups = (n as u32).div_ceil(WG1);
    let gx = groups.min(32768);
    let gy = groups.div_ceil(gx);
    (gx * WG1, [gx, gy, z])
}

impl<'a> Graph<'a> {
    pub fn new(ctx: &'a GpuContext) -> Self {
        Self {
            ctx,
            steps: Vec::new(),
            tensors: Vec::new(),
        }
    }

    fn tensor(&mut self, c: usize, h: usize, w: usize) -> Result<T> {
        let t = T {
            id: self.tensors.len(),
            c,
            h,
            w,
        };
        if c == 0 || h == 0 || w == 0 {
            return Err(GpuError::Shape("empty feature map".into()));
        }
        let bytes = t.len4() as u64 * 16;
        if bytes > self.ctx.max_binding_bytes() {
            return Err(GpuError::TooLarge {
                needed: bytes,
                limit: self.ctx.max_binding_bytes(),
            });
        }
        self.tensors.push(Info {
            bytes,
            last: self.steps.len(),
            pinned: false,
            external: None,
        });
        Ok(t)
    }

    /// A map the caller fills with [`Plan::write`] before running.
    pub fn input(&mut self, c: usize, h: usize, w: usize) -> Result<T> {
        let t = self.tensor(c, h, w)?;
        self.tensors[t.id].pinned = true;
        Ok(t)
    }

    /// A map living in a buffer the caller owns (HWC4, at least `len4 * 16` bytes).
    pub fn external(&mut self, buffer: &wgpu::Buffer, c: usize, h: usize, w: usize) -> Result<T> {
        let t = self.tensor(c, h, w)?;
        if buffer.size() < t.len4() as u64 * 16 {
            return Err(GpuError::Shape("external buffer too small".into()));
        }
        self.tensors[t.id].external = Some(buffer.clone());
        Ok(t)
    }

    /// Keep `t` alive to the end of the plan so [`Plan::read`] can fetch it.
    pub fn output(&mut self, t: T) {
        self.tensors[t.id].pinned = true;
    }

    fn push(
        &mut self,
        key: &str,
        src: impl FnOnce() -> String,
        params: &[u32],
        binds: Vec<Bind>,
        dispatch: [u32; 3],
    ) {
        let mut p = [0u32; 16];
        p[..params.len()].copy_from_slice(params);
        let at = self.steps.len();
        for b in &binds {
            if let Bind::Tensor(id) = b {
                self.tensors[*id].last = at;
            }
        }
        self.steps.push(Step {
            label: key.to_string(),
            pipeline: self.ctx.pipeline(key, src),
            params: p,
            binds,
            dispatch,
        });
    }

    /// Convolution with extra zero rows / columns at the bottom / right (`extra`), optional
    /// ReLU6 on the input samples and a fused output activation.
    pub fn conv_ex(
        &mut self,
        x: T,
        l: &GpuConv,
        extra: (usize, usize),
        pre_relu6: bool,
        act: Act,
    ) -> Result<T> {
        self.conv_full(x, l, extra, pre_relu6, act, None)
    }

    /// Convolution whose store fuses a second operand of the output's shape: `Res::Add` for a
    /// residual sum, `Res::Gate` for the ResAU gate. Same arithmetic as the separate
    /// [`Graph::pointwise`] step, one dispatch and one round trip through memory fewer.
    pub fn conv_res(&mut self, x: T, l: &GpuConv, act: Act, op: Res, res: T) -> Result<T> {
        self.conv_full(x, l, (0, 0), false, act, Some((op, res)))
    }

    fn conv_full(
        &mut self,
        x: T,
        l: &GpuConv,
        extra: (usize, usize),
        pre_relu6: bool,
        act: Act,
        res: Option<(Res, T)>,
    ) -> Result<T> {
        if x.c != l.in_ch {
            return Err(GpuError::Shape(format!(
                "conv: {} input channels, layer wants {}",
                x.c, l.in_ch
            )));
        }
        let oh = (x.h + extra.0 + 2 * l.pad.0)
            .checked_sub(l.k.0)
            .map(|v| v / l.stride + 1);
        let ow = (x.w + extra.1 + 2 * l.pad.1)
            .checked_sub(l.k.1)
            .map(|v| v / l.stride + 1);
        let (Some(oh), Some(ow)) = (oh, ow) else {
            return Err(GpuError::Shape(
                "conv: input smaller than the kernel".into(),
            ));
        };
        let y = self.tensor(l.out_ch, oh, ow)?;
        if let Some((_, r)) = res
            && (r.c, r.h, r.w) != (y.c, oh, ow)
        {
            return Err(GpuError::Shape(
                "conv: fused operand has a different shape from the output".into(),
            ));
        }
        // Measured on an RTX 2080: a 16 x 16 workgroup beats 8 x 8 on the layers with many
        // input blocks (HOP's 1 x 1 convolutions 147.5 -> 117.3 ms, its 3 x 3 family -13%) and
        // loses on the layers with few (SOP 560x888 goes 3.7 -> 5.1 ms if applied to all of
        // them), so it is chosen per layer.
        let wg = if l.icg4 >= 16 && ow >= 128 && oh >= 128 {
            2 * WG
        } else {
            WG
        };
        let v = ConvVariant {
            kh: l.k.0 as u32,
            kw: l.k.1 as u32,
            stride: l.stride as u32,
            ob: l.ob as u32,
            wg,
            act,
            pre_relu6,
            res: res.map(|(op, _)| op),
        };
        let (gx, gy) = ((ow as u32).div_ceil(wg), (oh as u32).div_ceil(wg));
        let mut binds = vec![
            Bind::Tensor(x.id),
            Bind::Tensor(y.id),
            Bind::Buffer(l.weight.clone()),
            Bind::Buffer(l.bias.clone()),
        ];
        if let Some((_, r)) = res {
            binds.push(Bind::Tensor(r.id));
        }
        self.push(
            &v.key(),
            || kernels::conv(&v),
            &[
                x.h as u32,
                x.w as u32,
                x.c4() as u32,
                oh as u32,
                ow as u32,
                y.c4() as u32,
                l.pad.0 as u32,
                l.pad.1 as u32,
                l.icg4 as u32,
                l.ocg4 as u32,
            ],
            binds,
            [gx, gy, (y.c4() / l.ob) as u32],
        );
        Ok(y)
    }

    pub fn conv(&mut self, x: T, l: &GpuConv) -> Result<T> {
        self.conv_ex(x, l, (0, 0), false, Act::None)
    }

    pub fn conv_transpose(&mut self, x: T, l: &GpuConvTranspose) -> Result<T> {
        if x.c != l.in_ch {
            return Err(GpuError::Shape("transposed conv: channel mismatch".into()));
        }
        let f = |n: usize| (n - 1) * l.stride + l.k + l.out_pad - 2 * l.pad;
        let (oh, ow) = (f(x.h), f(x.w));
        let y = self.tensor(l.out_ch, oh, ow)?;
        let (k, s, ob) = (l.k as u32, l.stride as u32, l.ob as u32);
        let (gx, gy) = grid2(ow, oh);
        self.push(
            &format!("convt_k{k}_s{s}_ob{ob}"),
            || kernels::conv_transpose(k, s, ob),
            &[
                x.h as u32,
                x.w as u32,
                x.c4() as u32,
                oh as u32,
                ow as u32,
                y.c4() as u32,
                l.pad as u32,
            ],
            vec![
                Bind::Tensor(x.id),
                Bind::Tensor(y.id),
                Bind::Buffer(l.weight.clone()),
                Bind::Buffer(l.bias.clone()),
            ],
            [gx, gy, (y.c4() / l.ob) as u32],
        );
        Ok(y)
    }

    pub fn depthwise(&mut self, x: T, l: &GpuDepthwise) -> Result<T> {
        if x.c != l.ch {
            return Err(GpuError::Shape("depthwise conv: channel mismatch".into()));
        }
        let y = self.tensor(x.c, x.h, x.w)?;
        let (gx, gy) = grid2(x.w, x.h);
        self.push(
            "depthwise3x3",
            kernels::depthwise3x3,
            &[x.h as u32, x.w as u32, x.c4() as u32],
            vec![
                Bind::Tensor(x.id),
                Bind::Tensor(y.id),
                Bind::Buffer(l.weight.clone()),
            ],
            [gx, gy, x.c4() as u32],
        );
        Ok(y)
    }

    /// General windowed channel copy into an existing map.
    #[allow(clippy::too_many_arguments)]
    fn copy_into(
        &mut self,
        src: T,
        (sx, sy): (usize, usize),
        sc4: usize,
        n4: usize,
        dst: T,
        dc4: usize,
        h: usize,
        w: usize,
    ) {
        let (gx, gy) = grid2(w, h);
        self.push(
            "copy_channels",
            kernels::copy_channels,
            &[
                h as u32,
                w as u32,
                n4 as u32,
                src.w as u32,
                src.c4() as u32,
                sx as u32,
                sy as u32,
                sc4 as u32,
                dst.w as u32,
                dst.c4() as u32,
                dc4 as u32,
            ],
            vec![Bind::Tensor(src.id), Bind::Tensor(dst.id)],
            [gx, gy, 1],
        );
    }

    /// `x[:, y0 .. y0 + h, x0 .. x0 + w]`.
    pub fn window(&mut self, x: T, x0: usize, y0: usize, w: usize, h: usize) -> Result<T> {
        if x0 + w > x.w || y0 + h > x.h {
            return Err(GpuError::Shape("window outside the map".into()));
        }
        if (x0, y0, w, h) == (0, 0, x.w, x.h) {
            return Ok(x);
        }
        let y = self.tensor(x.c, h, w)?;
        self.copy_into(x, (x0, y0), 0, x.c4(), y, 0, h, w);
        Ok(y)
    }

    /// Top-left crop to at most `h x w`.
    pub fn crop(&mut self, x: T, h: usize, w: usize) -> Result<T> {
        self.window(x, 0, 0, w.min(x.w), h.min(x.h))
    }

    /// Channel concatenation. Every part but the last must have a multiple of 4 channels.
    pub fn cat(&mut self, parts: &[T]) -> Result<T> {
        let (h, w) = (parts[0].h, parts[0].w);
        let c = parts.iter().map(|p| p.c).sum();
        let y = self.tensor(c, h, w)?;
        let mut at = 0;
        for (i, p) in parts.iter().enumerate() {
            if (p.h, p.w) != (h, w) || (i + 1 < parts.len() && p.c % 4 != 0) {
                return Err(GpuError::Shape("cat: size or channel alignment".into()));
            }
            self.copy_into(*p, (0, 0), 0, p.c4(), y, at, h, w);
            at += p.c4();
        }
        Ok(y)
    }

    /// In-place pointwise operation on `a`.
    pub fn pointwise(&mut self, op: Pointwise, a: T, others: &[T]) -> Result<()> {
        if others.len() != op.inputs() || others.iter().any(|o| (o.c, o.h, o.w) != (a.c, a.h, a.w))
        {
            return Err(GpuError::Shape(format!("{op:?}: operand shapes differ")));
        }
        let (row, dispatch) = grid1(a.len4(), 1);
        let mut binds = vec![Bind::Tensor(a.id)];
        binds.extend(others.iter().map(|o| Bind::Tensor(o.id)));
        self.push(
            &format!("pw_{op:?}"),
            || kernels::pointwise(op),
            &[a.len4() as u32, row],
            binds,
            dispatch,
        );
        Ok(())
    }

    /// An uninitialised map for operations that fill it piecewise.
    pub fn alloc(&mut self, c: usize, h: usize, w: usize) -> Result<T> {
        self.tensor(c, h, w)
    }

    /// `dst[dst_c4off ..][.. a.c4] = elu(a) * b` (whole maps `a`, `b` of equal shape).
    pub fn elu_gate_into(&mut self, a: T, b: T, dst: T, dst_c4off: usize) -> Result<()> {
        if (a.c, a.h, a.w) != (b.c, b.h, b.w)
            || (a.h, a.w) != (dst.h, dst.w)
            || !a.c.is_multiple_of(4)
            || dst_c4off + a.c4() > dst.c4()
        {
            return Err(GpuError::Shape("elu gate: operand shapes".into()));
        }
        self.elu_gate_raw(a, 0, b, 0, a.c4(), dst, dst_c4off);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn elu_gate_raw(
        &mut self,
        a: T,
        a_off: usize,
        b: T,
        b_off: usize,
        n4: usize,
        dst: T,
        off: usize,
    ) {
        let n = a.h * a.w;
        let (row, dispatch) = grid1(n, 1);
        self.push(
            "elu_gate",
            kernels::elu_gate,
            &[
                n as u32,
                row,
                n4 as u32,
                a.c4() as u32,
                a_off as u32,
                b.c4() as u32,
                b_off as u32,
                dst.c4() as u32,
                off as u32,
            ],
            vec![Bind::Tensor(a.id), Bind::Tensor(b.id), Bind::Tensor(dst.id)],
            dispatch,
        );
    }

    /// `elu(x[:c/2]) * x[c/2:]`.
    pub fn elu_gate(&mut self, x: T) -> Result<T> {
        if !x.c.is_multiple_of(8) {
            return Err(GpuError::Shape("elu gate: channel count".into()));
        }
        let y = self.tensor(x.c / 2, x.h, x.w)?;
        self.elu_gate_raw(x, 0, x, y.c4(), y.c4(), y, 0);
        Ok(y)
    }

    pub fn pixel_shuffle(&mut self, x: T, r: usize) -> Result<T> {
        if !x.c.is_multiple_of(r * r) {
            return Err(GpuError::Shape("pixel shuffle: channel count".into()));
        }
        let y = self.tensor(x.c / (r * r), x.h * r, x.w * r)?;
        let (gx, gy) = grid2(y.w, y.h);
        self.push(
            &format!("pixel_shuffle_{r}"),
            || kernels::pixel_shuffle(r as u32),
            &[
                y.h as u32,
                y.w as u32,
                y.c as u32,
                y.c4() as u32,
                x.w as u32,
                x.c4() as u32,
            ],
            vec![Bind::Tensor(x.id), Bind::Tensor(y.id)],
            [gx, gy, y.c4() as u32],
        );
        Ok(y)
    }

    /// Layer norm over channels.
    pub fn layer_norm(&mut self, x: T, l: &GpuLayerNorm) -> Result<T> {
        if x.c != l.ch {
            return Err(GpuError::Shape("layer norm: channel mismatch".into()));
        }
        let y = self.tensor(x.c, x.h, x.w)?;
        let n = x.h * x.w;
        let (row, dispatch) = grid1(n, 1);
        self.push(
            "layer_norm",
            kernels::layer_norm,
            &[n as u32, row, x.c as u32, x.c4() as u32],
            vec![
                Bind::Tensor(x.id),
                Bind::Buffer(l.wb.clone()),
                Bind::Tensor(y.id),
            ],
            dispatch,
        );
        Ok(y)
    }

    /// Channel attention of the transformer block: `qkv` holds q, k, v (each `dim` channels,
    /// `heads` heads); `temperature` is one `vec4` (4 heads).
    pub fn channel_attention(
        &mut self,
        qkv: T,
        heads: usize,
        temperature: &wgpu::Buffer,
    ) -> Result<T> {
        if heads != 4 || !qkv.c.is_multiple_of(3 * heads * 4) {
            return Err(GpuError::Shape(
                "attention: 4 heads of a multiple of 4 channels".into(),
            ));
        }
        let dim = qkv.c / 3;
        let hc = dim / heads;
        let hc4 = hc / 4;
        let n = qkv.h * qkv.w;
        let chunks = n.div_ceil(kernels::CHUNK as usize);
        let pairs = heads * hc4 * hc4;
        let (qoff, koff, voff) = (0, dim / 4, 2 * dim / 4);

        // Scratch maps are typed as 4-channel 1 x N "feature maps": one mat4x4 is 4 vec4.
        let mat = |g: &mut Self, count: usize| g.tensor(16, 1, count);
        let gram = |g: &mut Self, a: usize, b: usize| -> Result<T> {
            let part = mat(g, pairs * chunks)?;
            let (gx, gy) = grid2(chunks, pairs);
            g.push(
                "gram_chunks",
                kernels::gram_chunks,
                &[
                    n as u32,
                    chunks as u32,
                    pairs as u32,
                    qkv.c4() as u32,
                    a as u32,
                    b as u32,
                    hc4 as u32,
                ],
                vec![Bind::Tensor(qkv.id), Bind::Tensor(part.id)],
                [gx, gy, 1],
            );
            let out = mat(g, pairs)?;
            g.push(
                "gram_reduce",
                kernels::gram_reduce,
                &[chunks as u32, pairs as u32],
                vec![Bind::Tensor(part.id), Bind::Tensor(out.id)],
                [(pairs as u32).div_ceil(WG1), 1, 1],
            );
            Ok(out)
        };
        let qk = gram(self, qoff, koff)?;
        let qq = gram(self, qoff, qoff)?;
        let kk = gram(self, koff, koff)?;
        let attn = mat(self, pairs)?;
        self.push(
            "attention_softmax",
            kernels::attention_softmax,
            &[(heads * hc) as u32, hc as u32, hc4 as u32],
            vec![
                Bind::Tensor(qk.id),
                Bind::Tensor(qq.id),
                Bind::Tensor(kk.id),
                Bind::Buffer(temperature.clone()),
                Bind::Tensor(attn.id),
            ],
            [((heads * hc) as u32).div_ceil(WG1), 1, 1],
        );
        let y = self.tensor(dim, qkv.h, qkv.w)?;
        let (row, dispatch) = grid1(n, y.c4() as u32);
        self.push(
            "attention_apply",
            kernels::attention_apply,
            &[
                n as u32,
                row,
                qkv.c4() as u32,
                voff as u32,
                hc4 as u32,
                y.c4() as u32,
            ],
            vec![
                Bind::Tensor(qkv.id),
                Bind::Tensor(attn.id),
                Bind::Tensor(y.id),
            ],
            dispatch,
        );
        Ok(y)
    }

    /// Final step of a synthesis transform; see [`kernels::emit`]. `rec` is the planar picture
    /// buffer `[planes, pic_h, pic_w]`; `x` has `planes_out * r * r` channels.
    #[allow(clippy::too_many_arguments)]
    pub fn emit(
        &mut self,
        x: T,
        r: usize,
        planes_out: usize,
        plane0: usize,
        rec: &wgpu::Buffer,
        (pic_h, pic_w): (usize, usize),
        core: (usize, usize, usize, usize),
        offset: (usize, usize),
    ) -> Result<()> {
        let (cx, cy, cw, ch) = core;
        if x.c < planes_out * r * r
            || offset.0 + cw > x.w * r
            || offset.1 + ch > x.h * r
            || cx + cw > pic_w
            || cy + ch > pic_h
        {
            return Err(GpuError::Shape(
                "emit: core outside the tile or the picture".into(),
            ));
        }
        let (gx, gy) = grid2(cw, ch);
        self.push(
            &format!("emit_{r}"),
            || kernels::emit(r as u32),
            &[
                cw as u32,
                ch as u32,
                cx as u32,
                cy as u32,
                offset.0 as u32,
                offset.1 as u32,
                x.w as u32,
                x.c4() as u32,
                pic_w as u32,
                pic_h as u32,
                plane0 as u32,
            ],
            vec![Bind::Tensor(x.id), Bind::Buffer(rec.clone())],
            [gx, gy, planes_out as u32],
        );
        Ok(())
    }

    /// Planar YUV picture buffer to an `rgba8unorm` storage texture view.
    pub fn yuv_to_rgba(
        &mut self,
        rec: &wgpu::Buffer,
        (pic_h, pic_w): (usize, usize),
        (h, w): (usize, usize),
        view: &wgpu::TextureView,
    ) {
        let (gx, gy) = grid2(w, h);
        self.push(
            "yuv_to_rgba",
            kernels::yuv_to_rgba,
            &[w as u32, h as u32, pic_w as u32, pic_h as u32],
            vec![Bind::Buffer(rec.clone()), Bind::Texture(view.clone())],
            [gx, gy, 1],
        );
    }

    /// Assign buffers, upload the uniform blocks, create the bind groups.
    pub fn finish(self, pool: &Arc<Mutex<Pool>>) -> Result<Plan> {
        let ctx = self.ctx;
        let dev = &ctx.device;
        let n_steps = self.steps.len();

        // Greedy interval colouring, best fit by size.
        let mut born = vec![usize::MAX; self.tensors.len()];
        for (s, step) in self.steps.iter().enumerate() {
            for b in &step.binds {
                if let Bind::Tensor(id) = b {
                    born[*id] = born[*id].min(s);
                }
            }
        }
        let mut slot_bytes: Vec<u64> = Vec::new();
        let mut free: Vec<usize> = Vec::new();
        let mut slot_of = vec![usize::MAX; self.tensors.len()];
        let take = |bytes: u64, free: &mut Vec<usize>, slot_bytes: &mut Vec<u64>| -> usize {
            let fit = free
                .iter()
                .enumerate()
                .filter(|&(_, &s)| slot_bytes[s] >= bytes)
                .min_by_key(|&(_, &s)| slot_bytes[s])
                .or_else(|| free.iter().enumerate().max_by_key(|&(_, &s)| slot_bytes[s]))
                .map(|(i, _)| i);
            match fit {
                Some(i) => {
                    let s = free.swap_remove(i);
                    slot_bytes[s] = slot_bytes[s].max(bytes);
                    s
                }
                None => {
                    slot_bytes.push(bytes);
                    slot_bytes.len() - 1
                }
            }
        };
        // Pinned tensors first: they live for the whole plan.
        for (id, info) in self.tensors.iter().enumerate() {
            if info.pinned && info.external.is_none() {
                slot_of[id] = take(info.bytes, &mut free, &mut slot_bytes);
            }
        }
        for s in 0..n_steps {
            for (id, info) in self.tensors.iter().enumerate() {
                if born[id] == s && slot_of[id] == usize::MAX && info.external.is_none() {
                    slot_of[id] = take(info.bytes, &mut free, &mut slot_bytes);
                }
            }
            for (id, info) in self.tensors.iter().enumerate() {
                if info.last == s && born[id] <= s && !info.pinned && info.external.is_none() {
                    free.push(slot_of[id]);
                }
            }
        }

        let buffers: Vec<wgpu::Buffer> = {
            let mut pool = pool.lock().unwrap_or_else(|e| e.into_inner());
            pool.fit(dev, &slot_bytes)
        };

        let align = dev.limits().min_uniform_buffer_offset_alignment.max(64) as u64;
        let mut blob = vec![0u8; (align * n_steps.max(1) as u64) as usize];
        for (s, step) in self.steps.iter().enumerate() {
            blob[s * align as usize..][..64].copy_from_slice(bytemuck::cast_slice(&step.params));
        }
        let uniforms = dev.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenjpegai plan params"),
            size: blob.len() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        ctx.write_buffer(&uniforms, &blob);

        let buffer_of = |id: usize| -> &wgpu::Buffer {
            match &self.tensors[id].external {
                Some(b) => b,
                None => &buffers[slot_of[id]],
            }
        };
        let mut steps = Vec::with_capacity(n_steps);
        for (s, step) in self.steps.iter().enumerate() {
            let mut entries = vec![wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniforms,
                    offset: s as u64 * align,
                    size: wgpu::BufferSize::new(64),
                }),
            }];
            for (i, b) in step.binds.iter().enumerate() {
                let resource = match b {
                    Bind::Tensor(id) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: buffer_of(*id),
                        offset: 0,
                        size: wgpu::BufferSize::new(self.tensors[*id].bytes),
                    }),
                    Bind::Buffer(b) => b.as_entire_binding(),
                    Bind::Texture(v) => wgpu::BindingResource::TextureView(v),
                };
                entries.push(wgpu::BindGroupEntry {
                    binding: i as u32 + 1,
                    resource,
                });
            }
            let bind_group = dev.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &step.pipeline.get_bind_group_layout(0),
                entries: &entries,
            });
            steps.push((step.pipeline.clone(), bind_group, step.dispatch));
        }
        let labels = self
            .steps
            .iter()
            .map(|s| {
                let [x, y, z] = s.dispatch;
                format!("{} [{x}x{y}x{z}]", s.label)
            })
            .collect();
        let tensor_buffers = (0..self.tensors.len())
            .map(|id| {
                (born[id] != usize::MAX || self.tensors[id].pinned).then(|| buffer_of(id).clone())
            })
            .collect();
        Ok(Plan {
            steps,
            labels,
            tensor_buffers,
            activation_bytes: slot_bytes.iter().sum(),
            largest_binding_bytes: self.tensors.iter().map(|t| t.bytes).max().unwrap_or(0),
        })
    }
}

/// A recorded network, ready to replay.
pub struct Plan {
    steps: Vec<(wgpu::ComputePipeline, wgpu::BindGroup, [u32; 3])>,
    labels: Vec<String>,
    tensor_buffers: Vec<Option<wgpu::Buffer>>,
    /// Sum of the activation slot sizes this plan needs.
    pub activation_bytes: u64,
    /// Largest single feature map (what the device's storage binding limit has to cover).
    pub largest_binding_bytes: u64,
}

impl Plan {
    pub fn dispatches(&self) -> usize {
        self.steps.len()
    }

    /// One label per dispatch: kernel key and workgroup grid.
    pub fn labels(&self) -> &[String] {
        &self.labels
    }

    /// Record the plan with **one compute pass per dispatch**, each timed by a begin / end
    /// timestamp pair at `first + 2 * i`. Only for profiling: splitting the pass adds per-pass
    /// overhead (the whole-plan total measured this way is a few percent above
    /// [`Plan::encode`]'s), and it needs `2 * dispatches()` queries in the set.
    pub fn encode_profiled(&self, enc: &mut wgpu::CommandEncoder, q: &wgpu::QuerySet, first: u32) {
        for (i, (pipeline, bind_group, [x, y, z])) in self.steps.iter().enumerate() {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(&self.labels[i]),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: q,
                    beginning_of_pass_write_index: Some(first + 2 * i as u32),
                    end_of_pass_write_index: Some(first + 2 * i as u32 + 1),
                }),
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(*x, *y, *z);
        }
    }

    /// Record the whole plan as one compute pass. `timestamps`: query set and the index of the
    /// begin query (end = begin + 1).
    pub fn encode(
        &self,
        enc: &mut wgpu::CommandEncoder,
        timestamps: Option<(&wgpu::QuerySet, u32)>,
    ) {
        let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("zenjpegai plan"),
            timestamp_writes: timestamps.map(|(query_set, i)| wgpu::ComputePassTimestampWrites {
                query_set,
                beginning_of_pass_write_index: Some(i),
                end_of_pass_write_index: Some(i + 1),
            }),
        });
        for (pipeline, bind_group, [x, y, z]) in &self.steps {
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.dispatch_workgroups(*x, *y, *z);
        }
    }

    fn buffer(&self, t: T) -> Result<&wgpu::Buffer> {
        self.tensor_buffers
            .get(t.id)
            .and_then(|b| b.as_ref())
            .ok_or_else(|| GpuError::Shape("tensor is not part of this plan".into()))
    }

    /// Upload a planar `[C, H, W]` map into input `t`.
    pub fn write(&self, ctx: &GpuContext, t: T, planar: &[f32]) -> Result<()> {
        let packed = crate::layers::pack_hwc4(planar, t.c, t.h, t.w)?;
        ctx.write_buffer(self.buffer(t)?, bytemuck::cast_slice(&packed));
        Ok(())
    }

    /// Run the plan and read output `t` back as planar `[C, H, W]`.
    pub async fn run_and_read(&self, ctx: &GpuContext, t: T) -> Result<Vec<f32>> {
        let bytes = t.len4() as u64 * 16;
        let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("zenjpegai readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = ctx.device.create_command_encoder(&Default::default());
        self.encode(&mut enc, None);
        enc.copy_buffer_to_buffer(self.buffer(t)?, 0, &staging, 0, bytes);
        ctx.queue.submit([enc.finish()]);
        ctx.map_read(&staging, bytes).await?;
        let view = staging
            .slice(0..bytes)
            .get_mapped_range()
            .map_err(|e| GpuError::Device(e.to_string()))?;
        let packed: &[f32] = bytemuck::cast_slice(&view);
        Ok(crate::layers::unpack_hwc4(packed, t.c, t.h, t.w))
    }
}
