// Shared entry points for the libFuzzer targets and `tests/fuzz_regression.rs`.
//
// Each fuzz bin does `mod common;` and the stable regression test `include!`s
// this file, so the replay harness runs exactly the code libFuzzer runs. Plain
// std Rust only — no libfuzzer-sys imports. (Plain `//` header: `include!`
// cannot carry `//!` doc comments.)
//
// The decoder needs model checkpoints before it can run the float networks.
// Rather than shipping megabytes of trained weights, `SynthModels` serves
// all-zero `ZJM1` checkpoints generated on the fly in the exact tensor layout
// the loaders require (`VM_common_int/{Y,UV}_<beta>.pth`,
// `VM_{sop,bop,hop}/decoder_{Y,UV}_<beta>.pth`). Entropy output of those models
// is meaningless, but every allocation, table lookup and shape check of the
// decode pipeline runs — which is what this harness stresses.

use std::borrow::Cow;
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Mutex, OnceLock};

use zenjpegai::container::{Codestream, Marker, split_threads};
use zenjpegai::decoder::{self, Decoder};
use zenjpegai::header::{PictureHeader, RenderingInfo, ToolHeader};
use zenjpegai::mans::AnsTables;
use zenjpegai::model::{CommonModel, ModelSource};
use zenjpegai::nn::fast::{Engine, Tier};
use zenjpegai::weights::Checkpoint;
use zenjpegai::{Error, Limits};

// ---------------------------------------------------------------------------
// Cancellation token: stops after a fixed number of `check()` calls.
// ---------------------------------------------------------------------------

/// `enough::Stop` that cancels after `remaining` checks (fuzz input selects the
/// budget so some runs exercise the `Error::Cancelled` paths mid-decode).
pub struct StopAfter(pub AtomicU32);

impl StopAfter {
    pub fn new(remaining: u32) -> Self {
        Self(AtomicU32::new(remaining))
    }
}

impl enough::Stop for StopAfter {
    fn check(&self) -> Result<(), enough::StopReason> {
        match self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| n.checked_sub(1))
        {
            Ok(_) => Ok(()),
            Err(_) => Err(enough::StopReason::Cancelled),
        }
    }
    fn should_stop(&self) -> bool {
        self.0.load(Ordering::Relaxed) == 0
    }
}

// ---------------------------------------------------------------------------
// Minimal ZJM1 writer (format documented in `src/weights/packed.rs`).
// ---------------------------------------------------------------------------

const DT_F32: u8 = 1;
const DT_I32: u8 = 4;
const DT_I8: u8 = 6;
const DT_BOOL: u8 = 8;

struct TensorSpec {
    name: String,
    dtype: u8,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

fn spec(name: &str, dtype: u8, shape: &[usize], bytes: Vec<u8>) -> TensorSpec {
    TensorSpec {
        name: name.to_string(),
        dtype,
        shape: shape.to_vec(),
        bytes,
    }
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn f32_t(name: &str, shape: &[usize], fill: f32) -> TensorSpec {
    let mut bytes = Vec::with_capacity(numel(shape) * 4);
    for _ in 0..numel(shape) {
        bytes.extend_from_slice(&fill.to_le_bytes());
    }
    spec(name, DT_F32, shape, bytes)
}

fn i32_t(name: &str, shape: &[usize], fill: i32) -> TensorSpec {
    let mut bytes = Vec::with_capacity(numel(shape) * 4);
    for _ in 0..numel(shape) {
        bytes.extend_from_slice(&fill.to_le_bytes());
    }
    spec(name, DT_I32, shape, bytes)
}

fn i8_t(name: &str, shape: &[usize], fill: i8) -> TensorSpec {
    spec(name, DT_I8, shape, vec![fill as u8; numel(shape)])
}

fn bool_t(name: &str, v: bool) -> TensorSpec {
    spec(name, DT_BOOL, &[1], vec![v as u8])
}

/// Serialize specs into a `ZJM1` checkpoint (one u32 header block, per-tensor
/// entries, 64-byte aligned payloads).
fn zjm1(tensors: &[TensorSpec]) -> Vec<u8> {
    let mut table = 16usize;
    for t in tensors {
        table += 2 + t.name.len() + 2 + 8 * t.shape.len() + 16;
    }
    let mut offset = table.next_multiple_of(64);
    let mut out = Vec::new();
    out.extend_from_slice(b"ZJM1");
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    for t in tensors {
        out.extend_from_slice(&(t.name.len() as u16).to_le_bytes());
        out.extend_from_slice(t.name.as_bytes());
        out.push(t.dtype);
        out.push(t.shape.len() as u8);
        for &d in &t.shape {
            out.extend_from_slice(&(d as u64).to_le_bytes());
        }
        out.extend_from_slice(&(offset as u64).to_le_bytes());
        out.extend_from_slice(&(t.bytes.len() as u64).to_le_bytes());
        offset = (offset + t.bytes.len()).next_multiple_of(64);
    }
    out.resize(out.len().next_multiple_of(64), 0);
    for t in tensors {
        out.extend_from_slice(&t.bytes);
        out.resize(out.len().next_multiple_of(64), 0);
    }
    out
}

// ---------------------------------------------------------------------------
// Synthetic model checkpoints (zero weights, correct shapes).
// ---------------------------------------------------------------------------

fn conv3(t: &mut Vec<TensorSpec>, pfx: &str, i: usize, o: usize, bias: bool) {
    t.push(f32_t(&format!("{pfx}.weight"), &[o, i, 3, 3], 0.0));
    if bias {
        t.push(f32_t(&format!("{pfx}.bias"), &[o], 0.0));
    }
}

fn conv1(t: &mut Vec<TensorSpec>, pfx: &str, i: usize, o: usize, bias: bool) {
    t.push(f32_t(&format!("{pfx}.weight"), &[o, i, 1, 1], 0.0));
    if bias {
        t.push(f32_t(&format!("{pfx}.bias"), &[o], 0.0));
    }
}

/// Transposed conv, weight `[in, out, k, k]`, always biased.
fn conv_t(t: &mut Vec<TensorSpec>, pfx: &str, i: usize, o: usize, k: usize) {
    t.push(f32_t(&format!("{pfx}.weight"), &[i, o, k, k], 0.0));
    t.push(f32_t(&format!("{pfx}.bias"), &[o], 0.0));
}

/// `ResAU`: grouped 3x3 `conv` + 1x1 `conv2`, both bias-free. `groups` must
/// divide `c`; group count is read from the weight's dim 1.
fn res_au(t: &mut Vec<TensorSpec>, pfx: &str, c: usize, groups: usize) {
    t.push(f32_t(&format!("{pfx}.conv.weight"), &[c, c / groups, 3, 3], 0.0));
    t.push(f32_t(&format!("{pfx}.conv2.weight"), &[c, c, 1, 1], 0.0));
}

/// `ResidualBlock`: conv1 + conv2 3x3 with bias.
fn res_block(t: &mut Vec<TensorSpec>, pfx: &str, c: usize) {
    conv3(t, &format!("{pfx}.conv1"), c, c, true);
    conv3(t, &format!("{pfx}.conv2"), c, c, true);
}

/// Quantized conv of the hyper-scale decoder.
fn quant_conv(t: &mut Vec<TensorSpec>, pfx: &str, out: usize, i: usize, k: usize) {
    t.push(bool_t(&format!("{pfx}.is_quantized"), true));
    t.push(i8_t(&format!("{pfx}.weight"), &[out, i, k, k], 0));
    // Bias 4096 with shift 5 -> mid-range sigma index 128, inside the gain check.
    t.push(i32_t(&format!("{pfx}.bias"), &[out], 4096));
    t.push(i8_t(&format!("{pfx}.per_channel_shifts"), &[out], 5));
}

fn layer_norm(t: &mut Vec<TensorSpec>, pfx: &str, dim: usize) {
    t.push(f32_t(&format!("{pfx}.weight"), &[dim], 1.0));
    t.push(f32_t(&format!("{pfx}.bias"), &[dim], 0.0));
}

/// `TransformerBlock` of the HOP attention (`TABs.*`).
fn transformer_block(t: &mut Vec<TensorSpec>, pfx: &str, dim: usize) {
    let hidden = dim * 4;
    layer_norm(t, &format!("{pfx}.prep_data.norm1"), dim);
    conv1(t, &format!("{pfx}.prep_data.conv1"), dim, 3 * dim, false);
    // depthwise: groups = c -> weight [c, 1, 3, 3]
    t.push(f32_t(
        &format!("{pfx}.prep_data.conv2.weight"),
        &[3 * dim, 1, 3, 3],
        0.0,
    ));
    t.push(f32_t(&format!("{pfx}.attn.temperature"), &[4, 1, 1], 1.0));
    conv1(t, &format!("{pfx}.attn.project_out"), dim, dim, false);
    layer_norm(t, &format!("{pfx}.ffn.norm1"), dim);
    conv1(t, &format!("{pfx}.ffn.project_in"), dim, 2 * hidden, false);
    t.push(f32_t(
        &format!("{pfx}.ffn.dwconv.weight"),
        &[2 * hidden, 1, 3, 3],
        0.0,
    ));
    conv1(t, &format!("{pfx}.ffn.project_out"), hidden, dim, false);
}

/// `CAB` channel-attention block at `dim` channels.
fn cab(t: &mut Vec<TensorSpec>, pfx: &str, dim: usize) {
    res_block(t, &format!("{pfx}.residual_trunk.0"), dim);
    res_block(t, &format!("{pfx}.residual_trunk.1"), dim);
    conv3(t, &format!("{pfx}.subscale"), dim, dim, true);
    res_block(t, &format!("{pfx}.residual_mask2.0"), dim);
    res_block(t, &format!("{pfx}.residual_mask3.0"), dim);
    conv_t(t, &format!("{pfx}.upscale"), dim, dim, 3);
}

/// `TAM`: two transformer blocks, optionally around a stride-2 resample.
fn tam(t: &mut Vec<TensorSpec>, pfx: &str, dim: usize, downsample: bool) {
    if downsample {
        conv3(t, &format!("{pfx}.ds_conv"), dim, dim, true);
        conv_t(t, &format!("{pfx}.us_conv"), dim, dim, 3);
    }
    transformer_block(t, &format!("{pfx}.TABs.0"), dim);
    transformer_block(t, &format!("{pfx}.TABs.1"), dim);
}

/// `VM_common_int` file: hyper-entropy, hyper-scale decoder, hyper decoder.
/// `chs` is 160 for Y, 96 for UV (`header::LATENT_CHANNELS`).
fn common_ckpt(chs: usize) -> Vec<u8> {
    let mut t = Vec::new();
    t.push(bool_t("hyper_entropy.is_quantized", true));
    t.push(i32_t("hyper_entropy.freqs_int", &[chs, 63], 4));
    t.push(f32_t("vr_vec.c", &[chs, 1], 1.0));
    quant_conv(&mut t, "hyper_scale_decoder.conv1", chs, chs, 1);
    quant_conv(&mut t, "hyper_scale_decoder.depthwise", chs, chs, 3);
    quant_conv(&mut t, "hyper_scale_decoder.pointwise", chs * 16, chs, 1);
    conv1(&mut t, "hyper_decoder.conv1", chs, chs, false);
    conv_t(&mut t, "hyper_decoder.conv2", chs, chs, 4);
    conv3(&mut t, "hyper_decoder.conv3", chs, chs, true);
    conv3(&mut t, "hyper_decoder.conv4", chs, 4 * chs, true);
    zjm1(&t)
}

/// `VM_sop/decoder_Y_*` (primary, SOP).
fn sop_prim() -> Vec<u8> {
    let mut t = Vec::new();
    conv3(&mut t, "first_stage.0.0.conv1", 160, 160, true);
    t.push(f32_t("first_stage.1.conv.weight", &[256, 160, 2, 2], 0.0));
    res_au(&mut t, "first_stage.2", 64, 4);
    t.push(f32_t("conv2_t.conv.weight", &[128, 64, 2, 2], 0.0));
    res_au(&mut t, "iact2", 32, 2);
    conv3(&mut t, "conv3", 32, 32, true);
    res_au(&mut t, "iact3", 32, 2);
    conv1(&mut t, "conv4", 32, 16, false);
    zjm1(&t)
}

/// `VM_sop/decoder_UV_*` (secondary, SOP).
fn sop_sec() -> Vec<u8> {
    let mut t = Vec::new();
    conv3(&mut t, "first_stage.conv1", 256, 48, true);
    t.push(f32_t("conv2_t.conv.weight", &[128, 144, 2, 2], 0.0));
    res_au(&mut t, "iact2", 32, 2);
    conv3(&mut t, "conv3", 32, 32, true);
    res_au(&mut t, "iact3", 32, 2);
    conv1(&mut t, "conv4", 32, 128, false);
    zjm1(&t)
}

/// `VM_bop/decoder_Y_*` (primary, BOP).
fn bop_prim() -> Vec<u8> {
    let mut t = Vec::new();
    conv3(&mut t, "first_stage.0.0.conv1", 160, 160, true);
    conv_t(&mut t, "first_stage.1", 160, 64, 4);
    res_au(&mut t, "first_stage.2", 64, 4);
    conv_t(&mut t, "conv2_t", 64, 64, 4);
    res_au(&mut t, "iact2", 64, 4);
    conv3(&mut t, "conv3", 64, 96, true);
    res_au(&mut t, "iact3", 96, 6);
    conv1(&mut t, "conv4", 96, 16, false);
    zjm1(&t)
}

/// `VM_bop/decoder_UV_*` (secondary, BOP).
fn bop_sec() -> Vec<u8> {
    let mut t = Vec::new();
    conv3(&mut t, "first_stage.conv1", 256, 48, true);
    conv_t(&mut t, "conv2_t", 144, 64, 4);
    res_au(&mut t, "iact2", 64, 4);
    conv3(&mut t, "conv3", 64, 128, true);
    res_au(&mut t, "iact3", 128, 8);
    conv1(&mut t, "conv4", 128, 128, false);
    zjm1(&t)
}

/// `VM_hop/decoder_Y_*` (primary, HOP; hidden width 128).
fn hop_prim() -> Vec<u8> {
    let c = 128;
    let mut t = Vec::new();
    res_block(&mut t, "first_stage.0.0", 160);
    conv_t(&mut t, "first_stage.1", 160, c, 3);
    res_au(&mut t, "first_stage.2", c, 4);
    conv_t(&mut t, "conv2_t", c, c, 3);
    cab(&mut t, "CAB", c);
    res_au(&mut t, "iact2", c, 4);
    conv1(&mut t, "conv3_t", c, 4 * c, true);
    tam(&mut t, "TAM", c, true);
    res_au(&mut t, "iact3", c, 4);
    conv_t(&mut t, "conv4_t", c, 1, 3);
    zjm1(&t)
}

/// `VM_hop/decoder_UV_*` (secondary, HOP; hidden width 64).
fn hop_sec() -> Vec<u8> {
    let c = 64;
    let mut t = Vec::new();
    conv3(&mut t, "first_stage.conv1", 256, 48, true);
    conv_t(&mut t, "conv2_t", 144, c, 3);
    cab(&mut t, "CAB", c);
    res_au(&mut t, "iact2", c, 4);
    conv1(&mut t, "conv3_t", c, 4 * c, true);
    tam(&mut t, "TAM", c, false);
    res_au(&mut t, "iact3", c, 4);
    conv_t(&mut t, "conv4_t", c, 8, 3);
    zjm1(&t)
}

/// `ModelSource` that generates the decoder's checkpoint files on demand.
/// Filter networks (`models_in_op`) are not served: decode takes the
/// `Error::Model` path for streams that enable them.
#[derive(Default)]
pub struct SynthModels {
    cache: Mutex<BTreeMap<String, Vec<u8>>>,
}

impl SynthModels {
    fn build(rel: &str) -> Result<Vec<u8>, Error> {
        let (dir, file) = rel
            .rsplit_once('/')
            .ok_or_else(|| Error::Model(format!("{rel}: not a model path")))?;
        let Some(stem) = file.strip_suffix(".pth") else {
            return Err(Error::Model(format!("{rel}: not in the model bundle")));
        };
        let bytes = if dir == "VM_common_int" {
            match stem.split_once('_').map(|(c, _)| c) {
                Some("Y") => common_ckpt(160),
                Some("UV") => common_ckpt(96),
                _ => return Err(Error::Model(format!("{rel}: not in the model bundle"))),
            }
        } else {
            let Some(name) = stem.strip_prefix("decoder_") else {
                return Err(Error::Model(format!("{rel}: not in the model bundle")));
            };
            let prim = name.starts_with("Y_");
            match (dir, prim) {
                ("VM_sop", true) => sop_prim(),
                ("VM_sop", false) => sop_sec(),
                ("VM_bop", true) => bop_prim(),
                ("VM_bop", false) => bop_sec(),
                ("VM_hop", true) => hop_prim(),
                ("VM_hop", false) => hop_sec(),
                _ => return Err(Error::Model(format!("{rel}: not in the model bundle"))),
            }
        };
        Ok(bytes)
    }
}

impl ModelSource for SynthModels {
    fn read(&self, rel: &str) -> Result<Cow<'_, [u8]>, Error> {
        let mut cache = self
            .cache
            .lock()
            .map_err(|_| Error::Model("model cache poisoned".into()))?;
        if !cache.contains_key(rel) {
            cache.insert(rel.to_string(), Self::build(rel)?);
        }
        // One copy per read; the decoder itself caches the loaded models, so
        // each file is only read once per (model_id, operating point).
        Ok(Cow::Owned(cache[rel].clone()))
    }
}

// ---------------------------------------------------------------------------
// Shared state: ANS tables, fixed small models, the decoder itself.
// ---------------------------------------------------------------------------

fn tables() -> &'static AnsTables {
    static T: OnceLock<AnsTables> = OnceLock::new();
    T.get_or_init(AnsTables::new)
}

/// The common models the entropy stage runs against (160/96 channels, like the
/// trained `VM_common_int` pair).
fn common_models() -> &'static [CommonModel; 2] {
    static M: OnceLock<[CommonModel; 2]> = OnceLock::new();
    M.get_or_init(|| {
        let eng = Engine::with(Tier::detect(), false);
        let load = |rel: &str, chs: usize| {
            let src = SynthModels::default();
            let bytes = src.read(rel).expect("synth model").into_owned();
            let ck = Checkpoint::parse(&bytes).expect("synth checkpoint parses");
            CommonModel::load(&ck, chs, &eng).expect("synth common model loads")
        };
        [
            load("VM_common_int/Y_0.012.pth", 160),
            load("VM_common_int/UV_0.012.pth", 96),
        ]
    })
}

fn decoder_instance() -> &'static Decoder {
    static D: OnceLock<Decoder> = OnceLock::new();
    D.get_or_init(|| {
        Decoder::with_source(
            Box::new(SynthModels::default()),
            Engine::with(Tier::detect(), false),
        )
        .limits(
            Limits::none()
                .with_max_dimensions(1024, 1024)
                .with_max_pixels(1 << 20)
                .with_max_input_bytes(1 << 20)
                .with_max_memory(512 << 20),
        )
    })
}

// ---------------------------------------------------------------------------
// Entry points — one per fuzz target, all replayed by tests/fuzz_regression.rs.
// ---------------------------------------------------------------------------

/// Container + header parsing: `Codestream::parse`, PIH/TON/RDI/UDI headers
/// (incl. region / tiling / qmap syntax), header writers, conformance, and the
/// thread/region split helpers.
pub fn run_container_headers(data: &[u8]) {
    let Ok(cs) = Codestream::parse(data) else {
        return;
    };
    if let Some(pih) = cs.find(Marker::Pih) {
        // Parse headers even when the picture is non-conforming: conformance is
        // checked inside read_headers, but ToolHeader::parse must be safe for
        // every PictureHeader that parses.
        if let Ok(pic) = PictureHeader::parse(pih) {
            if let Some(ton) = cs.find(Marker::Ton) {
                let _ = ToolHeader::parse(ton, &pic);
            }
            let _ = pic.write();
            let _ = pic.check_conformance();
        }
    }
    if let Some(rdi) = cs.find(Marker::Rdi)
        && let Ok(r) = RenderingInfo::parse(rdi)
    {
        let _ = r.write();
    }
    // The full validated path (conformance + tool headers + memory estimate).
    if let Ok(headers) = decoder::read_headers(&cs) {
        let _ = headers.picture.write();
        let _ = headers.tools.write(&headers.picture);
        let _ = headers.rendering.write();
        for op in [
            zenjpegai::header::OperatingPoint::Sop,
            zenjpegai::header::OperatingPoint::Bop,
            zenjpegai::header::OperatingPoint::Hop,
        ] {
            let _ = zenjpegai::estimate_memory(&headers.picture, op);
        }
    }
    // Thread splits are the framing the entropy stage relies on: probe each
    // payload at every legal thread count.
    for s in &cs.substreams {
        for n in [1usize, 2, 4, 8, 16] {
            let _ = split_threads(s.payload, n);
        }
    }
}

/// me-tANS + entropy stage: z decode, hyper-scale decoder, gain/RVS/GRFS,
/// sigma map and residual symbols, driven by arbitrary codestreams and a fixed
/// small model pair. The last input byte rotates through the three public
/// `decode_entropy_stage*` entry points; the byte before it is the
/// cancellation budget where a `Stop` is taken.
pub fn run_entropy(data: &[u8]) {
    let Ok(cs) = Codestream::parse(data) else {
        return;
    };
    let Some(pih) = cs.find(Marker::Pih) else {
        return;
    };
    let Ok(hdr) = PictureHeader::parse(pih) else {
        return;
    };
    // Geometry cap: latent allocations scale with the coded size. Conformance
    // is deliberately not required — the stage must reject, never panic.
    if hdr.width > 512 || hdr.height > 512 {
        return;
    }
    let models = common_models();
    let sel = data.last().copied().unwrap_or(0);
    let budget = data
        .len()
        .checked_sub(2)
        .and_then(|i| data.get(i))
        .copied()
        .unwrap_or(0) as u32;
    match sel % 4 {
        0 => {
            let _ =
                decoder::decode_entropy_stage(tables(), &cs, &hdr, [&models[0], &models[1]]);
        }
        1 => {
            let _ = decoder::decode_entropy_stage_with(
                tables(),
                &cs,
                &hdr,
                [&models[0], &models[1]],
                &StopAfter::new(budget.saturating_mul(64)),
            );
        }
        _ => {
            let m = if data.len() >= 3 {
                Some(u16::from_le_bytes([
                    data[data.len() - 2],
                    data[data.len() - 3],
                ]))
            } else {
                None
            };
            let _ = decoder::decode_entropy_stage_progressive(
                tables(),
                &cs,
                &hdr,
                [&models[0], &models[1]],
                [m, m],
                &StopAfter::new(u32::MAX),
            );
        }
    }
}

/// Full `Decoder::decode` on arbitrary bytes and mutated real streams: tight
/// `Limits`, cooperative cancellation, in-memory synthetic checkpoints.
pub fn run_decode(data: &[u8]) {
    let dec = decoder_instance();
    // The last byte picks the cancellation budget: 0 -> no limit, else
    // b * 32 checks (small budgets cancel inside the entropy stage).
    let budget = data.last().copied().unwrap_or(0) as u32;
    if budget == 0 {
        let _ = dec.decode_picture_with(data, &enough::Unstoppable);
    } else {
        let _ = dec.decode_picture_with(data, &StopAfter::new(budget.saturating_mul(32)));
    }
}
