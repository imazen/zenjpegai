//! Shared helpers for the `reference-tests` integration tests.
#![allow(dead_code)]

use std::path::PathBuf;

/// Root of the upstream reference checkout. Hard failure if unset: these tests only compile with
/// `--features reference-tests`, and the caller (see `just test-ref`) owns that decision.
pub fn ref_root() -> PathBuf {
    let root = std::env::var_os("ZENJPEGAI_REF").expect("ZENJPEGAI_REF must point at the jpeg-ai-reference-software checkout (run via `just test-ref`)");
    let root = PathBuf::from(root);
    assert!(
        root.join("models").is_dir(),
        "{} has no models/ directory (git lfs pull?)",
        root.display()
    );
    root
}

pub fn read_model(rel: &str) -> Vec<u8> {
    let path = ref_root().join("models").join(rel);
    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

pub fn fnv1a64(bytes: impl IntoIterator<Item = u8>) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Root of the reference vector tree (`ZENJPEGAI_VECTORS`, or the default location); encoder
/// inputs live in its sibling `inputs/` directory.
pub fn vectors_root() -> PathBuf {
    std::env::var_os("ZENJPEGAI_VECTORS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"))
}

/// Directory holding the reference decoder dumps produced by
/// `scripts/ref_vectors/make_reference_streams.sh`. Hard failure if a vector is missing.
pub fn vector_dir(name: &str) -> PathBuf {
    let dir = vectors_root().join(name);
    assert!(
        dir.join("stream.bits").is_file(),
        "{} is missing: run scripts/ref_vectors/make_reference_streams.sh",
        dir.display()
    );
    dir
}

/// Dump of the reference decoder run with its known defects patched at runtime
/// (`dump_decode.py --contiguous-masks --fix-qmap-header`, in `<vector>/fixed_decoder/`): the
/// oracle for streams the stock decoder mis-decodes (regions) or cannot decode (quality maps).
pub fn load_fixed_decoder_dump(
    dir: &std::path::Path,
) -> std::collections::HashMap<String, RefTensor> {
    let fixed = dir.join("fixed_decoder");
    assert!(
        fixed.join("manifest.txt").is_file(),
        "{} is missing: run scripts/ref_vectors/make_reference_streams.sh",
        fixed.display()
    );
    load_dump(&fixed)
}

/// `IcciHeader` rebuilt from a `filters_lef_icci` dump's machine-readable selection tensors
/// (`eicci.selection`, `eicci.signalled`, `eicci.short_list`, written by
/// `scripts/ref_vectors/dump_filters_lef_icci.py`). Needed for the forced-4:2:0 eICCI vector,
/// whose tool header is not readable by a conformant parser (`icci_enable_flag` and the header
/// are not part of the 4:2:0 syntax, but the forced stream carries them anyway).
pub fn icci_header_from_dump(
    dump: &std::collections::HashMap<String, RefTensor>,
) -> zenjpegai::header::IcciHeader {
    let sel = &dump["eicci.selection"];
    let sig = &dump["eicci.signalled"];
    let short_list = dump["eicci.short_list"].bytes[0] != 0;
    assert_eq!(sel.shape[1..], [3]);
    assert_eq!(sig.shape[1..], [2]);
    assert_eq!(sel.shape[0], sig.shape[0]);
    let tiles = sel
        .i32()
        .as_chunks::<3>()
        .0
        .iter()
        .zip(sig.i32().as_chunks::<2>().0)
        .map(|(s, g)| zenjpegai::header::IcciTile {
            use_yuv: [s[0] != 0, s[1] != 0, s[2] != 0],
            short_list,
            index_y: g[0] as u8,
            index_uv: g[1] as u8,
        })
        .collect();
    zenjpegai::header::IcciHeader {
        tiling: None,
        tiles,
    }
}

/// One tensor of a reference dump.
pub struct RefTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl RefTensor {
    pub fn i8(&self) -> Vec<i8> {
        assert_eq!(self.dtype, "i8");
        self.bytes.iter().map(|&b| b as i8).collect()
    }
    /// Signed 64-bit integer tensor (the dump format's `i64`).
    pub fn i64(&self) -> Vec<i64> {
        assert_eq!(self.dtype, "i64");
        self.bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| i64::from_le_bytes(*b))
            .collect()
    }
    /// f64 tensor (the dump format's `f64`).
    pub fn f64(&self) -> Vec<f64> {
        assert_eq!(self.dtype, "f64");
        self.bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|b| f64::from_le_bytes(*b))
            .collect()
    }
    /// Integer tensor of any width, widened to i32.
    pub fn i32(&self) -> Vec<i32> {
        match self.dtype.as_str() {
            "i32" => self
                .bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| i32::from_le_bytes(*b))
                .collect(),
            "i16" => self
                .bytes
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| i16::from_le_bytes(*b) as i32)
                .collect(),
            "i8" => self.bytes.iter().map(|&b| b as i8 as i32).collect(),
            // The reference's scale map turns float once RVS adds its (float-typed) table to it.
            // The values are still integers; anything else is a failure, not a rounding matter.
            "f32" => self
                .f32()
                .into_iter()
                .map(|v| {
                    assert!(v.fract() == 0.0 && v.abs() < 1e9, "non-integer value {v}");
                    v as i32
                })
                .collect(),
            t => panic!("not an integer tensor: {t}"),
        }
    }
    pub fn f32(&self) -> Vec<f32> {
        assert_eq!(self.dtype, "f32");
        self.bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()
    }
}

/// The staged decode of `tests/decode_ref.rs`, parameterised on the checkpoint source so the
/// f16-weight tests can run the same pipeline from a packed bundle (`ZJB2`) as from `.pth`
/// files. Fields are exactly what `decode_ref`'s `Decoded` carries.
pub struct StagedDecode {
    pub psi: [zenjpegai::tensor::Tensor<f32>; 2],
    pub y_hat: [zenjpegai::tensor::Tensor<f32>; 2],
    pub planes: zenjpegai::decoder::reconstruct::Planes,
    /// `planes` after the post-filters the stream enables.
    pub filtered: zenjpegai::decoder::reconstruct::Planes,
}

/// Parse + read headers + staged decode.
pub fn decode_staged(
    src: &dyn zenjpegai::model::ModelSource,
    stream: &[u8],
    eng: &zenjpegai::nn::fast::Engine,
) -> (zenjpegai::header::PictureHeader, StagedDecode) {
    let cs = zenjpegai::container::Codestream::parse(stream).unwrap();
    let headers = zenjpegai::decoder::read_headers(&cs).unwrap();
    decode_staged_with(src, &cs, eng, &headers)
}

/// The staged decode with caller-supplied headers (the forced-4:2:0 eICCI vector's tool
/// header is not readable by a conformant parser; its test rebuilds it from the dump).
pub fn decode_staged_with(
    src: &dyn zenjpegai::model::ModelSource,
    cs: &zenjpegai::container::Codestream<'_>,
    eng: &zenjpegai::nn::fast::Engine,
    headers: &zenjpegai::decoder::Headers,
) -> (zenjpegai::header::PictureHeader, StagedDecode) {
    use zenjpegai::decoder::reconstruct::{post_process_latent, reconstruct_latent, synthesize};
    use zenjpegai::decoder::{decode_entropy_stage, output::to_source_format};
    let hdr = headers.picture.clone();
    let id = hdr.model_id as usize;
    let op = hdr.synthesis_transforms[0];
    let (ym, uvm) = (
        zenjpegai::model::load_common(src, id, 0, eng).unwrap(),
        zenjpegai::model::load_common(src, id, 1, eng).unwrap(),
    );
    let syn_y = zenjpegai::model::load_synthesis_primary(src, id, op, eng).unwrap();
    let syn_uv = zenjpegai::model::load_synthesis_secondary(src, id, op, eng).unwrap();
    let ent =
        decode_entropy_stage(&zenjpegai::mans::AnsTables::new(), cs, &hdr, [&ym, &uvm]).unwrap();
    let mut ly = reconstruct_latent(eng, &hdr, 0, &ym, &ent[0]).unwrap();
    let mut luv = reconstruct_latent(eng, &hdr, 1, &uvm, &ent[1]).unwrap();
    post_process_latent(&hdr, &headers.tools, 0, &ent[0], &mut ly).unwrap();
    post_process_latent(&hdr, &headers.tools, 1, &ent[1], &mut luv).unwrap();
    let planes = synthesize(eng, &hdr, &syn_y, &syn_uv, [&ly.y_hat, &luv.y_hat]).unwrap();
    // `model.decompress` hands the planes over in the source's chroma format.
    let planes = to_source_format(&hdr, planes).unwrap();
    let ctx = zenjpegai::filters::FilterContext {
        eng,
        hdr: &hdr,
        tools: &headers.tools,
        luma_scale_log: &ent[0].scale_log,
        models: src,
        op,
        icci_nets: &Default::default(),
        stop: &enough::Unstoppable,
    };
    let filtered = zenjpegai::filters::apply(&ctx, planes.clone()).unwrap();
    let psi = [ly.psi, luv.psi];
    (
        hdr,
        StagedDecode {
            psi,
            y_hat: [ly.y_hat, luv.y_hat],
            planes,
            filtered,
        },
    )
}

/// All tensors of the reference *decoder* dump, by name (`manifest.txt` + `tensors.bin`).
pub fn load_dump(dir: &std::path::Path) -> std::collections::HashMap<String, RefTensor> {
    load_dump_files(dir, "manifest.txt", "tensors.bin")
}

/// All tensors of the reference *encoder* dump (`enc_manifest.txt` + `enc_tensors.bin`).
pub fn load_encoder_dump(dir: &std::path::Path) -> std::collections::HashMap<String, RefTensor> {
    load_dump_files(dir, "enc_manifest.txt", "enc_tensors.bin")
}

fn load_dump_files(
    dir: &std::path::Path,
    manifest: &str,
    blob: &str,
) -> std::collections::HashMap<String, RefTensor> {
    let manifest = std::fs::read_to_string(dir.join(manifest))
        .unwrap_or_else(|e| panic!("{}/{manifest}: {e}", dir.display()));
    let blob = std::fs::read(dir.join(blob)).unwrap();
    let mut out = std::collections::HashMap::new();
    for line in manifest
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
    {
        let f: Vec<&str> = line.split_whitespace().collect();
        let ndim: usize = f[2].parse().unwrap();
        let shape: Vec<usize> = f[3..3 + ndim].iter().map(|d| d.parse().unwrap()).collect();
        let offset: usize = f[3 + ndim].parse().unwrap();
        let nbytes: usize = f[4 + ndim].parse().unwrap();
        let t = RefTensor {
            dtype: f[1].to_string(),
            shape,
            bytes: blob[offset..offset + nbytes].to_vec(),
        };
        out.insert(f[0].to_string(), t);
    }
    out
}
