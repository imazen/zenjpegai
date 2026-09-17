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

/// Directory holding the reference decoder dumps produced by
/// `scripts/ref_vectors/make_reference_streams.sh`. Hard failure if a vector is missing.
pub fn vector_dir(name: &str) -> PathBuf {
    let root = std::env::var_os("ZENJPEGAI_VECTORS")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/mnt/v/output/zenjpegai/reference/vectors"));
    let dir = root.join(name);
    assert!(
        dir.join("manifest.txt").is_file(),
        "{} is missing: run scripts/ref_vectors/make_reference_streams.sh",
        dir.display()
    );
    dir
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
