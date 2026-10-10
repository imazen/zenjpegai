//! Cross-check against the reference software's bitstream probe: for every reference
//! stream the probe lists the substreams in file order with their payload sizes; the
//! inventory must list the same substreams with the same sizes. Run through
//! `just inventory-oracle`, which sets `INVENTORY_ORACLE_PROBE` (the reference checkout's
//! python) and `ZENJPEGAI_REF`. Both must be set: the test fails loudly when they are not.
#![cfg(feature = "zencodec")]

use std::path::{Path, PathBuf};
use std::process::Command;

use zencodec::decode::{DecodeJob, DecoderConfig};
use zencodec::inventory::PartTag;
use zenjpegai::codec::JpegAiDecoderConfig;

fn bits_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap() {
            let p = e.unwrap().path();
            if p.is_dir() {
                stack.push(p);
            } else if p.extension().is_some_and(|x| x == "bits") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// Bytes of the `ue(v)` size field for a payload of `n` bytes.
fn ue_bytes(n: u64) -> u64 {
    let k = 63 - (n + 1).leading_zeros() as u64;
    (2 * k + 1).div_ceil(8)
}

fn marker_of(name: &str) -> u8 {
    match name {
        "picture_header" => 0x82,
        "tool_header" => 0x83,
        "rendering_information" => 0x84,
        "z_substream" => 0x88,
        "r_prim_substream" => 0x89,
        "r_sec_substream" => 0x8A,
        "quality_map" => 0x8B,
        "udi" => 0x8C,
        other => panic!("unknown probe substream name {other:?}"),
    }
}

#[test]
#[ignore = "needs the reference checkout: run `just inventory-oracle`"]
fn reference_probe_lists_the_same_substreams() {
    let reference = PathBuf::from(
        std::env::var("ZENJPEGAI_REF").expect("ZENJPEGAI_REF must name the reference checkout"),
    );
    let vectors = PathBuf::from(
        std::env::var("INVENTORY_ORACLE_VECTORS")
            .expect("INVENTORY_ORACLE_VECTORS must name the vectors dir"),
    );
    let python = std::env::var("INVENTORY_ORACLE_PROBE")
        .expect("INVENTORY_ORACLE_PROBE must name the python");
    let files = bits_files(&vectors);
    assert!(
        files.len() >= 20,
        "only {} streams under {}",
        files.len(),
        vectors.display()
    );
    let (mut compared, mut substreams) = (0, 0);
    let mut probe_failed: Vec<String> = Vec::new();
    let mut probe_failed: Vec<String> = Vec::new();
    for f in &files {
        let data = std::fs::read(f).unwrap();
        let out = Command::new(&python)
            .current_dir(&reference)
            .env("PYTHONPATH", ".")
            .arg("scripts/bitstream_probe.py")
            .arg(f)
            .output()
            .expect("run probe");
        let text = String::from_utf8_lossy(&out.stdout);
        let mut probe: Vec<(u8, u64)> = Vec::new();
        let mut current: Option<u8> = None;
        for line in text.lines() {
            if let Some(rest) = line.strip_prefix("Substream '") {
                current = Some(marker_of(rest.split('\'').next().unwrap()));
            } else if let Some(m) = current {
                let cells: Vec<&str> = line.split('|').map(str::trim).collect();
                if cells.len() >= 3 && cells[1] == "Size" {
                    probe.push((m, cells[2].parse().unwrap()));
                }
            }
        }
        let inv = JpegAiDecoderConfig::default()
            .job()
            .inventory(&data)
            .unwrap()
            .unwrap();
        inv.validate().unwrap();
        let mine: Vec<(u8, u64)> = inv
            .children(None)
            .into_iter()
            .map(|id| inv.get(id).unwrap())
            .filter_map(|p| match p.tag {
                PartTag::Marker(m) if matches!(m, 0x82..=0x8C) => {
                    let total = p.range.end - p.range.start - 2;
                    // payload n with 2 + ue_bytes(n) + n == len
                    let n = (0..=total)
                        .rev()
                        .find(|&n| ue_bytes(n) + n == total)
                        .expect("consistent size field");
                    Some((m, n))
                }
                _ => None,
            })
            .collect();
        // The probe prints substreams grouped by marker; compare as sorted (marker, size) lists.
        if probe.is_empty() {
            // The reference probe itself crashes on some streams (quality maps:
            // `'int' object has no attribute 'item'`); the inventory must still tile.
            probe_failed.push(f.display().to_string());
            continue;
        }
        let (mut mine, mut probe) = (mine, probe);
        mine.sort_unstable();
        probe.sort_unstable();
        assert_eq!(mine, probe, "{}\n{inv}", f.display());
        compared += 1;
        substreams += mine.len();
    }
    eprintln!(
        "reference probe crashed on {} streams: {probe_failed:?}",
        probe_failed.len()
    );
    assert!(compared >= 20, "only {compared} streams compared");
    eprintln!(
        "{compared} reference streams, {substreams} substreams: marker and payload size agree"
    );
}
