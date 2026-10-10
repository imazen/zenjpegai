//! Cross-check against the reference software's bitstream probe. For every reference
//! stream the probe prints each substream (grouped by type, in file order) with its
//! region and payload size; the inventory must list the same substreams by role: the
//! k-th substream of a marker is the probe's region k of that type, with the same
//! payload size and, for the entropy-coded ones, `ImageData`. Run through
//! `just inventory-oracle`, which sets `INVENTORY_ORACLE_PROBE`, `ZENJPEGAI_REF` and
//! `INVENTORY_ORACLE_VECTORS`; the test fails loudly when they are not set.
//!
//! The probe itself crashes on some streams; the exact set is pinned below.
#![cfg(feature = "zencodec")]

use std::path::{Path, PathBuf};
use std::process::Command;

use zencodec::decode::{DecodeJob, DecoderConfig};
use zencodec::inventory::{Disposition, PartTag};
use zenjpegai::codec::JpegAiDecoderConfig;

/// Streams the reference probe cannot read, and why (a substring of its stderr).
const PROBE_FAILS: &[(&str, &str)] = &[
    (
        "enc_img30_bop_m1_b0_qmap",
        "'int' object has no attribute 'item'",
    ),
    (
        "enc_img30_bop_m1_b0_qmap_rvs",
        "'int' object has no attribute 'item'",
    ),
    (
        "img30_base_qmap_bpp050",
        "'int' object has no attribute 'item'",
    ),
    (
        "img30_base_qmap_rvs_bpp025",
        "'int' object has no attribute 'item'",
    ),
    (
        "img30_base_qmap_threads8_bpp100",
        "'int' object has no attribute 'item'",
    ),
    // eICCI tool headers fail inside the probe's own TON read (not the quality-map crash):
    // the 4:2:0 stream with `ExceptionReadHeader`, the 4:2:2 one with an `IndexError` in
    // `eICCI/model_idxes.py decode_tile`.
    ("img30yuv420_base_eicci", "ExceptionReadHeader"),
    (
        "img30yuv422_base_eicci",
        "IndexError: list index out of range",
    ),
];

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

/// `(marker, region, payload size)` per probe entry, in the probe's order.
fn parse_probe(text: &str) -> Vec<(u8, usize, u64)> {
    let mut out = Vec::new();
    let (mut marker, mut region) = (None, 0usize);
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix("Substream '") {
            marker = Some(marker_of(rest.split('\'').next().unwrap()));
        } else if let Some(rest) = line.strip_prefix("= region_") {
            region = rest.trim_end_matches(':').parse().unwrap();
        } else if let (Some(m), cells) =
            (marker, line.split('|').map(str::trim).collect::<Vec<_>>())
            && cells.len() >= 3
            && cells[1] == "Size"
        {
            out.push((m, region, cells[2].parse().unwrap()));
        }
    }
    out
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
    let mut probe_failed: Vec<(String, String)> = Vec::new();
    for f in &files {
        let dir_name = f
            .parent()
            .unwrap()
            .file_name()
            .unwrap()
            .to_string_lossy()
            .into_owned();
        let data = std::fs::read(f).unwrap();
        let out = Command::new(&python)
            .current_dir(&reference)
            .env("PYTHONPATH", ".")
            .arg("scripts/bitstream_probe.py")
            .arg(f)
            .output()
            .expect("run probe");
        let text = String::from_utf8_lossy(&out.stdout);
        let probe = parse_probe(&text);
        let inv = JpegAiDecoderConfig::default()
            .job()
            .inventory(&data)
            .unwrap()
            .unwrap();
        inv.validate().unwrap();
        // The inventory's substreams in file order: (marker, payload size, payload part).
        let mine: Vec<(u8, u64, Disposition)> = inv
            .children(None)
            .into_iter()
            .map(|id| (id, inv.get(id).unwrap()))
            .filter_map(|(id, p)| match p.tag {
                PartTag::Marker(m) if matches!(m, 0x82..=0x8C) => {
                    let kids = inv.children(Some(id));
                    let header_end = kids
                        .first()
                        .map(|&k| inv.get(k).unwrap().range.end)
                        .unwrap_or(p.range.start);
                    // The payload part: the first child that is not the framing header.
                    let body = kids
                        .iter()
                        .map(|&k| inv.get(k).unwrap())
                        .find(|k| k.range.start >= header_end && k.kind.name() == "block");
                    Some((
                        m,
                        p.range.end - header_end,
                        body.map_or(p.disposition, |b| b.disposition),
                    ))
                }
                _ => None,
            })
            .collect();
        if probe.is_empty() {
            let err = String::from_utf8_lossy(&out.stderr);
            let why = PROBE_FAILS
                .iter()
                .find(|(n, _)| *n == dir_name)
                .unwrap_or_else(|| panic!("{dir_name}: probe failed unexpectedly:\n{err}"));
            assert!(
                err.contains(why.1),
                "{dir_name}: probe failed differently:\n{err}"
            );
            probe_failed.push((dir_name, why.1.to_string()));
            continue;
        }
        assert!(
            !PROBE_FAILS.iter().any(|(n, _)| *n == dir_name),
            "{dir_name} is pinned as a probe failure but the probe worked"
        );
        // Region k of a type is the k-th substream of that marker in the file.
        let mut seen = std::collections::HashMap::<u8, usize>::new();
        for (m, k, size) in &probe {
            let nth = seen.entry(*m).or_default();
            let (_, psize, disp) = mine
                .iter()
                .filter(|(mm, _, _)| mm == m)
                .nth(*nth)
                .unwrap_or_else(|| panic!("{dir_name}: no substream {m:#x} #{nth}\n{inv}"));
            assert_eq!(*k, *nth, "{dir_name}: probe region numbering for {m:#x}");
            assert_eq!(psize, size, "{dir_name}: {m:#x} region {k} size\n{inv}");
            if matches!(m, 0x88 | 0x89 | 0x8A) {
                assert_eq!(
                    *disp,
                    Disposition::ImageData,
                    "{dir_name}: {m:#x} region {k}\n{inv}"
                );
            }
            *nth += 1;
            substreams += 1;
        }
        // Every substream the inventory lists is one the probe lists, and the type order
        // (first appearance) is the file order.
        assert_eq!(
            mine.len(),
            probe.len(),
            "{dir_name}: substream count\n{inv}"
        );
        let order = |it: &mut dyn Iterator<Item = u8>| {
            let mut v: Vec<u8> = Vec::new();
            for m in it {
                if !v.contains(&m) {
                    v.push(m);
                }
            }
            v
        };
        let mut types_mine: Vec<u8> = mine.iter().map(|(m, _, _)| *m).collect();
        let order_mine = order(&mut types_mine.drain(..));
        let order_probe = order(&mut probe.iter().map(|(m, _, _)| *m));
        let mut a = order_mine.clone();
        let mut b = order_probe.clone();
        a.sort_unstable();
        b.sort_unstable();
        assert_eq!(a, b, "{dir_name}: same set of substream types");
        compared += 1;
    }
    let mut want: Vec<_> = PROBE_FAILS
        .iter()
        .map(|(n, w)| (n.to_string(), w.to_string()))
        .collect();
    want.sort();
    probe_failed.sort();
    assert_eq!(probe_failed, want, "the pinned probe-failure list");
    assert!(compared >= 20, "only {compared} streams compared");
    eprintln!(
        "{compared} reference streams, {substreams} substreams: role, order, size and disposition agree; probe fails on {} (pinned)",
        probe_failed.len()
    );
}
