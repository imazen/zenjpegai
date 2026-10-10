//! Structural inventory of JPEG AI codestreams (`DecodeJob::inventory`). Needs no models: the
//! walker reads only the picture header (for the quality-map flag) and the rendering info.
#![cfg(feature = "zencodec")]

use zencodec::decode::{DecodeJob, DecoderConfig};
use zencodec::inventory::{Inventory, PartId};
use zenjpegai::bitio::BitWriter;
use zenjpegai::codec::JpegAiDecoderConfig;
use zenjpegai::container::{CodestreamWriter, Marker};
use zenjpegai::header::{Cicp, Clli, RenderingInfo};

const PIH_HEX: &str = "011103401f00338000008a32c5001800";

fn pih() -> Vec<u8> {
    PIH_HEX
        .as_bytes()
        .chunks(2)
        .map(|p| u8::from_str_radix(std::str::from_utf8(p).unwrap(), 16).unwrap())
        .collect()
}

fn rdi() -> Vec<u8> {
    RenderingInfo {
        cicp: Some(Cicp {
            colour_primaries: 9,
            transfer_characteristics: 16,
            matrix_coefficients: 0,
            full_range: true,
            chroma420_sample_loc_type: 0,
        }),
        clli: Some(Clli {
            max_content_light_level: 1000,
            max_frame_average_light_level: 400,
        }),
        ..Default::default()
    }
    .write()
    .unwrap()
}

/// A raw substream with any marker, bypassing the writer's marker enum.
fn raw_substream(marker: u16, payload: &[u8]) -> Vec<u8> {
    let mut v = marker.to_be_bytes().to_vec();
    let mut w = BitWriter::new();
    w.write_ue(payload.len() as u64);
    v.extend_from_slice(&w.finish());
    v.extend_from_slice(payload);
    v
}

fn fixture() -> Vec<u8> {
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih()).unwrap();
    w.substream(Marker::Soq, &[9; 5]).unwrap(); // PIH declares no quality map
    w.substream(Marker::Soz, &[1; 20]).unwrap();
    w.substream(Marker::Sorp, &[2; 30]).unwrap();
    w.substream(Marker::Sors, &[3; 10]).unwrap();
    w.substream(Marker::Udi, b"made by Jane Doe's camera")
        .unwrap();
    w.substream(Marker::Udi, &[0xDE, 0xAD, 0xBE, 0xEF]).unwrap(); // repeated UDI
    w.substream(Marker::Rdi, &rdi()).unwrap();
    let mut bytes = w.finish();
    // An unknown substream in the reserved range, placed before EOC.
    let eoc = bytes.split_off(bytes.len() - 2);
    bytes.extend_from_slice(&raw_substream(0xFF90, b"vendor"));
    bytes.extend_from_slice(&eoc);
    bytes.extend_from_slice(b"appended after EOC");
    bytes
}

fn inventory_of(data: &[u8]) -> Inventory {
    let inv = JpegAiDecoderConfig::default()
        .job()
        .inventory(data)
        .unwrap()
        .expect("capability declared");
    inv.validate().unwrap_or_else(|e| panic!("{e}\n{inv}"));
    inv
}

fn render(inv: &Inventory, parent: Option<PartId>, depth: usize, out: &mut Vec<String>) {
    for id in inv.children(parent) {
        let p = inv.get(id).unwrap();
        let mut s = format!(
            "{}{} {} {}..{} {}",
            "  ".repeat(depth),
            p.kind.name(),
            p.tag,
            p.range.start,
            p.range.end,
            p.disposition
        );
        if let Some(l) = &p.label {
            s.push_str(&format!(" {l:?}"));
        }
        out.push(s);
        render(inv, Some(id), depth + 1, out);
    }
}

#[test]
fn pinned_every_marker_a_repeat_an_unknown_one_and_trailing_data() {
    let data = fixture();
    let inv = inventory_of(&data);
    let mut got = Vec::new();
    render(&inv, None, 0, &mut got);
    let want: Vec<String> = EXPECTED.iter().map(|s| s.to_string()).collect();
    assert_eq!(got, want, "\n{}\n{inv}", got.join("\n"));
    zencodec_testkit::check_inventory(JpegAiDecoderConfig::default(), &data).unwrap();
}

const EXPECTED: &[&str] = &[
    "header FF80 0..2 structure \"SOC\"",
    "segment FF82 2..22 structure \"PIH\"",
    "  header - 2..6 structure",
    "  block FF82 6..22 structure \"PIH\"",
    "segment FF8B 22..30 structure \"SOQ\"",
    "  header - 22..25 structure",
    "  block FF8B 25..30 skipped \"SOQ\"",
    "segment FF88 30..54 structure \"SOZ\"",
    "  header - 30..34 structure",
    "  block FF88 34..54 image-data \"SOZ\"",
    "segment FF89 54..88 structure \"SORP\"",
    "  header - 54..58 structure",
    "  block FF89 58..88 image-data \"SORP\"",
    "segment FF8A 88..101 structure \"SORS\"",
    "  header - 88..91 structure",
    "  block FF8A 91..101 image-data \"SORS\"",
    "segment FF8C 101..130 structure \"made by Jane Doe's camera\"",
    "  header - 101..105 structure",
    "  block FF8C 105..130 skipped \"made by Jane Doe's camera\"",
    "segment FF8C 130..137 dropped \"UDI\"",
    "  header - 130..133 dropped",
    "  block FF8C 133..137 dropped \"UDI\"",
    "segment FF84 137..149 structure \"RDI\"",
    "  header - 137..140 structure",
    "  block FF84 140..149 metadata(cicp) \"RDI\"",
    "segment FF90 149..158 structure \"unknown\"",
    "  header - 149..152 structure",
    "  block FF90 152..158 unknown \"unknown\"",
    "segment FF81 158..160 structure \"EOC\"",
    "trailer - 160..178 trailing",
];

#[test]
fn capability_is_declared() {
    assert!(JpegAiDecoderConfig::capabilities().inventory());
}

#[test]
fn malformed_streams_still_tile() {
    for data in [
        &[][..],
        &[0xFF][..],
        &[0xFF, 0x80][..],
        &[0xFF, 0x80, 0xFF][..],
        &[0xFF, 0x80, 0xFF, 0x82, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF][..],
        &[0xFF, 0x80, 0x12, 0x34, 0x56][..],
        b"not a jpeg ai stream at all",
    ] {
        inventory_of(data);
    }
    let data = fixture();
    for n in 0..data.len() {
        inventory_of(&data[..n]);
    }
}

#[test]
fn bytes_after_the_header_fields_are_split_off() {
    let mut w = CodestreamWriter::new();
    let mut padded = pih();
    padded.extend_from_slice(&[0xAA, 0xBB, 0xCC]);
    w.substream(Marker::Pih, &padded).unwrap();
    w.substream(Marker::Soz, &[1; 8]).unwrap();
    let data = w.finish();
    let inv = inventory_of(&data);
    let tail = inv
        .parts()
        .iter()
        .find(|p| p.detail.as_deref() == Some("bytes after the last field the parser reads"))
        .expect("a tail part");
    assert_eq!(tail.range, 22..25);
    assert_eq!(tail.disposition, zencodec::inventory::Disposition::Dropped);
}
