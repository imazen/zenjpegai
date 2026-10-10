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
    "segment FF8C 130..137 structure \"UDI\"",
    "  header - 130..133 structure",
    "  block FF8C 133..137 dropped \"UDI\"",
    "segment FF84 137..149 structure \"RDI\"",
    "  header - 137..140 structure",
    "  field rdi-flags 140..141 structure",
    "  field cicp 141..143 metadata(cicp) \"RDI\"",
    "  field matrix-coefficients 143..144 dropped",
    "  field cicp-range 144..145 metadata(cicp)",
    "  field clli 145..149 dropped",
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

// ── review round 1 ───────────────────────────────────────────────────

use zencodec::inventory::{Disposition, Part};
use zenjpegai::header::{Mdcv, PictureHeader, QualityMapHeader, Regions, ToolHeader};

fn part_at(inv: &Inventory, at: u64) -> &Part {
    inv.parts()
        .iter()
        .filter(|p| p.range.start <= at && at < p.range.end)
        .min_by_key(|p| p.range.end - p.range.start)
        .unwrap()
}

/// A PIH with independent regions (2 x 1) and a quality map, written back out.
fn region_pih() -> Vec<u8> {
    let mut h = PictureHeader::parse(&pih()).unwrap();
    h.regions = Some(Regions {
        num_ver: 2,
        num_hor: 1,
        independent: true,
        hyper_decoder_overlap: 0,
        mcm_overlap: 0,
    });
    h.quality_map = Some(QualityMapHeader {
        num_threads: 1,
        entropy_index: 0,
    });
    let bytes = h.write().unwrap();
    assert_eq!(PictureHeader::parse(&bytes).unwrap(), h);
    bytes
}

fn full_rdi() -> Vec<u8> {
    RenderingInfo {
        cicp: Some(Cicp {
            colour_primaries: 9,
            transfer_characteristics: 16,
            matrix_coefficients: 0,
            full_range: true,
            chroma420_sample_loc_type: 0,
        }),
        mdcv: Some(Mdcv {
            primaries: [(1, 2), (3, 4), (5, 6)],
            white_point: (7, 8),
            max_luminance: 1000,
            min_luminance: 5,
        }),
        clli: Some(Clli {
            max_content_light_level: 1000,
            max_frame_average_light_level: 400,
        }),
        display_mapping: Some((3, b"SERIAL=AB12345 owner=jane@example.com".to_vec())),
    }
    .write()
    .unwrap()
}

/// TON + qmap PIH + SOQ + two independent regions + a full RDI.
fn full_fixture() -> Vec<u8> {
    let pih_bytes = region_pih();
    let hdr = PictureHeader::parse(&pih_bytes).unwrap();
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih_bytes).unwrap();
    w.substream(Marker::Ton, &ToolHeader::default().write(&hdr).unwrap())
        .unwrap();
    w.substream(Marker::Soq, &[7; 6]).unwrap();
    w.substream(Marker::Soz, &[1; 12]).unwrap();
    w.substream(Marker::Sorp, &[1, 2, 2, 2, 2]).unwrap(); // region 1
    w.substream(Marker::Sorp, &[0, 3, 3, 3]).unwrap(); // region 0
    w.substream(Marker::Sors, &[0, 4, 4]).unwrap();
    w.substream(Marker::Sors, &[1, 5, 5]).unwrap();
    w.substream(Marker::Rdi, &full_rdi()).unwrap();
    let mut b = w.finish();
    b.extend_from_slice(b"after");
    b
}

fn render_all(inv: &Inventory) -> Vec<String> {
    let mut got = Vec::new();
    render(inv, None, 0, &mut got);
    got
}

#[test]
fn pinned_full_fixture_ton_qmap_regions_and_rdi() {
    let data = full_fixture();
    let inv = inventory_of(&data);
    let got = render_all(&inv);
    let want: Vec<String> = EXPECTED_FULL.iter().map(|s| s.to_string()).collect();
    assert_eq!(got, want, "\n{}\n{inv}", got.join("\n"));
    zencodec_testkit::check_inventory(JpegAiDecoderConfig::default(), &data).unwrap();
}

const EXPECTED_FULL: &[&str] = &[
    "header FF80 0..2 structure \"SOC\"",
    "segment FF82 2..25 structure \"PIH\"",
    "  header - 2..6 structure",
    "  block FF82 6..25 structure \"PIH\"",
    "segment FF83 25..29 structure \"TON\"",
    "  header - 25..28 structure",
    "  block FF83 28..29 structure \"TON\"",
    "segment FF8B 29..38 structure \"SOQ\"",
    "  header - 29..32 structure",
    "  block FF8B 32..38 image-data \"SOQ\"",
    "segment FF88 38..53 structure \"SOZ\"",
    "  header - 38..41 structure",
    "  block FF88 41..53 image-data \"SOZ\"",
    "segment FF89 53..61 structure \"SORP\"",
    "  header - 53..56 structure",
    "  field region-index 56..57 structure",
    "  block FF89 57..61 image-data \"SORP\"",
    "segment FF89 61..68 structure \"SORP\"",
    "  header - 61..64 structure",
    "  field region-index 64..65 structure",
    "  block FF89 65..68 image-data \"SORP\"",
    "segment FF8A 68..74 structure \"SORS\"",
    "  header - 68..71 structure",
    "  field region-index 71..72 structure",
    "  block FF8A 72..74 image-data \"SORS\"",
    "segment FF8A 74..80 structure \"SORS\"",
    "  header - 74..77 structure",
    "  field region-index 77..78 structure",
    "  block FF8A 78..80 image-data \"SORS\"",
    "segment FF84 80..157 structure \"RDI\"",
    "  header - 80..84 structure",
    "  field rdi-flags 84..85 structure",
    "  field cicp 85..87 metadata(cicp) \"RDI\"",
    "  field matrix-coefficients 87..88 dropped",
    "  field cicp-range 88..89 metadata(cicp)",
    "  field mdcv 89..113 dropped",
    "  field clli 113..117 dropped",
    "  field dm-type 117..118 dropped",
    "  field dm-size 118..120 structure",
    "  field dm-data 120..157 dropped",
    "segment FF81 157..159 structure \"EOC\"",
    "trailer - 159..164 trailing",
];

// finding 1
#[test]
fn independent_region_substreams_are_image_data() {
    let data = full_fixture();
    let inv = inventory_of(&data);
    let image: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| p.kind.name() == "block" && p.disposition == Disposition::ImageData)
        .filter(|p| matches!(p.tag, zencodec::inventory::PartTag::Marker(0x89 | 0x8A)))
        .collect();
    assert_eq!(image.len(), 4, "{inv}");
    assert!(
        inv.parts()
            .iter()
            .all(|p| p.disposition != Disposition::Dropped
                || !matches!(p.tag, zencodec::inventory::PartTag::Marker(0x89 | 0x8A)))
    );
    // The region index byte is its own Structure field.
    let idx = inv
        .parts()
        .iter()
        .filter(|p| matches!(&p.tag, zencodec::inventory::PartTag::Name(n) if n == "region-index"))
        .count();
    assert_eq!(idx, 4);
}

#[test]
fn bad_region_indices_and_second_residual_without_regions_are_malformed() {
    let pih_bytes = region_pih();
    let hdr = PictureHeader::parse(&pih_bytes).unwrap();
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih_bytes).unwrap();
    w.substream(Marker::Ton, &ToolHeader::default().write(&hdr).unwrap())
        .unwrap();
    w.substream(Marker::Soq, &[7; 6]).unwrap();
    w.substream(Marker::Soz, &[1; 12]).unwrap();
    w.substream(Marker::Sorp, &[0, 2, 2]).unwrap();
    w.substream(Marker::Sorp, &[0, 2, 2]).unwrap(); // duplicate region 0
    w.substream(Marker::Sorp, &[9, 2, 2]).unwrap(); // out of range
    w.substream(Marker::Sorp, &[]).unwrap(); // empty
    let inv = inventory_of(&w.finish());
    let bad: Vec<_> = inv
        .parts()
        .iter()
        .filter(|p| {
            matches!(p.kind.name(), "block" | "header")
                && p.disposition == Disposition::Malformed
                && p.detail.as_deref().is_some_and(|d| d.contains("rejects"))
        })
        .map(|p| p.detail.clone().unwrap())
        .collect();
    assert_eq!(bad.len(), 3, "{inv}");
    assert!(
        bad[0].contains("duplicate") && bad[1].contains("out of range") && bad[2].contains("empty")
    );

    // No regions: the second SORP makes the decoder reject the file.
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih()).unwrap();
    w.substream(Marker::Soz, &[1; 12]).unwrap();
    w.substream(Marker::Sorp, &[2; 5]).unwrap();
    w.substream(Marker::Sorp, &[2; 5]).unwrap();
    let inv = inventory_of(&w.finish());
    let p = part_at(&inv, inv.input_len() - 6 - 1 - 2 + 4);
    assert_eq!(p.disposition, Disposition::Malformed, "{inv}");
    assert!(
        p.detail
            .as_deref()
            .unwrap()
            .contains("the decoder rejects the file")
    );
}

// finding 2
#[test]
fn rdi_fields_are_split_and_native_only_ones_are_dropped() {
    let rdi = full_rdi();
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih()).unwrap();
    w.substream(Marker::Soz, &[1; 4]).unwrap();
    w.substream(Marker::Rdi, &rdi).unwrap();
    let data = w.finish();
    let inv = inventory_of(&data);
    let at = data.windows(5).position(|x| x == b"SERIA").unwrap() as u64;
    let secret = part_at(&inv, at);
    assert_eq!(secret.disposition, Disposition::Dropped, "{inv}");
    assert!(
        secret
            .detail
            .as_deref()
            .unwrap()
            .contains("native RenderingInfo only")
    );
    let cicp = inv
        .parts()
        .iter()
        .find(|p| matches!(&p.tag, zencodec::inventory::PartTag::Name(n) if n == "cicp"))
        .unwrap();
    assert!(matches!(cicp.disposition, Disposition::Metadata(_)));
    assert_eq!(cicp.len(), 2);
    for name in ["matrix-coefficients", "mdcv", "clli", "dm-type", "dm-data"] {
        let p = inv
            .parts()
            .iter()
            .find(|p| matches!(&p.tag, zencodec::inventory::PartTag::Name(n) if n == name))
            .unwrap();
        assert_eq!(p.disposition, Disposition::Dropped, "{name}");
    }
    assert!(inv.unconsumed().any(|(_, p)| p.range.contains(&at)));
}

// finding 3
#[test]
fn many_substreams_walk_in_linear_time() {
    // 2n substreams stay under the walker's substream bound, so the walk completes.
    let n = 120_000usize;
    let mut d = vec![0xFF, 0x80];
    d.extend(raw_substream(0xFF82, &pih()));
    for _ in 0..n {
        d.extend([0xFF, 0x8C, 0x80]);
    }
    for _ in 0..n {
        d.extend([0xFF, 0x88, 0x80]);
    }
    d.extend([0xFF, 0x81]);
    let t = std::time::Instant::now();
    let r = JpegAiDecoderConfig::default().job().inventory(&d);
    let secs = t.elapsed().as_secs_f64();
    eprintln!("n={n}: {} bytes, {secs:.3}s, ok={}", d.len(), r.is_ok());
    assert!(r.is_ok(), "{r:?}");
    assert!(
        secs < 3.0,
        "walk took {secs}s (the quadratic lookup needed several seconds at this size)"
    );
}

// finding 4
#[test]
fn rejected_streams_are_marked() {
    // TON that does not parse.
    let pih_bytes = pih();
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih_bytes).unwrap();
    w.substream(Marker::Ton, &[0xFF, 0xFF]).unwrap();
    w.substream(Marker::Soz, &[1; 4]).unwrap();
    let inv = inventory_of(&w.finish());
    let ton = inv
        .parts()
        .iter()
        .find(|p| {
            p.kind.name() == "block" && matches!(p.tag, zencodec::inventory::PartTag::Marker(0x83))
        })
        .unwrap();
    assert_eq!(ton.disposition, Disposition::Malformed, "{inv}");
    assert!(
        ton.detail
            .as_deref()
            .unwrap()
            .contains("rejects the tool header")
    );
    // No EOC: the note is on SOC (the writer's finish would add EOC, so cut it).
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih_bytes).unwrap();
    let mut b = w.finish();
    b.truncate(b.len() - 2);
    let inv = inventory_of(&b);
    assert!(
        part_at(&inv, 0)
            .detail
            .as_deref()
            .unwrap()
            .contains("no EOC")
    );
    // First substream is not PIH.
    let mut b = vec![0xFF, 0x80];
    b.extend(raw_substream(0xFF8C, b"hello world"));
    b.extend(raw_substream(0xFF82, &pih_bytes));
    b.extend([0xFF, 0x81]);
    let inv = inventory_of(&b);
    assert!(
        part_at(&inv, 0)
            .detail
            .as_deref()
            .unwrap()
            .contains("not PIH")
    );
    // A PIH that fails to parse.
    let mut b = vec![0xFF, 0x80];
    b.extend(raw_substream(0xFF82, &[0xFF; 3]));
    b.extend([0xFF, 0x81]);
    let inv = inventory_of(&b);
    let p = inv
        .parts()
        .iter()
        .find(|p| p.kind.name() == "block")
        .unwrap();
    assert_eq!(p.disposition, Disposition::Malformed);
}

// finding 5
#[test]
fn memory_is_bounded_by_the_part_cap() {
    // More substreams than the cap allows; `INVENTORY_MEM_N=22000000` gives the 64 MiB input
    // used for the `/usr/bin/time -v` measurement in the review report.
    let n: usize = std::env::var("INVENTORY_MEM_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2 << 20);
    let mut d = vec![0xFF, 0x80];
    d.extend(raw_substream(0xFF82, &pih()));
    d.reserve(n * 3);
    for _ in 0..n {
        d.extend([0xFF, 0x8C, 0x80]);
    }
    let e = JpegAiDecoderConfig::default()
        .job()
        .inventory(&d)
        .unwrap_err();
    assert!(e.to_string().contains("part cap"), "{e}");
}

#[test]
fn inventory_honours_the_input_limit() {
    use zencodec::ResourceLimits;
    let data = fixture();
    let job = JpegAiDecoderConfig::default()
        .job()
        .with_limits(ResourceLimits::none().with_max_input_bytes(10));
    assert!(job.inventory(&data).is_err());
}

// finding 6
#[test]
fn udi_labels_use_the_longest_printable_run() {
    let label_of = |payload: &[u8]| -> String {
        let mut w = CodestreamWriter::new();
        w.substream(Marker::Pih, &pih()).unwrap();
        w.substream(Marker::Udi, payload).unwrap();
        let inv = inventory_of(&w.finish());
        inv.parts()
            .iter()
            .find(|p| {
                p.kind.name() == "block"
                    && matches!(p.tag, zencodec::inventory::PartTag::Marker(0x8C))
            })
            .unwrap()
            .label
            .clone()
            .unwrap()
            .to_string()
    };
    assert_eq!(
        label_of(b"zenjpegai UDI test \0\x01\xff payload"),
        "zenjpegai UDI test"
    );
    assert_eq!(
        label_of(b"Owner: Jane Doe, serial 12345\0"),
        "Owner: Jane Doe, serial 12345"
    );
    assert_eq!(label_of(b"\x07Owner: Jane"), "Owner: Jane");
    assert_eq!(
        label_of("Owner: Jörg Müller".as_bytes()),
        "Owner: Jörg Müller"
    );
    assert_eq!(label_of(&[0xDE, 0xAD, 0xBE, 0xEF]), "UDI");
}

// ── review round 2 ───────────────────────────────────────────────────

fn soc_detail(data: &[u8]) -> String {
    let inv = inventory_of(data);
    part_at(&inv, 0).detail.clone().unwrap_or_default()
}

// F2: bytes the decoder reads are not Dropped.
#[test]
fn repeated_substream_framing_and_dm_size_are_structure() {
    let data = fixture();
    let inv = inventory_of(&data);
    // The repeated UDI at 130..137: framing 130..133 is read by Codestream::parse.
    assert_eq!(
        part_at(&inv, 130).disposition,
        Disposition::Structure,
        "{inv}"
    );
    assert_eq!(part_at(&inv, 134).disposition, Disposition::Dropped);
    let full = full_fixture();
    let inv = inventory_of(&full);
    let dm_size = inv
        .parts()
        .iter()
        .find(|p| matches!(&p.tag, zencodec::inventory::PartTag::Name(n) if n == "dm-size"))
        .unwrap();
    assert_eq!(dm_size.disposition, Disposition::Structure);
    assert_eq!(dm_size.len(), 2);
}

// F3: whole-file rejections the walker can see are noted on SOC.
#[test]
fn whole_file_rejections_are_noted_on_soc() {
    // No SOZ, no SORP/SORS (no regions).
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih()).unwrap();
    let d = soc_detail(&w.finish());
    assert!(
        d.contains("no SOZ") && d.contains("no SORP") && d.contains("no SORS"),
        "{d}"
    );
    // Quality map declared, no SOQ.
    let pih_bytes = region_pih();
    let hdr = PictureHeader::parse(&pih_bytes).unwrap();
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih_bytes).unwrap();
    w.substream(Marker::Ton, &ToolHeader::default().write(&hdr).unwrap())
        .unwrap();
    w.substream(Marker::Soz, &[1; 12]).unwrap();
    w.substream(Marker::Sorp, &[0, 2, 2]).unwrap();
    w.substream(Marker::Sors, &[0, 3, 3]).unwrap();
    let d = soc_detail(&w.finish());
    assert!(d.contains("quality map but there is no SOQ"), "{d}");
    // Any Malformed part.
    let mut w = CodestreamWriter::new();
    w.substream(Marker::Pih, &pih()).unwrap();
    w.substream(Marker::Ton, &[0xFF, 0xFF]).unwrap();
    w.substream(Marker::Soz, &[1; 4]).unwrap();
    w.substream(Marker::Sorp, &[2; 4]).unwrap();
    w.substream(Marker::Sors, &[3; 4]).unwrap();
    let d = soc_detail(&w.finish());
    assert!(d.contains("a part is Malformed"), "{d}");
    // A complete stream carries no note.
    assert_eq!(soc_detail(&full_fixture()), "");
}
