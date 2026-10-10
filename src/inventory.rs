//! Structural inventory of a JPEG AI codestream (`zencodec::inventory`).
//!
//! The container is `SOC | PIH | [TON] | [SOQ] | SOZ | SORP.. | SORS.. | [UDI] | [RDI] | EOC`,
//! each substream `marker (2 bytes, big-endian) | size ue(v), byte-aligned | payload`
//! ([`crate::container`]). The walker reports every substream and what the decode path does
//! with it. [`crate::container::Codestream::parse`] rejects unknown markers and ignores bytes
//! after EOC; the walker reports an unknown marker in the reserved `FF80..=FF9F` range as an
//! `Unknown` substream and anything else as `Malformed`.
//!
//! The decoder looks substreams up with `Codestream::find`, which returns the *first* one with
//! a marker, so a repeated PIH, TON, SOZ, SOQ, RDI or UDI is never read. SORP and SORS are
//! different: with independent regions every one of them is a region and is decoded; with
//! dependent or no regions a second one makes the decoder reject the file
//! (`container::split_regions`). A PIH, TON or RDI that does not parse makes the decoder reject
//! the file, so its payload is `Malformed`.
//!
//! Entropy-coded payloads (SOZ, SORP, SORS, SOQ) are `ImageData` as a whole. The ANS decoder
//! reads each thread backwards from its end, so bytes in front of where decoding stops (the
//! start of each thread) are never read; separating them needs the entropy decode and is not
//! done.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;
use core::ops::Range;

use zencodec::inventory::{
    DEFAULT_MAX_PARTS, Disposition, Inventory, InventoryError, MetadataKind, Part, PartId,
    PartKind, PartTag,
};

use crate::bitio::BitReader;
use crate::codec::jpegai_format;
use crate::container::Marker;
use crate::header::{PictureHeader, RenderingInfo, ToolHeader};

const SOC: u16 = Marker::Soc as u16;
const EOC: u16 = Marker::Eoc as u16;
const LABEL_MAX: usize = 64;
/// Parts one substream can produce: the segment, its framing, its payload and a tail split.
const PARTS_PER_SUBSTREAM: usize = 4;
const MAX_SUBSTREAMS: usize = DEFAULT_MAX_PARTS as usize / PARTS_PER_SUBSTREAM;

struct Sub {
    raw: u16,
    marker: Option<Marker>,
    start: u64,
    payload: Range<u64>,
    /// First substream with this marker (the one `Codestream::find` returns).
    is_first: bool,
}

fn name(raw: u16) -> &'static str {
    match Marker::from_u16(raw) {
        Some(Marker::Soc) => "SOC",
        Some(Marker::Eoc) => "EOC",
        Some(Marker::Pih) => "PIH",
        Some(Marker::Ton) => "TON",
        Some(Marker::Rdi) => "RDI",
        Some(Marker::Soz) => "SOZ",
        Some(Marker::Sorp) => "SORP",
        Some(Marker::Sors) => "SORS",
        Some(Marker::Soq) => "SOQ",
        Some(Marker::Udi) => "UDI",
        None => "unknown",
    }
}

/// The label for a UDI: the longest run of at least four printable characters (valid UTF-8,
/// no control characters) in the first [`LABEL_MAX`] bytes, like `strings`.
fn udi_label(payload: &[u8]) -> Option<String> {
    let window = &payload[..payload.len().min(LABEL_MAX)];
    let mut best: Option<&str> = None;
    let mut best_chars = 3usize;
    for chunk in window.utf8_chunks() {
        for run in chunk.valid().split(|c: char| c.is_control()) {
            let n = run.chars().count();
            if n > best_chars {
                best_chars = n;
                best = Some(run);
            }
        }
    }
    best.map(|r| String::from(r.trim()))
}

fn tag(raw: u16) -> PartTag {
    PartTag::Marker(raw as u8)
}

const NATIVE_ONLY: &str = "native RenderingInfo only; not reported through zencodec";

pub(crate) fn walk(data: &[u8]) -> Result<Inventory, InventoryError> {
    let len = data.len() as u64;
    let mut inv = Inventory::new(jpegai_format(), len);
    if data.is_empty() {
        return Ok(inv);
    }
    let too_many = || InventoryError::TooManyParts {
        max: DEFAULT_MAX_PARTS,
    };
    let malformed = |inv: &mut Inventory, from: u64, why: String| -> Result<(), InventoryError> {
        if from < len {
            inv.push(
                None,
                Part::new(
                    PartKind::Gap,
                    PartTag::None,
                    from..len,
                    Disposition::Malformed,
                )
                .with_detail(why),
            )?;
        }
        Ok(())
    };
    if len < 2 || u16::from_be_bytes([data[0], data[1]]) != SOC {
        malformed(&mut inv, 0, "codestream does not start with SOC".into())?;
        return Ok(inv);
    }

    // Split at markers like `Codestream::parse`, bounded by the part cap (each substream
    // becomes up to PARTS_PER_SUBSTREAM parts).
    let mut subs: Vec<Sub> = Vec::new();
    let mut first_of = [false; 13];
    let mut pos: u64 = 2;
    let mut eoc_end = None;
    let mut stop_reason: Option<(u64, String)> = None;
    while pos < len {
        if pos + 2 > len {
            stop_reason = Some((pos, "truncated marker".into()));
            break;
        }
        let raw = u16::from_be_bytes([data[pos as usize], data[pos as usize + 1]]);
        if raw == EOC {
            eoc_end = Some(pos + 2);
            break;
        }
        let marker = Marker::from_u16(raw);
        if raw == SOC || (marker.is_none() && !(0xFF80..=0xFF9F).contains(&raw)) {
            let why = if raw == SOC {
                "SOC inside the codestream".to_owned()
            } else {
                format!("unknown marker {raw:#06X}")
            };
            stop_reason = Some((pos, why));
            break;
        }
        let mut r = BitReader::new(&data[pos as usize + 2..]);
        let size = match r.read_ue() {
            Ok(s) => s,
            Err(e) => {
                stop_reason = Some((pos, format!("unreadable substream size: {e}")));
                break;
            }
        };
        let payload_start = pos + 2 + r.bytes_consumed() as u64;
        let Some(payload_end) = payload_start.checked_add(size) else {
            stop_reason = Some((pos, "substream size overflows".into()));
            break;
        };
        if subs.len() >= MAX_SUBSTREAMS {
            return Err(too_many());
        }
        let is_first = match marker {
            Some(_) => {
                let slot = &mut first_of[usize::from(raw - 0xFF80)];
                let first = !*slot;
                *slot = true;
                first
            }
            None => true,
        };
        if payload_end > len {
            subs.push(Sub {
                raw,
                marker,
                start: pos,
                payload: payload_start..len,
                is_first,
            });
            stop_reason = Some((
                u64::MAX,
                "substream payload runs past the end of the file".into(),
            ));
            break;
        }
        subs.push(Sub {
            raw,
            marker,
            start: pos,
            payload: payload_start..payload_end,
            is_first,
        });
        pos = payload_end;
    }

    let slice = |s: &Sub| &data[s.payload.start as usize..s.payload.end as usize];
    let find_first = |m: Marker| subs.iter().find(|s| s.marker == Some(m) && s.is_first);

    // What the decoder reads: the first substream of each marker.
    let pih = find_first(Marker::Pih).map(|s| {
        PictureHeader::parse_consumed(slice(s))
            .and_then(|(h, n)| h.check_conformance().map(|()| (h, n)))
            .map_err(|e| e.to_string())
    });
    let pih_ok = pih.as_ref().and_then(|r| r.as_ref().ok());
    let ton = match (find_first(Marker::Ton), pih_ok) {
        (Some(s), Some((h, _))) => {
            Some(ToolHeader::parse_consumed(slice(s), h).map_err(|e| e.to_string()))
        }
        _ => None,
    };
    let rdi = find_first(Marker::Rdi)
        .map(|s| RenderingInfo::parse_consumed(slice(s)).map_err(|e| e.to_string()));
    let regions = pih_ok.and_then(|(h, _)| h.regions);
    let independent = regions.is_some_and(|r| r.independent);
    let num_regions = pih_ok.map_or(1, |(h, _)| h.num_regions());
    let quality_map = pih_ok.map(|(h, _)| h.quality_map.is_some());
    // Region indices seen so far (independent regions), per residual component.
    let mut seen_region = [
        vec![false; num_regions.min(128 * 128)],
        vec![false; num_regions.min(128 * 128)],
    ];

    // SOC, with a note when the decoder would reject the stream as a whole.
    let mut soc_notes: Vec<String> = Vec::new();
    if eoc_end.is_none() {
        soc_notes.push("no EOC substream".into());
    }
    if subs.first().is_some_and(|s| s.marker != Some(Marker::Pih)) || subs.is_empty() {
        soc_notes.push("the first substream is not PIH".into());
    }
    // Substreams the decoder requires once the picture header parses.
    if pih_ok.is_some() {
        let has = |m: Marker| subs.iter().any(|s| s.marker == Some(m));
        if !has(Marker::Soz) {
            soc_notes.push("no SOZ substream".into());
        }
        if !independent {
            for (m, n) in [(Marker::Sorp, "SORP"), (Marker::Sors, "SORS")] {
                if !has(m) {
                    soc_notes.push(format!("no {n} substream (missing residual substream)"));
                }
            }
        }
        if quality_map == Some(true) && !has(Marker::Soq) {
            soc_notes.push("the picture header declares a quality map but there is no SOQ".into());
        }
    }
    let soc_id = inv.push(
        None,
        Part::new(
            PartKind::Header,
            PartTag::Marker(0x80),
            0..2,
            Disposition::Structure,
        )
        .with_label("SOC"),
    )?;

    let truncated_last = matches!(&stop_reason, Some((u64::MAX, _)));
    for (i, s) in subs.iter().enumerate() {
        let end = s.payload.end;
        let payload = slice(s);
        let mut label: Cow<'static, str> = name(s.raw).into();
        // (disposition, detail) of the payload; Malformed when the decoder cannot parse it.
        let mut disposition = Disposition::Structure;
        let mut detail: Option<String> = None;
        let mut region_index_byte = false;
        let mut consumed: Option<usize> = None;
        let mut rdi_fields: Option<&RenderingInfo> = None;
        let mut dropped_whole = false;
        match s.marker {
            None => {
                disposition = Disposition::Unknown;
                detail = Some("unknown substream marker; the decoder rejects the file".into());
            }
            Some(Marker::Sorp | Marker::Sors) => {
                let ccs = usize::from(s.marker == Some(Marker::Sors));
                if independent {
                    disposition = Disposition::ImageData;
                    region_index_byte = true;
                    detail = Some("independent region: each SORP/SORS is decoded".into());
                    match payload.first().map(|&b| usize::from(b)) {
                        None => {
                            disposition = Disposition::Malformed;
                            region_index_byte = false;
                            detail =
                                Some("empty region substream: the decoder rejects the file".into());
                        }
                        Some(idx) if idx >= num_regions => {
                            disposition = Disposition::Malformed;
                            detail = Some(format!(
                                "region index {idx} out of range ({num_regions} regions): the decoder rejects the file"
                            ));
                        }
                        Some(idx) if seen_region[ccs][idx] => {
                            disposition = Disposition::Malformed;
                            detail = Some(format!(
                                "duplicate region index {idx}: the decoder rejects the file"
                            ));
                        }
                        Some(idx) => seen_region[ccs][idx] = true,
                    }
                } else if s.is_first {
                    disposition = Disposition::ImageData;
                } else {
                    disposition = Disposition::Malformed;
                    detail = Some(
                        "a second SORP/SORS without independent regions: the decoder rejects the file"
                            .into(),
                    );
                }
            }
            Some(_) if !s.is_first => {
                disposition = Disposition::Dropped;
                dropped_whole = true;
                detail = Some(
                    "repeated substream: the decoder reads only the first with this marker".into(),
                );
            }
            Some(Marker::Pih) => match &pih {
                Some(Ok((_, n))) => consumed = Some(*n),
                Some(Err(e)) => {
                    disposition = Disposition::Malformed;
                    detail = Some(format!("the decoder rejects the picture header: {e}"));
                }
                None => {}
            },
            Some(Marker::Ton) => match &ton {
                Some(Ok((_, n))) => consumed = Some(*n),
                Some(Err(e)) => {
                    disposition = Disposition::Malformed;
                    detail = Some(format!("the decoder rejects the tool header: {e}"));
                }
                // The picture header failed first; the tool header is not reached.
                None => {}
            },
            Some(Marker::Soz) => disposition = Disposition::ImageData,
            Some(Marker::Soq) => match quality_map {
                Some(true) => disposition = Disposition::ImageData,
                _ => {
                    disposition = Disposition::Skipped;
                    detail = Some(
                        "the picture header declares no quality map, so SOQ is not read".into(),
                    );
                }
            },
            Some(Marker::Rdi) => match &rdi {
                Some(Ok((ri, n))) => {
                    consumed = Some(*n);
                    rdi_fields = Some(ri);
                }
                Some(Err(e)) => {
                    disposition = Disposition::Malformed;
                    detail = Some(format!(
                        "the decoder rejects the rendering information: {e}"
                    ));
                }
                None => {}
            },
            Some(Marker::Udi) => {
                disposition = Disposition::Skipped;
                match udi_label(payload) {
                    Some(l) => {
                        label = Cow::Owned(l);
                        detail = Some(format!(
                            "user-defined data ({} bytes): kept only in the native Headers.user_data",
                            payload.len()
                        ));
                    }
                    None => {
                        detail = Some(format!(
                            "user-defined data ({} bytes, no printable text): kept only in the native Headers.user_data",
                            payload.len()
                        ));
                    }
                }
            }
            Some(Marker::Soc | Marker::Eoc) => {}
        }
        if matches!(
            s.marker,
            Some(Marker::Soz | Marker::Sorp | Marker::Sors | Marker::Soq)
        ) && disposition == Disposition::ImageData
        {
            let d = detail.map(|d| format!("{d}; ")).unwrap_or_default();
            detail = Some(format!(
                "{d}the decoder reads each ANS thread backwards from its end, so unread bytes sit at the start of each thread; thread-size framing and those bytes are not distinguished"
            ));
        }
        if truncated_last && i + 1 == subs.len() {
            let d = detail.map(|d| format!("{d}; ")).unwrap_or_default();
            detail = Some(format!(
                "{d}truncated: the payload runs past the end of the file"
            ));
            disposition = Disposition::Malformed;
        }
        // `Codestream::parse` reads every marker and size field, repeated substreams
        // included, so only a repeat's payload is Dropped.
        let _ = dropped_whole;
        let framing_disposition = if disposition == Disposition::Malformed {
            Disposition::Malformed
        } else {
            Disposition::Structure
        };
        let seg = inv.push(
            None,
            Part::new(
                PartKind::Segment,
                tag(s.raw),
                s.start..end,
                framing_disposition,
            )
            .with_label(label.clone())
            .with_body(s.start..end),
        )?;
        if s.start < s.payload.start {
            inv.push(
                Some(seg),
                Part::new(
                    PartKind::Header,
                    PartTag::None,
                    s.start..s.payload.start,
                    framing_disposition,
                )
                .with_detail(match (&detail, s.payload.start >= s.payload.end) {
                    // An empty payload has no part of its own to carry the reason.
                    (Some(d), true) => format!("marker and size field; {d}"),
                    _ => "marker and size field".to_string(),
                }),
            )?;
        }
        if s.payload.start >= s.payload.end {
            continue;
        }
        let mut at = s.payload.start;
        if region_index_byte {
            inv.push(
                Some(seg),
                Part::new(
                    PartKind::Field,
                    PartTag::Name("region-index".into()),
                    at..at + 1,
                    Disposition::Structure,
                ),
            )?;
            at += 1;
        }
        let tail_at = consumed
            .map(|n| s.payload.start + n as u64)
            .filter(|&t| t < s.payload.end);
        let body_end = tail_at.unwrap_or(s.payload.end);
        if let Some(ri) = rdi_fields {
            at =
                push_rendering_fields(&mut inv, seg, ri, label.clone(), at, body_end, &mut detail)?;
        }
        if at < body_end {
            let mut p =
                Part::new(PartKind::Block, tag(s.raw), at..body_end, disposition).with_label(label);
            if let Some(d) = detail {
                p = p.with_detail(d);
            }
            inv.push(Some(seg), p)?;
        }
        if let Some(t) = tail_at {
            inv.push(
                Some(seg),
                Part::new(
                    PartKind::Gap,
                    PartTag::None,
                    t..s.payload.end,
                    Disposition::Dropped,
                )
                .with_detail("bytes after the last field the parser reads"),
            )?;
        }
    }

    // EOC, then whatever follows it.
    if let Some(end) = eoc_end {
        inv.push(
            None,
            Part::new(
                PartKind::Segment,
                PartTag::Marker(0x81),
                end - 2..end,
                Disposition::Structure,
            )
            .with_label("EOC"),
        )?;
        if end < len {
            inv.push(
                None,
                Part::new(
                    PartKind::Trailer,
                    PartTag::None,
                    end..len,
                    Disposition::Trailing,
                )
                .with_detail("after EOC; the decoder ignores it"),
            )?;
        }
    } else if let Some((at, why)) = stop_reason
        && at != u64::MAX
    {
        malformed(&mut inv, at, why)?;
    }
    inv.fill_gaps(None, Disposition::Malformed)?;
    // One note on SOC covers every consumed claim of a file the decoder rejects.
    if inv
        .parts()
        .iter()
        .any(|p| p.disposition == Disposition::Malformed)
    {
        soc_notes.push("a part is Malformed (see its detail)".into());
    }
    if !soc_notes.is_empty() {
        inv.set_detail(
            soc_id,
            format!("the decoder rejects the file: {}", soc_notes.join("; ")),
        );
    }
    Ok(inv)
}

/// Push the fixed-offset fields of a parsed RDI (flags, CICP, MDCV, CLLI, display mapping)
/// as children of `seg`, clipped to `limit`. Returns where the remaining bytes start.
#[allow(clippy::too_many_arguments)]
fn push_rendering_fields(
    inv: &mut Inventory,
    seg: PartId,
    ri: &RenderingInfo,
    label: Cow<'static, str>,
    mut at: u64,
    limit: u64,
    detail: &mut Option<String>,
) -> Result<u64, InventoryError> {
    let field = |inv: &mut Inventory,
                 at: &mut u64,
                 n: u64,
                 name: &'static str,
                 d: Disposition,
                 why: Option<&'static str>|
     -> Result<(), InventoryError> {
        let end = (*at + n).min(limit);
        if *at < end {
            let mut p = Part::new(PartKind::Field, PartTag::Name(name.into()), *at..end, d);
            if name == "cicp" {
                p = p.with_label(label.clone());
            }
            if let Some(w) = why {
                p = p.with_detail(w);
            }
            inv.push(Some(seg), p)?;
        }
        *at = end;
        Ok(())
    };
    field(inv, &mut at, 1, "rdi-flags", Disposition::Structure, None)?;
    if ri.cicp.is_some() {
        field(
            inv,
            &mut at,
            2,
            "cicp",
            Disposition::Metadata(MetadataKind::Cicp),
            Some("colour primaries and transfer characteristics"),
        )?;
        field(
            inv,
            &mut at,
            1,
            "matrix-coefficients",
            Disposition::Dropped,
            Some("forced to 0 through zencodec; native RenderingInfo only"),
        )?;
        field(
            inv,
            &mut at,
            1,
            "cicp-range",
            Disposition::Metadata(MetadataKind::Cicp),
            Some(
                "full-range flag; the 3 chroma sample location bits in this byte are native RenderingInfo only",
            ),
        )?;
    }
    if ri.mdcv.is_some() {
        field(
            inv,
            &mut at,
            24,
            "mdcv",
            Disposition::Dropped,
            Some(NATIVE_ONLY),
        )?;
    }
    if ri.clli.is_some() {
        field(
            inv,
            &mut at,
            4,
            "clli",
            Disposition::Dropped,
            Some(NATIVE_ONLY),
        )?;
    }
    if let Some((_, dm)) = &ri.display_mapping {
        field(
            inv,
            &mut at,
            1,
            "dm-type",
            Disposition::Dropped,
            Some(NATIVE_ONLY),
        )?;
        field(
            inv,
            &mut at,
            2,
            "dm-size",
            Disposition::Structure,
            Some("read to find the end of the rendering information"),
        )?;
        field(
            inv,
            &mut at,
            dm.len() as u64,
            "dm-data",
            Disposition::Dropped,
            Some(NATIVE_ONLY),
        )?;
    }
    *detail = None;
    Ok(at)
}
