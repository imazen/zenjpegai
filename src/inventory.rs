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
//! a marker, so a repeated PIH, TON, SOZ, SOQ, RDI or UDI is never read.

use alloc::borrow::Cow;
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use zencodec::inventory::{
    Disposition, Inventory, InventoryError, MetadataKind, Part, PartKind, PartTag,
};

use crate::bitio::BitReader;
use crate::codec::jpegai_format;
use crate::container::Marker;
use crate::header::{PictureHeader, RenderingInfo, ToolHeader};

const SOC: u16 = Marker::Soc as u16;
const EOC: u16 = Marker::Eoc as u16;
const LABEL_MAX: usize = 64;

struct Sub {
    raw: u16,
    marker: Option<Marker>,
    start: u64,
    payload: core::ops::Range<u64>,
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

fn printable(bytes: &[u8]) -> bool {
    !bytes.is_empty()
        && bytes
            .iter()
            .all(|&b| (0x20..0x7F).contains(&b) || matches!(b, b'\n' | b'\r' | b'\t'))
}

pub(crate) fn walk(data: &[u8]) -> Result<Inventory, InventoryError> {
    let len = data.len() as u64;
    let mut inv = Inventory::new(jpegai_format(), len);
    if data.is_empty() {
        return Ok(inv);
    }
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
    inv.push(
        None,
        Part::new(
            PartKind::Header,
            PartTag::Marker(0x80),
            0..2,
            Disposition::Structure,
        )
        .with_label("SOC"),
    )?;

    // Split at markers like `Codestream::parse`.
    let mut subs: Vec<Sub> = Vec::new();
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
            inv.push(
                None,
                Part::new(
                    PartKind::Segment,
                    PartTag::Marker(0x81),
                    pos..pos + 2,
                    Disposition::Structure,
                )
                .with_label("EOC"),
            )?;
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
        if payload_end > len {
            subs.push(Sub {
                raw,
                marker,
                start: pos,
                payload: payload_start..len,
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
        });
        pos = payload_end;
    }

    // What the decoder reads: the first substream of each marker.
    let first = |m: Marker| subs.iter().position(|s| s.marker == Some(m));
    let slice = |s: &Sub| &data[s.payload.start as usize..s.payload.end as usize];
    let pih_parsed =
        first(Marker::Pih).and_then(|i| PictureHeader::parse_consumed(slice(&subs[i])).ok());
    let quality_map = pih_parsed.as_ref().map(|(h, _)| h.quality_map.is_some());
    let rendering_parsed =
        first(Marker::Rdi).and_then(|i| RenderingInfo::parse_consumed(slice(&subs[i])).ok());
    let rendering = rendering_parsed.as_ref().map(|(r, _)| r.clone());
    // Bytes the field parsers read; whatever follows is never looked at.
    let ton_consumed = match (first(Marker::Ton), &pih_parsed) {
        (Some(i), Some((h, _))) => ToolHeader::parse_consumed(slice(&subs[i]), h)
            .ok()
            .map(|(_, n)| n),
        _ => None,
    };
    let consumed_of = |m: Option<Marker>| -> Option<usize> {
        match m {
            Some(Marker::Pih) => pih_parsed.as_ref().map(|(_, n)| *n),
            Some(Marker::Ton) => ton_consumed,
            Some(Marker::Rdi) => rendering_parsed.as_ref().map(|(_, n)| *n),
            _ => None,
        }
    };

    let truncated_last = matches!(&stop_reason, Some((u64::MAX, _)));
    for (i, s) in subs.iter().enumerate() {
        let end = s.payload.end;
        let is_first = s.marker.is_none_or(|m| first(m) == Some(i));
        let (disposition, detail): (Disposition, Option<String>) = match s.marker {
            None => (
                Disposition::Unknown,
                Some("unknown substream marker; the decoder rejects the file".into()),
            ),
            Some(_) if !is_first => (
                Disposition::Dropped,
                Some(
                    "repeated substream: the decoder reads only the first with this marker".into(),
                ),
            ),
            Some(Marker::Pih | Marker::Ton) => (Disposition::Structure, None),
            Some(Marker::Soz | Marker::Sorp | Marker::Sors) => (Disposition::ImageData, None),
            Some(Marker::Soq) => match quality_map {
                Some(true) => (Disposition::ImageData, None),
                _ => (
                    Disposition::Skipped,
                    Some("the picture header declares no quality map, so SOQ is not read".into()),
                ),
            },
            Some(Marker::Rdi) => match &rendering {
                Some(ri) => {
                    let mut native = Vec::new();
                    if ri.mdcv.is_some() {
                        native.push("mastering display");
                    }
                    if ri.clli.is_some() {
                        native.push("content light level");
                    }
                    if ri.display_mapping.is_some() {
                        native.push("display mapping");
                    }
                    let d = if ri.cicp.is_some() {
                        Disposition::Metadata(MetadataKind::Cicp)
                    } else {
                        Disposition::Dropped
                    };
                    let detail = (!native.is_empty()).then(|| {
                        format!(
                            "only CICP reaches zencodec; native-only: {}",
                            native.join(", ")
                        )
                    });
                    (d, detail)
                }
                None => (
                    Disposition::Dropped,
                    Some(
                        "rendering information does not parse; the decoder rejects the file".into(),
                    ),
                ),
            },
            Some(Marker::Udi) => (
                Disposition::Skipped,
                Some("user-defined data: kept only in the native Headers.user_data".into()),
            ),
            Some(Marker::Soc | Marker::Eoc) => (Disposition::Structure, None),
        };
        // The segment is framing (marker, size field); its payload is a child with the
        // disposition below, split further where the payload has an internal end.
        let payload = slice(s);
        let label_part = |p: Part| -> Part {
            if s.marker == Some(Marker::Udi) && printable(payload) {
                let n = payload.len().min(LABEL_MAX);
                p.with_label(Cow::Owned(
                    String::from_utf8_lossy(&payload[..n]).into_owned(),
                ))
            } else {
                p.with_label(name(s.raw))
            }
        };
        let mut detail = detail;
        if s.marker == Some(Marker::Udi) && !printable(payload) {
            let d = detail.unwrap_or_default();
            detail = Some(format!("{d}; {} bytes, not printable", payload.len()));
        }
        let mut disposition = disposition;
        if truncated_last && i + 1 == subs.len() {
            let d = detail.map(|d| format!("{d}; ")).unwrap_or_default();
            detail = Some(format!(
                "{d}truncated: the payload runs past the end of the file"
            ));
            disposition = Disposition::Malformed;
        }
        if matches!(
            s.marker,
            Some(Marker::Soz | Marker::Sorp | Marker::Sors | Marker::Soq)
        ) && is_first
            && disposition == Disposition::ImageData
        {
            let d = detail.map(|d| format!("{d}; ")).unwrap_or_default();
            detail = Some(format!(
                "{d}thread-size and region framing and the bytes after each ANS stream's end are not distinguished"
            ));
        }
        let seg_disposition = if disposition == Disposition::Malformed {
            Disposition::Malformed
        } else if is_first || s.marker.is_none() {
            Disposition::Structure
        } else {
            Disposition::Dropped
        };
        let seg = inv.push(
            None,
            label_part(
                Part::new(
                    PartKind::Segment,
                    PartTag::Marker(s.raw as u8),
                    s.start..end,
                    seg_disposition,
                )
                .with_body(s.start..end),
            ),
        )?;
        // Framing bytes: marker and size field.
        if s.start < s.payload.start {
            inv.push(
                Some(seg),
                Part::new(
                    PartKind::Header,
                    PartTag::None,
                    s.start..s.payload.start,
                    if is_first || s.marker.is_none() {
                        Disposition::Structure
                    } else {
                        Disposition::Dropped
                    },
                )
                .with_detail("marker and size field"),
            )?;
        }
        if s.payload.start < s.payload.end {
            let consumed = if is_first {
                consumed_of(s.marker)
            } else {
                None
            };
            let split = consumed
                .map(|n| s.payload.start + n as u64)
                .filter(|&at| at < s.payload.end);
            let body_end = split.unwrap_or(s.payload.end);
            let mut p = label_part(Part::new(
                PartKind::Block,
                PartTag::Marker(s.raw as u8),
                s.payload.start..body_end,
                disposition,
            ));
            if let Some(d) = detail {
                p = p.with_detail(d);
            }
            inv.push(Some(seg), p)?;
            if let Some(at) = split {
                inv.push(
                    Some(seg),
                    Part::new(
                        PartKind::Gap,
                        PartTag::None,
                        at..s.payload.end,
                        Disposition::Dropped,
                    )
                    .with_detail("bytes after the last field the parser reads"),
                )?;
            }
        }
    }

    match (eoc_end, stop_reason) {
        (Some(end), _) => {
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
        }
        (None, Some((at, why))) if at != u64::MAX => malformed(&mut inv, at, why)?,
        (None, Some(_)) => {}
        (None, None) => {}
    }
    inv.fill_gaps(None, Disposition::Malformed)?;
    Ok(inv)
}
