//! Decoder pipeline.

#[cfg(feature = "std")]
mod api;
pub mod entropy;
pub mod output;
pub mod reconstruct;

#[cfg(feature = "std")]
pub use api::Decoder;

use crate::container::{Codestream, Marker, RegionLayout, split_regions, split_threads};
use crate::error::{Error, Result};
use crate::header::{PictureHeader, RenderingInfo, ToolHeader};
use crate::mans::AnsTables;
use crate::model::CommonModel;

/// Parsed headers of a codestream.
#[derive(Clone, Debug)]
pub struct Headers {
    pub picture: PictureHeader,
    pub tools: ToolHeader,
    pub rendering: RenderingInfo,
    /// User-defined information (UDI substream): opaque bytes, passed through untouched.
    pub user_data: Option<alloc::vec::Vec<u8>>,
}

/// Parse and validate the non-entropy-coded substreams.
pub fn read_headers(cs: &Codestream<'_>) -> Result<Headers> {
    let pih = cs
        .find(Marker::Pih)
        .ok_or(Error::InvalidData("no picture header"))?;
    let picture = PictureHeader::parse(pih)?;
    picture.check_conformance()?;
    let tools = cs
        .find(Marker::Ton)
        .map(|ton| ToolHeader::parse(ton, &picture))
        .transpose()?
        .unwrap_or_default();
    let rendering = cs
        .find(Marker::Rdi)
        .map(RenderingInfo::parse)
        .transpose()?
        .unwrap_or_default();
    Ok(Headers {
        picture,
        tools,
        rendering,
        user_data: cs.find(Marker::Udi).map(<[u8]>::to_vec),
    })
}

/// Run the entropy stage for both components (luma, then chroma).
pub fn decode_entropy_stage(
    tables: &AnsTables,
    cs: &Codestream<'_>,
    hdr: &PictureHeader,
    models: [&CommonModel; 2],
) -> Result<[entropy::ComponentEntropy; 2]> {
    decode_entropy_stage_with(tables, cs, hdr, models, &enough::Unstoppable)
}

/// [`decode_entropy_stage`] that checks `stop` per component, region and channel chunk.
pub fn decode_entropy_stage_with(
    tables: &AnsTables,
    cs: &Codestream<'_>,
    hdr: &PictureHeader,
    models: [&CommonModel; 2],
    stop: &dyn enough::Stop,
) -> Result<[entropy::ComponentEntropy; 2]> {
    decode_entropy_stage_progressive(tables, cs, hdr, models, [None, None], stop)
}

/// Progressive decode: read only the first `max_channels[ccs]` latent channels of each component
/// (`num_decode_chs` in the reference); the remaining channels decode as zero residual. `None`
/// reads every coded channel.
pub fn decode_entropy_stage_progressive(
    tables: &AnsTables,
    cs: &Codestream<'_>,
    hdr: &PictureHeader,
    models: [&CommonModel; 2],
    max_channels: [Option<u16>; 2],
    stop: &dyn enough::Stop,
) -> Result<[entropy::ComponentEntropy; 2]> {
    let soz = cs
        .find(Marker::Soz)
        .ok_or(Error::InvalidData("no z substream"))?;
    let z_threads = split_threads(soz, hdr.num_threads_z as usize)?;
    let mut z_dec = tables.decoder(&z_threads)?;

    // The quality map has its own substream and applies to both components.
    let quality_map = match &hdr.quality_map {
        Some(q) => {
            let soq = cs
                .find(Marker::Soq)
                .ok_or(Error::InvalidData("no quality map substream"))?;
            let (lh, lw) = hdr.latent_size(0);
            Some(crate::tools::qualmap::QualityMap::decode(
                tables,
                soq,
                q,
                lh as usize,
                lw as usize,
            )?)
        }
        None => None,
    };

    let layout = match hdr.regions {
        Some(r) if r.independent => RegionLayout::Independent,
        _ => RegionLayout::Dependent,
    };
    let mut out = alloc::vec::Vec::with_capacity(2);
    for (ccs, marker) in [Marker::Sorp, Marker::Sors].into_iter().enumerate() {
        let payloads: alloc::vec::Vec<&[u8]> = cs.find_all(marker).collect();
        if payloads.is_empty() && layout == RegionLayout::Dependent {
            return Err(Error::InvalidData("missing residual substream"));
        }
        let regions = split_regions(&payloads, layout, hdr.num_regions())?;
        out.push(entropy::decode_component(
            tables,
            hdr,
            ccs,
            models[ccs],
            &mut z_dec,
            &regions,
            quality_map.as_ref(),
            max_channels[ccs],
            stop,
        )?);
    }
    let [y, uv] = <[entropy::ComponentEntropy; 2]>::try_from(out)
        .map_err(|_| Error::InvalidData("internal: component count"))?;
    Ok([y, uv])
}
