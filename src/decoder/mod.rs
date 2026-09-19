//! Decoder pipeline.

use alloc::vec::Vec;

#[cfg(feature = "std")]
mod api;
#[cfg(feature = "std")]
pub mod budget;
pub mod entropy;
pub mod limits;
pub mod output;
pub mod reconstruct;
pub(crate) mod stats;

#[cfg(feature = "std")]
pub use api::Decoder;
#[cfg(feature = "std")]
pub use budget::{BudgetGuard, BudgetPolicy, MemoryBudget};
#[cfg(feature = "std")]
pub use stats::DecodeStats;

use crate::container::{Codestream, Marker, RegionLayout, split_regions, split_threads};
use crate::error::{Error, Result};
use crate::header::{PictureHeader, RenderingInfo, ToolHeader};
use crate::mans::AnsTables;
use crate::model::CommonModel;
use crate::nn::fast::Engine;
use crate::tensor::Tensor;
use entropy::ComponentEntropy;
use stats::{Gate, Probe, Tick, join2, record};

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

/// Entropy-stage options beyond the headers.
#[derive(Clone, Copy, Debug)]
pub struct EntropyOptions {
    /// `num_decode_chs` per component (progressive decode); `None` reads every coded channel.
    pub max_channels: [Option<u16>; 2],
    /// Run independent units — the two components, residual regions, ANS threads — on the
    /// rayon pool. Output is identical either way; this only picks where the work runs.
    pub parallel: bool,
}

impl Default for EntropyOptions {
    fn default() -> Self {
        Self {
            max_channels: [None; 2],
            parallel: cfg!(feature = "parallel"),
        }
    }
}

/// The serial front of the entropy stage: both components' `z` (one shared substream,
/// decoded in order — a true data dependency) and the shared quality map, plus each
/// component's residual region payloads.
struct EntropyFront<'a> {
    z: [Tensor<i8>; 2],
    quality_map: Option<crate::tools::qualmap::QualityMap>,
    regions: [alloc::vec::Vec<Option<&'a [u8]>>; 2],
}

fn entropy_front<'a>(
    tables: &'a AnsTables,
    cs: &'a Codestream<'_>,
    hdr: &PictureHeader,
    models: [&'a CommonModel; 2],
    parallel: bool,
    probe: Option<&'a Probe>,
    stop: &dyn enough::Stop,
) -> Result<EntropyFront<'a>> {
    let soz = cs
        .find(Marker::Soz)
        .ok_or(Error::InvalidData("no z substream"))?;
    let z_threads = split_threads(soz, hdr.num_threads_z as usize)?;
    let mut z_dec = tables.decoder(&z_threads)?;
    // z symbols are one ANS stream covering both components in order — inherently serial —
    // but the quality map has its own substream and decodes concurrently with them.
    let (zr, qr) = join2(
        parallel,
        || -> Result<[Tensor<i8>; 2]> {
            let mut one = |ccs: usize| -> Result<Tensor<i8>> {
                let (hz, wz) = hdr.hyper_latent_size(ccs);
                let t = Tick::now();
                let z = entropy::decode_z(&mut z_dec, models[ccs], hz as usize, wz as usize)?;
                record(Probe::field(probe, ccs, |p| &p.z), t);
                stop.check()?;
                Ok(z)
            };
            Ok([one(0)?, one(1)?])
        },
        || -> Result<Option<crate::tools::qualmap::QualityMap>> {
            let Some(q) = &hdr.quality_map else {
                return Ok(None);
            };
            let t = Tick::now();
            let soq = cs
                .find(Marker::Soq)
                .ok_or(Error::InvalidData("no quality map substream"))?;
            let (lh, lw) = hdr.latent_size(0);
            let qm = crate::tools::qualmap::QualityMap::decode(
                tables,
                soq,
                q,
                lh as usize,
                lw as usize,
            )?;
            record(probe.map(|p| &p.quality_map), t);
            Ok(Some(qm))
        },
    );
    let [z0, z1] = zr?;
    let quality_map = qr?;

    let layout = match hdr.regions {
        Some(r) if r.independent => RegionLayout::Independent,
        _ => RegionLayout::Dependent,
    };
    let mut regions: [alloc::vec::Vec<Option<&[u8]>>; 2] = [Vec::new(), Vec::new()];
    for (ccs, marker) in [Marker::Sorp, Marker::Sors].into_iter().enumerate() {
        let payloads: alloc::vec::Vec<&[u8]> = cs.find_all(marker).collect();
        if payloads.is_empty() && layout == RegionLayout::Dependent {
            return Err(Error::InvalidData("missing residual substream"));
        }
        regions[ccs] = split_regions(&payloads, layout, hdr.num_regions())?;
    }
    Ok(EntropyFront {
        z: [z0, z1],
        quality_map,
        regions,
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
    decode_entropy_impl(
        tables,
        cs,
        hdr,
        models,
        EntropyOptions::default(),
        None,
        stop,
    )
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
    decode_entropy_impl(
        tables,
        cs,
        hdr,
        models,
        EntropyOptions {
            max_channels,
            ..Default::default()
        },
        None,
        stop,
    )
}

fn decode_entropy_impl(
    tables: &AnsTables,
    cs: &Codestream<'_>,
    hdr: &PictureHeader,
    models: [&CommonModel; 2],
    opts: EntropyOptions,
    probe: Option<&Probe>,
    stop: &dyn enough::Stop,
) -> Result<[entropy::ComponentEntropy; 2]> {
    let gate = Gate::new(stop);
    let stop = &gate;
    let front = entropy_front(tables, cs, hdr, models, opts.parallel, probe, stop)?;
    let [z0, z1] = front.z;
    let [regions0, regions1] = front.regions;
    let body = |ccs: usize, z: Tensor<i8>, regions: &[Option<&[u8]>]| -> Result<ComponentEntropy> {
        entropy::decode_component_body(
            tables,
            hdr,
            ccs,
            models[ccs],
            z,
            regions,
            front.quality_map.as_ref(),
            opts.max_channels[ccs],
            opts.parallel,
            probe,
            stop,
        )
    };
    let (y, uv) = join2(
        opts.parallel,
        || body(0, z0, &regions0),
        || body(1, z1, &regions1),
    );
    Ok([y?, uv?])
}

/// Entropy stage + latent reconstruction + LSBS as two per-component chains
/// (luma ‖ chroma), the arrangement [`crate::Decoder`] uses. Chaining keeps the chroma
/// entropy decode overlapped with the luma reconstruction — a bigger overlap than decoding
/// both entropies and then both latents. When `luma_synth` is given, the luma chain continues
/// into its synthesis (it only needs the luma latent) while the chroma chain is still
/// running. Returns the latents and the luma scale map (the post-filters read it).
#[allow(clippy::too_many_arguments)]
pub(crate) fn decode_components(
    tables: &AnsTables,
    cs: &Codestream<'_>,
    hdr: &PictureHeader,
    tools: &ToolHeader,
    eng: &Engine,
    models: [&CommonModel; 2],
    max_channels: [Option<u16>; 2],
    keep_luma_scale: bool,
    luma_synth: Option<reconstruct::LumaSynth<'_>>,
    probe: Option<&Probe>,
    stop: &dyn enough::Stop,
) -> Result<(reconstruct::Latent, reconstruct::Latent, Tensor<i32>)> {
    let parallel = eng.parallel;
    let front = entropy_front(tables, cs, hdr, models, parallel, probe, stop)?;
    let [z0, z1] = front.z;
    let [regions0, regions1] = front.regions;
    let chain = |ccs: usize,
                 z: Tensor<i8>,
                 regions: &[Option<&[u8]>],
                 luma_synth: Option<reconstruct::LumaSynth<'_>>|
     -> Result<(reconstruct::Latent, Tensor<i32>)> {
        let mut e = entropy::decode_component_body(
            tables,
            hdr,
            ccs,
            models[ccs],
            z,
            regions,
            front.quality_map.as_ref(),
            max_channels[ccs],
            parallel,
            probe,
            stop,
        )?;
        // Reconstruction reads the hyper-latent and the residual (LSBS also the `likely`
        // map); the scale map leaves only for the luma post-filters.
        let scale_log = if ccs == 0 && keep_luma_scale {
            core::mem::replace(&mut e.scale_log, Tensor::zeros(0, 0, 0)?)
        } else {
            e.scale_log = Tensor::zeros(0, 0, 0)?; // free now, before the network run
            Tensor::zeros(0, 0, 0)?
        };
        e.skip_scale_log = Tensor::zeros(0, 0, 0)?;
        e.mask = Tensor::zeros(0, 0, 0)?;
        e.residual_q = Tensor::zeros(0, 0, 0)?;
        let t = Tick::now();
        let mut l =
            reconstruct::reconstruct_latent_timed(eng, hdr, ccs, models[ccs], &e, stop, probe)?;
        let t2 = Tick::now();
        reconstruct::post_process_latent_par(parallel, hdr, tools, ccs, &e, &mut l)?;
        record(Probe::field(probe, ccs, |p| &p.lsbs), t2);
        record(Probe::field(probe, ccs, |p| &p.latent), t);
        l.psi = Tensor::zeros(0, 0, 0)?;
        if let Some(s) = luma_synth {
            let t = Tick::now();
            reconstruct::synthesize_luma_into(
                eng, s.tiles, s.model, &l.y_hat, s.out_h, s.out_w, s.rec_y, stop,
            )?;
            record(probe.map(|p| &p.synthesis_luma), t);
        }
        Ok((l, scale_log))
    };
    let (y, uv) = join2(
        parallel,
        || chain(0, z0, &regions0, luma_synth),
        || chain(1, z1, &regions1, None),
    );
    let (ly, scale_log) = y?;
    let (luv, _) = uv?;
    Ok((ly, luv, scale_log))
}
