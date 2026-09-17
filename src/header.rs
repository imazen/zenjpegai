//! Picture header (PIH), tool header (TON), rendering information (RDI) and user data (UDI).
//!
//! In the reference these substreams are written by walking the tool tree
//! (`CoderEngine.encode_header_recursively`); every tool appends its own fields to the substream
//! named by its `stream_header_part`. The walk order is fixed by the order in which the tools are
//! attached to their parents, so the syntax below is the flattened result of that walk:
//!
//! ```text
//! PIH: CodingEngine | colour_transform | model_id | CcsGvaeSGMM (threads z, beta[0], regions,
//!      beta[1]) | per component: threads r, num_chs, cube flags, rvs/grfs, synthesis tiling
//!      | quality map
//! TON: lsbs[0] | lsbs[1] | EFE linear | eICCI | EFE non-linear | LEF
//! ```
//!
//! Sources: `coding_engine.py`, `colour_transformation.py`, `multitools/engine.py`,
//! `ccs_sgmm_tool.py`, `quantization.py`, `sep_chan_tool.py`, `skip_mode.py`,
//! `res_var_scale.py`, `tiling.py`, `quality_map.py`, `lsbs_scale_mode.py`, `rdi.py`, `udi.py`.

use alloc::vec::Vec;

use crate::bitio::{BitReader, BitWriter};
use crate::error::{Error, Result};

/// Bit depths selectable by `bit_depth_idc`. The reference decoder only accepts the first two.
const BIT_DEPTHS: [u8; 5] = [8, 10, 12, 14, 16];
/// `img_width_minus64` / `img_height_minus64` are bounded by this value.
const MAX_DIM_MINUS64: u32 = (1 << 16) - 1;
/// `MultiToolsEngine(max_models_count=16)`: `model_id` is coded with 4 bits.
const MAX_MODELS: u32 = 16;
/// Luma / chroma latent channel counts (`CcsGvaeSGMM.__init__`).
pub const LATENT_CHANNELS: [usize; 2] = [160, 96];
/// Skip-mode cube geometry (`SkipModeCoder`): 8x8 spatial cubes over all channels of one MCM
/// phase, flags sent in groups of 8.
const CUBE_SIZE: usize = 8;
const CUBE_GROUP_SIZE: usize = 8;

/// Synthesis transform selected by `synthesis_transform_id`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum OperatingPoint {
    /// Simple operating point (id 0).
    Sop = 0,
    /// Base operating point (id 1).
    Bop = 1,
    /// High operating point (id 2).
    Hop = 2,
}

impl OperatingPoint {
    fn from_id(id: u32) -> Result<Self> {
        match id {
            0 => Ok(Self::Sop),
            1 => Ok(Self::Bop),
            2 => Ok(Self::Hop),
            _ => Err(Error::InvalidData("unknown synthesis_transform_id")),
        }
    }
}

/// `colour_transform_idx` and its parameters.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ColourTransform {
    /// 0: the coded components are output as they are (YUV in, YUV out).
    None,
    /// 1: BT.709 YCbCr, converted to RGB on output.
    Bt709,
    /// 2: user-defined 3x3 matrix and offsets (8 bits each).
    Custom { matrix: [u8; 9], offset: [u8; 3] },
}

/// Region partitioning of the residual substreams.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Regions {
    pub num_ver: u8,
    pub num_hor: u8,
    /// `region_residual_in_its_own_substream_flag`.
    pub independent: bool,
    /// `hyper_decoder_overlap_in_latent_samples` (dependent regions only; else 0).
    pub hyper_decoder_overlap: u8,
    /// `mcm_overlap_in_latent_samples` (dependent regions only; else 0).
    pub mcm_overlap: u8,
}

/// Synthesis tiling parameters (`synthesis_tile_enable`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SynthesisTiling {
    /// Tile size in luma samples (multiple of 16).
    pub tile_size: u32,
    /// Overlap between neighbouring tiles in luma samples (multiple of 16).
    pub overlap: u32,
}

/// Per-component (luma, chroma) part of the picture header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ComponentHeader {
    /// ANS threads of the residual substream: 1, 2, 4, 8 or 16.
    pub num_threads_r: u8,
    /// Number of latent channels present in the stream (`num_chs`).
    pub num_chs: u16,
    /// Skip-mode cube flags, `[phase][cube_y][cube_x]` flattened; `None` means all cubes skip.
    pub cube_flags: Option<Vec<bool>>,
    pub rvs_enabled: bool,
    /// Channel-wise gain flags (`grfs_channel_flag`), one per coded channel, if enabled.
    pub grfs_channel_flags: Option<Vec<bool>>,
    pub synthesis_tiling: Option<SynthesisTiling>,
}

/// Quality-map (spatially varying quantisation) signalling.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QualityMapHeader {
    pub num_threads: u8,
    /// `quality_map_entropy_index`: selects the sigma used to code the map.
    pub entropy_index: u8,
}

/// Decoded picture header.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PictureHeader {
    pub stream_profile_idc: u8,
    /// 0 simple, 1 base, 2 high.
    pub decoder_profile_id: u8,
    /// Synthesis transforms the stream may be decoded with; the first is the default.
    pub synthesis_transforms: Vec<OperatingPoint>,
    pub level_idc: u8,
    /// Coded picture size in luma samples.
    pub width: u32,
    pub height: u32,
    /// Columns / rows at the right / bottom of the coded picture that are not displayed.
    pub diff_display_width: u8,
    pub diff_display_height: u8,
    pub bit_depth: u8,
    /// Source chroma subsampling factors (1 or 2).
    pub s_ver: u8,
    pub s_hor: u8,
    /// Coded chroma subsampling factors (1 or 2).
    pub c_ver: u8,
    pub c_hor: u8,
    pub colour_transform: ColourTransform,
    /// Index of the trained model (beta) the stream was coded with.
    pub model_id: u8,
    pub num_threads_z: u8,
    /// `beta_displacement_log` per component (already offset by -2048).
    pub beta_displacement_log: [i32; 2],
    pub regions: Option<Regions>,
    pub components: [ComponentHeader; 2],
    pub quality_map: Option<QualityMapHeader>,
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

impl PictureHeader {
    /// Size of component `ccs`'s picture as the core model sees it: the luma size for luma,
    /// half of it (rounded up) for chroma (`SepChannelsSGMMTool.get_processed_img_shape`).
    pub fn component_size(&self, ccs: usize) -> (u32, u32) {
        if ccs == 0 {
            (self.height, self.width)
        } else {
            (ceil_div(self.height, 2), ceil_div(self.width, 2))
        }
    }

    /// Latent (`y`) size of a component: picture size / 16 for luma, half-size picture / 8 for
    /// chroma (`get_size_on_depth(h, w, 4, skip_depth_step)`).
    pub fn latent_size(&self, ccs: usize) -> (u32, u32) {
        let (h, w) = self.component_size(ccs);
        let div = if ccs == 0 { 16 } else { 8 };
        (ceil_div(h, div), ceil_div(w, div))
    }

    /// Hyper-latent (`z`) size: one quarter of the latent size, rounded up from the picture.
    pub fn hyper_latent_size(&self, ccs: usize) -> (u32, u32) {
        let (h, w) = self.component_size(ccs);
        let div = if ccs == 0 { 64 } else { 32 };
        (ceil_div(h, div), ceil_div(w, div))
    }

    /// Number of regions the residual substreams are split into.
    pub fn num_regions(&self) -> usize {
        self.regions
            .map_or(1, |r| r.num_ver as usize * r.num_hor as usize)
    }

    /// Number of skip-mode cube flags of a component: 4 phases of `ceil(ceil(h/2)/8)` rows.
    fn cube_flag_count(&self, ccs: usize) -> usize {
        let (h, w) = self.latent_size(ccs);
        let cube_h = (h as usize).div_ceil(2).div_ceil(CUBE_SIZE);
        let cube_w = (w as usize).div_ceil(2).div_ceil(CUBE_SIZE);
        4 * cube_h * cube_w
    }

    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(payload);

        // CodingEngine.decode_header
        let stream_profile_idc = r.read_bits(4)? as u8;
        let decoder_profile_id = r.read_bits(4)? as u8;
        let n_transforms = r.read_bits(4)? as usize + 1;
        let mut synthesis_transforms = Vec::with_capacity(n_transforms);
        for _ in 0..n_transforms {
            synthesis_transforms.push(OperatingPoint::from_id(r.read_bits(4)?)?);
        }
        let level_idc = r.read_bits(8)? as u8;
        let width = r.read_bounded(MAX_DIM_MINUS64)? + 64;
        let height = r.read_bounded(MAX_DIM_MINUS64)? + 64;
        let diff_display_width = r.read_bits(6)? as u8;
        let diff_display_height = r.read_bits(6)? as u8;
        let bit_depth_idc = r.read_bounded(BIT_DEPTHS.len() as u32 - 1)? as usize;
        if bit_depth_idc > 1 {
            return Err(Error::InvalidData(
                "bit_depth_idc: only 8 and 10 bit are defined",
            ));
        }
        let bit_depth = BIT_DEPTHS[bit_depth_idc];
        let s_ver = r.read_bits(1)? as u8 + 1;
        let s_hor = r.read_bits(1)? as u8 + 1;
        let c_ver = if s_ver == 1 {
            r.read_bits(1)? as u8 + 1
        } else {
            2
        };
        let c_hor = if s_hor == 1 {
            r.read_bits(1)? as u8 + 1
        } else {
            2
        };

        // ColourTransformation.decode_header
        let colour_transform = match r.read_bounded(2)? {
            0 => ColourTransform::None,
            1 => ColourTransform::Bt709,
            2 => {
                let mut matrix = [0u8; 9];
                for m in &mut matrix {
                    *m = r.read_bits(8)? as u8;
                }
                let mut offset = [0u8; 3];
                for o in &mut offset {
                    *o = r.read_bits(8)? as u8;
                }
                ColourTransform::Custom { matrix, offset }
            }
            _ => return Err(Error::InvalidData("colour_transform_idx out of range")),
        };

        // MultiToolsEngine.decode_header
        let model_id = r.read_bounded(MAX_MODELS - 1)? as u8;

        // CcsGvaeSGMM.decode_header
        let num_threads_z = read_thread_count(&mut r)?;
        let beta0 = r.read_bits(12)? as i32 - 2048;
        let regions = if r.read_bit()? {
            let num_ver = r.read_bits(7)? + 1;
            let num_hor = r.read_bits(7)? + 1;
            if num_ver > ceil_div(height, 256) || num_hor > ceil_div(width, 512) {
                return Err(Error::InvalidData(
                    "more regions than the picture size allows",
                ));
            }
            let independent = r.read_bit()?;
            let (hd, mcm) = if independent {
                (0, 0)
            } else {
                (r.read_bits(2)? as u8, r.read_bits(4)? as u8)
            };
            Some(Regions {
                num_ver: num_ver as u8,
                num_hor: num_hor as u8,
                independent,
                hyper_decoder_overlap: hd,
                mcm_overlap: mcm,
            })
        } else {
            None
        };
        let beta1 = if r.read_bit()? {
            r.read_bits(12)? as i32 - 2048
        } else {
            beta0
        };

        let mut hdr = PictureHeader {
            stream_profile_idc,
            decoder_profile_id,
            synthesis_transforms,
            level_idc,
            width,
            height,
            diff_display_width,
            diff_display_height,
            bit_depth,
            s_ver,
            s_hor,
            c_ver,
            c_hor,
            colour_transform,
            model_id,
            num_threads_z,
            beta_displacement_log: [beta0, beta1],
            regions,
            components: [ComponentHeader::empty(), ComponentHeader::empty()],
            quality_map: None,
        };

        for ccs in 0..2 {
            hdr.components[ccs] = ComponentHeader::parse(&mut r, &hdr, ccs)?;
        }

        // QualityMap (`gain_3D_enable_flag`)
        if r.read_bit()? {
            // The reference encoder writes log2_num_threads_q_minus1 with 2 bits and its decoder
            // reads 1; the encoder (and the z / r fields) win. See PORTING.md.
            let num_threads = read_thread_count(&mut r)?;
            let entropy_index = r.read_bounded(7)? as u8;
            hdr.quality_map = Some(QualityMapHeader {
                num_threads,
                entropy_index,
            });
        }
        Ok(hdr)
    }

    pub fn write(&self) -> Result<Vec<u8>> {
        let mut w = BitWriter::new();
        let n = self.synthesis_transforms.len();
        if n == 0 || n > 16 {
            return Err(Error::InvalidArgument(
                "1..=16 synthesis transforms required",
            ));
        }
        if !(64..=MAX_DIM_MINUS64 + 64).contains(&self.width)
            || !(64..=MAX_DIM_MINUS64 + 64).contains(&self.height)
        {
            return Err(Error::InvalidArgument("picture dimensions out of range"));
        }
        if self.diff_display_width > 63 || self.diff_display_height > 63 {
            return Err(Error::InvalidArgument("diff_display_* must be below 64"));
        }
        let bit_depth_idc = match self.bit_depth {
            8 => 0,
            10 => 1,
            _ => return Err(Error::InvalidArgument("bit depth must be 8 or 10")),
        };
        w.write_bits(self.stream_profile_idc as u32, 4);
        w.write_bits(self.decoder_profile_id as u32, 4);
        w.write_bits(n as u32 - 1, 4);
        for &t in &self.synthesis_transforms {
            w.write_bits(t as u32, 4);
        }
        w.write_bits(self.level_idc as u32, 8);
        w.write_bounded(self.width - 64, MAX_DIM_MINUS64);
        w.write_bounded(self.height - 64, MAX_DIM_MINUS64);
        w.write_bits(self.diff_display_width as u32, 6);
        w.write_bits(self.diff_display_height as u32, 6);
        w.write_bounded(bit_depth_idc, BIT_DEPTHS.len() as u32 - 1);
        w.write_bits(self.s_ver as u32 - 1, 1);
        w.write_bits(self.s_hor as u32 - 1, 1);
        if self.s_ver == 1 {
            w.write_bits(self.c_ver as u32 - 1, 1);
        }
        if self.s_hor == 1 {
            w.write_bits(self.c_hor as u32 - 1, 1);
        }
        match &self.colour_transform {
            ColourTransform::None => w.write_bounded(0, 2),
            ColourTransform::Bt709 => w.write_bounded(1, 2),
            ColourTransform::Custom { matrix, offset } => {
                w.write_bounded(2, 2);
                for &m in matrix.iter().chain(offset) {
                    w.write_bits(m as u32, 8);
                }
            }
        }
        w.write_bounded(self.model_id as u32, MAX_MODELS - 1);
        write_thread_count(&mut w, self.num_threads_z)?;
        let beta = |b: i32| -> Result<u32> {
            u32::try_from(b + 2048)
                .ok()
                .filter(|&v| v < 4096)
                .ok_or(Error::InvalidArgument("beta_displacement_log out of range"))
        };
        w.write_bits(beta(self.beta_displacement_log[0])?, 12);
        match &self.regions {
            None => w.write_bit(false),
            Some(reg) => {
                w.write_bit(true);
                w.write_bits(reg.num_ver as u32 - 1, 7);
                w.write_bits(reg.num_hor as u32 - 1, 7);
                w.write_bit(reg.independent);
                if !reg.independent {
                    w.write_bits(reg.hyper_decoder_overlap as u32, 2);
                    w.write_bits(reg.mcm_overlap as u32, 4);
                }
            }
        }
        let independent_beta = self.beta_displacement_log[0] != self.beta_displacement_log[1];
        w.write_bit(independent_beta);
        if independent_beta {
            w.write_bits(beta(self.beta_displacement_log[1])?, 12);
        }
        for (ccs, c) in self.components.iter().enumerate() {
            c.write(&mut w, self, ccs)?;
        }
        match &self.quality_map {
            None => w.write_bit(false),
            Some(q) => {
                w.write_bit(true);
                write_thread_count(&mut w, q.num_threads)?;
                w.write_bounded(q.entropy_index as u32, 7);
            }
        }
        Ok(w.finish())
    }
}

/// `multi_threading_*` flag followed by `log2_num_threads_*_minus1` (2 bits) when set.
fn read_thread_count(r: &mut BitReader<'_>) -> Result<u8> {
    if r.read_bit()? {
        Ok(1 << (r.read_bits(2)? + 1))
    } else {
        Ok(1)
    }
}

fn write_thread_count(w: &mut BitWriter, n: u8) -> Result<()> {
    match n {
        1 => w.write_bit(false),
        2 | 4 | 8 | 16 => {
            w.write_bit(true);
            w.write_bits(n.trailing_zeros() - 1, 2);
        }
        _ => {
            return Err(Error::InvalidArgument(
                "thread count must be 1, 2, 4, 8 or 16",
            ));
        }
    }
    Ok(())
}

impl ComponentHeader {
    fn empty() -> Self {
        Self {
            num_threads_r: 1,
            num_chs: 0,
            cube_flags: None,
            rvs_enabled: false,
            grfs_channel_flags: None,
            synthesis_tiling: None,
        }
    }

    fn parse(r: &mut BitReader<'_>, hdr: &PictureHeader, ccs: usize) -> Result<Self> {
        // SepChannelsSGMMTool.decode_header
        let num_threads_r = read_thread_count(r)?;
        let num_chs = (r.read_bits(8)? as usize).min(LATENT_CHANNELS[ccs]) as u16;

        // SkipModeCoder.decode_header: flags are sent in groups of eight behind a group flag;
        // a cleared group flag means "all eight set".
        let cube_flags = if r.read_bit()? {
            let count = hdr.cube_flag_count(ccs);
            let mut flags = alloc::vec![true; count];
            for group in flags.chunks_mut(CUBE_GROUP_SIZE) {
                if r.read_bit()? {
                    for f in group {
                        *f = r.read_bit()?;
                    }
                }
            }
            Some(flags)
        } else {
            None
        };

        // ResVarScale.decode_header
        let rvs_enabled = r.read_bit()?;
        let grfs_channel_flags = if r.read_bit()? {
            let mut flags = Vec::with_capacity(num_chs as usize);
            for _ in 0..num_chs {
                flags.push(r.read_bit()?);
            }
            Some(flags)
        } else {
            None
        };

        // TileManager (synthesis) enable flag + decode_header
        let synthesis_tiling = if r.read_bit()? {
            let tile_size = r.read_bits(8)? * 16;
            let overlap = r.read_bits(5)? * 16;
            if tile_size == 0 {
                return Err(Error::InvalidData("synthesis_tile_size is zero"));
            }
            Some(SynthesisTiling { tile_size, overlap })
        } else {
            None
        };

        Ok(Self {
            num_threads_r,
            num_chs,
            cube_flags,
            rvs_enabled,
            grfs_channel_flags,
            synthesis_tiling,
        })
    }

    fn write(&self, w: &mut BitWriter, hdr: &PictureHeader, ccs: usize) -> Result<()> {
        write_thread_count(w, self.num_threads_r)?;
        if self.num_chs as usize > LATENT_CHANNELS[ccs] {
            return Err(Error::InvalidArgument(
                "num_chs exceeds the latent channel count",
            ));
        }
        w.write_bits(self.num_chs as u32, 8);
        match &self.cube_flags {
            None => w.write_bit(false),
            Some(flags) => {
                if flags.len() != hdr.cube_flag_count(ccs) {
                    return Err(Error::InvalidArgument(
                        "cube flag count does not match picture",
                    ));
                }
                w.write_bit(true);
                for group in flags.chunks(CUBE_GROUP_SIZE) {
                    let all_set = group.iter().all(|&f| f);
                    w.write_bit(!all_set);
                    if !all_set {
                        for &f in group {
                            w.write_bit(f);
                        }
                    }
                }
            }
        }
        w.write_bit(self.rvs_enabled);
        match &self.grfs_channel_flags {
            None => w.write_bit(false),
            Some(flags) => {
                if flags.len() != self.num_chs as usize {
                    return Err(Error::InvalidArgument(
                        "one grfs flag per coded channel required",
                    ));
                }
                w.write_bit(true);
                for &f in flags {
                    w.write_bit(f);
                }
            }
        }
        match &self.synthesis_tiling {
            None => w.write_bit(false),
            Some(t) => {
                if t.tile_size % 16 != 0 || t.overlap % 16 != 0 || t.tile_size == 0 {
                    return Err(Error::InvalidArgument(
                        "tile size/overlap must be multiples of 16",
                    ));
                }
                if t.tile_size / 16 > 255 || t.overlap / 16 > 31 {
                    return Err(Error::InvalidArgument("tile size/overlap out of range"));
                }
                w.write_bit(true);
                w.write_bits(t.tile_size / 16, 8);
                w.write_bits(t.overlap / 16, 5);
            }
        }
        Ok(())
    }
}

/// Decoded tool header. Absent substream == everything off.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ToolHeader {
    /// Latent-space bias shift per component.
    pub lsbs_enabled: [bool; 2],
    pub efe_linear_enabled: bool,
    pub icci_enabled: bool,
    pub efe_nonlinear_enabled: bool,
    pub lef_enabled: bool,
}

impl ToolHeader {
    /// Parse the TON payload. The post-filter parameter syntax is not ported yet: a stream that
    /// enables any of the four filters is rejected as unsupported rather than mis-decoded.
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(payload);
        let lsbs_enabled = [r.read_bit()?, r.read_bit()?];
        let mut t = ToolHeader {
            lsbs_enabled,
            ..Default::default()
        };
        t.efe_linear_enabled = r.read_bit()?;
        if t.efe_linear_enabled {
            return Err(Error::Unsupported("EFE linear post-filter"));
        }
        t.icci_enabled = r.read_bit()?;
        if t.icci_enabled {
            return Err(Error::Unsupported("eICCI post-filter"));
        }
        t.efe_nonlinear_enabled = r.read_bit()?;
        if t.efe_nonlinear_enabled {
            return Err(Error::Unsupported("EFE non-linear post-filter"));
        }
        t.lef_enabled = r.read_bit()?;
        if t.lef_enabled {
            return Err(Error::Unsupported("LEF post-filter"));
        }
        Ok(t)
    }

    pub fn write(&self) -> Result<Vec<u8>> {
        if self.efe_linear_enabled
            || self.icci_enabled
            || self.efe_nonlinear_enabled
            || self.lef_enabled
        {
            return Err(Error::Unsupported("post-filter headers"));
        }
        let mut w = BitWriter::new();
        w.write_bit(self.lsbs_enabled[0]);
        w.write_bit(self.lsbs_enabled[1]);
        for _ in 0..4 {
            w.write_bit(false);
        }
        Ok(w.finish())
    }
}

/// CICP colour description (`cicp_info_present_flag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cicp {
    pub colour_primaries: u8,
    pub transfer_characteristics: u8,
    pub matrix_coefficients: u8,
    pub full_range: bool,
    pub chroma420_sample_loc_type: u8,
}

/// Mastering display colour volume (`mdcv_info_present_flag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mdcv {
    /// `(x, y)` chromaticity of the three primaries.
    pub primaries: [(u16, u16); 3],
    pub white_point: (u16, u16),
    pub max_luminance: u32,
    pub min_luminance: u32,
}

/// Content light level (`clli_info_present_flag`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Clli {
    pub max_content_light_level: u16,
    pub max_frame_average_light_level: u16,
}

/// Rendering information substream. Absent substream == nothing present.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RenderingInfo {
    pub cicp: Option<Cicp>,
    pub mdcv: Option<Mdcv>,
    pub clli: Option<Clli>,
    /// `(dm_type, dm_data)`: opaque display-mapping payload.
    pub display_mapping: Option<(u8, Vec<u8>)>,
}

impl RenderingInfo {
    pub fn parse(payload: &[u8]) -> Result<Self> {
        let mut r = BitReader::new(payload);
        let cicp_present = r.read_bit()?;
        let mdcv_present = r.read_bit()?;
        let clli_present = r.read_bit()?;
        let dm_present = r.read_bit()?;
        if r.read_bits(4)? != 0 {
            return Err(Error::InvalidData("ri_reserved_zero_4bits is not zero"));
        }
        let mut info = RenderingInfo::default();
        if cicp_present {
            let cicp = Cicp {
                colour_primaries: r.read_bits(8)? as u8,
                transfer_characteristics: r.read_bits(8)? as u8,
                matrix_coefficients: r.read_bits(8)? as u8,
                full_range: r.read_bit()?,
                chroma420_sample_loc_type: r.read_bits(3)? as u8,
            };
            if r.read_bits(4)? != 0 {
                return Err(Error::InvalidData("ri_reserved_zero_4bits is not zero"));
            }
            info.cicp = Some(cicp);
        }
        if mdcv_present {
            let mut primaries = [(0u16, 0u16); 3];
            for p in &mut primaries {
                *p = (r.read_bits(16)? as u16, r.read_bits(16)? as u16);
            }
            info.mdcv = Some(Mdcv {
                primaries,
                white_point: (r.read_bits(16)? as u16, r.read_bits(16)? as u16),
                max_luminance: r.read_bits(32)?,
                min_luminance: r.read_bits(32)?,
            });
        }
        if clli_present {
            info.clli = Some(Clli {
                max_content_light_level: r.read_bits(16)? as u16,
                max_frame_average_light_level: r.read_bits(16)? as u16,
            });
        }
        if dm_present {
            let dm_type = r.read_bits(8)? as u8;
            let dm_size = r.read_bits(16)? as usize;
            let mut data = Vec::with_capacity(dm_size);
            for _ in 0..dm_size {
                data.push(r.read_bits(8)? as u8);
            }
            info.display_mapping = Some((dm_type, data));
        }
        Ok(info)
    }

    pub fn write(&self) -> Result<Vec<u8>> {
        let mut w = BitWriter::new();
        w.write_bit(self.cicp.is_some());
        w.write_bit(self.mdcv.is_some());
        w.write_bit(self.clli.is_some());
        w.write_bit(self.display_mapping.is_some());
        w.write_bits(0, 4);
        if let Some(c) = &self.cicp {
            w.write_bits(c.colour_primaries as u32, 8);
            w.write_bits(c.transfer_characteristics as u32, 8);
            w.write_bits(c.matrix_coefficients as u32, 8);
            w.write_bit(c.full_range);
            if c.chroma420_sample_loc_type > 7 {
                return Err(Error::InvalidArgument(
                    "chroma420_sample_loc_type out of range",
                ));
            }
            w.write_bits(c.chroma420_sample_loc_type as u32, 3);
            w.write_bits(0, 4);
        }
        if let Some(m) = &self.mdcv {
            for &(x, y) in &m.primaries {
                w.write_bits(x as u32, 16);
                w.write_bits(y as u32, 16);
            }
            w.write_bits(m.white_point.0 as u32, 16);
            w.write_bits(m.white_point.1 as u32, 16);
            w.write_bits(m.max_luminance, 32);
            w.write_bits(m.min_luminance, 32);
        }
        if let Some(c) = &self.clli {
            w.write_bits(c.max_content_light_level as u32, 16);
            w.write_bits(c.max_frame_average_light_level as u32, 16);
        }
        if let Some((dm_type, data)) = &self.display_mapping {
            let size = u16::try_from(data.len())
                .map_err(|_| Error::InvalidArgument("display mapping data too long"))?;
            w.write_bits(*dm_type as u32, 8);
            w.write_bits(size as u32, 16);
            for &b in data {
                w.write_bits(b as u32, 8);
            }
        }
        Ok(w.finish())
    }
}

/// Conformance limits (`cfg/profiles/*.json`, `levels.json`; `CodingEngine.check_complience`).
impl PictureHeader {
    /// Check the stream against its declared profile and level.
    pub fn check_conformance(&self) -> Result<()> {
        use OperatingPoint::{Bop, Hop, Sop};
        if self.stream_profile_idc != 0 {
            return Err(Error::NonConforming("unknown stream_profile_idc"));
        }
        let allowed: &[OperatingPoint] = match self.decoder_profile_id {
            0 => &[Sop],
            1 => &[Bop, Sop],
            2 => &[Hop, Bop, Sop],
            _ => return Err(Error::NonConforming("unknown decoder_profile_id")),
        };
        let default = self
            .synthesis_transforms
            .first()
            .ok_or(Error::NonConforming("no synthesis transform"))?;
        if !allowed.contains(default) {
            return Err(Error::NonConforming(
                "default synthesis transform not in profile",
            ));
        }
        let max_pic_size: u64 = match self.level_idc / 10 {
            1 => 6_220_800,
            2 => 24_883_200,
            3 => 99_532_800,
            4 => 149_817_600,
            5 => 398_131_200,
            _ => return Err(Error::NonConforming("unknown picture-size level")),
        };
        if self.width as u64 * self.height as u64 > max_pic_size {
            return Err(Error::NonConforming("picture larger than the level allows"));
        }
        let models: &[u8] = match self.level_idc % 10 {
            0 => &[2],
            1 => &[2, 3],
            2 => &[0, 1, 2, 3],
            _ => return Err(Error::NonConforming("unknown model-set level")),
        };
        if !models.contains(&self.model_id) {
            return Err(Error::NonConforming("model_id not permitted at this level"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Header payloads cut from streams written by the reference encoder (upstream b9e573f) for
    // data/test/00030_TE_560x888_8bit_sRGB.png at 0.5 bpp; expected values are what the
    // reference's scripts/bitstream_probe.py prints for the same streams.
    const PIH_BASE_TOOLS_OFF: &str = "011103401f00338000008a32c5001800";
    const PIH_HIGH_TOOLS_ON: &str = "0222103401f003380000091830503fffffff9744000000000000000000\
        00000000000181ffe00000000000000000000000";
    const TON_HIGH_TOOLS_ON: &str = "eb3c5a73888880007fffc0013fff3ffc3fff0888bffc000553035a1000\
        0ffff86002806605410105100680565aa8695aa408a206";

    #[test]
    fn pih_base_profile_tools_off() {
        let payload = hex(PIH_BASE_TOOLS_OFF);
        let h = PictureHeader::parse(&payload).unwrap();
        assert_eq!(h.stream_profile_idc, 0);
        assert_eq!(h.decoder_profile_id, 1);
        assert_eq!(
            h.synthesis_transforms,
            [OperatingPoint::Bop, OperatingPoint::Sop]
        );
        assert_eq!(h.level_idc, 52);
        assert_eq!((h.width, h.height), (496 + 64, 824 + 64));
        assert_eq!((h.diff_display_width, h.diff_display_height), (0, 0));
        assert_eq!(h.bit_depth, 8);
        assert_eq!((h.s_ver, h.s_hor, h.c_ver, h.c_hor), (1, 1, 1, 1));
        assert_eq!(h.colour_transform, ColourTransform::Bt709);
        assert_eq!(h.model_id, 1);
        assert_eq!(h.num_threads_z, 1);
        assert_eq!(h.beta_displacement_log, [2251 - 2048, 2251 - 2048]);
        assert_eq!(h.regions, None);
        for (c, chs) in h.components.iter().zip([160, 96]) {
            assert_eq!(c.num_threads_r, 1);
            assert_eq!(c.num_chs, chs);
            assert_eq!(c.cube_flags, None);
            assert!(!c.rvs_enabled);
            assert_eq!(c.grfs_channel_flags, None);
            assert_eq!(c.synthesis_tiling, None);
        }
        assert_eq!(h.quality_map, None);
        h.check_conformance().unwrap();

        assert_eq!(h.latent_size(0), (56, 35));
        assert_eq!(h.latent_size(1), (56, 35));
        assert_eq!(h.hyper_latent_size(0), (14, 9));
        assert_eq!(h.hyper_latent_size(1), (14, 9));

        assert_eq!(
            h.write().unwrap(),
            payload,
            "PIH must re-serialise to the same bytes"
        );
    }

    #[test]
    fn pih_high_profile_tools_on() {
        let payload = hex(&PIH_HIGH_TOOLS_ON.replace(char::is_whitespace, ""));
        let h = PictureHeader::parse(&payload).unwrap();
        assert_eq!(h.decoder_profile_id, 2);
        use OperatingPoint::{Bop, Hop, Sop};
        assert_eq!(h.synthesis_transforms, [Hop, Bop, Sop]);
        assert_eq!(h.model_id, 2);
        assert_eq!(h.beta_displacement_log, [1548 - 2048, 1548 - 2048]);
        let y = &h.components[0];
        assert!(y.rvs_enabled);
        let grfs = y.grfs_channel_flags.as_ref().unwrap();
        assert_eq!(grfs.len(), 160);
        // probe: 29 ones, 0, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, then zeros
        let ones: Vec<usize> = grfs
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
            .map(|(i, _)| i)
            .collect();
        let mut want: Vec<usize> = (0..29).collect();
        want.extend([31, 33, 34, 35, 37, 41]);
        assert_eq!(ones, want);
        let uv = &h.components[1];
        assert!(uv.rvs_enabled);
        let grfs = uv.grfs_channel_flags.as_ref().unwrap();
        assert_eq!(grfs.len(), 96);
        assert_eq!(grfs.iter().filter(|f| **f).count(), 10);
        assert!(grfs[..10].iter().all(|f| *f));
        assert_eq!(h.quality_map, None);
        h.check_conformance().unwrap();
        assert_eq!(h.write().unwrap(), payload);
    }

    #[test]
    fn ton_flags() {
        assert_eq!(ToolHeader::parse(&[0x00]).unwrap(), ToolHeader::default());
        assert_eq!(ToolHeader::default().write().unwrap(), [0x00]);
        // lsbs on for both components, then EFE linear enabled: not ported yet.
        let payload = hex(&TON_HIGH_TOOLS_ON.replace(char::is_whitespace, ""));
        assert_eq!(
            ToolHeader::parse(&payload),
            Err(Error::Unsupported("EFE linear post-filter"))
        );
        let t = ToolHeader {
            lsbs_enabled: [true, true],
            ..Default::default()
        };
        assert_eq!(ToolHeader::parse(&t.write().unwrap()).unwrap(), t);
    }

    #[test]
    fn rdi_roundtrip() {
        assert_eq!(
            RenderingInfo::parse(&[0x00]).unwrap(),
            RenderingInfo::default()
        );
        let info = RenderingInfo {
            cicp: Some(Cicp {
                colour_primaries: 9,
                transfer_characteristics: 16,
                matrix_coefficients: 9,
                full_range: true,
                chroma420_sample_loc_type: 2,
            }),
            mdcv: Some(Mdcv {
                primaries: [(34000, 16000), (13250, 34500), (7500, 3000)],
                white_point: (15635, 16450),
                max_luminance: 10_000_000,
                min_luminance: 50,
            }),
            clli: Some(Clli {
                max_content_light_level: 1000,
                max_frame_average_light_level: 400,
            }),
            display_mapping: Some((3, alloc::vec![1, 2, 3, 250])),
        };
        let bytes = info.write().unwrap();
        assert_eq!(RenderingInfo::parse(&bytes).unwrap(), info);
        assert!(
            RenderingInfo::parse(&[0x01]).is_err(),
            "reserved bits must be zero"
        );
    }

    #[test]
    fn pih_roundtrip_with_everything() {
        let mut h = PictureHeader::parse(&hex(PIH_BASE_TOOLS_OFF)).unwrap();
        h.width = 4000;
        h.height = 3000;
        h.bit_depth = 10;
        h.s_ver = 2;
        h.s_hor = 2;
        h.c_ver = 2;
        h.c_hor = 2;
        h.colour_transform = ColourTransform::Custom {
            matrix: [1, 2, 3, 4, 5, 6, 7, 8, 9],
            offset: [16, 128, 128],
        };
        h.num_threads_z = 8;
        h.beta_displacement_log = [-1069, 702];
        h.regions = Some(Regions {
            num_ver: 3,
            num_hor: 4,
            independent: false,
            hyper_decoder_overlap: 2,
            mcm_overlap: 5,
        });
        let n = h.cube_flag_count(0);
        let flags: Vec<bool> = (0..n)
            .map(|i| i % 11 != 0 && !(40..56).contains(&i))
            .collect();
        h.components[0].cube_flags = Some(flags);
        h.components[0].num_threads_r = 16;
        h.components[0].synthesis_tiling = Some(SynthesisTiling {
            tile_size: 1024,
            overlap: 64,
        });
        h.components[1].num_chs = 48;
        h.components[1].grfs_channel_flags = Some((0..48).map(|i| i % 3 == 0).collect());
        h.quality_map = Some(QualityMapHeader {
            num_threads: 4,
            entropy_index: 5,
        });
        let bytes = h.write().unwrap();
        assert_eq!(PictureHeader::parse(&bytes).unwrap(), h);
        // truncation anywhere must be an error, never a panic
        for cut in 0..bytes.len() {
            assert!(PictureHeader::parse(&bytes[..cut]).is_err(), "cut at {cut}");
        }
    }

    #[test]
    fn conformance_limits() {
        let mut h = PictureHeader::parse(&hex(PIH_BASE_TOOLS_OFF)).unwrap();
        h.level_idc = 10; // 6.2 MP, model 2 only
        assert_eq!(
            h.check_conformance(),
            Err(Error::NonConforming("model_id not permitted at this level"))
        );
        h.model_id = 2;
        h.check_conformance().unwrap();
        h.width = 4000;
        h.height = 3000;
        assert!(h.check_conformance().is_err());
        h.level_idc = 20;
        h.check_conformance().unwrap();
        h.decoder_profile_id = 0; // simple profile: SOP only, but the default transform is BOP
        assert!(h.check_conformance().is_err());
    }
}
