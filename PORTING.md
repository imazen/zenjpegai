# PORTING.md — the Python/C++ → Rust map

Auditability index: for each Rust module, the upstream reference-software source it ports and
the parity test that gates it. Upstream is the JPEG AI reference software at commit `b9e573f`
(`https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software`), referred to below as `ref/`.

A row is only marked **ported** when the code is called from the main path and a parity test
against reference-produced data passes. "stub" and "partial" mean what they say.

| Rust module | Upstream source | Status | Parity gate |
| --- | --- | --- | --- |
| `bitio` | `entropy_coding/bits_coders.py`, `binarizers.py`, `cpp_exts/direct/` | ported (reader + writer, ue/se) | unit tests (known codewords, round trips); exercised by every header test once headers land |
| `container` | `bitstream_structure/{layouts_def,substream,bitstream_structure,aemem}.py` | ported: marker split, region split (independent + dependent), thread split, writer | unit round trips only so far; no reference-stream gate yet |
| `mans::tables` | `lib_wrappers/mans/utils.py`, `ec_lib_mans.py::init_quant_params` | ported, integer-only | `mans::tables::tests::tables_match_reference` (FNV-1a of the reference-built encode/decode/state-map tables) |
| `mans::decoder` | `cpp_exts/mans/decompressor.{h,cpp}` | ported: residual + z, 1..16 threads (threads run sequentially for now) | `tests/mans_vectors.rs`: decodes payloads written by the reference C++ `ANSEncoder` |
| `mans::encoder` | `cpp_exts/mans/compressor.{h,cpp}` | ported: residual + z, 1..16 threads | `tests/mans_vectors.rs`: output bytes and thread sizes equal the reference's |
| `weights` | `torch.load` of `models/**/*.pth` (no upstream source: replaces PyTorch's unpickler) | ported: stored-ZIP reader (zip64 aware), data-only pickle interpreter, lazy strided tensor materialisation | `tests/weights_ref.rs` (`reference-tests` feature): tensor hashes equal `torch.load`'s for int8/int32/int64/bool/f32 tensors; all 68 upstream checkpoints parse |
| `weights::packed` | (no upstream source: a container of our own) | done: `ZJM1` (one checkpoint: name / dtype / shape / offset table + 64-byte aligned little-endian tensor bytes) read through the same `Checkpoint` accessors as `.pth`; `ZJB1` bundle of such files + the upstream licence text, served zero-copy as a `ModelSource` (`PackedBundle`, mergeable); `Recorder` + `zenjpegai pack-models` keep only the tensors the loaders look up (`model::with_checkpoint` reports them). Lossless: no f16 / quantised option exists. Bundle sizes per (model, operating point): SOP 15.05 MB, BOP 16.22 MB, HOP 28.76 MB (the `.pth` files they come from: 22..128 MB); split: common part of a model 12.36 MB, synthesis part SOP 2.69 / BOP 3.86 / HOP 16.40 MB; all 4 models x 3 points 141 MB from 353 MB. gzip -9 saves only 9 % (float weights). **Post-filter checkpoints are not packed yet: their loaders must go through `model::with_checkpoint`** | `tests/packed_bundle.rs`: SOP / BOP / HOP decodes from a bundle (whole and split) equal the `.pth` decodes byte for byte; repacking a bundle reproduces it; damaged bundles are rejected |
| `header` | `coding_engine.py`, `colour_transformation.py`, `multitools/engine.py`, `ccs_sgmm_tool.py`, `quantization.py`, `sep_chan_tool.py`, `skip_mode.py`, `res_var_scale.py`, `tiling.py`, `quality_map.py`, `rdi.py` (header functions only) | partial: PIH complete (parse + write + profile/level conformance); RDI complete; TON complete (LSBS flags, EFE linear filter sets, eICCI tiling + per-tile model selection, EFE non-linear masks / tiles / weights, LEF channel), parse + write; UDI is exposed as opaque bytes (`decoder::Headers::user_data`) | `header::tests`: two reference-encoder PIH payloads parse to the values `scripts/bitstream_probe.py` prints and re-serialise byte-identically; a reference TON with all four post-filters on parses to its end and re-serialises byte-identically |
| `tensor` | (PyTorch tensors) | minimal `[C,H,W]` container | - |
| `model::hsd` | `components/autoencoder_hyper/decoder_scale/basic.py`, `base_layers/conv_quant_layers.py` | ported. Convolutions run through `nn::fast::PackedIntConv` (i16 pairs + `madd`, wrapping i32; exact by construction because integer sums are order-free); the plain scalar loops stay as `forward_reference` | `tests/entropy_ref.rs` (sigma maps equal the reference's); `tests/fast_vs_reference.rs::int_conv_matches_scalar_loops` (every tier, threads on/off, odd channel counts, clamped inputs, accumulator wrap-around) |
| `model::common` | `core_models/CCS_SGMM/common_modules.py::build_model`, `gain_unit.py::get_gain_vector_log`, `custom_prob_wrapper.py::normalize_z` | ported: z CDFs, HSD weights, gain vector. Hyper-decoder / MCM weights not loaded yet | `tests/entropy_ref.rs` |
| `tools::gain` | `quantization/gain_unit/gain_unit.py` | ported (decoder side) | `tests/entropy_ref.rs` (dequantised residual floats bit-identical). `scaler_from_log` (libm `expf`) is bit-identical to the reference's `torch.exp` on every reachable input: `gain::tests::scaler_matches_torch_exp_on_every_reachable_input` checks all 1.34M pairs over `tests/vectors/gain_scaler.bin` (`scripts/ref_vectors/gen_gain_scaler_vectors.py`) — every `gain_vector_log` entry of the eight `VM_common_int` checkpoints times every `beta_displacement_log` the 12-bit header field can signal (-2048..=2047; the -1069..702 clip is encoder-side only, `QuantizerHeaderBaseFuncs.enc_flag`) |
| `tools::skip` | `skip_ls/skip_mode.py` (mask + cube-flag expansion) | ported (decoder side) | `tests/entropy_ref.rs` (threshold mask + cube flags, bit-exact): `enc_img30_bop_m0_bm1069` and `enc_img30_hop_m3_bm1069` carry `use_cube_flags = 1` with false flags — the latter on both components, including a cleared `cube_group_flag` (decode dumps: `make_reference_streams.sh cubeflags`). `tests/decode_ref.rs` decodes the BOP stream end to end; the HOP stream's luma `y_hat` peaks at 5.5e-4 (the 5e-4 gate) at that extreme rate, its planes and 8-bit output pass |
| `tools::log2lin` | `common/log2lin.py::Log2LinConvertion` | ported: the 4352-entry table is computed (`round(2^17 exp(i ln(100/0.11)/4352 + ln 0.11))`) instead of stored | unit test pins the FNV-1a hash of upstream's literal table |
| `tools::rvs` | `quantization/rvs/res_var_scale.py` (`buildTables`, `analyze`, `quantize_scale`, `dequantize_resi`), constants from `cfg/pipeline.json` | ported (decoder side): block-wise "likely" map, RVS scale tables, GRFS ("cwg") per-channel variants | `tests/entropy_ref.rs` (scale map and residual exact) on RVS+GRFS, RVS-only and GRFS-only streams |
| `tools::qualmap` | `quality_map/quality_map.py` (`decode`, `quantize_scale`, `dequantize_resi`) | ported (decoder side): ANS-coded delta plane, DPCM reconstruction, log-scale offset and residual step per position, shared by both components | `tests/entropy_ref.rs` + `tests/decode_ref.rs` on 3 quality-map streams (plain, with RVS, with 8 ANS threads); oracle = reference decoder with its header defect patched (below) |
| `tools::lsbs` | `ls_processing/lsbs/lsbs_scale_mode.py` (`buildTables`, `post_processing`), constants from `cfg/pipeline.json` | ported | `tests/decode_ref.rs` LSBS streams (`y_hat` within 5e-4 of the reference after scaling) |
| `tools::regions` | `tiling/tiling.py::TileManagerHyper` | ported: region grids for y / psi / z, with and without overlap extension | unit tests + `tests/entropy_ref.rs` region streams (latent grid only) |

| `encoder` | `ccs_sgmm_tool.py::compress`, `sep_chan_tool.py::compress`, `common_modules.py::{compress, _compress_z, encoder_get_scales, _compress_ar_scale, encoder_skip_and_cubeflag_for_tiles, encode, encode_z, encode_y, _ac_encode_y, _ac_encode_z}`, `image.py`/`colorspace.py` (`to_YUV_`, `to_format_`/`to_420_`/`to_422_`, `pad_`, `pixel_unshuffle`, `read_yuv`/`extract_info`, `convert_range_`), `filters/LEF/LEFfilter.py::analyze` | ported for: a fixed model and operating point (SOP / BOP / HOP) or a rate-matched target, sources up to 120 MP, any number of analysis tiles, regions, ANS threads and coding tools as listed below. Sources: interleaved RGB (8 or 16 bit, `colour_transform_idx` 1 / BT.709) and planar YUV (4:4:4 / 4:2:2 / 4:2:0, 8- and 10-bit verified, `colour_transform_idx` 0) — `SourceImage::{Rgb, Yuv}` / `SourceImage::read_yuv` (file-name metadata like `extract_info`); coded chroma is the source's subsampling or coarser via `EncodeParams::c_ver`/`c_hor` (`-c_ver_value`/`-c_hor_value`; a 4:2:2 source coded 4:2:0 is `Unsupported`, as upstream's `NotImplementedError`), and `diff_display` codes the non-displayed border. Chroma resampling is `nn::resize_bilinear`, bit-identical to PyTorch's `F.interpolate(mode="bilinear", align_corners=True)` channels-last kernel (weight products rounded once, then nested FMAs — probed with impulses, `tests/vectors/nn/bilinear_align_corners.bin`). `Encoder::encode(src, EncodeParams { model_id, beta_displacement_log, op, .. })` and `Encoder::encode_to_bpp(src, bpp, op)`; CLI `zenjpegai encode in.png|in.yuv [--model N --beta-disp B | --bpp R] [--c-ver 1|2 --c-hor 1|2 --diff-display W,H]` (16-bit PNGs code `bit_depth_idc` 4, as `read_png` does). Coding tools on the encode side: RVS, GRFS (the encoder derives the gain flags, `analyzeCWG`), LSBS, 1..16 ANS threads per substream, region partitioning (dependent and independent; the grid follows `calc_numHor_numVer_regions`) and the quality map (`Encoder::encode_with_quality_map`), plus the LEF flag (`EncodeParams::lef` / `--lef`; `LEF_chIdx` is the channel of the luma scale map with the highest mean — the filter itself is decoder-side). The EFE linear post-filter is decided and coded (`EncodeParams::efe_linear` / `--efe-linear`, row `encoder::filters::efe_linear`), and so is the eICCI per-tile model selection (`EncodeParams::eicci` / `--eicci`, row `encoder::filters::icci`) and the EFE non-linear post-filter (`EncodeParams::efe_nonlinear` / `--efe-nonlinear`, row `encoder::filters::efe_nonlinear`; it consumes EFE linear's up-sampled picture as its mask alternative when both are on). `num_chs` below the model's channel count is ported (`EncodeParams::num_chs` / `--num-chs y,uv`, the reference's `-model.CCS_SGMM.tools_common.model_{y,uv}.common_modules.num_chs`): the residual substream codes the first N channels of each component (`_ac_encode_y`'s `[:, :num_chs]` slices), the header signals N per component, uncoded channels keep a zero quantised residual, their cube-flag contribution follows the reference exactly (context-module luma: `diff[:, num_chs:] = 0`, none; context-free chroma: the full `|y - psi|` since `y_rec_resi[:, num_chs:] = 0`), GRFS ranks all channels and writes `cwgf[:num_chs]`, and `num_chs = 0` codes no residual at all. The rate matcher's `hyperopt` UV search is unreachable in b9e573f — see `encoder::rate`. | `tests/encode_ref.rs` on twenty-three fixed-model reference encodes (models 0..3, SOP / BOP / HOP, beta displacement -1069..+400, 2096x1400, one per coding tool, one per region mode, two with a quality map and five with reduced `num_chs`) plus the nine `formats` vectors: colour pre-processing bit-exact against the reference's own analysis inputs (`enc2` dumps, incl. odd 203x301 where the analysis luma is unpadded); `z_hat`, `skip_scale_log`, `scale_log` and the cube flags equal the reference encoder's exactly; twenty-five of thirty-two streams byte-identical to the reference's and the other seven the same length (the two 10-bit `formats` vectors move 5 of 1,254,400 luma residual symbols by 1 each — the rounding-boundary mechanism below); every stream decodes with `zenjpegai::Decoder` and with the reference decoder |

| `encoder::tiles` | `tiling.py::TileManager` (`setup_tiles_enc`, `_init_image_tiles_with_overlap`, `_get_latent_tile_from_image_tile`, the picture-border branch of `_get_core_of_overlapping_tile`), `sep_chan_tool.py::setup_enc_tile_managers_of_model`, `common_modules.py::compress_colocated_tiles` | ported: above 1 MP (luma) / 0.26 MP (chroma) the analysis transform and the hyper-encoder run per tile (1024 / overlap 64 luma, 512 / 32 chroma, from `cfg/CTC.json`) and only each tile's core is kept; the picture header's `synthesis_tiling` follows `tile_manager_synthesis`. **`_adjust_boundary_tiles` does not apply here** (the reference calls `setup_tiles_enc` without a minimum tile size) | unit test vs the layout `enc_img01_bop_m1_b0/encoder.log` logs, tile for tile; `tests/encode_ref.rs::colour_preprocessing_matches_reference` checks every tile's analysis input against the reference's own, and the 2096x1400 vector encodes to the reference's byte count |
| `tools::rvs` (encode) | `res_var_scale.py::{analyzeCWG, quantize_resi}` | ported: the forward scale table and `grfs_flags` (the `cnum_list[model]` channels with the highest mean sigma get the gain flag) | `tests/encode_ref.rs` on `enc_img30_bop_m1_b0_{rvs,rvsonly,grfsonly}`: byte-identical streams, so the derived flags are the reference's |
| `encoder` (regions) | `ccs_sgmm_tool.py::calc_numHor_numVer_regions`, `common_modules.py::{compress, compress_ar_scale_tile, merge_psi_overlaps_of_tiles}`, `tiling.py::TileManagerHyper` | ported: the region grid from the picture size, per-region hyper-decoder and context model merged exactly as `decoder::reconstruct::reconstruct_latent_with` merges them, per-region residual substreams (dependent: sizes at the head of one substream; independent: one substream per region behind its index), and the region-aware analysis tiling independent regions force (`cfg_update_for_conformance`) | `tests/encode_ref.rs` on `enc_img01_bop_m1_b0_{depregions,indregions}`: the dependent stream is byte-identical to the reference's given its own latents; both are its length from the PNG, and the patched reference decoder reads both |
| `tools::qualmap` (encode) | `quality_map.py::{encode, quantize_resi, generate_qp_map_index, generate_qp_map_byROI_map}` | ported: the ROI-mask map (`qp_map_type = 3`, the only generator that works on this code path), the entropy-index choice from the zero-delta fraction, the DPCM deltas and their SOQ substream, and the forward quantiser step. **The other three map generators are not ported** (the reference crashes on them here) | `tests/encode_ref.rs` on `enc_img30_bop_m1_b0_qmap{,_rvs}`: both streams byte-identical to the reference's |
| `encoder::filters::efe_linear` | `filters/EFElinear/EFElinear.py`: `compress`, `SplitDecide`/`searchCore` (region-split candidates x filter lengths, loss = SSE with `lossModifier`), `LumaAidedUpsampler_encoder` + `linearSolve` (unfold design matrices, the >2 M-sample four-slice subsample-averaging), `integerize` (round-half-even `w*2048 + 32767`), `minSymbol`/`maxSymbol`, `conv_UV_fix`, `means`, and the second "up-sampled" filter set for EFE non-linear | ported: `decide`/`split_decide` produce the TON EFE linear sets (normal + up-sampled) for every coded chroma format — coded 4:4:4 reuses one phase's filter across all four, coded 4:2:2 / 4:2:0 solve all four phases plus the luma-aided refine; `DCTIF_only` emits empty sets with the correct header. Runs once on the final reconstruction after the bitrate matcher, as `coding_engine.py::compress` does. The least-squares solve is a **deterministic f64 pivoted-QR `gelsy`** (column pivoting, rank estimate on the R diagonal at `eps*max(m,n)`, `tzrzf`/`ormrz` completion), *not* MKL's f32 `gelsy`: on rank-borderline systems MKL's rank estimate is nondeterministic across identical calls (observed rank 30 vs 32) and its residuals run up to ~2x the true minimum, so its coded weights cannot be reproduced there by any deterministic solver — on well-posed systems our integerised codes are identical, and our residual is never worse than the reference's | oracle tests (`encoder::filters::efe_linear::tests::oracle`, `reference-tests` — missing dumps panic): `scripts/ref_vectors/dump_efe_solves.py` replays `SplitDecide` on a vector's `EFElinear.in.*` planes and records every `lstsq` triple + `integerize` pair into `<vector>/efe_solves/` (`make_reference_streams.sh efesolves`; the `DUMPS` table pins per-dump solve/int/excused counts): on `crop277_efe_f4c5_f3c6_nl` all 12 solves integerise to the reference's codes and every forced spec replays exactly, solve order identical; on `img30_c420_efe_f3c5_f4c7_nl` 50/130 solves code-exact, 80 excused on ill-posed systems (`smin/smax <= 1e-2`; all design matrices bit-identical); every dumped `integerize` pair is exact on all three dumps; on `img30_base_efelin_bpp050` the free search hit two MKL-truncated solves, so our stream codes a better filter than the reference's draw — 28,023 B vs 27,944 B (+0.28 %) and **+0.099 dB** through the reference decoder |
| `encoder::filters::icci` + `encoder::msssim` | `filters/eICCI/icci_filter.py::compress` (per-tile candidate search, `compress_tile`, `get_current_error`, `calculate_{mse,msssim,mixed}`, `skip_luma`/`skip_msssim`), `model_idxes.py::{map_idx, encode_header}`, `tiling.py::setup_tiles_enc`; `pytorch_msssim==0.2.1`'s `ssim.py` | ported: `select` runs the per-tile luma/chroma search (`zip` of the active maps, strict-improvement luma, best-of-`[u+v, u, v]`-gain chroma with `argmax`'s first-maximum) and signals `IcciHeader` exactly as `encode_header` writes it; the shipped `mixed` loss, pure `mse` and `ms-ssim`, and the loss weights are configurable (`EicciConfig`), the short lists or the whole ten-network bank (`process_short_list`), the filter's own tiling (`numSamplesPerTile`, overlap, 176 boundary rule) and the `s_ver/s_hor == 1` gate. The networks, Haar front-end and tile layout are the decoder's (`filters::icci`/`model::icci`); the picture the search scores is the EFE-linear output when both tools are on (`FiltersComposite` order). `ms_ssim` keeps the reference's elementwise f32 op order and reduces everything (11-tap convolutions, per-map means, 2x2 pools) in f64 — bit-identical on every tier and thread count. **Unexercised by an oracle:** the long list (no shipped config uses it — self-consistency test only), the HOP/SOP bank and short lists, EFE-linear + eICCI chained (no vector enables both), non-8-bit sources | `tests/encode_ref.rs`: `msssim_matches_pytorch_msssim` on the `--msssim` dump (5 planes 161x161..1024x1024) — max abs error **2.0e-6** vs `pytorch_msssim`; `eicci_selection_matches_reference`: **0 of 9** tile selections differ from the reference on `img30_base_eicci_bpp050`, `img30_base_on_bpp025`/`bpp100` and the six-tile `img01_base_eiccitiles_lef_bpp050` (fed the reference's own `eicci.in` planes — no ties needed excusing); `eicci_encode_end_to_end`: our streams carry the identical `IcciHeader` and decode with worst diff 0; `eicci_tiers_agree_bit_for_bit`: the whole stream is byte-identical on every SIMD tier, threaded or not |

| `encoder::filters::efe_nonlinear` | `filters/EFEnonlinear/EFEnonlinear.py::compress`: `LumaAidedAdaptiveNonlinearFilter_encoder` (the 1200-px tile grid over the downsampled luma snapped to 64, `lumaMin`/`lumaMax` as `round(x*100)`, the eight ReLU-hinge design columns solved per tile per chroma plane), `integerize` (raw u16 codes — `encode_header`'s fold pins `minSymbol`/`maxSymbol` to 0/65535), `calculateOnOff`/`apply_OnoffSwitch` (the W5 block search, `block_sizes`/`base_model_beta` by `model_id`, `candNum`), the `lossModifier`-scaled per-plane enable and strictly-positive mask keep rules | ported: `decide` runs once on the post-eICCI picture with the EFE-linear up-sampled picture as the mask alternative (`FiltersComposite` order), downsamples the luma to the chroma lattice, solves and applies per tile, then searches masks only when the alternative exists; fills `EfeNonlinearHeader` (mask flags + geometry + masks, per-plane enables, tile grid, `numTiles`, luma bounds, `candNum`, raw weight codes). The weight solve reuses `efe_linear`'s deterministic f64 pivoted-QR `gelsy`; same MKL caveat as E1 but sharper — the hinge matrices are rank-borderline in practice, so MKL's f32 draws are not reproducible run-to-run (the coded stream and a `compress` replay disagree on enable flags on identical inputs) | oracle test `tests/encode_ref.rs::efe_nonlinear_decisions_match_reference` (`reference-tests`, missing dumps panic): `scripts/ref_vectors/dump_efe_nonlinear.py` replays `compress` on a vector's `filters/` planes into `<vector>/efe_nonlinear/` and re-solves every tile in f64 (`dgelsy`); wired into `make_reference_streams.sh` (the `efe` set + the `on`/`efenl` base vectors), 17 dumps. Tile grid, luma bounds, mask geometry and `numTiles` asserted exactly; **272/272 tile weight codes exact** against the f64 oracle (the MKL draw differs on 262). Downstream of the solve parity is measured: 11/34 plane-enable outliers vs *both* reference draws, 3 mask keep/drop diffs, 289/1234 mask blocks, output max abs diff 19.2 (0..255). `reference_decoder_accepts_efe_nonlinear_stream` (`--ignored`): our `efe_linear + efe_nonlinear` stream decodes under the stock reference decoder within 1 |

| `encoder::rate` | `bitrate_matcher/bitrate_matcher.py::{match_luma, beta_linear_interpolation}` | ported: model pre-selection at displacement 0, log-rate interpolation, +/-100 bisection, per-model `BDL_range`. Two trial measures (`EncodeParams::rate_estimate`): `Coded` (default) measures the real codestream at the +/-1 % parameter-default tolerance; `Likelihood` is the reference's `ECLibLH` estimate with the reference's effective tolerances (`cfg/BRM/regen_list.json`: -10 % / +5 %, and its unclamped +/-100 bisection window) and picks the reference's `(model, beta)` exactly. **`find_UV_beta_with_hyperopt` is not ported — unreachable in b9e573f**: the branch (`bitrate_matcher.py:831-833`, taken when `independent_beta_UV = 0`) imports `hyperopt`, which is not in `requirements.txt` and not installed in the reference venv — an encode forced onto that path (`-model.bitrate_matcher.independent_beta_UV 0` under `--set_target_bpp`, which pulls in `cfg/BRM/regen_list.json`) dies with `ModuleNotFoundError` before writing a stream. Even with the package installed it would fail on the next line: `MSSSIM()` at `bitrate_matcher.py:399` is a bare name with no import anywhere in the module. See "Reference dead code: the rate matcher's UV hyperopt search" | `tests/encode_ref.rs::rate_matching_hits_the_target` (coded path): same model as the reference at all five CTC rates, and closer to the target at every one (table below); `likelihood_estimate_picks_the_references_displacements`: same `(model, beta)` as the reference at all five CTC rates on both test pictures, estimate within 2e-3 bpp of the reference's logged per-trial estimates |
| `model::mcm` (compress) | `context.py::{forward, pred, gen_skip_cubeflag, convert_cubeflag_map, _mask_redundant_padding_*}` | ported: per stage quantise with the sigma-threshold mask, decide the stage's cube flags from its own reconstruction error, re-quantise with the cubes that must not be skipped | as above (`y.residual_quant`, `y.cube_flag`) |
| `decoder::entropy` | `common_modules.py::decode/decode_z/decode_y/_ac_decode_y/_cal_step_size`, `gm.py::build_indexes` | ported for: all 4 models, 1..16 threads, no regions / dependent / independent regions, RVS, GRFS, the quality map, and progressive decode (`num_decode_chs`: a caller-chosen prefix of the latent channels) | `tests/entropy_ref.rs`: 21 reference streams (6 with RVS and/or GRFS, 3 with a quality map); z_hat, sigma, quantised residual exact, dequantised residual bit-identical |
| `nn` + `nn::reference` | `torch.nn.functional` conv2d / conv_transpose2d / pixel_shuffle / ReLU / ReLU6 | ported as plain loops that *define* the crate's numeric contract (FMA accumulation in `(ic, ky, kx)` order). Kept as the oracle and as the fallback for geometries the fast engine does not cover (stride-2 convolutions) | `tests/nn_vectors.rs`: 11 tiny PyTorch-computed cases (groups, depthwise, stride 2, 2x2, 1x3/3x1, both transposed geometries) within 2e-6 relative; pixel shuffle exact |
| `nn::fast` | (replaces oneDNN under PyTorch) | NCHWc blocked tensors (16 lanes on AVX-512, 8 on AVX2 / NEON / WebAssembly SIMD128 / scalar); register-blocked FMA micro-kernel for stride-1 and stride-2 convolutions (any group count with block-aligned groups, up to 9 taps) reading the input in place with virtual zero padding; depthwise 3x3; stride-2 transposed convolution (k <= 4); int8 `madd` convolution for the HSD; ReLU/ReLU6/gate/sigmoid/ELU-gate, layer norm, channel attention, pixel shuffle on blocked data; rayon over output rows; bounded buffer pool. `exp` is one fixed algorithm (Cephes `expf`) and long sums run in `f64` with a fixed order, so every tier agrees bit for bit. **Not done: Winograd, int8 VNNI (no safe `vpdpwssd`/interleave abstraction in archmage/magetypes 0.9.29 for the packed layout), padding-free transposed convolution; Wasm128 kernel retuned to B=4 x V=8 with register-resident accumulators (pass 2, `benchmarks/wasm_kernel_2026-09-18.md`); remaining per-iteration overhead is V8 safepoint/load artifacts, not reachable from safe Rust. Native retune (pass 3, `benchmarks/conv_kernels_2026-09-18.md`): V4 tile B=24 (B=28 spilled every accumulator each iteration), one overlapping full-width tail block instead of a narrow-block cascade, const-tap-count unrolling, register-resident int accumulators — ~40 % off single-thread decode, bit-identical** | `tests/fast_vs_reference.rs`: every tier, with and without threads, bit-identical to `nn::reference`; `tests/math_tiers.rs`; `tests/decode_ref.rs::tiers_and_threads_agree_bit_for_bit` on whole SOP / BOP / HOP decodes |
| `model::hyper_decoder` | `components/autoencoder_hyper/decoder/base.py` | ported | `tests/decode_ref.rs` (`psi` within 5e-4 abs of the reference) |
| `model::mcm` | `components/contexts/{context,MCM_phases,fusion_pred_net,utils}.py` (decoder direction) | ported: 4-phase context model + the context-free chroma path | `tests/decode_ref.rs` (`y_hat` within 5e-4) |
| `model::synthesis` | `components/autoencoder_data/decoder/{sop,bop,hop}_{prim,sec}.py`, `activations/resau.py`, `base_layers/conv_layers.py` | ported: SOP, BOP and HOP, luma and chroma | `tests/decode_ref.rs` (planes within 3e-3 on a 0..255 scale) |
| `model::attention` | `base_layers/{cab.py, tam.py}`, `conv_layers.py::ResidualBlock` | ported: residual block, CAB (sigmoid-gated trunk, mask at half resolution), TAM (two transformer blocks: layer norm, 1x1 + depthwise 3x3, 4-head channel attention with L2-normalised q/k and learned temperature, ELU-gated feed-forward; optional stride-2 / transposed resampling). The decoder never passes CAB's `gama`, so it is fixed at 1 | `tests/decode_ref.rs::img30_high_off_bpp050` (HOP stream end to end); `tests/math_tiers.rs` (attention math bit-identical across 11 CPU-feature permutations) |
| `tools::tiles` | `tiling/tiling.py::TileManager` (`_init_image_tiles_with_overlap`, `_init_image_tiles` + `_add_overlap`, `_get_latent_tile_from_image_tile`, both branches of `_get_core_of_overlapping_tile` for picture tiles) | ported. `minimum_tile_size` / `_adjust_boundary_tiles` is encoder/metric-side only and not ported | unit tests reproduce the layouts the reference logs (2096x1400: tile 1024 / overlap 64, and tile 640 / overlap 128 with two independent regions); `tests/decode_ref.rs` img01 streams |
| `model::analysis` | `components/autoencoder_data/encoder/{bop,hop}_{prim,sec}.py`, `base_layers/utils.py::normalize/padding_layer` | ported, called from `Encoder` (tiled above ~1 MP — `encoder::tiles`): BOP and HOP, luma and chroma (12-plane half-resolution input), replicate padding before every stride-2 convolution, TAM / CAB for HOP. `feature_clipping` not ported (no checkpoint has `clip_thres`; the reference runs `clipping_mode = 0`) | `tests/encode_ref.rs` (oracle: `dump_encode.py --enc2`, 560x888): `y` max abs error BOP 4.6e-5 luma / 1.1e-5 chroma, HOP 1.1e-4 / 4.6e-5 (asserted < 5e-4); tiers and thread counts bit-identical |
| `model::hyper_encoder` | `components/autoencoder_hyper/encoder/basic.py` (`abs_in_hyperprior = 1`, LeakyReLU 0.01), weights `hyper_encoder.*` in `VM_common_int` | ported, called from `Encoder` | `tests/encode_ref.rs`: unrounded `z` within 3.3e-5 of the reference (asserted < 2e-4); after clamp + round-half-even every `z_hat` symbol equals the reference's on both vectors, both components |
| `decoder::reconstruct` | `ccs_sgmm_tool.py::forward/decompress`, `common_modules.py::hyper_decode_tile/merge_psi_overlaps_of_tiles/extract_psi_for_mcm/decompress_ar_scale_tile/merge_y_hat_overlaps_of_tiles/extract_y_hat_for_synthesis_tiles/decompress_y_hat_to_image_tile` | ported: dependent and independent regions, synthesis tiling (luma and chroma must be tiled identically, as the reference assumes). Latent post-processing (LSBS) included | `tests/decode_ref.rs`: 16 streams incl. 3 region streams and 6 tool streams (oracle for those: the reference decoder run with a contiguous skip mask, see below) |
| `decoder::output` | `ccs_sgmm_tool.py::decompress` (`to_format_`), `common/image.py::to_444_/to_RGB_/clip_data_/write_yuv`, `pytorch_ops.py::resize_tensor`, `colorspace.py` (BT.709), `image_io.py::write_png` quantisation | ported: coded → source chroma format (bicubic 4:2:0 / 4:2:2 → 4:4:4 with `align_corners=True`, reproducing PyTorch's compiled kernel including where its build fuses multiply-adds), BT.709 → RGB, YUV output for YUV sources (4:4:4 / 4:2:2 / 4:2:0), 8 and 10 bit, non-displayed border. **Rejected (`Error::Unsupported`): the user-defined colour transform (`colour_transform_idx = 2`)** is dead, self-inconsistent code upstream — see "Reference dead code" below; there is no trustworthy oracle | `tests/nn_vectors.rs::bicubic_align_corners_matches_torch` (bit-identical to PyTorch on 6 shapes); `tests/decode_ref.rs`: 9 format streams (YUV 420/422/444, 10-bit 420/444, odd 203x301, RGB coded 4:2:0 and 4:2:2, display crop) |
| `filters::efe_linear` | `filters/EFElinear/EFElinear.py`: `decompress`, `SplitApply`, `LumaAidedUpsampler_apply`, `pixelUnshuffleGeneral`, `pixelShuffleGeneral`, `deinteger` | ported for every chroma format the reference can produce: 4:4:4 source coded 4:4:4 / 4:2:2 / 4:2:0 (the latter two with the 4x4 DCT-IF kernels and four coded phases, incl. `DCTIF_only` = no coded filters), 4:2:2 and 4:2:0 sources; filter lengths 1..4, all 8 region splits, odd picture sizes, the second ("up-sampled") picture for the non-linear filter's switch. **Rejected with `Error::Unsupported` (no oracle, the reference fails on them too):** a plane signalled as not filtered (`best_cand_idx = 0`) in a picture coded at the source's chroma resolution; vertical-only subsampling (`*_ver = 2, *_hor = 1`); 4:2:2 source coded 4:2:0. Note that the *decoder* around it still only outputs 4:4:4-coded 4:4:4 pictures (`decoder::output`, and the bicubic `to_format_` between synthesis and filters is not ported), so the subsampled branches are verified in isolation only | `tests/filters_efe_ref.rs`: 17 reference streams, filter run on the reference's own input planes, output within 2e-4 (0..255) of the reference's, measured max 9.2e-5; identical bits on every tier, threaded or not. `tests/decode_ref.rs`: 7 EFE streams through the whole decoder, plus upstream's two `tools_on` streams (all four filters chained: the second picture travels EFE linear → eICCI → EFE non-linear) |
| `filters::efe_nonlinear` | `filters/EFEnonlinear/EFEnonlinear.py`: `decompress`, `LumaAidedAdaptiveNonlinearFilter_apply`, `apply_OnoffSwitch`, `downsample`, `deinteger` | ported: per-tile two-layer 1x1 network (1 and 4 tiles checked), U-only / V-only / both, on/off masks with values 0, 1, 2 and block sizes 112 / 96, 4:4:4 / 4:2:2 / 4:2:0 sources, odd sizes. Same `Unsupported` formats as EFE linear | `tests/filters_efe_ref.rs`: **bit-identical** to the reference on all 16 streams that enable it (max abs error 0); `tests/decode_ref.rs` as above |
| `filters::lef` | `filters/LEF/LEFfilter.py` (`decompress`, `adptive_sharpness`), nearest up-sampling as `torch.nn.functional.interpolate` does it (`floor(dst * (in / out))` in f32) | ported: luma only, so every chroma format takes the same path. **Unverified: bit depths other than 8** (the range is taken as `2^bit_depth - 1`; the decoder rejects 10-bit streams before it gets here) | `tests/filters_lef_icci_ref.rs`: fed the reference's own input plane, the output is **bit-identical** to the reference's on 4 streams (`model_id` 1 and 2: two of the four constant rows are exercised), on every tier, threaded or not. `tests/decode_ref.rs::img30_base_lef_bpp050` and `decoder_api_on_filter_streams`: whole decode |
| `filters::icci` + `model::icci` | `filters/eICCI/{icci_filter.py, icci_models.py, model_idxes.py, params.py}`, `base_layers/conv_layers.py::ResidualBlock_BN_RectKernel`, `tiling.py::_adjust_boundary_tiles`, `image.py::{to_444_, to_format_}`, short lists from `cfg/pipeline.json` | ported for every chroma format the reference implements: 4:4:4, and 4:2:0 / 4:2:2 by up-sampling chroma to 4:4:4 (bicubic, the verified `decoder::output::resize_bicubic`), filtering at luma size, then bilinear down-sampling back (`nn::resize_bilinear` reproduces PyTorch's channels-last kernel bit for bit — weight products rounded once, then nested fused multiply-adds). Per-tile network selection (long and short lists, all three operating points' banks), the filter's own overlapping tiling with the 176-sample boundary adjustment, two-level Haar transform, both trunks on `nn::fast` (batch norm and residual scales folded into the convolutions), networks cached in `Decoder`. **Missing: eICCI tiling combined with a non-displayed border (`Unsupported`, unverified), the HOP network bank (code path shared, no oracle stream selects it), bit depths other than 8** | `tests/filters_lef_icci_ref.rs`: fed the reference's input planes, output within 1.1e-4 (0..255) on 6 streams incl. a 2096x1400 one with six filter tiles and both subsampled forced-encode vectors; bit-identical across tiers. `tests/decode_ref.rs`: `img30_base_eicci_bpp050`, `img30yuv422_base_eicci`, `img01_base_eiccitiles_lef_bpp050`, and `img30yuv420_base_eicci` (non-conformant forced stream, headers rebuilt in the test) |
| `gpu/` (`zenjpegai-gpu`, separate workspace member; the core crate does not depend on wgpu) | same sources as `model::synthesis` / `model::attention`; runs them as WGSL compute shaders through wgpu 30 | ported: SOP, BOP, HOP synthesis (luma + chroma), synthesis tiling and independent regions, `GpuDecoder` (CPU entropy / latent stage and output stage, GPU synthesis), readback-free `rgba8unorm` presentation; builds for wasm32. **Runs in a browser (verified on hardware, 2026-09-18 — row `B1`); no f16**; one tuning pass done (-10% to -22% of device time on BOP and HOP, SOP unchanged), the convolutions still run at a few per cent of the card's f32 peak (`gpu/README.md` "Status") | `gpu/tests/kernels.rs`: every kernel vs `nn::reference` / `nn::fast::math`, max relative error 1.7e-6 (conv, transposed conv), 2.5e-7 (attention), exact copies / shuffles. `gpu/tests/decode_ref.rs`: planes vs reference 3.2e-4 (SOP), 4.3e-4 (BOP), 3.4e-4 (HOP), 6.6e-4 / 5.6e-4 (2096x1400, 6 tiles / 12 tiles in 2 independent regions), bound 3e-3; 8-bit output differs by 1 in 50 / 59 / 35 of 1,491,840 and 348 / 373 of 8,803,200 samples; vs the CPU engine planes differ by at most 9.2e-4. All measured on an **RTX 2080** (NVIDIA 580.178.04, `ZENJPEGAI_GPU_ADAPTER=GeForce just gpu-test`), feature `gpu-tests`; the numbers above are that run and are unchanged by the tuning of 711233ad and 0120c800, which keeps every output's summation order. 8-bit output differs by 1 in 53 / 54 / 33 of 1,491,840 (SOP / BOP / HOP) and 348 / 365 of 8,803,200 (2096x1400, 6 tiles / 12 tiles in 2 independent regions) samples; the `rgba8unorm` presentation texture differs from the CPU output stage in 28 of 1,491,840 samples, all by one step (it was 46,533 until the shader rounded half-to-even itself instead of trusting the driver's float-to-unorm conversion — llvmpipe had agreed, NVIDIA did not). Timings: `benchmarks/gpu_decode_2026-09-17_rtx2080.{tsv,meta}`, `gpu_profile_2026-09-17_rtx2080.{tsv,meta}`, reference software on the same card `gpu_reference_2026-09-17.{tsv,meta}` |

## Accuracy of the float path (measured, 560x888 test image 00030, upstream b9e573f, torch 1.10.2 CPU)

| stream | psi max abs | y_hat max abs | planes max abs (0..255) | 8-bit samples differing (of 1,491,840) |
| --- | --- | --- | --- | --- |
| base profile (BOP), 0.12 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 51, all by 1 |
| base profile (BOP), 0.50 bpp | 8.9e-7 | 8.0e-5 | 4.7e-4 | 73, all by 1 |
| base profile (BOP), 1.00 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 67, all by 1 |
| 2096x1400 image 00001, base profile, 0.50 bpp, 6 synthesis tiles | < 5e-4 | < 5e-4 | < 3e-3 | 398 of 8,803,200, all by 1 |
| same image, 2 dependent regions / 2 independent regions / 2 independent regions + 8 ANS threads (model 2) | < 5e-4 | < 5e-4 | < 3e-3 | 404 / 404 / 396, all by 1 |
| simple profile (SOP), 0.50 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 49, all by 1 |
| high profile (HOP), 0.50 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 41, all by 1 |
| base / simple profile with LSBS, RVS, GRFS in six combinations | < 5e-4 | < 5e-4 | < 3e-3 | 47 .. 57, all by 1 |
| YUV sources 4:2:0 / 4:2:2 / 4:4:4 (8 bit) | < 5e-4 | < 5e-4 | < 3e-3 | 21 of 745,920 / 19 of 994,560 / 33 of 1,491,840 |
| YUV sources, 10 bit, 4:2:0 / 4:4:4 (10-bit samples) | < 5e-4 | < 5e-4 | < 3e-3 | 66 of 745,920 / 87 of 1,491,840 |
| RGB coded 4:2:0 / 4:2:2 (bicubic up-sampling), display crop 523x883 | < 5e-4 | < 5e-4 | < 3e-3 | 51 / 50 / 64 |
| base profile + EFE linear (stock encoder), and 4 streams with EFE linear + non-linear forced to filter lengths 2..4 / every region split | < 5e-4 | < 5e-4 | < 3e-3 (before filters) | 47 .. 66, all by 1 (the two filters add at most 9.2e-5 of their own) |
| 2096x1400 image 00001 with EFE linear + non-linear (4 tiles); 277x201 crop (odd size) | < 5e-4 | < 5e-4 | < 3e-3 | 384 of 8,803,200; 5 of 167,031, all by 1 |
| base profile + LEF, 0.50 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 58, all by 1 |
| base profile + eICCI (U and V filtered), 0.50 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 53, all by 1 |
| upstream `tools_on` (RVS, GRFS, LSBS, all four post-filters), 0.25 / 1.00 bpp, through `Decoder` | < 5e-4 | < 5e-4 | < 3e-3 | 65 / 56, all by 1 |
| 2096x1400, base profile + eICCI in six filter tiles (Y, U, V filtered) + LEF, 0.50 bpp | < 5e-4 | < 5e-4 | < 3e-3 | 427 of 8,803,200, all by 1 |
| base profile + eICCI on YUV 4:2:2 / 4:2:0 sources (forced-encode vectors; the 4:2:0 one is non-conformant and decodes with rebuilt headers, see below) | < 5e-4 | < 5e-4 | < 3e-3 | 24 of 994,560 / 19 of 745,920, all by 1 |

Post-filters on their own (input = the reference's planes entering the filter, 0..255 scale):
LEF 0 (bit-identical) on 4 streams; eICCI max abs error Y 9.2e-5, U 9.2e-5, V 1.1e-4 over 3
4:4:4 filtering streams and Y 9.2e-5, U 6.1e-5, V 9.2e-5 over the two subsampled ones
(asserted bound 5e-4). The `planes` column above is the synthesis output
before the filters.

Entries written `< x` are the bounds `tests/decode_ref.rs` asserts, not individually recorded
maxima; the other numbers were read off `examples/dbg_recon`.

The differences come from convolution summation order (PyTorch/oneDNN vs this crate's fixed
order); they land on 8-bit rounding boundaries in about 0.005 % of samples.

## Encoder parity (measured 2026-09-17, 560x888 test image 00030, eighteen fixed-model encodes; `num_chs` added 2026-09-20)

Vectors `enc_img30_{bop_m1_b0, hop_m2_b0, bop_m1_bm300, sop_m0_bm300, bop_m3_b400,
bop_m0_bm1069, hop_m3_bm1069}`, `enc_img01_bop_m1_b0` (2096x1400, 6 analysis tiles),
`enc_img30_bop_m1_b0_{threads8, rvs, rvsonly, grfsonly, lsbs, lef, qmap, qmap_rvs}` and
`enc_img01_bop_m1_b0_{depregions, indregions}` (`make_reference_streams.sh encoder`), plus
the reduced-channel vectors `enc_img30_bop_m1_{b0_y64u32, bm300_y96u48, b0_y96u48_rvs,
b0_uv0}` and `enc_img01_bop_m1_b0_y80u40` (the `numchs` set).

| what | result |
| --- | --- |
| colour pre-processing vs the reference's own analysis inputs | bit-identical (every sample, every analysis tile) |
| `z_hat`, `skip_scale_log`, `scale_log`, cube flags | identical on all twenty-three, both components |
| stream bytes vs the reference encoder's | identical on eighteen of twenty-three; the other five are the same length |
| `num_chs` vectors (`num_chs` = 64/32, 96/48, 96/48+RVS, 160/0, 80/40) | all five: the integer stage identical (`z_hat`, both scale maps, skip mask, cube flags — including `use_cube_flags = 1` on the `uv0` chroma), signalled `num_chs` and truncated `grfs_channel_flag` equal the reference's; four streams byte-identical, `enc_img01_bop_m1_b0_y80u40` moves 1 of 1,228,800 luma symbols by 1 (same length, 181,981 bytes) |
| `LEF_chIdx` (the channel of the luma scale map with the highest mean) | the reference's choice on all five LEF streams (`tests/encode_ref.rs::lef_channel_matches_reference`; `dump_encode.py --lef` dumps `lef.ch_idx` + `lef.avg_sig`) |
| `enc_img30_bop_m3_b400` (178,753 bytes, the exception) | same length; 41 of 313,600 luma and 1 of 188,160 chroma residual symbols move by 1; the two decodes differ in 2,455 of 1,491,840 samples, all by 1 |
| `enc_img01_bop_m1_b0` (205,110 bytes, 2096x1400) | byte-identical from the reference's own latents; from the PNG, 1 of 1,844,480 luma symbols moves and the stream is still 205,110 bytes |
| `enc_img01_bop_m1_b0_depregions` (205,171 bytes) | byte-identical from the reference's own latents; 1 symbol moves from the PNG |
| `enc_img01_bop_m1_b0_indregions` (205,087 bytes) | 1 symbol moves even from the reference's latents (the psi of a region boundary); 205,088 bytes from the PNG |
| our streams through the **reference decoder** vs through ours | 43-61 of 1,491,840 samples differ (435-504 of 8,803,200 at 2096x1400), all by 1: the decoder's own float-network gap, table above. Region and quality-map streams use the patched decoder, because the stock one mis-decodes every region stream (its own included) and cannot read a quality map at all |
| analysis transform vs the reference's `y` | 4.6e-5 (BOP luma), 1.1e-4 (HOP luma); hyper-encoder 3.3e-5 |

Why a symbol can move at all: it is `round((y - mean) * scaler)`, and `mean` comes out of the
hyper-decoder and, for luma, the context model - float networks whose sums this crate orders
differently from oneDNN (`psi` differs by up to 8.9e-7, `y_hat` by 8.0e-5). A symbol whose
unrounded value sits within that of a `.5` boundary lands on the other side. At the rates the
reference's stock configuration produces this never happens; it takes beta displacement +400
(0.34 bpp -> 2.9 bpp) before 42 of 501,760 symbols move.

### Rate matching (`--bpp`), 560x888 test image 00030, base profile

Two trial measures (`EncodeParams::rate_estimate`): `Coded` (the default) codes every trial
and measures the real codestream; `Likelihood` is the reference's own measure — `ECLibLH`,
`-sum(log2 p)` over `z_hat` (the factorized model's float `forward`, per-channel tables in
`src/encoder/lh.rs`) and the residual (`GMProbModel.forward`'s unquantised Gaussian) — at the
reference's effective tolerances (`cfg/BRM/regen_list.json`, always appended by
`--set_target_bpp`: -10 % / +5 %, not the +/-1 % parameter defaults this port's coded path
kept) and its unclamped +/-100 bisection window. `Likelihood` picks the reference's
`(model, beta)` exactly on all ten CTC points (five rates, both test pictures); its estimate
reproduces the reference encoder's logged per-trial estimates to the four decimals the log
prints. Note the reference's own estimator misses the coded size by up to ~1 % at these rates
(img30 0.75: est 0.7866 vs coded 0.7946) — the ideal likelihood under-prices the quantised
ANS tables, exp-Golomb tails, flush and container.

| target bpp | our model / displacement (Coded) | our bpp | reference model / displacement (= Likelihood pick) | reference bpp |
| --- | --- | --- | --- | --- |
| 0.12 | 0 / +259 | 0.1165 (-3.0 %) | 0 / +211 | 0.1081 (-9.9 %) |
| 0.25 | 1 / -222 | 0.2523 (+0.9 %) | 1 / -189 | 0.2643 (+5.7 %) |
| 0.50 | 1 / +259 | 0.4827 (-3.5 %) | 1 / +203 | 0.4493 (-10.1 %) |
| 0.75 | 2 / -248 | 0.7568 (+0.9 %) | 2 / -209 | 0.7946 (+5.9 %) |
| 1.00 | 2 / -9   | 1.0088 (+0.9 %) | 2 / 0    | 1.0198 (+2.0 %) |

The model choice is the reference's at every rate on either measure. The two coded-path picks
that miss +/-1 % are capped by the model's own `BDL_range` (`cfg/BRM/default.json`): model 0
and model 1 stop at +259, which is as high a rate as they can reach. 12-15 trial encodes,
0.55 s of process wall time.

### `tools_on` (`EncodeParams::tools_on`, CLI `--tools-on`), measured 2026-09-18

The preset mirrors `cfg/tools_on.json`: RVS, GRFS, LSBS, all four post-filters. Gate
(`tests/encode_ref.rs::tools_on_ctc_matches_reference`, vectors `img30_base_on_bpp*` /
`img01_base_on_bpp*` from `make_reference_streams.sh toolson`): same `(model, beta)` as
the reference's signalled pick on all ten CTC points, every tool signalled, and the
entropy-coded payload compared **symbol by symbol**. The brief's nominal gate — 0.5 %
size, 0.02 dB — does **not** hold as written; the measured numbers are:

| vector | reference bytes | ours | size | TON bytes ref → ours | symbols moved | PSNR delta |
| --- | --- | --- | --- | --- | --- | --- |
| img30 0.12 bpp | 7,818 | 7,969 | +1.93 % | 58 → 209 | none | +0.1222 dB |
| img30 0.25 bpp | 16,546 | 16,671 | +0.76 % | 57 → 182 | none | +0.1108 dB |
| img30 0.50 bpp | 28,238 | 28,362 | +0.44 % | 58 → 182 | none | +0.0876 dB |
| img30 0.75 bpp | 49,592 | 49,894 | +0.61 % | 80 → 381 | 1 residual by 1 | +0.0484 dB |
| img30 1.00 bpp | 65,805 | 65,929 | +0.19 % | 341 → 465 | 1 residual by 1 | +0.0173 dB |
| img01 0.12 bpp | 45,489 | 45,704 | +0.47 % | 159 → 373 | 1 residual by 1 | +0.0285 dB |
| img01 0.25 bpp | 97,430 | 97,677 | +0.25 % | 390 → 637 | none | +0.0381 dB |
| img01 0.50 bpp | 194,184 | 194,321 | +0.07 % | 472 → 610 | 12 residuals by 1 | +0.0532 dB |
| img01 0.75 bpp | 250,386 | 250,614 | +0.09 % | 382 → 610 | 4 residuals by 1 | +0.0805 dB |
| img01 1.00 bpp | 389,142 | 389,564 | +0.11 % | 506 → 930 | 1 `z_hat` by 1, 37 residuals by ≤1 | +0.1050 dB |

Every breach of the nominal gate is the same documented mechanism: the post-filter
*decisions* differ because MKL's f32 `lstsq` is nondeterministic on these
rank-borderline hinge matrices (E1/E2; the reference's own `compress` vs coded-stream
replay disagree on enable flags). Our deterministic f64 `gelsy` solve lands a different
— on every vector, **better** — filter set: PSNR is higher on all ten through our
decoder *and* through the reference decoder (identical deltas; the `--ignored` leg
`reference_decoder_tools_on_psnr` sends each of our streams through
`src.reco.coders.decoder`). The size excess is entirely the bigger TON header that
codes the chosen filters (img30 0.12: +151 bytes = 209 − 58, payload symbol-identical);
one byte at img30 0.75 is the `ue`-coded substream size prefix crossing a code-length
boundary. The residual-symbol moves are the float-path rounding-boundary mechanism of
the table above (`mean` differs by ~1e-7, a symbol at a `.5` boundary flips); the single
`z_hat` move at img01 1.00 cascades through the scale map into 5,674 `scale_log` cells
and is what moves those 37 residuals. At img01 0.12 the GRFS chroma flags pick a
different set among tied channel means (`torch.sort` is unstable on ties; ours takes
lowest indices first — same count; the divergence is only *which* of the tied-mean
channels fill the lower-ranked slots). Asserted in the
test: the whole signature is **pinned per vector** (`TOOLS_ON_SIGNATURE` in
`tests/encode_ref.rs` — `z_hat`/residual move counts, the worst move, and the size excess
outside the TON, all exact), and a breach of the nominal gate additionally requires the PSNR
delta to be positive through our decoder *and* the reference decoder
(`reference_decoder_tools_on_psnr`, `--ignored`).

### Speed (`benchmarks/encode_end_to_end_2026-09-18.{tsv,meta}`)

Best of three steady-state runs of one fixed-model encode pass, `encode_ms`:

| case | reference, 1 thread | reference, 32 | ours, 1 thread | ours, 32 |
| --- | --- | --- | --- | --- |
| 560x888 SOP model 1 | 596 | 5671 | 301 | 63.8 |
| 560x888 BOP model 1 | 698 | 5577 | 307 | 73.3 |
| 560x888 HOP model 2 | 5595 | 6177 | 1715 | 413 |
| 2096x1400 BOP model 1 (6 analysis tiles) | 3997 | 5672 | 2007 | 423 |

Whole-process wall time, one encode, checkpoints read from `.pth`: **0.13 s** against the
reference's **1.5 s** at 560x888. The reference pins torch to one thread by default and gets
*slower* when allowed more on this box, so its one-thread column is the fair comparison.

With every tool on (`tools_on`; `benchmarks/encode_tools_on_2026-09-18.{tsv,meta}`) the
reference's post-filter searches dominate its encode — its rate-matched `tools_on` encode of
the 2096x1400 picture takes ~35 s. Best of three, `encode_ms`:

| case | reference, 1 thread | reference, 32 | ours, 1 thread | ours, 32 |
| --- | --- | --- | --- | --- |
| 560x888 BOP model 1, tools on | 1408 | 24835 | 1364 | 621 |
| 560x888 BOP 0.50 bpp, tools on | 4966 | 5131 | 3078 | 958 |
| 2096x1400 BOP model 1, tools on | 17321 | 55662 | 9694 | 5077 |
| 2096x1400 BOP 0.50 bpp, tools on | 34974 | 20844 | 21392 | 6970 |

Not started (decoder): nothing left of the float path. Decided against porting: the
user-defined colour transform (`colour_transform_idx = 2`) — dead, self-inconsistent code
upstream, see "Reference dead code".
Not started (encoder): nothing left. `num_chs` below the model's channel count is ported
(`EncodeParams::num_chs`); the rate matcher's `find_UV_beta_with_hyperopt` UV search is
unreachable in b9e573f (missing `hyperopt` dependency, unresolved `MSSSIM` — "Reference dead
code" below).


## WebAssembly numeric policy (measured 2026-09-17)

WebAssembly has no deterministic fused multiply-add: `f32::mul_add` becomes a soft-float `fmaf`
call and relaxed-simd's `relaxed_madd` fuses or not depending on the host CPU. On
`target_arch = "wasm32"` the contract's multiply-add is therefore **unfused** (`w * x + acc`,
two roundings; `nn::fmadd`) in `nn::reference`, the scalar tier and the `Wasm128` tier alike.
Native targets are unchanged (fused everywhere).

- Wasm output is bit-identical across the `Wasm128` and `Scalar` tiers, across builds with and
  without `simd128`, and across engines (checked: node 26 / V8, wasmtime; PNG bytes equal).
  `tests/fast_vs_reference.rs`, `tests/math_tiers.rs`, `tests/nn_vectors.rs` and all of
  `tests/decode_ref.rs` (incl. `tiers_and_threads_agree_bit_for_bit`) pass on `wasm32-wasip1`
  under node (`just test-wasi`).
- Wasm output differs from native output by rounding only: 18..183 8-bit samples per picture, each
  by 1. Against the reference decoder wasm sits where native does
  (`benchmarks/wasm_parity_2026-09-17.tsv`, `scripts/wasm/parity.sh`):

| stream | samples | wasm vs reference | native vs reference | wasm vs native |
| --- | --- | --- | --- | --- |
| img30 BOP 0.12 / 0.25 / 0.50 / 0.75 / 1.00 bpp | 1,491,840 | 54 / 59 / 73 / 58 / 55 | 51 / 64 / 73 / 53 / 67 | 19 / 29 / 30 / 37 / 22 |
| img30 SOP 0.50 bpp, HOP 0.50 bpp | 1,491,840 | 49, 50 | 49, 41 | 24, 21 |
| img30 RVS+GRFS, RVS only, GRFS only | 1,491,840 | 43, 48, 51 | 51, 54, 51 | 18, 20, 24 |
| img01 2096x1400, plain / 8 ANS threads | 8,803,200 | 419 / 419 | 398 / 398 | 153 / 153 |
| img01 dependent / independent regions / independent + 8 threads | 8,803,200 | 443 / 412 / 376 | 404 / 404 / 396 | 183 / 176 / 170 |

  Every difference is by 1; the worst case (443 of 8,803,200) is a quarter of the whole-picture
  gate (1 in 5000).
- Speed, one thread, 560x888 BOP 0.5 bpp stream, node 26 on a Ryzen 9 9950X3D: fused soft-float
  6.0 s; unfused without SIMD 1.42 s; unfused `Wasm128` 0.31 s (wasmtime: 0.37 s). Native
  AVX-512, one thread: 0.093 s. The `Wasm128` micro-kernel keeps 4 output positions x 8 channels
  in registers (3 / 4 / 6 / 8 / 12 positions measured: 428 / 352 / 385 / 457 / 457 ms).
- 2026-09-18 profile + speed pass (`benchmarks/wasm_profile_2026-09-18.md`,
  `wasm_threads_2026-09-18.tsv`): `conv::run_row` is 88% of a single-threaded decode; the kernel
  was believed to be at its bit-identity floor (unfused mul+add, fixed order, 16 v128
  registers — B=5/B=6 measured slower; pass 2 below disproved the "floor"), so the levers were
  outside it: the rayon pool now defaults to
  `min(hardwareConcurrency, 16)` children — measured optimum on a 32-hwc host where 32 children
  regressed (162 vs 132 ms mean) — the per-row border-column scratch is `thread_local` (allocator
  calls under shared memory are atomic-linked), and wasm-bindgen init uses the non-deprecated
  object form. Browser medians over the 16 demo streams (threads package):
  chromium 204 -> 104 ms, firefox 227 -> 107 ms, webkit 238 -> 131 ms
  (`benchmarks/wasm_decode_2026-09-18.tsv` run 0 vs run 3; box is shared, the run-0 baseline
  predates several sessions, treat the exact ratio as approximate). ~12% of decode stays serial
  (entropy + latent reconstruction), which caps threaded decode at ~90-130 ms regardless.
- 2026-09-18 kernel pass 2 (`benchmarks/wasm_kernel_2026-09-18.md`): inspecting V8's TurboFan
  code showed the "floor" claim above was wrong — accumulators were spilled per input block and
  every tap carried bounds checks. Restructuring `block()` (accumulators live across the whole
  channel loop, bias inits `acc` directly, tap rows sliced once to `(B-1)*S+1` with a single
  `assert!`, the `vin == V` input-channel loop unrolled, padding taps point at a static
  `ZERO_PAD` instead of a per-call `Vec`) gives 356.6 -> 307.1 ms (-13.9%) single-threaded
  under V8 and ~920 -> 365 ms (-60%) under wasmtime; output stays bit-identical across wasm
  tiers and thread counts. The residual per-`(channel, tap)` overhead is V8 artifacts
  unreachable from safe Rust (safepoint poll, a wasted `leaq` before each `vbroadcastss`).
  The native pass 3's constant-tap-count unrolling did *not* carry over to wasm: it cost
  Cranelift ~47 % (spilled v128 registers) while buying V8 ~6 %, so it is
  `cfg(not(target_arch = "wasm32"))` — recheck table appended to the same file.

## Reference behaviour that differs from its own configuration

- **`skip_cube_thr` is 3, not the tool's default of 1.** `cfg/oper_point/common.json` sets it on
  `model.CCS_SGMM.tools_common.model_common.common_modules.skip_mode`, a path that *does* reach
  `SkipModeParams` (unlike `sigma_quant_level` below). With 1 the encoder would flag two cubes of
  `enc_img30_bop_m1_b0` that the reference leaves unflagged (their worst latent reconstruction
  error is 1.01 and 1.13, computed from the reference's own `mcm_y.0.mean*` dumps); with 3 the
  flags agree on all seven encoder vectors. `encoder::SKIP_CUBE_THR`.
- **`sigma_quant_level` is 32, not 35.** `cfg/pipeline.json` sets `sigma_quant_level: 35`,
  `sigma_quant_max: 100` on `tools_common`, a node that owns no such parameter, so the values
  never reach `CommonEncDecModules`. The reference runs with the defaults (32 levels,
  `sigma_quant_max = 54.82`, `log_k = 0.20036548400207613`); so does this port. Checked at
  runtime (`hyper_scale_decoder.sigma_idx_max_value == 3967`) and by the entropy-stage parity
  tests, which fail with 35.

## Reference decoder bug: region streams

With `region_partitioning_flag = 1` the reference **decoder** mis-decodes residuals (its own
encoder and decoder print different reconstruction MD5s for such streams). Cause:
`SgtProbWrapper.decode` passes `masks.cpu().numpy()` to the C++ coder; for a region that is a
sub-rectangle of the latent this is a non-contiguous view, and the C++ side reads it linearly,
ignoring strides. The encoder side calls `.copy()` first and is correct. (When cube flags are
present the mask goes through `torch.logical_or`, which happens to make it contiguous.)

This port decodes region streams the way the encoder wrote them. `tests/entropy_ref.rs` therefore
checks region streams against tensors dumped from the reference *encoder*
(`scripts/ref_vectors/dump_encode.py`): z_hat, sigma and residual match exactly for dependent
regions, independent regions, and independent regions with 8 ANS threads.

For the stages behind the entropy decoder the oracle is the reference decoder run with that one
defect patched at runtime (`scripts/ref_vectors/dump_decode.py --contiguous-masks`, dumps in
`<vector>/fixed_decoder/`). With the patch its residual equals the encoder's bit for bit on all
three region streams (checked), so its psi / y_hat / reconstruction are what the encoder intended.

## LEF / eICCI: speed, and what is still missing

Measured 2026-09-17 on a busy 9950X3D (load 16; `benchmarks/filters_lef_icci_2026-09-17.tsv`,
`scripts/bench/filters_lef_icci.sh`), filter time only, same input planes:

| stream | filter | reference (1 thread, its stock setting) | zenjpegai 1 thread | zenjpegai 32 threads |
| --- | --- | --- | --- | --- |
| 560x888, eICCI on U+V | eICCI | 38 .. 46 ms | 32.4 ms | 7.8 ms |
| 560x888, tools_on (eICCI on Y+U+V) | eICCI | 73 .. 98 ms | 49.1 ms | 10.3 ms |
| 560x888 | LEF | 9.8 .. 15 ms | 0.54 ms | 0.28 .. 0.37 ms |
| 2096x1400, 6 eICCI tiles, Y+U+V | eICCI | 424 .. 518 ms | 310 ms | 70 ms |
| 2096x1400 | LEF | 93 .. 110 ms | 3.5 ms | 2.1 ms |

(The reference allowed 32 threads is 10x *slower* on this loaded box; rows in the TSV.)

Missing, most important first, with the next concrete step for each:

1. **eICCI tiling together with `diff_display_img_*` != 0** (`Error::Unsupported`). The reference
   relies on Python slices clamping. Next step: a stream with a non-multiple-of-16 picture *and*
   `numSamplesPerTile` small enough to tile (`scripts/ref_vectors/cfg/eicci_tiles.json` on a
   cropped input), then clamp tile and core areas to the displayed size in `tile_layout`.
2. **Unexercised but shared code paths** (no oracle stream selects them): the HOP network bank
   (`eicci_hop_*`), SOP/HOP short lists, LEF constants of `model_id` 0 and 3, per-tile
   *different* selections, tiles with only one chroma plane filtered. Next step: encode HOP /
   SOP / very low and very high rate streams, add them to `tests/filters_lef_icci_ref.rs`.
3. **Bit depths other than 8** for both filters (range taken as `2^bit_depth - 1`, as
   `Image.data_range` suggests, never compared). Next step: a 10-bit source stream with LEF and
   eICCI on, same per-filter test.
4. **eICCI multi-thread scaling** is about 4.5x on 16 cores: the network is 140x222 at 48
   channels, too small for the engine's row-level split to fill the pool. Next step: run the
   luma and chroma trunks (independent) as two rayon tasks, or tile-level parallelism when the
   filter is tiled (tiles must still be *written* in raster order, see below, so only the
   inference of tiles whose inputs are final can overlap).

## Reference behaviour of the post-filters worth knowing

- **eICCI filters its tiles in place.** The reference *decoder* (`EfficientICCIFilter.decompress`)
  cuts each tile out of the picture it is writing filtered cores into, so a tile's overlap margin
  already holds its left / upper neighbours' output. The reference *encoder's* model search
  (`compress`) evaluates every tile on the unfiltered picture instead; its reconstruction is
  nevertheless produced by calling `decompress`, so encoder and decoder agree on the picture.
  The effect is large (0.87 on a 0..255 scale on the tiled test stream when the margin is read
  unfiltered); this port does what the decoder does. Raster order is therefore normative.
- **eICCI tiles: `_adjust_boundary_tiles` breaks on small layouts.** With one tile column (or
  row) narrower than 176 it indexes `[-2]` onto the same tile and produces a negative position;
  with tiles not larger than 176 + overlap the neighbour becomes empty. The port rejects both as
  `InvalidData` instead of reproducing the accident.
- **eICCI always rescales.** Even with no plane selected in any tile the picture goes through
  `/ 255`, clamp to `[0, 1]`, `* 255`. The port does the same (checked: `img30_base_on_bpp100`).
- **eICCI tile layout vs picture size.** The layout is computed from the coded picture size
  (`get_original_img_shape`), the filter runs on the displayed picture (cropped by
  `diff_display_img_*`). With one tile this is harmless (Python slicing clamps); with tiling on
  it relies on slices clamping silently. The port returns `Unsupported` for that combination.
- **eICCI on subsampled pictures.** The filter itself supports them: the decoder up-samples
  chroma to 4:4:4 (`Image.to_444_`, bicubic), filters at luma size, clamps/scales, then
  down-samples back (`to_format_`, bilinear). The reference *encoder* never selects it there
  (`compress` switches the filter off when `s_ver != 1 or s_hor != 1`), so the oracle vectors
  are forced encodes (`scripts/ref_vectors/force_icci_encode.py`, set `icci420` of
  `make_reference_streams.sh`). The 4:2:2 vector is a conformant stream — `icci_enable_flag`
  is coded whenever at least one subsampling factor is 1 — and the stock reference decoder
  decodes it. **The 4:2:0 vector is deliberately non-conformant**: for `s_ver == s_hor == 2`
  the flag and the header are not part of the syntax at all
  (`EfficientICCIFilter.auto_enableflag_detected_value` auto-detects `False` because it tests
  `and` where `compress` tests `or`), so a decoder that plays by the grammar skips those bits
  and desynchronises — the stock reference decoder fails with `ExceptionReadHeader`, this port
  with `Error::UnexpectedEof`. The dump scripts therefore patch that one method at runtime
  (`dump_decode.py` / `dump_filters_lef_icci.py` `--patch-icci420`, lambda returning `None`);
  the patched decoder's reconstruction MD5 equals the forced encoder's. Our tests feed the
  entropy payloads (conformant, checked bit-exact) and run the staged decode / filter with the
  eICCI header rebuilt from the dump's `eicci.selection` / `eicci.signalled` /
  `eicci.short_list` tensors. `ToolHeader::write` still refuses to *emit* eICCI for 4:2:0 —
  the syntax has no place for it.
- **LEF clamps the whole luma plane** to the data range, the unfiltered one-sample border
  included; chroma is passed through unclamped.

## Reference constants that are not in the bitstream

- EFE linear: the eight region splits (`cands`, with `0.33` / `0.66` fractions and borders
  rounded to multiples of 32 phase samples by Python's round-half-even), the DCT-IF 4-tap
  kernels, the weight scale (`(code - 32767) / 2^11`). EFE non-linear: 8 luma bins per tile.
  `filters::efe_linear` / `efe_nonlinear` carry copies.

- LEF: the three sigma-index thresholds and sharpening gains per `model_id`
  (`LEF.recSharpThrList` / `recSharpMagList`, Python literals; `filters::lef::{THRESHOLDS,
  MAGNITUDES}`), and the 3x3 kernel `[[5,5,5],[5,24,5],[5,5,5]] / 64`.
- eICCI: the per-operating-point, per-`model_id` short lists (`cfg/pipeline.json`; the Python defaults in
  `params.py` hold the same numbers but under integer keys, which the string-keyed lookup would miss), the order
  of the 10 networks of a bank (`ckpt_files`), the use of the BOP bank for SOP, the network
  shape (`nf = 48`, `nbY = 2`, `nbUV = 4`; `params.py` defaults to `nbUV = 2`), the minimum tile
  size 176 (`tile_min_size_for_MSSSIM`, an encoder-side MS-SSIM constraint that shapes the
  decoder's tile layout), and the batch-norm epsilon `1e-5`.

- RVS thresholds / scale lists and LSBS scale lists are per-model configuration
  (`cfg/pipeline.json`, `tools_N`), and the LSBS thresholds are the tool's Python defaults. A
  decoder has to know them; `tools::rvs` and `tools::lsbs` carry copies.

- With *dependent* regions, each region's context-model output loses `48 / 2 / 16 = 1` latent
  sample on every side facing another region before it is merged, and merged regions overwrite
  each other in raster order. The 48 is the default `numSamplesTileOverlap` of the reference's
  internal hyper-decoder / context tile managers; nothing signals it. `decoder::reconstruct`
  hard-codes it (`HD_MCM_TILE_OVERLAP`).

## Reference decoder bug: quality maps

The stock reference decoder cannot decode a stream with a quality map:
`QualityMap.decode_header` calls `.item()` on a Python `int` and raises, and it reads
`log2_num_threads_q_minus1` with 1 bit where the encoder writes 2. Of the encoder's map
generators only the mask-file one (`qp_map_type = 3`) survives this code path; the others
downscale a shape that is already latent-sized and crash in `quantize_scale`.
`scripts/ref_vectors/dump_decode.py --fix-qmap-header` replaces that one method at runtime; with
it the decoder's reconstruction MD5 equals the encoder's on all three quality-map vectors, so that
patched decoder is the oracle for them (`<vector>/fixed_decoder/`).

## Reference dead code: the user-defined colour transform (`colour_transform_idx = 2`)

The claim that "upstream's inverse uses the first row of the inverse matrix for all three
components" is confirmed — and it is worse than that. At b9e573f the path cannot run at all:
`ColourTransformation.pre_processing` and `post_processing` call `Image.convert_range_(0, 1)`
while `convert_range_` takes a single tuple argument, so the encoder raises `TypeError` before
writing a stream. `scripts/ref_vectors/probe_colour_transform2.py` patches only that call (the
tuple every other call site passes) and forces `-colour_processing.colour_transform.
colour_transform_idx 2`. Then:

- *Both* directions apply `inv_matrix[0, :]` — the first row of the **inverse** matrix — to all
  three components (`colour_transformation.py` lines 82-84 and 120-122). The forward direction
  of a user-defined transform should presumably be `clr_tr_matrix` row-wise; as written the
  encoder codes the same linear mixture in all three planes, so colour is destroyed before
  compression, and `post_processing` has no inverse to apply anyway.
- Measured on an identity-matrix stream: the reference encoder's own reconstruction hashes
  identically on all three components; the reference decoder emits three identical planes
  (then converts the "wrong format" to RGB with a warning) — 12.45 dB PSNR against the source.

The header syntax itself is fine (`header::ColourTransform::Custom` round-trips the matrix and
offsets, verified on the probe stream). `decoder::output::finish` rejects such streams as
`Error::Unsupported`. Decision: **not ported** — there is no self-consistent reference
behaviour to match.

## Reference dead code: the rate matcher's UV hyperopt search (`find_UV_beta_with_hyperopt`)

`bitrate_matcher.py` can in principle pick the chroma displacement independently of the luma
one: when `independent_beta_UV = 0` it calls `find_UV_beta_with_hyperopt` (line 831-833),
which builds a `hyperopt` search over `beta_disp_UV` scored by an `MSSSIM`-weighted
distortion. The only reachable trigger is a `--set_target_bpp` encode (which appends
`cfg/BRM/regen_list.json`) with `-model.bitrate_matcher.independent_beta_UV 0` — no shipped
config sets it. At b9e573f the path cannot produce a stream:

- `requirements.txt` has `hyperopt` commented out and the reference venv does not have it:
  the forced encode dies at the branch's own `from hyperopt import atpe, fmin, hp` with
  `ModuleNotFoundError: No module named 'hyperopt'` (verified by running it).
- With the package installed it would still fail: `MSSSIM()` at `bitrate_matcher.py:399` is
  a bare name — nothing in the module imports or defines it, so the scoring function raises
  `NameError` on the first trial.

Decision: **not ported — unreachable in b9e573f**. The `independent_beta_uv` *signalling* is
ported (`header.rs` reads and writes the bit plus the second 12-bit displacement whenever the
UV beta differs from Y's; `EncodeParams::beta_displacement_log` takes a per-component pair, so
a caller that sets `[y, uv]` explicitly produces a conforming independent-beta stream), and
the ordinary UV displacement reuse — `beta_disp_UV` defaulting to the matched Y value — is
what every reachable reference encode does.

## EFE filters: reference behaviour worth knowing

- **The on/off switch looks at the U mask only.** `EFElinear.decompress` builds the second
  picture, and `EFEnonlinear.decompress` applies `apply_OnoffSwitch`, only when `mask1` (U) is
  present. A stream with `mask1_enabled_flag = 0, mask2_enabled_flag = 1` decodes with the V mask
  silently ignored. Ported as is (`efe_nonlinear::has_first_mask`).
- **"Plane not filtered" crashes the reference at full chroma resolution.** With
  `best_cand_idx = 0` `SplitApply` indexes `cands[-1]` and, when the picture is coded at the
  source's chroma resolution, dereferences the missing luma weights (`self.weightsY_UV.device`).
  The reference *encoder* dies the same way with `DCTIF_only = 1` on such a picture, so no such
  stream exists. This crate returns `Error::Unsupported`. With chroma coded below the source's
  resolution the same signalling works (DCT-IF taps only) and is ported + verified.
- **Masks without the second picture crash the reference** (`img_alt` is `None` when EFE linear
  is off or its first set filters neither plane): `Error::InvalidData` here. Likewise a mask whose
  size is not `ceil(plane / bS)`, zero `bS`, zero tile size, fewer tile parameters than tiles.
- **Empty regions.** On pictures below about 64 luma samples a three-way split leaves regions
  empty; PyTorch rejects the empty convolution for kernels above 1x1. Skipped here (no oracle).
- **`icci_enable_flag` is absent for 4:2:0 sources** (`auto_enableflag_detected_value`): the tool
  header goes straight from the EFE linear data to `EFE_nonlinear_filter_enabled_flag`. Found
  while making 4:2:0 vectors; `header::ToolHeader` follows it (`PictureHeader::icci_flag_coded`).
- The shadowed loop variable `j` in `LumaAidedAdaptiveNonlinearFilter_apply` (plane index reused
  as tile column index) is harmless: the plane is chosen before the inner loop runs.
- Formats the header can spell but the reference cannot run: vertical-only subsampling
  (`Image.get_format_from_subsampling` returns `None`), 4:2:2 source coded 4:2:0
  (`to_420_` raises `NotImplementedError`).
- The encoder's RDO nearly always lands on the smallest EFE choice (1x1 filter, one region,
  non-linear filter off). `scripts/ref_vectors/force_efe_encode.py` overrides the *decisions*
  (not the weights, syntax or decoder) to get streams for the other branches.

Timing, 560x888, EFE linear (3x3 + 4x4 filters, 4 / 6 regions, two pictures) + EFE non-linear
(U and V, masks), `examples/prof_filters_efe`, best of 40 on a busy box: 6.0 ms + 2.7 ms on one
thread, 2.2 ms + 0.4 ms threaded; the reference's `decompress` calls take 10.6 ms + 10.6 ms
(one torch thread, `dump_filters.py` `timing.txt`, one run). 2096x1400: 33 + 18 ms one thread,
8.3 + 1.7 ms threaded, reference 159 + 94 ms.

## Deliberate divergences from the reference

- **Container strictness.** The reference reader skips unknown two-byte words and spins forever on
  a truncated file. `container::Codestream::parse` requires SOC, known markers, PIH first, and EOC.
- **`log2_num_threads_q_minus1` width.** The reference encoder writes it with 2 bits, its decoder
  reads 1 (so the reference cannot decode its own multi-threaded quality maps). `header` uses
  2 bits, like the `z` and residual thread fields.
- **Corrupt ANS payloads.** Where the C++ decoder would read in front of its buffer, `mans`
  reports `Error::InvalidData`.

## Reference environment

The reference runs locally from `~/work/zen/jpeg-ai-reference-software` (see `CLAUDE.md` for
the environment recipe). Reference bitstreams, decoded images and intermediate tensor dumps
used by the parity tests live under `/mnt/v/output/zenjpegai/reference/`; they are produced by
`scripts/make_reference_vectors.sh` and are not committed (size).

## Work queue

Each item is sized for one agent, names its upstream source, the files it owns, the oracle and
the gate that closes it. "Byte-identical" means our stream equals the reference encoder's for the
same decisions; where the reference's float search can legitimately land elsewhere, the gate is
"same length, decodes with the reference decoder, size/PSNR within the stated tolerance". Add new
reference streams through `scripts/ref_vectors/make_reference_streams.sh` (new set), dump
intermediate tensors through additive flags of `dump_encode.py` / `dump_decode.py`, and record
measured numbers in the status table above when you close an item.

### Encoder (decoder-supported features the encoder cannot produce yet)

- ~~**E1 EFE linear, encode side.**~~ **Done 2026-09-18** (row `encoder::filters::efe_linear`):
  `EncodeParams::efe_linear` + `--efe-linear` run `decide` on the final reconstruction — the
  reference's `compress`/`SplitDecide`/`searchCore`/`LumaAidedUpsampler_encoder`/`linearSolve`,
  both filter sets, all coded chroma formats, `DCTIF_only`. The solver is a deterministic f64
  pivoted-QR `gelsy`, not MKL's f32 one. **Documented deviation (the gate's "documented
  deviation" branch):** on rank-borderline systems MKL 2020's rank estimate is nondeterministic
  across identical calls (rank 30 vs 32 measured) and lands on worse-than-minimum solutions
  (residual up to ~2x), so its coded weights are unreproducible there by construction — the
  oracle tests assert exact codes only on well-posed systems (64/146 dumped solves across
  three pictures; every design matrix is bit-identical). On `img30_base_efelin_bpp050` our
  stream picks a better filter than the reference's draw: +79 B (+0.28 %) and +0.099 dB
  through the reference decoder. With decisions pinned the coded weights are identical on
  every well-posed spec; the reference decoder reads our stream.
- ~~**E2 EFE non-linear, encode side.**~~ **Done 2026-09-18** (row
  `encoder::filters::efe_nonlinear`): `EncodeParams::efe_nonlinear` + `--efe-nonlinear` run
  `decide` on the post-eICCI reconstruction — the 1200-px tile grid snapped to 64,
  `lumaMin`/`lumaMax`, the eight-hinge per-tile weight solve, per-plane enable, the W5
  on/off-mask search and `candNum`, `EfeNonlinearHeader` (raw u16 codes; the header's
  `minSymbol`/`maxSymbol` fold pins 0/65535). Same solver caveat as E1, sharper: the hinge
  matrices are rank-borderline so MKL's f32 `gelsy` draws are nondeterministic on them
  (stream vs replay disagree on enable flags on identical inputs); the oracle dumps a
  deterministic f64 re-solve (`dgelsy`) and the test asserts our codes against it —
  **272/272 exact** — while enable/mask/output parity is measured (11/34 enable outliers,
  3 mask keep/drop diffs, 289/1234 mask blocks, max output diff 19.2/255). The reference
  decoder accepts our `efe_linear + efe_nonlinear` stream (decoded output within 1).
- ~~**E3 eICCI, encode side.**~~ **Done 2026-09-19** (row `encoder::filters::icci` +
  `encoder::msssim`): `EncodeParams::eicci` (`--eicci`, `--eicci-loss`, `--eicci-long-list`,
  `--eicci-tile-samples`) runs `select` once on the final reconstruction — the reference's
  `compress` (`zip` of the luma/chroma candidate maps, strict-improvement luma, chroma by the
  best of the `[u+v, u, v]` gains vs the tile's initial error, `argmax` first-maximum) plus
  `encode_header`'s `use_YUV` / `icci_use_shortList` / `icci_model_signalled_idx`, over the
  decoder's own networks and tile layout. All three losses (`mse` / `ms-ssim` / the shipped
  `mixed` with configurable `luma/chroma_loss_weights`), short and long candidate lists,
  `numSamplesPerTile` tiling with overlap 48 and the 176 boundary rule, the 4:4:4-source gate,
  and the EFE-linear → eICCI composite order. `ms_ssim` is a fixed-order `pytorch_msssim
  0.2.1` port (f32 elementwise in the reference's op order, every reduction in f64):
  **max abs error 2.0e-6** against the reference's own scores (the `dump_filters.py
  --msssim` oracle, 5 planes 161x161..1024x1024), bit-identical on every tier. Gate met:
  **0 of 9** tile selections differ from the reference across the four eICCI vectors — no
  ties needed excusing — our streams carry the identical `IcciHeader`, decode with worst
  diff 0, and are byte-identical across all SIMD tiers and thread modes. Left unverified by
  an oracle (recorded in the row): the long list, the HOP/SOP banks, EFE-linear chained
  into eICCI, non-8-bit sources.
- **E4 LEF, encode side.** DONE (2026-09-18): `EncodeParams::lef` + `--lef` signal
  `LEF_enabled_flag` and `LEF_chIdx`, the channel of the luma `scale_log` with the highest mean
  (`encoder::filters::lef::reference_channel`, porting `LEFfilter.py::analyze`). The filter
  itself is *not* applied on the encode side — the reference runs the post-filters once on the
  reconstruction after `model.compress` returns (`coding_engine.py::compress`), outside the
  bitrate matcher's trials. Gates met: `enc_img30_bop_m1_b0_lef` (new in the `encoder` set)
  byte-identical to the reference's stream; same `LEF_chIdx` as the reference on all five LEF
  vectors; the reference decoder reads our stream.
- ~~**E5 Post-filter flags + `tools_on` preset.**~~ **Done 2026-09-18**:
  `EncodeParams::tools_on` + CLI `--tools-on` (plus the individual `--efe-linear`,
  `--efe-nonlinear`, `--eicci` flags alongside the existing `--lef`/`--rvs`/`--grfs`/
  `--lsbs`) mirror `cfg/tools_on.json`. All ten CTC points (`img30`/`img01` x
  0.12-1.00 bpp, the `toolson` reference-vector set) pick the reference's `(model,
  beta)` and signal every tool; the coded payload is symbol-identical except
  rounding-boundary moves (worst: 1 `z_hat` + 37 residuals, all one step). The nominal
  0.5 % / 0.02 dB gate fails on the post-filter *decisions*: our deterministic f64
  solve picks a different, always-better filter set than MKL's nondeterministic f32
  `lstsq` (size +0.07..+1.93 %, all of it the larger TON; PSNR +0.017..+0.122 dB
  through our decoder and the reference's alike). Full table and mechanism under
  "Encoder parity → `tools_on`"; `benchmarks/encode_tools_on_2026-09-18.tsv` times the
  reference's rate-matched tools_on encode against ours.
- ~~**E6 Chroma-subsampled, 10-bit and YUV sources.**~~ **Done 2026-09-18**: `SourceImage` /
  `SourceMeta` / `read_yuv`, `nn::resize_bilinear` (bit-identical to the
  reference's `F.interpolate(mode="bilinear", align_corners=True)` channels-last kernel),
  `EncodeParams::{c_ver, c_hor, diff_display}`, CLI `.yuv` input and `--c-ver`/`--c-hor`/
  `--diff-display`, 16-bit PNG input. Gate result: the analysis-transform inputs are
  bit-for-bit the reference's on all nine `formats` vectors (`enc2` dumps added under each
  vector), seven streams byte-identical, the two 10-bit ones the same length with 5 of
  1,254,400 luma residual symbols moved by 1 (the rounding-boundary mechanism of the table
  above); the reference decoder reads all nine and its output agrees with ours within 1 LSB
  (`tests/encode_ref.rs::formats_streams_match_reference` and
  `reference_decoder_accepts_formats_streams`, the latter `--ignored`).
- ~~**E7 Likelihood-based rate estimation.**~~ **Done 2026-09-18**: `RateEstimate::Likelihood`
  (`EncodeParams::rate_estimate`, CLI `--rate-estimate likelihood`) runs the `ECLibLH` port in
  `src/encoder/lh.rs` — factorized `z` likelihood tables plus the `GMProbModel` Gaussian —
  with the reference's effective tolerances (-10 %/+5 %, `cfg/BRM/regen_list.json`) and its
  unclamped +/-100 window. Gate: same `(model, beta)` as the reference at all five CTC rates
  on both test pictures (10/10), estimates equal the reference's logged per-trial estimates to
  the four decimals printed, estimate within 1.0 % of the coded size — the reference's own
  estimator misses by that much (the brief's 0.5 % is unattainable: it prices the ideal
  likelihood, not the quantised ANS tables). `RateEstimate::Coded` stays the default.
- ~~**E8 Encoder limits and memory.**~~ **Done 2026-09-18**: `EncodeLimits` +
  `Encoder::limits` (max pixels / dimensions / estimated heap; defaults 120 MP and 4 GiB, as
  the decoder's), `estimate_encode_memory` / `Encoder::estimate_memory` and
  `Encoder::release_buffers` in `src/encoder/limits.rs`, mirroring `decoder::limits`. Bounds
  are judged on the source size before any picture-sized allocation or network run, and
  `max_memory_bytes` shrinks the shared recycled-buffer pool to fit under what is left.
  Rate matching now drops each losing model's latents as soon as model selection knows it is
  worse (only the best-so-far set survives the loop; ~45 MB held less at 3 MP), trial streams
  die as soon as their size is measured, latents move instead of deep-cloning, each
  component's residual payload is coded before the next component's compress runs, and
  non-traced encodes shed scales / masks / quantised residuals right after their payload
  (`encode_traced*` keeps them). heaptrack measurements (`scripts/bench/memory_encode.sh`):
  `benchmarks/memory_encode_2026-09-18.tsv` — calibrated on the `--pool-mb 0` rows the
  estimate lands at 1.30x–1.44x of the measured peak heap on both sizes, fixed and
  rate-matched (gate: within 1x–2x), pinned by
  `tests/api_ref.rs::encode_memory_estimate_covers_the_measured_peaks`. Pixels unchanged:
  `encoder_end_to_end_matches_reference`, `rate_matching_hits_the_target` and the `formats`
  vectors still pass. CLI `--pool-mb` now applies to `encode` too. Still unmeasured: HOP
  encodes (the estimate pads HOP's fixed cost from the decoder-side model ratio).
- ~~**E9 zencodec encoder trait.**~~ **Done 2026-09-18**: `JpegAiEncoderConfig` /
  `JpegAiEncodeJob` / `JpegAiEncoder` in `src/codec.rs`, the encode-side mirror of the decoder
  bridge (config = `Arc<dyn ModelSource>` + `Engine` + `EncodeParams` + `EncodeLimits`, cheap
  `Clone` sharing the encoder's model cache). `EncoderConfig::with_generic_quality` maps `q`
  to `q / 100` bpp and selects `Encoder::encode_to_bpp`; `estimate_encode_resources` reports
  E8's calibrated `estimate_encode_memory` (live → `peak_memory_bytes_est`, live + pool →
  `peak_memory_bytes_max`) plus a wall-time model from
  `benchmarks/encode_end_to_end_2026-09-18.tsv`. Per-job `with_limits` layers a check in front
  of the shared encoder (strictest of job vs config wins) plus `max_output_bytes` on the
  stream; `with_stop` drives the same `enough` token as the decode side. Inputs: interleaved
  RGB8/RGB16 full-range sRGB-or-untagged `PixelSlice`s (the `RDI` written carries no CICP, so
  other tagging is refused rather than mislabelled), `encode_srgba8` for padding-alpha RGBA8.
  Row-push, pull and animation encode are rejected via `UnsupportedOperation`. Byte-identity
  gate: `tests/codec_ref.rs` — `encode_matches_the_plain_api_byte_for_byte`,
  `encode_16bit_matches_the_plain_api`, `rate_matched_encode_matches_the_plain_api` (with the
  `RateMatch` surfaced as an output extension) and `encode_srgba8_strips_padding_alpha` all
  compare against `Encoder::encode` / `encode_to_bpp` on a decoded source; limits, stop and
  estimate paths covered alongside. Still open: forwarding a descriptor's CICP into `RDI`,
  and `Metadata` is dropped, not embedded (no ICC/EXIF/XMP carrier is declared).
- ~~**E10 `num_chs` + the UV beta hyperopt search.**~~ **Done 2026-09-20**:
  `EncodeParams::num_chs` + CLI `--num-chs y,uv` port the reference's
  `-model.CCS_SGMM.tools_common.model_{y,uv}.common_modules.num_chs`: the header signals the
  clamped count per component, the residual substream codes the first `num_chs` channels
  (`_ac_encode_y`'s `[:, :num_chs]` slices of residual / `scale_log` / mask), uncoded channels
  keep a zero quantised residual, and their cube-flag contribution matches the reference —
  none for the context-module luma (`context.py::pred` zeroes `diff[:, num_decode_chs:]` with
  `num_decode_chs = num_chs` on encode), the full `|y - psi|` for the context-free chroma
  (`_compress_ar_scale` zeroes `y_rec_resi[:, num_chs:]` before `gen_skip_cubeflag`, so the
  reconstruction error there is the whole residual). `analyzeCWG` still ranks all channels and
  `encode_header` writes `cwgf[:num_chs]`; the likelihood rate estimate prices the residual
  over `num_chs` channels (z is always full-channel). Five new vectors
  (`make_reference_streams.sh numchs`): 64/32, 96/48 at beta -300 (cube flags exercised),
  96/48 with RVS+GRFS, chroma `num_chs = 0` (signals `use_cube_flags`), and 80/40 on the
  tiled 2096x1400 picture. Gate met: four of five streams byte-identical to the reference's
  (the tiled one moves 1 symbol by 1, same length); integer stage identical on all five; our
  decoder and the reference decoder both read them. `find_UV_beta_with_hyperopt` is **not
  ported — unreachable in b9e573f**: it needs the `hyperopt` package (commented out of
  `requirements.txt`, absent from the venv — the forced encode dies with
  `ModuleNotFoundError`) and an `MSSSIM` the module never imports. See "Reference dead code:
  the rate matcher's UV hyperopt search". What remains unported on the encoder: nothing.

### Browser and GPU

- ~~**B1** WebGPU synthesis in the worker/polyfill/demo behind feature detection~~ **Done
  2026-09-18** (the `webgpu` agent, hardened on hardware by `b1bhw`): `pkg-webgpu` third wasm
  package (`gpu` cargo feature on `zenjpegai-wasm`: async `decode` with CPU fallback on
  `GpuError`, `initGpu`/`gpuStatus`/`present`/`disableGpu`), `DecoderPool` `gpu` option
  (`auto|on|software|force-software|off`), `decodeToCanvas` no-readback presentation through
  `Blitter`, `chromium-webgpu` Playwright project + `benchmark-gpu.spec.ts` ->
  `benchmarks/wasm_decode_2026-09-18_gpu.{tsv,meta}`, size delta in
  `benchmarks/wasm_size_2026-09-18.md` §7. The `create_buffer_init`/`mappedAtCreation` trap is
  fixed at the root in `gpu/` (unmapped `create_buffer` + chunked `queue.write_buffer`, native
  parity re-verified unchanged); **verified on hardware** — RTX 2080 through Dawn/Vulkan in
  headless Chromium (`WEBGPU_ADAPTER=hardware`, `isFallbackAdapter: false`), all 16 demo
  streams on `path: 'gpu'`, zero traps. ~~Measured crossover keeps `auto` on the CPU engine at
  demo sizes~~ — superseded by `b3gpu` below; `auto` now takes a hardware adapter.
  Dawn/SwiftShader still loses its device at model-2 load (per-call CPU fallback, no trap).
  **b3gpu done 2026-09-18** — the ~150 ms per-call gap was mostly *not* GPU orchestration:
  phase timings (`benchmarks/wasm_gpu_phases_2026-09-18.tsv`) showed ~120 ms of it was the
  serial single-threaded CPU latent stage in `pkg-webgpu`. New `pkg-webgpu-threads` package
  (webgpu + rayon, picked automatically on isolated pages) plus one command encoder per decode
  (`RunTail` in `gpu/src/synthesis.rs`: tile plans + timestamp resolve + optional RGBA convert
  and staged readback in a single submit) and one batched map round-trip
  (`GpuContext::map_read2`) took warm ~1 MP `decode_ms` from ~179 ms to ~55 ms on the RTX 2080 —
  ~1.9x faster than the `threads` CPU package (~98-104 ms) and ~13x faster than `simd`
  (~675-765 ms), so `auto` now prefers a non-software adapter on both page kinds. GPU parity
  unchanged (same kernels, same counts: 6+6 `gpu/tests`, 195/3,000,000 samples differ by 1
  in-browser). Detail in `web/README.md` §7.
  **B2 done 2026-09-18** — demo throttling +
  viewport-priority queue landed: one shared queue in `web/src/pool.js` (threads build: exactly
  one decode in flight; simd build: at most `min(navigator.hardwareConcurrency - 1, 4)` —
  one hardware thread is left for the page's main thread, because `hwc` busy workers on a
  4-vCPU runner pushed per-image decode past 2x solo), the
  polyfill and demo only enqueue decodes once an image is within one viewport height
  (`IntersectionObserver`, `rootMargin '100% 0px'`, visibility-ordered, re-evaluated function
  priorities), and every queued image shows a pre-sized placeholder. The AIC artwork swap also
  landed (corpus item 3008, flower-garland still life, CC0 — replaces the removed dead-chicken
  still life; `web/demo/IMAGES.md`, release `demo-assets-v1` updated and hash-verified).
  Measured on the demo page in chromium (`web/tests/scheduling.spec.ts`,
  `benchmarks/wasm_demo_scheduling_2026-09-18.tsv`): pre-fix vs post-fix simd path
  first-image 1123/1097 ms and all-eight-images 2073/2082 ms — on this 32-core box the change
  is about ordering, laziness and the bounded-inflight guarantee, not throughput; on a
  smaller-core-count machine the queue is what prevents decode oversubscription.
  **B4 done 2026-09-18** — cache-safe demo assets: `manifest.json` carries per-variant
  `file`/`sha256` and a `models` digest map (`web/scripts/update-demo-manifest.mjs`, re-uploaded
  to `demo-assets-v1`); `demo.js` fetches `streams/<file>?v=<sha256>` and `manifest.json` with
  `cache: 'no-cache'`; `worker.js` keys Cache API entries on `?v=` digests
  (`zenjpegai-models-v2`; v1 purged). `web/tests/cache-swap.spec.ts` proves a same-name asset
  swap renders new content without clearing storage.
  **B3** GPU kernels: after the tuning of 711233ad + 0120c800 (gate / residual fused into the
  convolution's store, the transposed convolution's dead taps skipped, the depthwise 3x3 tiled
  in workgroup memory, the workgroup edge chosen per layer — device time for one picture BOP
  1024x1024 23.9 -> 21.3 ms, HOP 560x888 259 -> 224 ms, HOP 1024x1024 635 -> 498 ms, SOP
  unchanged) the convolutions still reach only a few per cent of f32 peak. Profiling them needs
  Nsight Graphics (GPU Trace): they are Vulkan compute shaders, so `ncu` sees no kernels at all
  and this box's `nsys` cannot load its Vulkan importer; neither is a substitute for the
  per-dispatch timestamps in `gpu_bench --profile`. Also: the activation pool never shrinks, so a workspace that has
  synthesised a 4096x4096 picture runs the next 560x888 BOP picture at 32 ms instead of 10;
  8-bit readback (201 MB of f32 planes at 4096x4096 is 88 ms of the 227 ms wall); `f16`;
  cancellation and `max_channels` on the GPU path (details in `gpu/README.md` "Status").

### Decoder

- **D1** eICCI on 4:2:0 / 4:2:2 (in progress on a SWE-2 agent; needs forced-flag vectors).
- **D2** Speed: Winograd F(2x2,3x3) or int8-VNNI for the 3x3 convolutions, padding-free
  transposed convolution, SIMD `exp` for HOP's ELU gate; re-run `scripts/bench/decode_end_to_end.sh`
  and commit the TSV after each.

### History of agent notes

Earlier per-agent appendices (browser 2026-09-17, GPU passes, encoder groundwork) were folded
into the rows above and the items here; their numbers live in `benchmarks/` and the crate READMEs.

Appended by the WebGPU browser agent (`webgpu` workspace), 2026-09-18:

- Landed: `gpu` cargo feature on `zenjpegai-wasm` (`wasm/src/gpu.rs`: `initGpu`, `gpuStatus`,
  `present`, `disableGpu`; `decode` becomes async under the feature and falls back to the CPU
  engine on any `GpuError`), a third `pkg-webgpu` package (`build-wasm.sh webgpu`; `pkg-simd` and
  `pkg-threads` untouched), worker/pool wiring (`gpu` mode `auto|software|force-software|off`,
  `decodeToCanvas` with one-time `OffscreenCanvas` transfer and worker-side `Blitter`
  presentation — no CPU readback when `presented == 'gpu'`), a `chromium-webgpu` Playwright
  project with per-path reporting, `tests/benchmark-gpu.spec.ts` →
  `benchmarks/wasm_decode_2026-09-18_gpu.{tsv,meta}`, and `wasm_size_2026-09-18.md` §7
  (`pkg-webgpu` = 663,094 bytes post-`wasm-opt`, +76% over `pkg-simd`; never downloaded by
  browsers that only offer a software adapter). Verified in-browser through SwiftShader
  (`WEBGPU_ADAPTER=swiftshader`): GPU decode + `presented:'gpu'` blit both ran, 201/3,000,000
  samples differ from the reference PNG, all by 1 (same bound as the CPU tests, reported
  separately — GPU output is not bit-identical to CPU). All 96 Playwright tests pass across
  chromium / firefox / webkit / chromium-webgpu (83 run per project set, 13 project-gated
  skips).
- ~~Open: never run on a hardware GPU adapter; `create_buffer_init` trap~~ — closed by `b1bhw`
  2026-09-18: hardware run on the RTX 2080 (adapter `nvidia/turing`, non-fallback) through
  Dawn/Vulkan; `gpu/` uploads are all unmapped `create_buffer` + chunked `write_buffer` so the
  staging-limit trap cannot occur (worker `onerror`→`disableGpu`→CPU-retry kept as defence in
  depth). Remaining: Dawn/SwiftShader loses the device at model-2 load (per-call CPU fallback).
  `auto` on the CPU engine was superseded by the b3gpu pass (B1 row above): `auto` takes a
  non-software adapter when one exists (`web/README.md` §7).

