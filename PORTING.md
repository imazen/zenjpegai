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
| `header` | `coding_engine.py`, `colour_transformation.py`, `multitools/engine.py`, `ccs_sgmm_tool.py`, `quantization.py`, `sep_chan_tool.py`, `skip_mode.py`, `res_var_scale.py`, `tiling.py`, `quality_map.py`, `rdi.py` (header functions only) | partial: PIH complete (parse + write + profile/level conformance); RDI complete; TON complete (LSBS flags, EFE linear filter sets, eICCI tiling + per-tile model selection, EFE non-linear masks / tiles / weights, LEF channel), parse + write; **UDI not ported** | `header::tests`: two reference-encoder PIH payloads parse to the values `scripts/bitstream_probe.py` prints and re-serialise byte-identically; a reference TON with all four post-filters on parses to its end and re-serialises byte-identically |
| `tensor` | (PyTorch tensors) | minimal `[C,H,W]` container | - |
| `model::hsd` | `components/autoencoder_hyper/decoder_scale/basic.py`, `base_layers/conv_quant_layers.py` | ported. Convolutions run through `nn::fast::PackedIntConv` (i16 pairs + `madd`, wrapping i32; exact by construction because integer sums are order-free); the plain scalar loops stay as `forward_reference` | `tests/entropy_ref.rs` (sigma maps equal the reference's); `tests/fast_vs_reference.rs::int_conv_matches_scalar_loops` (every tier, threads on/off, odd channel counts, clamped inputs, accumulator wrap-around) |
| `model::common` | `core_models/CCS_SGMM/common_modules.py::build_model`, `gain_unit.py::get_gain_vector_log`, `custom_prob_wrapper.py::normalize_z` | ported: z CDFs, HSD weights, gain vector. Hyper-decoder / MCM weights not loaded yet | `tests/entropy_ref.rs` |
| `tools::gain` | `quantization/gain_unit/gain_unit.py` | ported (decoder side) | `tests/entropy_ref.rs` (dequantised residual floats bit-identical). `scaler_from_log` uses libm `expf` where the reference uses `torch.exp`; not yet checked exhaustively over the whole input range |
| `tools::skip` | `skip_ls/skip_mode.py` (mask + cube-flag expansion) | ported (decoder side); cube flags exercised only by header round trips so far, no reference stream with `use_cube_flags = 1` yet | `tests/entropy_ref.rs` (threshold mask) |
| `tools::log2lin` | `common/log2lin.py::Log2LinConvertion` | ported: the 4352-entry table is computed (`round(2^17 exp(i ln(100/0.11)/4352 + ln 0.11))`) instead of stored | unit test pins the FNV-1a hash of upstream's literal table |
| `tools::rvs` | `quantization/rvs/res_var_scale.py` (`buildTables`, `analyze`, `quantize_scale`, `dequantize_resi`), constants from `cfg/pipeline.json` | ported (decoder side): block-wise "likely" map, RVS scale tables, GRFS ("cwg") per-channel variants | `tests/entropy_ref.rs` (scale map and residual exact) on RVS+GRFS, RVS-only and GRFS-only streams |
| `tools::qualmap` | `quality_map/quality_map.py` (`decode`, `quantize_scale`, `dequantize_resi`) | ported (decoder side): ANS-coded delta plane, DPCM reconstruction, log-scale offset and residual step per position, shared by both components | `tests/entropy_ref.rs` + `tests/decode_ref.rs` on 3 quality-map streams (plain, with RVS, with 8 ANS threads); oracle = reference decoder with its header defect patched (below) |
| `tools::lsbs` | `ls_processing/lsbs/lsbs_scale_mode.py` (`buildTables`, `post_processing`), constants from `cfg/pipeline.json` | ported | `tests/decode_ref.rs` LSBS streams (`y_hat` within 5e-4 of the reference after scaling) |
| `tools::regions` | `tiling/tiling.py::TileManagerHyper` | ported: region grids for y / psi / z, with and without overlap extension | unit tests + `tests/entropy_ref.rs` region streams (latent grid only) |
| `decoder::entropy` | `common_modules.py::decode/decode_z/decode_y/_ac_decode_y/_cal_step_size`, `gm.py::build_indexes` | ported for: all 4 models, 1..16 threads, no regions / dependent / independent regions, RVS, GRFS and the quality map. **Missing: `num_decode_chs` (progressive decode)** — such streams are rejected as `Unsupported` | `tests/entropy_ref.rs`: 21 reference streams (6 with RVS and/or GRFS, 3 with a quality map); z_hat, sigma, quantised residual exact, dequantised residual bit-identical |
| `nn` + `nn::reference` | `torch.nn.functional` conv2d / conv_transpose2d / pixel_shuffle / ReLU / ReLU6 | ported as plain loops that *define* the crate's numeric contract (FMA accumulation in `(ic, ky, kx)` order). Kept as the oracle and as the fallback for geometries the fast engine does not cover (stride-2 convolutions) | `tests/nn_vectors.rs`: 11 tiny PyTorch-computed cases (groups, depthwise, stride 2, 2x2, 1x3/3x1, both transposed geometries) within 2e-6 relative; pixel shuffle exact |
| `nn::fast` | (replaces oneDNN under PyTorch) | NCHWc blocked tensors (16 lanes on AVX-512, 8 on AVX2 / NEON / WebAssembly SIMD128 / scalar); register-blocked FMA micro-kernel for stride-1 and stride-2 convolutions (any group count with block-aligned groups, up to 9 taps) reading the input in place with virtual zero padding; depthwise 3x3; stride-2 transposed convolution (k <= 4); int8 `madd` convolution for the HSD; ReLU/ReLU6/gate/sigmoid/ELU-gate, layer norm, channel attention, pixel shuffle on blocked data; rayon over output rows; bounded buffer pool. `exp` is one fixed algorithm (Cephes `expf`) and long sums run in `f64` with a fixed order, so every tier agrees bit for bit. **Not done: Winograd, int8 VNNI, padding-free transposed convolution** | `tests/fast_vs_reference.rs`: every tier, with and without threads, bit-identical to `nn::reference`; `tests/math_tiers.rs`; `tests/decode_ref.rs::tiers_and_threads_agree_bit_for_bit` on whole SOP / BOP / HOP decodes |
| `model::hyper_decoder` | `components/autoencoder_hyper/decoder/base.py` | ported | `tests/decode_ref.rs` (`psi` within 5e-4 abs of the reference) |
| `model::mcm` | `components/contexts/{context,MCM_phases,fusion_pred_net,utils}.py` (decoder direction) | ported: 4-phase context model + the context-free chroma path | `tests/decode_ref.rs` (`y_hat` within 5e-4) |
| `model::synthesis` | `components/autoencoder_data/decoder/{sop,bop,hop}_{prim,sec}.py`, `activations/resau.py`, `base_layers/conv_layers.py` | ported: SOP, BOP and HOP, luma and chroma | `tests/decode_ref.rs` (planes within 3e-3 on a 0..255 scale) |
| `model::attention` | `base_layers/{cab.py, tam.py}`, `conv_layers.py::ResidualBlock` | ported: residual block, CAB (sigmoid-gated trunk, mask at half resolution), TAM (two transformer blocks: layer norm, 1x1 + depthwise 3x3, 4-head channel attention with L2-normalised q/k and learned temperature, ELU-gated feed-forward; optional stride-2 / transposed resampling). The decoder never passes CAB's `gama`, so it is fixed at 1 | `tests/decode_ref.rs::img30_high_off_bpp050` (HOP stream end to end); `tests/math_tiers.rs` (attention math bit-identical across 11 CPU-feature permutations) |
| `tools::tiles` | `tiling/tiling.py::TileManager` (`_init_image_tiles_with_overlap`, `_init_image_tiles` + `_add_overlap`, `_get_latent_tile_from_image_tile`, both branches of `_get_core_of_overlapping_tile` for picture tiles) | ported. `minimum_tile_size` / `_adjust_boundary_tiles` is encoder/metric-side only and not ported | unit tests reproduce the layouts the reference logs (2096x1400: tile 1024 / overlap 64, and tile 640 / overlap 128 with two independent regions); `tests/decode_ref.rs` img01 streams |
| `decoder::reconstruct` | `ccs_sgmm_tool.py::forward/decompress`, `common_modules.py::hyper_decode_tile/merge_psi_overlaps_of_tiles/extract_psi_for_mcm/decompress_ar_scale_tile/merge_y_hat_overlaps_of_tiles/extract_y_hat_for_synthesis_tiles/decompress_y_hat_to_image_tile` | ported: dependent and independent regions, synthesis tiling (luma and chroma must be tiled identically, as the reference assumes). Latent post-processing (LSBS) included | `tests/decode_ref.rs`: 16 streams incl. 3 region streams and 6 tool streams (oracle for those: the reference decoder run with a contiguous skip mask, see below) |
| `decoder::output` | `common/image.py::to_RGB_/clip_data_`, `colorspace.py` (BT.709), `image_io.py::write_png` quantisation | partial: **4:4:4, BT.709 → RGB only.** Missing: 4:2:0 / 4:2:2 chroma upsampling (bicubic), custom colour transform, YUV output | `tests/decode_ref.rs`: 8-bit output differs from the reference decoder in 49..73 of 1,491,840 samples, each by 1 |
| `filters::efe_linear` | `filters/EFElinear/EFElinear.py`: `decompress`, `SplitApply`, `LumaAidedUpsampler_apply`, `pixelUnshuffleGeneral`, `pixelShuffleGeneral`, `deinteger` | ported for every chroma format the reference can produce: 4:4:4 source coded 4:4:4 / 4:2:2 / 4:2:0 (the latter two with the 4x4 DCT-IF kernels and four coded phases, incl. `DCTIF_only` = no coded filters), 4:2:2 and 4:2:0 sources; filter lengths 1..4, all 8 region splits, odd picture sizes, the second ("up-sampled") picture for the non-linear filter's switch. **Rejected with `Error::Unsupported` (no oracle, the reference fails on them too):** a plane signalled as not filtered (`best_cand_idx = 0`) in a picture coded at the source's chroma resolution; vertical-only subsampling (`*_ver = 2, *_hor = 1`); 4:2:2 source coded 4:2:0. Note that the *decoder* around it still only outputs 4:4:4-coded 4:4:4 pictures (`decoder::output`, and the bicubic `to_format_` between synthesis and filters is not ported), so the subsampled branches are verified in isolation only | `tests/filters_efe_ref.rs`: 17 reference streams, filter run on the reference's own input planes, output within 2e-4 (0..255) of the reference's, measured max 9.2e-5; identical bits on every tier, threaded or not. `tests/decode_ref.rs`: 7 EFE streams through the whole decoder |
| `filters::efe_nonlinear` | `filters/EFEnonlinear/EFEnonlinear.py`: `decompress`, `LumaAidedAdaptiveNonlinearFilter_apply`, `apply_OnoffSwitch`, `downsample`, `deinteger` | ported: per-tile two-layer 1x1 network (1 and 4 tiles checked), U-only / V-only / both, on/off masks with values 0, 1, 2 and block sizes 112 / 96, 4:4:4 / 4:2:2 / 4:2:0 sources, odd sizes. Same `Unsupported` formats as EFE linear | `tests/filters_efe_ref.rs`: **bit-identical** to the reference on all 16 streams that enable it (max abs error 0); `tests/decode_ref.rs` as above |

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
| base profile + EFE linear (stock encoder), and 4 streams with EFE linear + non-linear forced to filter lengths 2..4 / every region split | < 5e-4 | < 5e-4 | < 3e-3 (before filters) | 47 .. 66, all by 1 (the two filters add at most 9.2e-5 of their own) |
| 2096x1400 image 00001 with EFE linear + non-linear (4 tiles); 277x201 crop (odd size) | < 5e-4 | < 5e-4 | < 3e-3 | 384 of 8,803,200; 5 of 167,031, all by 1 |

Entries written `< x` are the bounds `tests/decode_ref.rs` asserts, not individually recorded
maxima; the other numbers were read off `examples/dbg_recon`.

The differences come from convolution summation order (PyTorch/oneDNN vs this crate's fixed
order); they land on 8-bit rounding boundaries in about 0.005 % of samples.

Not started (decoder): two of the four post-filters (eICCI, LEF: in progress; see the table for
EFE linear / non-linear),
chroma-subsampled and 10-bit output, custom colour transform, UDI, progressive (`num_decode_chs`)
decode. Not started (everything else): the whole encoder
above the entropy coder (analysis transforms, hyper-encoder, quantisation/RDO tools, bitrate
matching, header/stream assembly), CI.

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
  6.0 s; unfused without SIMD 1.42 s; unfused `Wasm128` 0.35 s (wasmtime: 0.51 s). Native
  AVX-512, one thread: 0.093 s. The `Wasm128` micro-kernel keeps 4 output positions x 8 channels
  in registers (3 / 4 / 6 / 8 / 12 positions measured: 428 / 352 / 385 / 457 / 457 ms).

## Reference behaviour that differs from its own configuration

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

## Reference constants that are not in the bitstream

- EFE linear: the eight region splits (`cands`, with `0.33` / `0.66` fractions and borders
  rounded to multiples of 32 phase samples by Python's round-half-even), the DCT-IF 4-tap
  kernels, the weight scale (`(code - 32767) / 2^11`). EFE non-linear: 8 luma bins per tile.
  `filters::efe_linear` / `efe_nonlinear` carry copies.

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
