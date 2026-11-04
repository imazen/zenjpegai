# zenjpegai work queue

The durable worklist — read after every compaction; mark items `[x]` in place when landed
(with the commit), add newly discovered gaps. Order is current priority; reorder freely when
a dependency or a discovery demands it. Conventions: "dead path" = upstream code that crashes
or is unreachable in the shipped config — all of it is in scope to port; "no oracle" =
upstream cannot produce reference output, so verify by unit test / self-consistency /
construction, and say so in `PORTING.md`.

## Landed

- [x] **UV-beta search** (`independent_beta_UV = 0`, upstream crashes on `import
  hyperopt`): `EncodeParams::uv_beta_search` + `--uv-beta-search` (`--bpp` only);
  `Encoder::find_uv_beta` sweeps ≤9 ints in `[min(βY/32,0.1·βY), max(5·βY,3·βY)]`,
  decode-scored `find_loss` (`10·distortion·(βY+2048)/2048·base_beta + bpp`,
  `bits_mismatch<0.09`). `Encoder::models_dir` retained for the trial `Decoder`.
  Commit `b0182c1`.
- [x] **feature_clipping** (`clipping_mode`): `weights` nested-dict flatten → `clip_thres.*`
  lists, `model::load::clip_bounds`/`clip_channels`, `forward_clip` on analysis prim/sec +
  hyper_encoder (E1B..E5B / E1H..E5H / HE1..HE5), `EncodeParams::clipping` + `--clipping`,
  `scripts/ref_vectors/inject_clip_thres.py`. Commits `06c1d94`, `68ae33e`, `83f5010`.
- [x] **qp_map_type 0/1/2 generators** (upstream crashes: `img` NameError, unset config):
  `QualityMap::from_variance`/`from_roi_rects`/`from_noise` + `--qp-map
  variance|noise:b,min,max|roi:x,y,w,h;...`. Commits `0144ba7`, `2c66628`, `55e5566`, `8e5858b`.

## Next up

- [~] **chroma / z RDLR** — chroma `process_y` DONE (c2fe1ec, `--num-ite-uv`); `process_z` pending (`cfg/tools/RDLR.json`: `numIteYLuma=8` shipped, `numIteYChroma`,
  `numIteZLuma`, `numIteZChroma = 0`; set them >0 in a reference cfg → the path RUNS, so there
  IS an oracle — dump + compare). Three sub-parts:
    1. **chroma `process_y`** (`rdlr_comp_UV.py::RDLRCompUV.rdlr_y_iteration` /
       `get_current_rate_point`): chroma `optimize_y` variant. Loss = mean of per-channel
       PSNR+rate direction projections (psnrU/psnrV vs rate_log_diff, `normal_vec` from
       `current_bpp_curve_slopes`) — PSNR, NOT luma MS-SSIM. `sup_info` = refined luma
       `y_hat[::2,::2]` concat'd into the chroma decoder input — OUR
       `SynthesisSecondary::forward(eng, luma_sup_decimated, chroma_yhat, h, w)` already
       takes it (`first_stage.conv1` = 256→48 = 160+96). `calc_additional_rate`
       (`get_luma_tile_rate`) adds the luma's rate into `expected_joined_rate`. Order: luma
       `process_y` first, then chroma with `sup_info`/`calc_additional_rate`
       (`rdlr_tool.py:256-283`).
       **`rate_point_uv` + `RatePointUV` + `psnr_plane` LANDED (`5382f2a`)** — the
       evaluator. Remaining pieces, mirroring `backward`/`optimize_y`:
       - `loss_upstream_uv`: `loss = -(nU·cU + nV·cV)/2`, `nP=[-sin,cos]`,
         `cP=[rate_log_diff, psnrP_diff]`, `rate_log_diff=log(rate_UV+Y_part)-log(ref)`.
         Gradients: `d(loss)/d(log_joined)=(sin_U+sin_V)/2`, `d(loss)/d(psnrP)=-cos_P/2`.
       - `backward_uv`: `d(resi)` via `rate_vjp` (GM term) + `d(rec_u)/d(rec_v)` via
         `psnr_vjp` parametrized on plane → `SynthesisSecondary::backward` through
         `cat(sup, y_hat)` (sup detached → grad on the 96 chroma part only; needs a
         secondary `forward_tape`/`backward` — check `synthesis.rs` Secondary tape) →
         `ctx.compress_backward` → `d(y)` + direct `+1` resi term.
       - `optimize_uv` driver: Adagrad on chroma `y` mirroring `optimize_y`, with
         `sup_info`/`calc_additional_rate`/`tile_rate_ref` refs.
       - `rdlr` integration (`encoder/mod.rs`/`rdlr.rs::rdlr`): luma `process_y` first →
         `sup_info` + `calc_additional_rate` → chroma `process_y`.
       RESOLVED geometry/spec (reference-verified):
       - `sup` = luma `y_hat` at FULL latent res (h/16) — the decoder (`ccs_sgmm_tool.py:470`,
         `supp_info_uv = tiles_synthesis['y_hat']`, crop only) is ground truth. The encode-side
         `sup_info = y_hat[:,:,::2,::2]` in `rdlr_tool.py` is a DEAD-CODE BUG (would fail the
         `cat` — chroma `y_hat` is at h/16, equal to luma's; `header.rs latent_size(1)` =
         `component_size(1)/8` = `(h/2)/8` = h/16). Do NOT decimate.
       - `ref_data` = `(joint_rate_ref, psnrU_ref, psnrV_ref, rate_ref_Y_part)`; element 0/3
         get `additional_rate` (=`get_luma_tile_rate` = luma `rate_point_y.bpp` at the
         refined `y`) added in `process_y`.
       - `expected_joined_rate = bpp_UV + rate_ref_Y_part`; `rate_log_diff = ln(joined)-ln(ref)`.
       - `loss = (lossU+lossV)/2`, `lossP = -dot([-sin sP, cos sP], [rate_log_diff, psnrP-P_ref])`.
         Gradients: `d/d(rate_UV) = (sin sU + sin sV)/(2·joined)`, `d/d(psnrP) = -cos sP/2`.
       - UV slopes `mean_psnrU_slopes` = {6:64.131075, 12:62.749463, 25:63.404102,
         50:58.369855, 75:44.556739, other:58.642247}; `mean_psnrV_slopes` = {6:64.006979,
         12:62.568593, 25:62.318648, 50:56.189655, 75:42.682759, other:57.553327}.
       - `rate_UV` = `(z_bits + GM residual bits)/luma_pixels` = `rate_point_uv.bpp`.
       - `original` for UV = 2ch source chroma planes at CHROMA res; `synthesis_uv` called
         with (ch, cw) = chroma res → rec 2ch at chroma res for `psnr_plane`.
       - `numIteYChroma` default 40 (shipped 0 → knob `EncodeParams` off-by-default);
         oracle = reference cfg with numIteYChroma>0 → dump.
    2. **`process_z`** (`rdlr_comp_base.py:77`): dead even upstream (NO callers). `Substract`
       filter (per-position `additive2` + scalar `additive`) + Adagrad(lr*0.9/iter) + STE
       `z_hat = z_new + clamp(round(z_new)-z_new, -1, 1)` optimizing `z_hat` toward
       `encode_z + encode_y` total bits, `numIteZ` iters (default 14), `nochange==7`.
       **DONE (both components, `a3c5e1b`/`a69e9ee`/`5374daa`):**
       `HyperDecoder::forward_tape`/`backward`/`forward_f32` + `HyperTape` +
       `load_hyper_decoder` (finite-diff verified), `mcm::upshuffle_psi_backward`,
       `lh::z_bits_vjp`, `rate_point_z`/`backward_z`/`optimize_z`/`ste_z`/`quant_z`/
       `ZRatePoint`, `compress_backward` → `d_psi` (MCM psi-part plumbing for luma),
       `num_ite_z` + `--num-ite-z` + `final_run` gating + `component_pass` re-deriving
       `scales`/`mask`/`q` from the optimized `z_hat`. Verified end-to-end (fires on
       both `z_Y`/`z_UV`, real `d_z`, decodes cleanly; identical output = legit
       no-improvement when the additive can't flip `z_hat` <0.5 in `num_ite_z`).
    3. **`optimize_z`/`rate_point_z`/`rate_point_uv`** in `rdlr.py` — the point evaluators
       the above two use.
- [x] **`colour_transform_idx = 2`** — RESOLVED as correctly-rejected: the upstream path is
  irrecoverably self-inconsistent (`pre`/`post_processing` both apply only the first row of
  the *inverse* matrix to all 3 planes AND crash unpatched on `convert_range_(0,1)`), so no
  trustworthy oracle exists — even the reference produces garbage/three-identical-planes. Our
  `ColourTransform::Custom` parses the 12-byte matrix/offset bits exactly, then
  `output.rs:379` returns `Err(Unsupported("user-defined colour transform"))` — the honest
  port (rejecting a stream the reference itself cannot decode).
- [ ] **More `Precision::Approximate` knobs**: Winograd F(2,3) for 3x3 convs, early-exit RDLR,
  f16-weight kernels (f16 `.pth` load already upcasts; a real f16 kernel is new). All opt-in,
  all parity-breaking — never under `Exact`.

## Also queued

- [ ] Niche coverage vectors: per-tile-different eICCI selections, single-chroma tiles,
  10-bit eICCI/LEF (needs reference decode support check first).
- [ ] GPU work (separate workspace): shader-f16 (B3), GPU pool ledger — parity-breaking.
- [ ] Sweep `PORTING.md` after every landing: row update + what remains missing.

## Continuation checkpoint (post-`process_z`)

`process_z` is DONE both components (`5374daa`): `compress_backward`→`d_psi`
(MCM psi-part plumbing for luma), `rate_point_z`/`backward_z`/`optimize_z`
`ccs`-parametric (`compress`/`compress_backward`+`d_psi` luma; `compress_context_free`
+`upshuffle_psi_backward` chroma), `num_ite_z`+`--num-ite-z`+`final_run` gating+
`component_pass` re-derives `scales`/`mask`/`q` from optimized `z_hat`. Verified
end-to-end (fires, real `d_z` ~36, valid stream decodes; identical output = legit
no-improvement — additive can't flip `z_hat` <0.5 in `num_ite_z` iters).
`colour_transform_idx=2` RESOLVED upstream-broken→rejected (no oracle exists).
ALL RDLR dead-paths now ported (`process_y` luma+chroma, `process_z` luma+chroma,
`feature_clipping`, QMAP 0/1/2, UV-beta, `Precision::Approximate` early-exit).

- [x] **`minimum_tile_size` / `_adjust_boundary_tiles`** — RESOLVED. The
  `_adjust_boundary_tiles` machinery is ported (`filters/icci.rs`,
  `MINIMUM_TILE_SIZE = 176`, oracle-verified `matches_reference_layout_with_adjusted_boundary`)
  — it's the eICCI `numSamplesPerTile` tiling path.
- [x] **Per-tile RDLR** (`3f2e3f3`, `1da69c9`) — `setup_tiles_enc` +
  `_init_image_tiles_with_overlap` + `init_latent_tiles` +
  `_get_core_of_overlapping_tile` + `_adjust_boundary_tiles` ported as
  `tools::tiles::rdlr_tiles` (`numSamplesPerLumaTile`=1M → 1024-tile,
  `numSamplesPerChromaTile`=0.5M → 720-tile, overlap 64, alignment 16,
  `minimum_tile_size`=240 when MS-SSIM loss mixed). `optimize_y`/`optimize_uv`
  take `optimize_area` — the Adagrad leaf covers only `core_lat_rel` (overlap
  frozen, matching upstream's `lat_tile_optimized_part`). `luma`/`chroma` loop
  the tile grid; `get_luma_tile_rate`/`calc_additional_rate` ported as
  `luma_tile_rate` (rate-only) → `ComponentPass::y_rate_parts`. `compress_regions`
  derives grids from the passed tile dims for `regions == None`. Gate raised to
  `numSamples`=30M. Upstream's multi-tile path is crash-dead (whole-picture
  `scale_log`/`psi`/`mask2` vs tile `y`; `ref_data[3]` IndexErrors even
  single-tile UV) — intended tile-local semantics implemented; no oracle exists.
  Verified end-to-end on 2048×1536 (multi-tile luma+chroma → valid stream).
Next high-value (unstarted):
- **`d_psi` — RESOLVED (unverifiable by finite-diff, transitively verified).**
  `d(psi_s)` extraction in `compress_backward` shares the `joined`→`d(t)`/`d_y`
  slice the reference oracle verifies. A `d(out)/d(psi)` finite-diff CANNOT
  isolate it: every `Compressed` output (`y_hat`, `resi_dq`, `residual_q`) is
  quantized/discrete, so the forward diff measures the true step function while
  the backward's straight-through estimator is a smooth surrogate — they
  legitimately differ (the point of STE). The continuous leaf (`means`/`mu`)
  isn't exposed in `Compressed`. No `psi_grad` oracle upstream. The `#[ignore]`d
  test in `mcm.rs` documents this.
- **Winograd F(2,3) `Precision::Approximate` conv** — **LANDED** (`conv.rs`):
  `wino_filter`/`wino_input`/`wino_output` transforms + `wino_loop`/`wino_block`/
  `wino_strip`/`run_wino` + `PackedConv.w16` (16 `G g G^T` planes packed at
  `new` for dense 3x3 stride-1 convs); `forward` routes to `forward_winograd`
  on `Precision::Approximate`. Correct + tier-identical (fixed GEMM order,
  `winograd_conv_matches_exact` bounds it ≤2e-3 of the exact kernel on every
  tier). **Measured: ~15–18% SLOWER than the direct kernel** (615–645 vs
  531 ms, 2048×1536 V3) — the per-strip transforms (`v16` gather + `wino_input`),
  `m16` write-back for the inverse transform, and `xr` pointer setup eat the
  2.25x mult reduction at these layer shapes; the direct `tap_loop_n` kernel is
  too well-tuned (zero-copy interior reads, 24-deep unrolled accumulators, no
  scratch traffic). Landed opt-in + correct; the honest path to a real wino
  win is **F(4,3)** (36 planes amortising transforms over 16 outputs not 4) or
  a batched `icb`-major driver — F(2,3) as implemented stays behind
  `Approximate` for experiment use. `PackedIntConv` (`int_conv.rs`) is
  int8-only (hyper-scale decoder) — not a float path.
  **Also measured `--precision approximate` vs `exact` before wino:** 540 vs
  531 ms — the accumulator-width knob alone does ~nothing.
- **Phase-decomposed stride-2 conv** — **LANDED opt-in** (`forward_s2`,
  `ZENJPEGAI_CONV_S2_PHASE`): repacks `x` into the 4 (row-parity, col-parity)
  planes of a `[icb][4*hp][wp]` buffer; the 3x3 s2 conv's 9 taps become
  unit-stride reads across 4 planes (Pan & Chen ASP-DAC'21 / DWM-style
  decomposition) with *identical* accumulation order → bit-identical on every
  `Precision`/tier (`conv2d_s2_phase_is_bit_identical`). **Measured ~8 %
  SLOWER end-to-end** (593 vs 549 ms) — see the profile note below. Kept
  opt-in as the substrate for a future phase-GEMM (the 4 planes are
  GEMM-shaped); the direct `tap_loop` strided read stays the default.
  Decode-side convs are already optimally covered: `PackedConvTranspose`
  runs the same phase trick on the output side (per output-column parity),
  `PackedDepthwise`/`PackedIntConv` own their paths, no `reference::`
  fallback runs on the decode forward path.
- **Conv-kernel bottleneck — MEASURED (bench_s2conv example, 128→128 s2
  3x3 conv on 512×352, i270/Core Ultra AVX2):** 158 ms/conv single-threaded
  vs a ~5–7 ms FMA port floor → **~3 % FMA utilization: memory-LATENCY-
  bound, not FMA-bound** (my earlier guess was wrong). 24 threads give only
  13.4× (→11.8 ms, ~40 % util) — latency-bound with a bandwidth ceiling,
  not compute-bound; ~2× off peak even threaded. Root cause: the conv
  re-reads the whole input per output-channel block (ocb) — a ~92 MB working
  set per ocb — so every splat-load misses L2/LLC (~17 cycles/step). **The
  real fix is ocb-tiling** (N-block the output channels: accumulate
  `acc[OC_B][B]` over several output blocks at once, so one input read feeds
  `OC_B` FMAs — input traffic ÷OC_B, the same miss-avoidance GEMM
  N-blocking does). Neither Winograd nor phase-decompose helps a
  latency-bound kernel — only fewer misses (tiling) or fewer FLOPs
  (F(4,3)/implicit-GEMM) or GPU do. The `phase` path's repack only added
  traffic.
- f16-weight kernels, niche coverage vectors (per-tile eICCI, single-chroma
  tiles, 10-bit LEF/eICCI), GPU (separate workspace).
