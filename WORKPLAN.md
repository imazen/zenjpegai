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
  — it's the eICCI `numSamplesPerTile` tiling path. For RDLR it's dormant by
  design: `numSamples` = `numSamplesPerLumaTile` = 4.2M forces a single-tile
  grid → nothing to adjust + whole-picture `ms_ssim` == per-tile-optimized-part.
  **Remaining structural gap (real, large):** per-tile RDLR for pictures
  >4.2MP — the reference's `numSamplesPerLumaTile`=1M/`numSamplesPerChromaTile`=
  0.5M makes `image_tiles`/`latent_tiles` multi-tile, `rdlr_y_iteration` per
  `lat_tile`/`im_tile`/`optimized_part` + `ms_ssim` per tile (non-additive at
  `ratio != 0`). Our `numSamples`=4.2M gate (vs reference 30M) is the deliberate
  bound making this unreachable — a resource/design choice, not an unported
  feature per se. A full per-tile RDLR port is a large restructure.
Next high-value (unstarted):
- **`d_psi` finite-diff** — `d(psi_s)` extraction in `compress_backward` is a
  subset of the oracle-verified `d_y` path but has NO reference oracle
  (upstream dumps no `psi_grad`). A `d(Σ forward-out)/d(psi)` finite-diff in
  `mcm.rs` (needs a real MCM fixture — `load_common`/`CommonModel.context`)
  would close the verification gap.
- **Winograd F(2,3) `Precision::Approximate` conv** — convs use `packed`/
  `depthwise` SIMD kernels (NOT `dot8`); a Winograd 3x3 kernel is the big
  remaining perf knob (parity-breaking, opt-in). ~250 lines + error-bounded test.
- f16-weight kernels, niche coverage vectors (per-tile eICCI, single-chroma
  tiles, 10-bit LEF/eICCI), GPU (separate workspace).
