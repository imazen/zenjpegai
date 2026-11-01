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

- [ ] **chroma / z RDLR** (`cfg/tools/RDLR.json`: `numIteYLuma=8` shipped, `numIteYChroma`,
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
         `sup_info = y_hat[::2,::2]` + `calc_additional_rate` → chroma `process_y`
    2. **`process_z`** (`rdlr_comp_base.py:77`): dead even upstream (NO callers). `Substract`
       filter (per-position `additive2` + scalar `additive`) + Adagrad(lr*0.9) + STE
       `z_hat = z_new + clamp(round(z_new)-z_new, -1, 1)` optimizing `z_hat` toward
       `encode_z + encode_y` total bits (`ECLibLH`), `numIteZ` iters (default 14),
       early-stop `nochange==7`. Needs the z-entropy VJP + the `z_hat→psi→scale_log→residual`
       chain — the hard part (like `optimize_y`'s backward but on z).
    3. **`optimize_z`/`rate_point_z`/`rate_point_uv`** in `rdlr.py` — the point evaluators
       the above two use.
- [ ] **`colour_transform_idx = 2`** (user-defined transform; decoder side self-inconsistent
  upstream — needs a decision on what the "correct" semantics are, then port).
- [ ] **More `Precision::Approximate` knobs**: Winograd F(2,3) for 3x3 convs, early-exit RDLR,
  f16-weight kernels (f16 `.pth` load already upcasts; a real f16 kernel is new). All opt-in,
  all parity-breaking — never under `Exact`.

## Also queued

- [ ] Niche coverage vectors: per-tile-different eICCI selections, single-chroma tiles,
  10-bit eICCI/LEF (needs reference decode support check first).
- [ ] GPU work (separate workspace): shader-f16 (B3), GPU pool ledger — parity-breaking.
- [ ] Sweep `PORTING.md` after every landing: row update + what remains missing.
