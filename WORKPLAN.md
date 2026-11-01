# zenjpegai work queue

The durable worklist — read after every compaction; mark items `[x]` in place when landed
(with the commit), add newly discovered gaps. Order is current priority; reorder freely when
a dependency or a discovery demands it. Conventions: "dead path" = upstream code that crashes
or is unreachable in the shipped config — all of it is in scope to port; "no oracle" =
upstream cannot produce reference output, so verify by unit test / self-consistency /
construction, and say so in `PORTING.md`.

## Landed

- [x] **feature_clipping** (`clipping_mode`): `weights` nested-dict flatten → `clip_thres.*`
  lists, `model::load::clip_bounds`/`clip_channels`, `forward_clip` on analysis prim/sec +
  hyper_encoder (E1B..E5B / E1H..E5H / HE1..HE5), `EncodeParams::clipping` + `--clipping`,
  `scripts/ref_vectors/inject_clip_thres.py`. Commits `06c1d94`, `68ae33e`, `83f5010`.
- [x] **qp_map_type 0/1/2 generators** (upstream crashes: `img` NameError, unset config):
  `QualityMap::from_variance`/`from_roi_rects`/`from_noise` + `--qp-map
  variance|noise:b,min,max|roi:x,y,w,h;...`. Commits `0144ba7`, `2c66628`, `55e5566`, `8e5858b`.

## Next up

- [ ] **UV-beta search** (`bitrate_matcher.py::find_UV_beta_with_hyperopt`, upstream crashes on
  `import hyperopt` when `independent_beta_UV = 0`; shipped config uses `1` = coupled UV=Y
  beta, already matched). Port: `EncodeParams` knob → after `rate_match` picks `beta_Y`, sweep
  `beta_UV` in `[min(beta_min_mult·βY, 0.1·βY), max(beta_max_mult·βY, 3·βY)]`
  (`beta_min/max_mult` defaults need reading); per trial `encode_from_latents([beta_Y,beta_uv])`
  → stream bpp → `Decoder::decode` → reco planes → `distortion = Σ_c MSE(reco_c, orig_c)·
  [0.8,0.1,0.1]·5000` → `loss = 10·distortion·(βY+2048)/2048·base_beta + bpp`; keep argmin with
  `bits_mismatch < 0.09`. `find_loss` in `bitrate_matcher.py:307–390` is the spec. Deterministic
  grid/golden search replaces hyperopt `fmin`/`atpe`. No oracle → self-consistency.
- [ ] **chroma / z RDLR** (`cfg/tools/RDLR.json`: `numIteYLuma=8` shipped, `numIteYChroma`,
  `numIteZLuma`, `numIteZChroma = 0`; `rdlr_comp_base.py::process_z/process_y` and
  `rdlr.py::RDLRCompUV/RDLRCompZ/optimize_z/rate_point_z/rate_point_uv` exist but are
  unexercised). `process_z`: `Substract` filter + Adagrad (lr ×0.9/iter) optimizing `z_hat`
  toward the `encode_z + encode_y` total-bits loss, `numIteZ` iters (default 14). Chroma
  `process_y` needs a chroma distortion term, not the luma MS-SSIM. Large port — start with
  `process_z`.
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
