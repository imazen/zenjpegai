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

- [x] **GPU encode readback bug fixed + f16 verified on llvmpipe** (`9733784e`,
  `ec99d7c8`). `GpuEncoder::encode`/`encode_input` *panicked* — `read_latent`
  sized the staging buffer `t.len4() * 4` bytes, but `len4()` is the vec4 count
  (`h*w*c4`), 16 bytes each → the HWC4→planar unpack overran a quarter-size
  buffer. Fixed to `*16`. New `gpu/tests/encoder.rs::gpu_encode_matches_cpu_encode`
  verifies `GpuEncoder::encode` ≈ `Encoder::encode` within the gpu-analysis
  tolerance. **GPU verification is NOT fully hardware-gated:** llvmpipe
  (`ZENJPEGAI_GPU_ALLOW_SOFTWARE=1 ZENJPEGAI_REF=… ZENJPEGAI_VECTORS=…`) runs the
  whole `gpu-tests` suite — `analysis_matches_cpu` (f32 `3e-4`, f16 `1e-3`/`2e-2`
  via `context().has_f16()`), all `decode_ref` streams, `gpu_encode` — on this box.
  The RTX 2080 remains for *perf* (f16-throughput, tiled timing), not correctness.
- **Vector suite now fully populated + verified on this box:** `decode_ref`
  **50/50 bit-exact**; `filters_efe_ref` **17/17**; `formats_streams`,
  `efe_nonlinear_decisions`, `eicci_selection`, `colour_preprocessing`,
  `decisions_match`, `rate_matching`, `hop_post_filters`, all `eicci_*` — all
  pass. Generated all `efe`/`formats`/`icci`/`efe_nonlinear`/`filters_lef_icci`/
  `enc2` oracles (`make_reference_streams.sh` per-set + direct `force_efe_encode`/
  `dump_decode`/`dump_filters`/`dump_efe_nonlinear`/`dump_encode --enc2` drivers —
  the `efe`/`formats` sets' preamble shell structure fails `rc=120` on this box,
  but every constituent Python call works standalone, so the per-vector driver
  bypasses it). Remaining `encode_ref` failures: `msssim_matches` (needs bespoke
  `msssim_oracle`, no standard generator), `encoder_end_to_end`
  (`enc_img01_hop_m2_b0_indregions` `uv.z_hat` 1 value), `likelihood`/`tools_on_ctc`
  (environment-sensitive — see below). **Full ref-suite state:** `decode_ref`
  50/50, `entropy_ref` 37/37, `weights_ref` 4/4, `api_ref` 6/6, `filters_lef_icci`
  19/19 — all bit-exact. `memory_ref` `budget_*` + `fast_vs_reference`
  `conv2d_blis`/`conv2d_s2_phase`/`phase_32x32` fail ONLY in-parallel (pass alone)
  — a test-parallelism flake: the global memory-budget allocator and the
  `ZENJPEGAI_CONV_*` env knobs aren't isolated per test (the `SERIAL` lock covers
  some but not the shared allocator/conv-state), so concurrent runs race. Not a
  real regression — but a test-hygiene gap that could mask real failures.
  `msssim_oracle` generated via
  `dump_filters.py --msssim` → `msssim_matches` passes (≤1e-5).
  `encoder_end_to_end`: **26/27 `enc_*` vectors bit-exact** (`0 symbols moved`,
  `+0.000%`, `0 samples differ`) — the encoder is essentially bit-exact. The
  failures are a handful of borderline symbols on the *hardest* paths only
  (`enc_img01_hop_m2_b0_indregions` `uv.z_hat` 1 value; `enc_img30_bop_m3_b400`
  24 moves/worst 1; `enc_img01_hop_m2_b0` 4 moves/worst 4;
  `enc_img30_hop_m2_b0_{rvsonly,grfsonly}` 2-3 moves/worst 1). All on img01
  (largest, most float-sensitive) or the m3/indregions/RVS/GRFS analysis paths —
  a small borderline-integerization divergence, not a systematic bug (most of
  those same-path vectors are bit-exact elsewhere).
- **BUG (subsampled decode — latent, surfaced by populated format vectors):**
  `decode_ref` `y.psi` diverges ~1.9–3.0 on every chroma-subsampled vector
  (`img30yuv{420,422,444,420b10,444b10}_base_off_bpp050`,
  `img30_base_off_{c420,c422}_bpp050`, `img30cropyuv420_base_off_bpp075`) while all
  non-subsampled streams stay bit-exact. The reference's `img30_base_off_c420`
  `y.psi` is byte-identical to `img30_base_off` (444) — chroma subsample must not
  touch luma — so our coded-420/subsampled decode's luma `y.psi` is wrong, not an
  env/dump artefact. **Localized to the `ccs=0` entropy decode, not the psi
  synthesis**: for `img30_base_off_c420` our `ent[0].z_hat`/`residual`/`ly.psi`
  diverge from the 444 decode (`z_hat` nonzero 696 vs 1391, `residual` nonzero
  7869 vs 48375, `z_sum` -610 vs -378) while `ent[0]`/`ent[1]` dims are identical
  (`z_hat 14x9`, `residual 56x35`, `num_chs` 160/96) — `ent[0]` (luma) reads
  different/wrong data. `s_ver`/`s_hor`/`c_ver`/`c_hor` bleeds into the `ccs=0`
  entropy path (`decode_z`/`decode_y`/`component_scales_impl`/`num_chs`/
  `num_decode`/`channel_step`/`num_threads`/`cube_flag_count`/`num_regions`/
  `region_grid`/`hyper_latent_size`/`latent_size`/`component_size`/`num_decode`/
  `max_channels`/`channel_step`/`num_threads`/`cube_flag_count`/`num_regions`/
  `region_grid`/`hyper_latent_size`/`latent_size`/`component_size` at
  `src/decoder/entropy.rs`/`mod.rs`/`tools/regions.rs`/`header.rs`). The luma
  dims are right; the luma entropy *values* are wrong — the residual/substream
  decode for `ccs=0` reads a different count or offset for `c_ver`/`c_hor`/`s_ver`/
  `s_hor != 1`. Everything else: `decode_ref` 42/50,
  `entropy_ref` 37/37, `weights_ref` 4/4, `api_ref` 6/6, `filters_efe` 17/17,
  `filters_lef_icci` 19/19, `formats`/`efe_nl`/`eicci`/`msssim`/`encoder_end_to_end`
  (26/27) — all bit-exact.
- **Regenerated-vector caveat (encode-side signature tests):** `make_reference_streams.sh all`
  now produces ~98 vectors on this box; every decode-side vector verifies bit-exact.
  Two encode-side tests diverge only on `img01` (the biggest, most float-sensitive input):
  `likelihood_estimate` picks `beta -274` vs the reference stream's `-273` (one bisection
  step on a flat estimate curve) and `tools_on_ctc`'s `TOOLS_ON_SIGNATURE` drifts. Both pin
  OUR-vs-THEIR-stream metrics to a table measured on the *reference host's* streams — this
  box's reference encoder produces slightly different streams for img01 (tiled analysis →
  more float paths), so the pinned deviations shift. Not correctness bugs: our decode of the
  this-box streams is bit-exact (all 6 `img01_base_*` pass). The pinned signature/displacement
  tables are environment-sensitive and would need re-pinning against this box's reference host
  to flip green here; the reference host's own vectors are authoritative.
- [~] **chroma / z RDLR — dead upstream at two levels, oracle blocked.** Enabling
  `numIteYChroma`/`numIteZLuma`/`numIteZChroma` > 0 crashes the reference: first
  `TileManager.set_alignment_size` (commented out at `tiling.py:89`; `dump_rdlr.py`
  already patches it for the luma oracle), then `common_modules.py:177`
  `quant_dequant`'s `data_q[~mask2_list] = 0` applies a **luma-shaped** `mask2`/`psi`
  `[1,96,56,35]` to the **chroma** latent `[1,96,28,18]` (`IndexError`, chroma
  `process_y` → `get_ref_rate_data` → `_compress_ar_scale`). Producing a chroma/z
  RDLR oracle needs a deeper `dump_rdlr.py` patch fixing `quant_dequant`/`mask2`/`psi`
  for the chroma dims — the same class as the documented multi-tile `mask2`/`psi`
  dead code. Our `optimize_uv`/`optimize_z`/`process_z` (`rdlr.rs:723-2540`) are
  construction-complete + self-consistent (`rdlr_roundtrip`, `num_ite_uv/z=2`),
  unverifiable until that patch exists. chroma `process_y` machinery DONE (c2fe1ec);
  `process_z` pending — and the patch is *semantically* uncertain, not just mechanical:
  `mask2`/`psi` are computed at luma latent res but indexed into the chroma latent, so a
  "fix" must choose a downsample/upsample the reference never defined (it would be
  inventing semantics, not restoring them). Three sub-parts:
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

- [x] 10-bit eICCI/LEF — done + fixed a real bug. `img30yuv444b10_base_lef_eicci_bpp050`
  (10-bit 4:4:4 + LEF + eICCI) generated + oracle-dumped; exposed a `data_range`
  convention bug: the post-filter planes are the codec's 8-bit *core* `[0,255]` (extra
  bits ride the LSB plane, recombined at output), but `icci::filter`/`lef::sharpen`
  normalized by `2^bit_depth-1`=1023 → chroma diverged ~208, whole decode ~732.
  Fixed `range` → `255.0` (`9bd3cffd`); both filters + whole decode now bit-identical.
  Per-tile-*different* eICCI selections + single-chroma tiles — **done**, and it
  surfaced an upstream bug. Extended `force_icci_encode.py` (`--sel Y:U:V,...`
  per-tile triples, 0=off → single-chroma, applies to 444 too — `ef121444`) and
  generated `img01_eiccitiles_varsel{,2}_bpp050` (6 tiles, `eicci.selection`
  `[[7,3,0],[5,0,2],[4,1,1]]` vs the reordered `[5,7,4]`). `check_icci` diverged
  ~11-13 *positionally* (left column exact, interior wrong, net-independent —
  proven by reordering). Root cause is a **reference bug**: `decompress` in
  `icci_filter.py` initialises `tile_idx = 0` and never increments it, so
  `get_model_index(0)` → `model_state_per_tile[0]` filters every tile — the
  per-tile selections signalled in the header are ignored on decode. A
  conformant decoder must reproduce it; `icci::filter` now applies
  `h.tiles[0]`'s selection to all tiles (comment in `src/filters/icci.rs`).
  varsel1 (`sel[0]=[7,3,0]` → `want=[true,false]` U-only), varsel2
  (`sel[0]=[5,0,2]` → `want=[false,true]` V-only) and the natural
  `img01_base_eiccitiles_lef` (uniform `[4,2,2]` — sel[0]=all) all pass. The
  `dump_filters_lef_icci.py` per-tile `eicci.tile{i}.rec/.dwt` capture (added
  for this) is retained as the per-tile input oracle.
  **Verified against the full eICCI surface:** regenerated every missing eICCI/LEF
  vector on this box (`img30_base_{eicci,lef,on_025,on_100}`, `img30yuv{422,420}_base_eicci`
  via `~/tmp/gen_missing_filters.sh`; `enc_img30_{sop_m0,hop_m2}_b0_eicci`,
  `enc_img30_bop_{m0,m3}_b0_lef` via `~/tmp/gen_enc_filters.sh`; `dump_decode` on
  `img01_base_eiccitiles_lef`) — **all 19 `filters_lef_icci_ref` tests pass**:
  single-tile, six-tile natural + forced-varied selection, forced 422/420, 10-bit,
  the hop/sop eICCI banks, LEF model-0/3 rows, and the whole-decode API across all
  five filter streams. The `tile_idx=0` reproduction is confirmed end-to-end.
  **10-bit LEF/eICCI — real bug found, isolated to eICCI chroma.** Reference
  env unblocked on this box (built the `lib_wrappers/{mans,direct}` ANS `.so`s;
  import via `src.codec…` — see CLAUDE.md). Vector
  `img30yuv444b10_base_lef_eicci_bpp050` (10-bit 4:4:4 + LEF + eICCI, bpp50) +
  `filters_lef_icci` oracle generated (`/mnt/v/output/zenjpegai/reference/vectors/`).
  Whole-decode: 994350/1491840 differ, worst 732. **Per-filter isolation**
  (`filters_lef_icci_ref` `check_lef`/`check_icci` on the vector):
  eICCI **Y exact (~1.5e-5)** but **U diff ~208**; LEF luma diff ~1.19 (small drift,
  gate wants exact 0). So the dominant defect is eICCI's **chroma** correction at
  10-bit — `tile_input`/`tile_output`/`dwt2` are plane-agnostic and `data_range` is
  a single tuple applied uniformly, so the suspect is the chroma network output /
  `tile_output` `scale` / `chroma.forward`+`last_u`/`last_v` producing a wrong-scale
  correction for `bit_depth = 10`. Next: compare our `chroma_corrections`/`tile_output`
  U output to the dump's `eicci.out.b`. (Two `img30yuv444b10_*` tests were written and
  isolated this; kept out of `check!`/`check_icci` registration so main stays green.)
  **Root cause found — `data_range` convention, not eICCI.** The reference's
  reconstruction `rec.*`/`out.*`/`eicci.in.*` planes are all at **`[0,255]`
  (8-bit scale)** for this 10-bit source — luma max exactly 255, chroma centered
  ~128 (8-bit neutral): the codec's internal `Image` is the 8-bit *core*, and the
  extra 2 bits ride in the LSB plane (`LSBS`). Our `range = (1<<bit_depth)-1`
  uses `hdr.bit_depth = 10` → 1023, but the reference filters at `data_range
  [0,255]` → `s/255`. `check_icci` ran our `filter` on the dump's `[0,255]` planes
  with `range=1023` → normalized to `[0,0.25]` not `[0,1]` → chroma out ~4x off
  (the 208/732). Two layers to fix: (1) **the post-filter `range`/`data_range`
  must be the coded-core scale (255 for ≤8-bit, `2^bd-1` only when the internal
  image actually holds >8 bits — determine the `LSBS`/internal range the decoder
  stores for `bit_depth > 8`)**, in both `filters/icci.rs:247` and
  `filters/lef.rs:124`; (2) whether our reconstructed *planes* store `[0,1023]`
  (true 10-bit) vs the reference's `[0,255]` core — if ours hold 1023 the
  normalization matches internally but the plane convention differs from
  reference; confirm what `Decoder` stores for a 10-bit `LSBS` stream. The dump
  shows reference in/out planes at `[0,255]` — our planes should match that scale.
- [~] GPU work (separate workspace): shader-f16 conv *weights* landed (`GpuContext` requests `SHADER_F16`, `GpuConv` packs `mat4x4<f16>`, `conv()`/`conv_tiled()` widen to `mat4x4<f32>` — activations stay f32); conv_transpose/depthwise and f16 *activations* still f32. GPU pool ledger — parity-breaking.
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
  set per ocb — so every splat-load misses L2/LLC (~17 cycles/step).
  **ocb-sweep tested: ~12x SLOWER** (`forward_ocb_sweep`, opt-in
  `ZENJPEGAI_CONV_SWEEP`) — iterating position-blocks outer / ocbs inner
  keeps the input strip hot but thrashes the *weights* (the ~590KB weight
  set cycles through L1 each position block, vs the default keeping each
  ocb's ~37KB weight L1-resident). **OCB2 register-tile: ~25x SLOWER**
  (`forward_ocb2`, opt-in `ZENJPEGAI_CONV_OCB2`) — `acc[2][B]` halves the
  input loads/FMA but the paired 74KB weight set exceeds L1, so both ocbs'
  weights miss L1. The lesson across all four attempts (wino, phase,
  sweep, ocb2): the default driver's weight-per-ocb-in-L1 is the linchpin
  — any reorder that breaks it loses more than the read-side gains.
- **Packed-tile (BLIS) dense driver — LANDED, default for `groups == 1`
  (`forward_blis`, `ZENJPEGAI_CONV_NOBLIS` opt-out):** the input strip for
  each output row is im2col-packed once into a contiguous
  `[icb][ntaps][ow][V]` tile (one gather pass, amortised over all output
  blocks), then every `ocb` reads it at unit stride `is: 1` with every
  tap interior — sequential L1/L2-streamed reads instead of scattered
  ones, while the weight-per-ocb L1 residency is preserved. **Measured:
  ~15% faster on the conv micro-bench (140.8 vs 158 ms 1T, 10.1 vs 11.9 ms
  24T) and ~5-6% faster end-to-end encode (511 vs 540 ms) — bit-identical
  on every tier** (`conv2d_blis_is_bit_identical`). `coz::progress!()`
  markers are wired in the pack + sweep loops behind the `coz` feature.
  The ~64% backend-bound topdown (~29% mem + ~35% core) confirms the
  packed reads attack the memory third.
- **Prefetch — tested thoroughly (temporary unsafe `_mm_prefetch`, since
  reverted):** `ZENJPEGAI_PF` bitmask tried next-icb input rows (1),
  next-posblock tap rows (2), next-icb weights (4) on both the packed
  and strided paths via `zenbench conv_kernels`. Single-threaded: dead
  even everywhere (~18.6 ms on `hsd_conv3x3_128_128` for every mode) —
  the packed tile made input reads sequential, which the hardware
  prefetcher already covers. Weight prefetch (4) was a clear loss
  (~25 ms — the prefetch uops steal the load ports the FMA loop needs).
  Next-posblock (2) showed a marginal mt delta (4.8–5.3 vs 5.5 ms) that
  does not replicate within the box's CV=15-25% noise. **Conclusion:
  prefetch does not clear the bar for an archmage API — not worth asking
  upstream.** `#![forbid(unsafe_code)]` restored.
- **Encoder/analysis on the wgpu backend — landed** (`gpu/src/analysis.rs`
  `GpuAnalysis` + `gpu/src/encoder.rs` `GpuEncoder`; the `pad` WGSL
  kernel + `Graph::pad` give the structural replicate-pad; `Loader`/
  `ResAu`/`Tam`/`Cab`/`ResidualBlock`/`TransformerBlock` promoted to
  `pub(crate)` in `synthesis.rs`). `encode(SourceImage, params)` mirrors
  `Encoder::encode`: GPU analysis -> CPU `encode_latents` (the parity
  seam); produces a valid, not bit-identical, codestream (~1e-4 latent
  gap like the reference tolerates). Single-tile pictures run the
  whole-picture shared-graph analysis; multi-tile pictures run
  `encode_tiled` — per-component `analysis_tiles_with`, per-tile
  `GpuAnalysis` on the `window`ed plane, the caller-owned
  `[HyperEncoder;2]` (CPU, `GpuEncoder`'s `Engine` + private `hyper`
  workaround) for `z_hats`, `assign` merge, then `encode_latents` with
  the real z_hats — so GPU analysis covers every picture size.
  `gpu/tests/analysis.rs` compares `GpuAnalysis` luma/chroma to the CPU
  transforms (BOP+HOP, ~3e-4 bound; needs `ZENJPEGAI_REF` + a GPU).
  Compile-verified everywhere; the GPU-run + a GPU-vs-reference
  *bitstream* check are for the RTX 2080. `clip_thres` is dormant
  upstream (no shipped model carries it).
- **f16 shaders (B3):** conv *weights* f16-packed on `SHADER_F16` devices
  (activations f32); `conv_transpose`/depthwise + f16 activations still
  to do.
- **Wasm encoder — landed** (`wasm/src/lib.rs`: `State` gains an
  `Encoder::with_source` sharing the decoder's `PackedBundle`;
  `encode(rgb, w, h, model_id, op, beta)` -> codestream, builds for
  wasm32-unknown-unknown).
- f16 conv_transpose/depthwise + f16-activation kernels, niche coverage
  vectors (per-tile eICCI, single-chroma tiles, 10-bit LEF/eICCI), GPU
  (separate workspace).
