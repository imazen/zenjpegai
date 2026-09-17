#!/usr/bin/env bash
# Encode + decode a matrix of configurations with the reference software and dump the decoder's
# intermediate tensors for the parity tests.
#
#   scripts/ref_vectors/make_reference_streams.sh [SET]      SET: smoke (default) | regions | tools | filters | efe | filtertiles | qmap | formats | all
#
# Output: $OUT/<name>/{stream.bits,encoder.log,tensors.bin,manifest.txt,decoded.png,stdout.log}
# with OUT=/mnt/v/output/zenjpegai/reference/vectors. Existing streams are kept (delete the
# directory to regenerate). Nothing here is committed: the vectors are several MB each.
set -euo pipefail

REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
OUT=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SET=${1:-smoke}

cd "$REF"
# shellcheck disable=SC1091
. .venv/bin/activate
mkdir -p "$OUT"

one() { # name image bpp cfg...
  local name=$1 image=$2 bpp=$3; shift 3
  local dir="$OUT/$name"
  if [ -f "$dir/manifest.txt" ] || [ -f "$dir/fixed_decoder/manifest.txt" ]; then echo "== $name: exists"; return; fi
  mkdir -p "$dir"
  echo "== $name: encoding ($(date -u +%H:%M:%S))"
  local input="data/test/$image"; case "$image" in /*) input="$image";; esac
  nice -n 19 python -m src.reco.coders.encoder "$input" "$dir/stream.bits" \
      --set_target_bpp "$bpp" --cfg "$@" -target_device cpu > "$dir/encoder.log" 2>&1
  echo "== $name: decoding + dumping"
  # The stock reference decoder cannot decode every stream its encoder writes (quality maps
  # crash it, region streams are mis-decoded; PORTING.md). Its failure is recorded, not fatal.
  nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir" > "$dir/dump.log" 2>&1 \
      || echo "   stock reference decoder FAILED on this stream (see $dir/dump.log)"
  # Same decode with those defects patched at runtime: the oracle for such streams.
  nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir/fixed_decoder" --contiguous-masks \
      --fix-qmap-header > "$dir/dump_fixed.log" 2>&1
  { grep -hs "^MD5" "$dir/encoder.log" "$dir/stdout.log" "$dir/fixed_decoder/stdout.log" || true; } \
      | sort | uniq -c | sed 's/^/   /'
  ls -la "$dir/stream.bits" | awk '{print "   stream bytes:", $5}'
}

# Like `one`, but with the bitrate matcher off and the model / beta displacement given directly.
# (The matcher's rate-estimation backend crashes when regions have unequal sizes; fixing the
# operating point sidesteps that and makes the encode a single pass.)
one_fixed() { # name image model_id beta_disp_log cfg...
  local name=$1 image=$2 tool=$3 beta=$4; shift 4
  local dir="$OUT/$name"
  if [ -f "$dir/manifest.txt" ]; then echo "== $name: exists"; return; fi
  mkdir -p "$dir"
  echo "== $name: encoding, fixed model $tool beta_disp $beta ($(date -u +%H:%M:%S))"
  # dump_encode.py runs the normal encoder and also saves the tensors it committed to the
  # stream (enc_manifest.txt / enc_tensors.bin): the oracle for region streams, which the
  # reference decoder itself mis-decodes (see PORTING.md).
  nice -n 19 python "$HERE/dump_encode.py" "$dir" -- "data/test/$image" "$dir/stream.bits" \
      --cfg "$@" -target_device cpu -model.bitrate_matcher.enabled 0 \
      -model.bitrate_matcher.target_tool_idx "$tool" -model.bitrate_matcher.target_beta_disp_Y "$beta" \
      > "$dir/encoder.log" 2>&1
  cat "$dir/enc_stdout.log" >> "$dir/encoder.log"
  echo "== $name: decoding + dumping"
  nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir" > "$dir/dump.log" 2>&1
  nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir/fixed_decoder" --contiguous-masks \
      > "$dir/dump_fixed.log" 2>&1
  # Encoder and (patched) decoder print the MD5 of their reconstructions: one line = they agree.
  { grep -hs "^MD5" "$dir/encoder.log" "$dir/stdout.log" "$dir/fixed_decoder/stdout.log" || true; } \
      | sort | uniq -c | sed 's/^/   /'
  ls -la "$dir/stream.bits" | awk '{print "   stream bytes:", $5}'
}

IMG30=00030_TE_560x888_8bit_sRGB.png
IMG01=00001_TE_2096x1400_8bit_sRGB.png

if [ "$SET" = smoke ] || [ "$SET" = all ]; then
  for bpp in 012 025 050 075 100; do
    one "img30_base_off_bpp$bpp" $IMG30 "$((10#$bpp))" cfg/tools_off.json cfg/profiles/base.json
  done
  one img30_simple_off_bpp050 $IMG30 50 cfg/tools_off.json cfg/profiles/simple.json
  one img30_high_off_bpp050 $IMG30 50 cfg/tools_off.json cfg/profiles/high.json
  # Progressive decode oracles: the same stream decoded with a channel limit (num_decode_chs).
  P=-model.CCS_SGMM.tools_common
  for lim in "64 32" "1 1" "37 0"; do
    set -- $lim; dir="$OUT/img30_base_off_bpp050/progressive_y$1_uv$2"
    [ -f "$dir/manifest.txt" ] || nice -n 19 python "$HERE/dump_decode.py" "$OUT/img30_base_off_bpp050/stream.bits" "$dir" \
        --decoder-args "$P.model_y.common_modules.num_decode_chs $1 $P.model_uv.common_modules.num_decode_chs $2" > "$dir.log" 2>&1
  done
fi

if [ "$SET" = regions ] || [ "$SET" = all ]; then
  one img01_base_off_bpp050 $IMG01 50 cfg/tools_off.json cfg/profiles/base.json
  one img01_base_off_threads8_bpp050 $IMG01 50 cfg/tools_off.json cfg/tools/ECThread8.json cfg/profiles/base.json
  one_fixed img01_base_off_depregions_m1 $IMG01 1 150 cfg/tools_off.json cfg/tools/DependentRegions.json cfg/profiles/base.json
  one_fixed img01_base_off_indregions_m1 $IMG01 1 150 cfg/tools_off.json cfg/tools/IndependentRegions.json cfg/profiles/base.json
  one_fixed img01_base_off_indregions_threads8_m2 $IMG01 2 -100 cfg/tools_off.json cfg/tools/IndependentRegions.json cfg/tools/ECThread8.json cfg/profiles/base.json
fi
if [ "$SET" = tools ] || [ "$SET" = all ]; then
  # One coding tool at a time on top of tools_off, then the pairs that interact.
  one img30_base_lsbs_bpp050 $IMG30 50 cfg/tools_off.json cfg/tools/LSBS.json cfg/profiles/base.json
  one img30_base_rvs_bpp050 $IMG30 50 cfg/tools_off.json cfg/tools/ResVarScale.json cfg/profiles/base.json
  one img30_base_lsbs_rvs_bpp025 $IMG30 25 cfg/tools_off.json cfg/tools/LSBS.json cfg/tools/ResVarScale.json cfg/profiles/base.json
  # RVS without gain flags and gain flags without RVS (configs of ours: upstream only ships both).
  one img30_base_rvsonly_bpp050 $IMG30 50 cfg/tools_off.json "$HERE/cfg/rvs_only.json" cfg/profiles/base.json
  one img30_base_grfsonly_bpp075 $IMG30 75 cfg/tools_off.json "$HERE/cfg/grfs_only.json" cfg/profiles/base.json
  one img30_simple_lsbs_rvs_bpp100 $IMG30 100 cfg/tools_off.json cfg/tools/LSBS.json cfg/tools/ResVarScale.json cfg/profiles/simple.json
fi
if [ "$SET" = filters ] || [ "$SET" = all ]; then
  # The four enhancement post-filters, one at a time, then upstream's full tools_on set.
  one img30_base_efelin_bpp050 $IMG30 50 cfg/tools_off.json cfg/tools/EFElinear.json cfg/profiles/base.json
  one img30_base_eicci_bpp050 $IMG30 50 cfg/tools_off.json cfg/tools/eICCI.json cfg/profiles/base.json
  one img30_base_efenl_bpp050 $IMG30 50 cfg/tools_off.json cfg/tools/EFEnonlinear.json cfg/profiles/base.json
  one img30_base_lef_bpp050 $IMG30 50 cfg/tools_off.json cfg/tools/LEF.json cfg/profiles/base.json
  one img30_base_on_bpp025 $IMG30 25 cfg/tools_on.json cfg/profiles/base.json
  one img30_base_on_bpp100 $IMG30 100 cfg/tools_on.json cfg/profiles/base.json
fi
if [ "$SET" = qmap ] || [ "$SET" = all ]; then
  # Quality map (spatially varying quantisation). Of upstream's map generators only the
  # mask-file one (qp_map_type 3) works on this code path (the others downscale a shape that is
  # already latent-sized and crash), and upstream does not ship its sample mask. The mask is
  # drawn by examples/make_roi_mask.rs; the config is written here because it needs the path.
  MASKS=$OUT/../masks; mkdir -p "$MASKS"
  (cd "$HERE/../.." && nice -n 19 cargo run -q --release --features cli --example make_roi_mask -- \
      560 888 "$MASKS/img30_roi.png" 96,160,208,304 352,560,128,160)
  cat > "$MASKS/qmap_img30.json" <<JSON
{ "model": { "tool": "CCS_SGMM", "CCS_SGMM": { "tools_common": { "qual_map": {
  "enabled": 1, "qp_map_type": 3, "adjust_qp": 1, "ROI_map_in_file": "$MASKS/img30_roi.png" } } } } }
JSON
  one img30_base_qmap_bpp050 $IMG30 50 cfg/tools_off.json "$MASKS/qmap_img30.json" cfg/profiles/base.json
  one img30_base_qmap_rvs_bpp025 $IMG30 25 cfg/tools_off.json cfg/tools/ResVarScale.json "$MASKS/qmap_img30.json" cfg/profiles/base.json
  one img30_base_qmap_threads8_bpp100 $IMG30 100 cfg/tools_off.json cfg/tools/ECThread8.json "$MASKS/qmap_img30.json" cfg/profiles/base.json
fi
# Per-filter picture dumps (`<vector>/filters/`): the picture before and after every enabled
# post-filter, for tests that run one filter in isolation (tests/filters_efe_ref.rs).
filters_dump() { # name
  local dir="$OUT/$1"
  if [ -f "$dir/filters/manifest.txt" ]; then return; fi
  echo "== $1: dumping the picture around each post-filter"
  nice -n 19 python "$HERE/dump_filters.py" "$dir/stream.bits" "$dir/filters" > "$dir/dump_filters.log" 2>&1
}

# EFE filter streams with the encoder's EFE decisions forced (force_efe_encode.py): the stock
# encoder nearly always picks a 1x1 filter, one region, non-linear filter off.
efe() { # name input model_id beta_disp_log "force args" cfg... [-- encoder overrides...]
  local name=$1 input=$2 tool=$3 beta=$4 force=$5; shift 5
  local dir="$OUT/$name"
  if [ ! -f "$dir/manifest.txt" ]; then
    mkdir -p "$dir"
    echo "== $name: encoding, EFE decisions forced: $force ($(date -u +%H:%M:%S))"
    # shellcheck disable=SC2086
    nice -n 19 python "$HERE/force_efe_encode.py" $force -- "$input" "$dir/stream.bits" \
        --cfg "$@" -target_device cpu -model.bitrate_matcher.enabled 0 \
        -model.bitrate_matcher.target_tool_idx "$tool" -model.bitrate_matcher.target_beta_disp_Y "$beta" \
        > "$dir/encoder.log" 2>&1
    echo "== $name: decoding + dumping"
    nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir" > "$dir/dump.log" 2>&1
    ls -la "$dir/stream.bits" | awk '{print "   stream bytes:", $5}'
  fi
  filters_dump "$name"
}

if [ "$SET" = efe ] || [ "$SET" = all ]; then
  for v in img30_base_efelin_bpp050 img30_base_on_bpp025 img30_base_on_bpp100; do
    [ -f "$OUT/$v/manifest.txt" ] && filters_dump "$v"
  done
  EFE="cfg/tools_off.json cfg/tools/EFElinear.json cfg/tools/EFEnonlinear.json cfg/profiles/base.json"
  IN="$OUT/_inputs"; mkdir -p "$IN"
  mk() { [ -f "$IN/$1" ] || python "$HERE/make_efe_inputs.py" "data/test/$IMG30" "$IN/$1" "$2" "$3"; }
  mk crop_277x201_8bit_sRGB.png 277 201
  mk crop_277x201_8bit_420.yuv 277 201
  mk crop_277x201_8bit_422.yuv 277 201
  mk crop_560x888_8bit_420.yuv 560 888
  # 4:4:4 source coded 4:4:4: every filter length and every region split, non-linear filter on.
  # shellcheck disable=SC2086
  {
  efe img30_efe_f2c1_f2c2_nl "data/test/$IMG30" 1 0 "--linear 2:1,2:2 --nonlinear" $EFE
  efe img30_efe_f3c3_f3c4_nl "data/test/$IMG30" 1 0 "--linear 3:3,3:4 --nonlinear" $EFE
  efe img30_efe_f3c5_f4c7_nl "data/test/$IMG30" 1 0 "--linear 3:5,4:7 --nonlinear" $EFE
  efe img30_efe_f4c6_f1c5 "data/test/$IMG30" 2 0 "--linear 4:6,1:5" $EFE
  efe crop277_efe_f4c5_f3c6_nl "$IN/crop_277x201_8bit_sRGB.png" 1 0 "--linear 4:5,3:6 --nonlinear" $EFE
  # 2096x1400: four non-linear tiles.
  efe img01_efe_f2c0_f3c0_nl "data/test/$IMG01" 1 0 "--linear 2:0,3:0 --nonlinear" $EFE
  # 4:4:4 source coded 4:2:0 / 4:2:2: the 4x4 DCT-IF kernels, four coded phases.
  efe img30_c420_efe_f1c0_f2c1 "data/test/$IMG30" 1 0 "--linear 1:0,2:1" $EFE -c_ver_value 2 -c_hor_value 2
  efe img30_c420_efe_f3c5_f4c7_nl "data/test/$IMG30" 1 0 "--linear 3:5,4:7 --nonlinear" $EFE -c_ver_value 2 -c_hor_value 2
  efe crop277_c420_efe_f4c6_f3c3_nl "$IN/crop_277x201_8bit_sRGB.png" 1 0 "--linear 4:6,3:3 --nonlinear" $EFE -c_ver_value 2 -c_hor_value 2
  # DCTIF_only: no coded filters at all, the fixed DCT-IF taps alone.
  efe img30_c420_efe_dctif "data/test/$IMG30" 1 0 "" $EFE -c_ver_value 2 -c_hor_value 2 -post_filters.EFElinear.DCTIF_only 1
  # (vertical-only subsampling, c_ver = 2 with c_hor = 1, is not a format of the reference.)
  efe img30_c422_efe_f3c2_f2c4 "data/test/$IMG30" 1 0 "--linear 3:2,2:4" $EFE -c_ver_value 1 -c_hor_value 2
  # 4:2:0 and 4:2:2 sources.
  efe crop277_s420_efe_f3c5_f4c7_nl "$IN/crop_277x201_8bit_420.yuv" 1 0 "--linear 3:5,4:7 --nonlinear" $EFE
  efe img30_s420_efe_f1c0_f2c1_nl "$IN/crop_560x888_8bit_420.yuv" 1 0 "--linear 1:0,2:1 --nonlinear" $EFE
  efe crop277_s422_efe_f3c6_f4c2_nl "$IN/crop_277x201_8bit_422.yuv" 1 0 "--linear 3:6,4:2 --nonlinear" $EFE
  # (a 4:2:2 source coded 4:2:0 is not implemented in the reference encoder.)
  }
fi
if [ "$SET" = filtertiles ] || [ "$SET" = all ]; then
  # eICCI with its own tiling on (upstream's threshold of 2048^2 samples never tiles the test
  # pictures): 1024 tiles, overlap 48, last column narrower than the filter's 176 minimum, so
  # `_adjust_boundary_tiles` runs. LEF on top. Needs dump_filters_lef_icci.py afterwards.
  one img01_base_eiccitiles_lef_bpp050 $IMG01 50 cfg/tools_off.json "$HERE/cfg/eicci_tiles.json" cfg/tools/LEF.json cfg/profiles/base.json
  for v in img01_base_eiccitiles_lef_bpp050 img30_base_lef_bpp050 img30_base_eicci_bpp050 img30_base_on_bpp025 img30_base_on_bpp100; do
    [ -f "$OUT/$v/filters_lef_icci/manifest.txt" ] || nice -n 19 python "$HERE/dump_filters_lef_icci.py" \
        "$OUT/$v/stream.bits" "$OUT/$v/filters_lef_icci" > "$OUT/$v/dump_filters_lef_icci.log" 2>&1
  done
fi
if [ "$SET" = formats ] || [ "$SET" = all ]; then
  # Chroma-subsampled and 10-bit sources: raw YUV written by the reference's own Image class
  # from test image 00030 (scripts/ref_vectors/make_yuv_inputs.py).
  IN=$OUT/../inputs
  [ -f "$IN/img30_560x888_8bit_420.yuv" ] || PYTHONPATH=. python "$HERE/make_yuv_inputs.py" "$IN"
  one img30yuv420_base_off_bpp050 "$IN/img30_560x888_8bit_420.yuv" 50 cfg/tools_off.json cfg/profiles/base.json
  one img30yuv422_base_off_bpp050 "$IN/img30_560x888_8bit_422.yuv" 50 cfg/tools_off.json cfg/profiles/base.json
  one img30yuv444_base_off_bpp050 "$IN/img30_560x888_8bit_444.yuv" 50 cfg/tools_off.json cfg/profiles/base.json
  one img30yuv420b10_base_off_bpp050 "$IN/img30_560x888_10bit_420.yuv" 50 cfg/tools_off.json cfg/profiles/base.json
  one img30yuv444b10_base_off_bpp050 "$IN/img30_560x888_10bit_444.yuv" 50 cfg/tools_off.json cfg/profiles/base.json
  one img30cropyuv420_base_off_bpp075 "$IN/img30crop_203x301_8bit_420.yuv" 75 cfg/tools_off.json cfg/profiles/base.json
  # RGB source coded with subsampled chroma: the decoder upsamples (bicubic) before RGB.
  one img30_base_off_c420_bpp050 $IMG30 50 cfg/tools_off.json cfg/profiles/base.json -c_ver_value 2 -c_hor_value 2
  one img30_base_off_c422_bpp050 $IMG30 50 cfg/tools_off.json cfg/profiles/base.json -c_hor_value 2
  # Non-displayed right/bottom border. The bitrate matcher cannot handle it (its loss compares
  # the cropped reconstruction with the uncropped source), hence the fixed model.
  # User-defined information: opaque bytes in their own substream.
  printf 'zenjpegai UDI test \000\001\377 payload' > "$IN/udi_payload.bin"
  one_fixed img30_base_off_udi_m1 $IMG30 1 100 cfg/tools_off.json cfg/profiles/base.json -udi.filepath "$IN/udi_payload.bin"
  one_fixed img30_base_off_display_m1 $IMG30 1 100 cfg/tools_off.json cfg/profiles/base.json -diff_display_img_width 37 -diff_display_img_height 5
fi
if [ "$SET" = encoder ] || [ "$SET" = all ]; then
  # Oracles for the ENCODER port: fixed-model encodes whose analysis-side tensors are dumped
  # into <vector>/enc2 (dump_encode.py --enc2). tests/encode_ref.rs reads the first two.
  enc_fixed() { # name image model_id beta_disp_log cfg...
    local name=$1 image=$2 tool=$3 beta=$4; shift 4
    local dir="$OUT/$name"
    if [ -f "$dir/enc2/enc_manifest.txt" ]; then echo "== $name: exists"; return; fi
    mkdir -p "$dir/enc2"
    echo "== $name: encoding + dumping analysis side, fixed model $tool beta_disp $beta"
    nice -n 19 python "$HERE/dump_encode.py" "$dir/enc2" --enc2 -- "data/test/$image" "$dir/stream.bits" \
        --cfg "$@" -target_device cpu -model.bitrate_matcher.enabled 0 \
        -model.bitrate_matcher.target_tool_idx "$tool" -model.bitrate_matcher.target_beta_disp_Y "$beta" \
        > "$dir/encoder.log" 2>&1
  }
  enc_fixed enc_img30_bop_m1_b0 $IMG30 1 0 cfg/tools_off.json cfg/profiles/base.json
  enc_fixed enc_img30_hop_m2_b0 $IMG30 2 0 cfg/tools_off.json cfg/profiles/high.json
  # Not generated yet (next steps of the encoder port, see PORTING.md "Work queue"): the other
  # models, beta displacements (e.g. -300, -100, 150, 400: low rates exercise the cube flags) and
  # the 2096x1400 picture (analysis tiling).
fi
echo "== done ($(date -u +%H:%M:%S))"
