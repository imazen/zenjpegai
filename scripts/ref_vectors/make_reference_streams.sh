#!/usr/bin/env bash
# Encode + decode a matrix of configurations with the reference software and dump the decoder's
# intermediate tensors for the parity tests.
#
#   scripts/ref_vectors/make_reference_streams.sh [SET]      SET: smoke (default) | regions | tools | filters | efe | efesolves | filtertiles | qmap | formats | icci420 | encoder | cubeflags | rate | toolson | all
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

if [ "$SET" = rate ] || [ "$SET" = all ]; then
  # E7: the second CTC picture at every CTC rate, for the likelihood-estimator parity gate
  # (img30's five are the `smoke` set; `regions` adds img01 at 0.50).
  for bpp in 012 025 075 100; do
    one "img01_base_off_bpp$bpp" $IMG01 "$((10#$bpp))" cfg/tools_off.json cfg/profiles/base.json
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

# EFE non-linear encode-side oracle (`<vector>/efe_nonlinear/`): replay `EFEnonlinear.compress`
# on the filters/ dumps and record its decisions, for tests/encode_ref.rs. No-op for vectors
# whose stream never enabled the tool.
nl_dump() { # name source_image model_id
  local dir="$OUT/$1"
  if [ -f "$dir/efe_nonlinear/manifest.txt" ]; then return; fi
  grep -q "EFEnonlinear.in.a" "$dir/filters/manifest.txt" 2>/dev/null || return 0
  echo "== $1: replaying EFEnonlinear.compress"
  nice -n 19 python "$HERE/dump_efe_nonlinear.py" "$dir/efe_nonlinear" "$dir/filters" "$2" "$3" \
      > "$dir/dump_efe_nonlinear.log" 2>&1
}

# `nl_dump` for streams the bitrate matcher coded: the model id comes back out of the PIH.
nl_dump_probe() { # name source_image
  local dir="$OUT/$1" mid
  mid=$(PYTHONPATH=. python scripts/bitstream_probe.py "$dir/stream.bits" 2>/dev/null \
      | awk -F'|' '$2 ~ /^ *model_id *$/ {gsub(/ /,"",$3); print $3; exit}')
  nl_dump "$1" "$2" "$mid"
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
  nl_dump "$name" "$input" "$tool"
}

if [ "$SET" = efe ] || [ "$SET" = all ]; then
  for v in img30_base_efelin_bpp050 img30_base_efenl_bpp050 img30_base_on_bpp025 img30_base_on_bpp100; do
    [ -f "$OUT/$v/manifest.txt" ] && filters_dump "$v"
  done
  for v in img30_base_efenl_bpp050 img30_base_on_bpp025 img30_base_on_bpp100; do
    [ -f "$OUT/$v/filters/manifest.txt" ] && nl_dump_probe "$v" "data/test/$IMG30"
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
# EFE-linear solver dumps (`<vector>/efe_solves/`): dump_efe_solves.py re-runs the
# reference's SplitDecide on the vector's own `EFElinear.in.*` planes and records every
# lstsq triple, every integerizeTensor pair, and each spec's decision — the oracle for
# `encoder::filters::efe_linear::tests::oracle`. Additive: writes only efe_solves/.
efe_solves() { # vector source c_ver c_hor spec...
  local vector=$1 source=$2 cver=$3 chor=$4; shift 4
  local dir="$OUT/$vector"
  if [ -f "$dir/efe_solves/manifest.txt" ]; then echo "== $vector/efe_solves: exists"; return; fi
  [ -f "$dir/filters/manifest.txt" ] || { echo "!! $vector/filters missing — run the efe set first"; return 1; }
  echo "== $vector: dumping EFE solver triples ($*)"
  EFE_CVER=$cver EFE_CHOR=$chor nice -n 19 python "$HERE/dump_efe_solves.py" \
      "$dir/efe_solves" "$dir/filters" "$source" "$@" > "$dir/dump_efe_solves.log" 2>&1
}
if [ "$SET" = efesolves ] || [ "$SET" = all ]; then
  # Specs mirror the table in encoder::filters::efe_linear::tests::oracle::DUMPS —
  # spec order on this command line fixes the dump's solve.N numbering, so a change
  # here must update that table (solve/int/excused counts).
  efe_solves crop277_efe_f4c5_f3c6_nl "$OUT/_inputs/crop_277x201_8bit_sRGB.png" 1 1 4:5 up2
  efe_solves img30_c420_efe_f3c5_f4c7_nl "data/test/$IMG30" 2 2 3:5 4:7 up2
  efe_solves img30_base_efelin_bpp050 "data/test/$IMG30" 1 1 search:1
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
if [ "$SET" = toolson ] || [ "$SET" = all ]; then
  # E5: upstream's cfg/tools_on.json (RVS+GRFS, LSBS, EFE linear + non-linear, eICCI, LEF)
  # at every CTC rate on both test pictures — the oracle for the `tools_on` preset gate.
  # The post-filter dumps (filters/, filters_lef_icci/, efe_nonlinear/) make them full
  # per-tool oracles like the two tools_on streams of the `filters` set.
  for bpp in 012 025 050 075 100; do
    one "img30_base_on_bpp$bpp" $IMG30 "$((10#$bpp))" cfg/tools_on.json cfg/profiles/base.json
    one "img01_base_on_bpp$bpp" $IMG01 "$((10#$bpp))" cfg/tools_on.json cfg/profiles/base.json
  done
  for img in img30 img01; do
    case $img in img30) SRC="data/test/$IMG30";; *) SRC="data/test/$IMG01";; esac
    for bpp in 012 025 050 075 100; do
      v=${img}_base_on_bpp$bpp
      [ -f "$OUT/$v/manifest.txt" ] && filters_dump "$v"
      if [ -f "$OUT/$v/manifest.txt" ] && [ ! -f "$OUT/$v/filters_lef_icci/manifest.txt" ]; then
        nice -n 19 python "$HERE/dump_filters_lef_icci.py" "$OUT/$v/stream.bits" \
            "$OUT/$v/filters_lef_icci" > "$OUT/$v/dump_filters_lef_icci.log" 2>&1
      fi
      [ -f "$OUT/$v/filters/manifest.txt" ] && nl_dump_probe "$v" "$SRC"
    done
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
if [ "$SET" = icci420 ] || [ "$SET" = all ]; then
  # eICCI on chroma-subsampled sources: the reference encoder never enables it for them, so it
  # is forced (force_icci_encode.py, model selection 7:3 on the long list). For a 4:2:2 source
  # `icci_enable_flag` is part of the syntax and the stream is conformant. For 4:2:0 it is not
  # (auto-detected off): the stream carries the flag + header anyway, and only the decoder
  # patched the same way (--patch-icci420) can read it — PORTING.md documents why that stream
  # is a per-filter oracle only.
  IN=$OUT/../inputs
  icci() { # name input [force args...]
    local name=$1 input=$2; shift 2
    local dir="$OUT/$name"
    if [ ! -f "$dir/manifest.txt" ] && [ ! -f "$dir/fixed_decoder/manifest.txt" ]; then
      mkdir -p "$dir"
      echo "== $name: encoding, eICCI forced on ($(date -u +%H:%M:%S))"
      nice -n 19 python "$HERE/force_icci_encode.py" "$@" -- "$input" "$dir/stream.bits" \
          --cfg cfg/tools_off.json cfg/tools/eICCI.json cfg/profiles/base.json -target_device cpu \
          -model.bitrate_matcher.enabled 0 -model.bitrate_matcher.target_tool_idx 1 \
          -model.bitrate_matcher.target_beta_disp_Y 0 > "$dir/encoder.log" 2>&1
      echo "== $name: decoding + dumping"
      nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir" > "$dir/dump.log" 2>&1 \
          || echo "   stock reference decoder FAILED on this stream (see $dir/dump.log)"
      nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir/fixed_decoder" \
          --contiguous-masks --patch-icci420 > "$dir/dump_fixed.log" 2>&1
      { grep -hs "^MD5" "$dir/encoder.log" "$dir/stdout.log" "$dir/fixed_decoder/stdout.log" || true; } \
          | sort | uniq -c | sed 's/^/   /'
      ls -la "$dir/stream.bits" | awk '{print "   stream bytes:", $5}'
    fi
  }
  icci img30yuv422_base_eicci "$IN/img30_560x888_8bit_422.yuv"
  icci img30yuv420_base_eicci "$IN/img30_560x888_8bit_420.yuv" --flag420
  for v in img30yuv422_base_eicci img30yuv420_base_eicci; do
    [ -f "$OUT/$v/filters_lef_icci/manifest.txt" ] || nice -n 19 python "$HERE/dump_filters_lef_icci.py" \
        "$OUT/$v/stream.bits" "$OUT/$v/filters_lef_icci" --patch-icci420 \
        > "$OUT/$v/dump_filters_lef_icci.log" 2>&1
  done
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
    nice -n 19 python "$HERE/dump_encode.py" "$dir/enc2" --enc2 --lef -- "data/test/$image" "$dir/stream.bits" \
        --cfg "$@" -target_device cpu -model.bitrate_matcher.enabled 0 \
        -model.bitrate_matcher.target_tool_idx "$tool" -model.bitrate_matcher.target_beta_disp_Y "$beta" \
        > "$dir/encoder.log" 2>&1
  }
  enc_fixed enc_img30_bop_m1_b0 $IMG30 1 0 cfg/tools_off.json cfg/profiles/base.json
  enc_fixed enc_img30_hop_m2_b0 $IMG30 2 0 cfg/tools_off.json cfg/profiles/high.json
  # Low rates: the reconstruction error inside a cube exceeds skip_cube_thr, so cube flags go
  # false and `use_cube_flags` is signalled (the only way to exercise that path).
  enc_fixed enc_img30_bop_m1_bm300 $IMG30 1 -300 cfg/tools_off.json cfg/profiles/base.json
  enc_fixed enc_img30_sop_m0_bm300 $IMG30 0 -300 cfg/tools_off.json cfg/profiles/simple.json
  enc_fixed enc_img30_bop_m3_b400 $IMG30 3 400 cfg/tools_off.json cfg/profiles/base.json
  # -1069 is the low end of BDL_clipping_range (`cfg`: [-1069, 702]).
  enc_fixed enc_img30_bop_m0_bm1069 $IMG30 0 -1069 cfg/tools_off.json cfg/profiles/base.json
  enc_fixed enc_img30_hop_m3_bm1069 $IMG30 3 -1069 cfg/tools_off.json cfg/profiles/high.json
  # 2096x1400: above the encoder's 1 MP threshold, so the analysis transform and the
  # hyper-encoder run per tile (1024 luma / 512 chroma, overlap 64 / 32).
  enc_fixed enc_img01_bop_m1_b0 $IMG01 1 0 cfg/tools_off.json cfg/profiles/base.json
  # Encode-side coding tools, one at a time on top of tools_off, at a fixed model.
  enc_fixed enc_img30_bop_m1_b0_threads8 $IMG30 1 0 cfg/tools_off.json cfg/tools/ECThread8.json cfg/profiles/base.json
  enc_fixed enc_img30_bop_m1_b0_rvs $IMG30 1 0 cfg/tools_off.json cfg/tools/ResVarScale.json cfg/profiles/base.json
  enc_fixed enc_img30_bop_m1_b0_rvsonly $IMG30 1 0 cfg/tools_off.json "$HERE/cfg/rvs_only.json" cfg/profiles/base.json
  enc_fixed enc_img30_bop_m1_b0_grfsonly $IMG30 1 0 cfg/tools_off.json "$HERE/cfg/grfs_only.json" cfg/profiles/base.json
  enc_fixed enc_img30_bop_m1_b0_lsbs $IMG30 1 0 cfg/tools_off.json cfg/tools/LSBS.json cfg/profiles/base.json
  enc_fixed enc_img30_bop_m1_b0_lef $IMG30 1 0 cfg/tools_off.json cfg/tools/LEF.json cfg/profiles/base.json
  enc_fixed enc_img01_bop_m1_b0_depregions $IMG01 1 0 cfg/tools_off.json cfg/tools/DependentRegions.json cfg/profiles/base.json
  enc_fixed enc_img01_bop_m1_b0_indregions $IMG01 1 0 cfg/tools_off.json cfg/tools/IndependentRegions.json cfg/profiles/base.json
  # Quality map: the same ROI mask the `qmap` set uses (drawn here so the sets are independent).
  MASKS=$OUT/../masks; mkdir -p "$MASKS"
  [ -f "$MASKS/img30_roi.png" ] || (cd "$HERE/../.." && nice -n 19 cargo run -q --release --features cli --example make_roi_mask -- \
      560 888 "$MASKS/img30_roi.png" 96,160,208,304 352,560,128,160)
  cat > "$MASKS/qmap_img30.json" <<JSON
{ "model": { "tool": "CCS_SGMM", "CCS_SGMM": { "tools_common": { "qual_map": {
  "enabled": 1, "qp_map_type": 3, "adjust_qp": 1, "ROI_map_in_file": "$MASKS/img30_roi.png" } } } } }
JSON
  enc_fixed enc_img30_bop_m1_b0_qmap $IMG30 1 0 cfg/tools_off.json "$MASKS/qmap_img30.json" cfg/profiles/base.json
  enc_fixed enc_img30_bop_m1_b0_qmap_rvs $IMG30 1 0 cfg/tools_off.json cfg/tools/ResVarScale.json "$MASKS/qmap_img30.json" cfg/profiles/base.json
  # Not generated yet (see PORTING.md "Work queue"): odd picture sizes.
fi
if [ "$SET" = cubeflags ] || [ "$SET" = all ]; then
  # Decoder-side dumps for the encoder set's use_cube_flags = 1 streams (beta -1069; the
  # streams themselves come from `make_reference_streams.sh encoder`). Non-region, so the
  # stock reference decoder is a correct oracle for them.
  for d in enc_img30_bop_m0_bm1069 enc_img30_hop_m3_bm1069; do
    [ -f "$OUT/$d/stream.bits" ] || { echo "== $d: missing stream, run the encoder set first" >&2; exit 1; }
    [ -f "$OUT/$d/manifest.txt" ] || nice -n 19 python "$HERE/dump_decode.py" "$OUT/$d/stream.bits" "$OUT/$d" > "$OUT/$d/dump.log" 2>&1
  done
fi
echo "== done ($(date -u +%H:%M:%S))"
