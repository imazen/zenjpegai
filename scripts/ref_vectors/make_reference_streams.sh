#!/usr/bin/env bash
# Encode + decode a matrix of configurations with the reference software and dump the decoder's
# intermediate tensors for the parity tests.
#
#   scripts/ref_vectors/make_reference_streams.sh [SET]      SET: smoke (default) | regions | tools | filters | all
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
  if [ -f "$dir/manifest.txt" ]; then echo "== $name: exists"; return; fi
  mkdir -p "$dir"
  echo "== $name: encoding ($(date -u +%H:%M:%S))"
  nice -n 19 python -m src.reco.coders.encoder "data/test/$image" "$dir/stream.bits" \
      --set_target_bpp "$bpp" --cfg "$@" -target_device cpu > "$dir/encoder.log" 2>&1
  echo "== $name: decoding + dumping"
  nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir" > "$dir/dump.log" 2>&1
  # Same decode with the skip mask made contiguous (what the reference encoder does): the
  # oracle for everything behind the entropy stage (tests/decode_ref.rs).
  nice -n 19 python "$HERE/dump_decode.py" "$dir/stream.bits" "$dir/fixed_decoder" --contiguous-masks \
      > "$dir/dump_fixed.log" 2>&1
  grep -h "^MD5" "$dir/encoder.log" "$dir/stdout.log" | sort | uniq -c | sed 's/^/   /'
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
  grep -h "^MD5" "$dir/encoder.log" "$dir/stdout.log" | sort | uniq -c | sed 's/^/   /'
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
echo "== done ($(date -u +%H:%M:%S))"
