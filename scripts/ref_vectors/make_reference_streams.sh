#!/usr/bin/env bash
# Encode + decode a matrix of configurations with the reference software and dump the decoder's
# intermediate tensors for the parity tests.
#
#   scripts/ref_vectors/make_reference_streams.sh [SET]      SET: smoke (default) | regions | all
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
echo "== done ($(date -u +%H:%M:%S))"
