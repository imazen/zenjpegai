#!/usr/bin/env bash
# Peak heap of one `zenjpegai encode` process per image, measured with heaptrack.
#   scripts/bench/memory_encode.sh [label] > benchmarks/memory_encode_$(date +%F).tsv
# Needs heaptrack, the upstream models, the reference test images and
# `cargo build --release --features cli`. BIN overrides the binary (before/after runs).
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
IMG=${ZENJPEGAI_IMAGES:-$REF/data/test}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/../../target/release/zenjpegai}
LABEL=${1:-current}
TMP=${TMPDIR:-$HOME/tmp}/zenjpegai-mem; mkdir -p "$TMP"
# name=file: the two sizes the E8 gate measures (the reference's own test images).
IMAGES=${IMAGES:-"img30_560x888=$IMG/00030_TE_560x888_8bit_sRGB.png img01_2096x1400=$IMG/00001_TE_2096x1400_8bit_sRGB.png"}
# encode = fixed model 1, beta 0, BOP; bpp050 = rate matched to 0.5 bpp.
MODES=${MODES:-"encode bpp050"}
# POOL_MB: cap of the recycled-buffer pool (`--pool-mb`); unset = the default 1 GiB.
EXTRA=""; [ -n "${POOL_MB:-}" ] && EXTRA="--pool-mb $POOL_MB"
[ -n "${NO_HEADER:-}" ] || printf 'label\tmode\tstream\tthreads\tpeak_heap_bytes\tpeak_rss_bytes\tallocations\ttemporary_allocations\n'
tobytes() { awk -v v="$1" 'BEGIN{n=v+0; u=v; sub(/^[0-9.]+/,"",u); m=1; if(u=="K")m=1e3; if(u=="M")m=1e6; if(u=="G")m=1e9; printf "%.0f", n*m}'; }
for pair in $IMAGES; do
  name=${pair%%=*}; img=${pair#*=}
  for mode in $MODES; do
    if [ "$mode" = encode ]; then set -- --model 1 --beta-disp 0 --op bop; else set -- --bpp 0.5 --op bop; fi
    for mt in "" "--single-thread"; do
      th=$(nproc); [ -n "$mt" ] && th=1
      out=$TMP/ht.$LABEL.$name.$mode.$th
      rm -f "$out".zst
      ZENJPEGAI_MODELS=$REF/models nice -n 19 heaptrack -o "$out" "$BIN" encode "$img" "$TMP/o.bits" "$@" $mt $EXTRA >/dev/null 2>&1
      rep=$(heaptrack_print -f "$out".zst 2>/dev/null | tail -12)
      peak=$(sed -n 's/^peak heap memory consumption: *//p' <<<"$rep")
      rss=$(sed -n 's/^peak RSS (including heaptrack overhead): *//p' <<<"$rep")
      calls=$(sed -n 's/^calls to allocation functions: *\([0-9]*\).*/\1/p' <<<"$rep")
      temp=$(sed -n 's/^temporary memory allocations: *\([0-9]*\).*/\1/p' <<<"$rep")
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$mode" "$name" "$th" "$(tobytes "$peak")" "$(tobytes "$rss")" "$calls" "$temp"
    done
  done
done
