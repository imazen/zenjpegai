#!/usr/bin/env bash
# Tracked-heap accounting vs heaptrack ground truth, one process per case.
#   scripts/bench/memory_tracked.sh [label] > benchmarks/memory_tracked_$(date +%F).tsv
#
# Each case runs the CLI twice: once plain with --memory-report (the library's own ledger,
# src/mem.rs) and once under heaptrack (every allocation, ground truth). The tracked peak
# must sit inside the untracked remainder — a tracked/heaptrack ratio under ~0.8 means a
# large allocation bypasses the Charge ledger.
#
# Needs heaptrack, the reference vectors, the upstream models and
# `cargo build --release --features cli`. BIN overrides the binary.
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VEC=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
IMG=${ZENJPEGAI_IMAGES:-$REF/data/test}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/../../target/release/zenjpegai}
LABEL=${1:-current}
TMP=${TMPDIR:-$HOME/tmp}/zenjpegai-mem; mkdir -p "$TMP"
# The six standard decode cases: small and large CTC picture at each operating point.
STREAMS=${STREAMS:-"img30_simple_off_bpp050 img30_base_off_bpp050 img30_high_off_bpp050 img01_simple_off_bpp050 img01_base_off_bpp050 img01_high_off_bpp050"}
# Encode: fixed model 1 BOP plus the rate-matched case, both pictures.
IMAGES=${IMAGES:-"img30_560x888=$IMG/00030_TE_560x888_8bit_sRGB.png img01_2096x1400=$IMG/00001_TE_2096x1400_8bit_sRGB.png"}
MODES=${MODES:-"encode bpp050"}
[ -n "${NO_HEADER:-}" ] || printf 'label\tcase\ttracked_peak_bytes\testimate_peak_bytes\theaptrack_peak_bytes\ttracked_ratio\n'

heaptrack_peak() { # outfile cmd... -> peak heap bytes
  local out=$1; shift
  rm -f "$out".zst
  ZENJPEGAI_MODELS=$REF/models nice -n 19 heaptrack -o "$out" "$@" >/dev/null 2>&1
  local rep peak
  rep=$(heaptrack_print -f "$out".zst 2>/dev/null | tail -12)
  peak=$(sed -n 's/^peak heap memory consumption: *//p' <<<"$rep")
  awk -v v="$peak" 'BEGIN{n=v+0; u=v; sub(/^[0-9.]+/,"",u); m=1; if(u=="K")m=1e3; if(u=="M")m=1e6; if(u=="G")m=1e9; printf "%.0f", n*m}'
}

row() { # case tracked_tsv heap_bytes
  local case=$1 tsv=$2 ht=$3
  local tracked est
  tracked=$(awk -F'\t' '$1=="tracked_peak_bytes"{print $2}' <<<"$tsv")
  est=$(awk -F'\t' '$1=="estimate_peak_bytes"{print $2}' <<<"$tsv")
  printf '%s\t%s\t%s\t%s\t%s\t%.3f\n' "$LABEL" "$case" "${tracked:-0}" "${est:-0}" "$ht" \
      "$(awk -v t="${tracked:-0}" -v h="$ht" 'BEGIN{print (h>0)? t/h : 0}')"
}

for s in $STREAMS; do
  tsv=$(ZENJPEGAI_MODELS=$REF/models "$BIN" decode "$VEC/$s/stream.bits" "$TMP/o.png" --discard --memory-report 2>&1 >/dev/null)
  ht=$(heaptrack_peak "$TMP/ht.$LABEL.$s" "$BIN" decode "$VEC/$s/stream.bits" "$TMP/o.png" --discard)
  row "decode:$s" "$tsv" "$ht"
done
for pair in $IMAGES; do
  name=${pair%%=*}; img=${pair#*=}
  for mode in $MODES; do
    if [ "$mode" = encode ]; then set -- --model 1 --beta-disp 0 --op bop; else set -- --bpp 0.5 --op bop; fi
    tsv=$(ZENJPEGAI_MODELS=$REF/models "$BIN" encode "$img" "$TMP/o.bits" "$@" --memory-report 2>&1 >/dev/null)
    ht=$(heaptrack_peak "$TMP/ht.$LABEL.$name.$mode" "$BIN" encode "$img" "$TMP/o.bits" "$@")
    row "encode:$name:$mode" "$tsv" "$ht"
  done
done
