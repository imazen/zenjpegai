#!/usr/bin/env bash
# Peak heap of one `zenjpegai decode` process per stream, measured with heaptrack.
#   scripts/bench/memory.sh [label] > benchmarks/memory_$(date +%F).tsv
# Needs heaptrack, the reference vectors, the upstream models and
# `cargo build --release --features cli`. BIN overrides the binary (before/after runs).
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VEC=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=${BIN:-$HERE/../../target/release/zenjpegai}
LABEL=${1:-current}
TMP=${TMPDIR:-$HOME/tmp}/zenjpegai-mem; mkdir -p "$TMP"
STREAMS=${STREAMS:-"img30_simple_off_bpp050 img30_base_off_bpp050 img30_high_off_bpp050 img01_base_off_bpp050"}
[ -n "${NO_HEADER:-}" ] || printf 'label\tstream\tthreads\tpeak_heap_bytes\tpeak_rss_bytes\tallocations\ttemporary_allocations\n'
for s in $STREAMS; do
  for mode in "" "--single-thread"; do
    th=$(nproc); [ -n "$mode" ] && th=1
    out=$TMP/ht.$LABEL.$s.$th
    rm -f "$out".zst
    ZENJPEGAI_MODELS=$REF/models nice -n 19 heaptrack -o "$out" "$BIN" decode "$VEC/$s/stream.bits" "$TMP/o.png" $mode >/dev/null 2>&1
    rep=$(heaptrack_print -f "$out".zst 2>/dev/null | tail -12)
    # heaptrack_print prints human units; the raw numbers come from its summary lines.
    peak=$(sed -n 's/^peak heap memory consumption: *//p' <<<"$rep")
    rss=$(sed -n 's/^peak RSS (including heaptrack overhead): *//p' <<<"$rep")
    calls=$(sed -n 's/^calls to allocation functions: *\([0-9]*\).*/\1/p' <<<"$rep")
    temp=$(sed -n 's/^temporary memory allocations: *\([0-9]*\).*/\1/p' <<<"$rep")
    tobytes() { awk -v v="$1" 'BEGIN{n=v+0; u=v; sub(/^[0-9.]+/,"",u); m=1; if(u=="K")m=1e3; if(u=="M")m=1e6; if(u=="G")m=1e9; printf "%.0f", n*m}'; }
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LABEL" "$s" "$th" "$(tobytes "$peak")" "$(tobytes "$rss")" "$calls" "$temp"
  done
done
