#!/usr/bin/env bash
# Per-stage decode breakdown (DecodeStats via `decode --stats`): streams x thread counts.
#   THREADS="1 8" RUNS=4 scripts/bench/decode_stages.sh > benchmarks/decode_stages_$(date +%F).tsv
# Needs `cargo build --release --features cli` and ZENJPEGAI_MODELS (defaults to the reference
# checkout's models dir). Run 0 includes model load; steady state is min over runs 1..
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VEC=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/../../target/release/zenjpegai
MODELS=${ZENJPEGAI_MODELS:-$REF/models}
RUNS=${RUNS:-4}
STREAMS=${STREAMS:-"img30_simple_off_bpp050 img30_base_off_bpp050 img30_high_off_bpp050 img01_base_off_bpp050"}
THREADS=${THREADS:-"1 8"}

REV=$(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || jj -R "$HERE/../.." log -r @- --no-graph -T 'commit_id.short()' 2>/dev/null || true)
echo "# commit $REV host-cpu $(lscpu | sed -n 's/^Model name: *//p') threads $(nproc) date $(date -u +%FT%TZ)"
echo "# stage rows are wall ms of each measured section; *_0=luma *_1=chroma; joined/chain rows are"
echo "# wall time of the overlapping pair, so they can be smaller than the sum of their parts."
printf 'stream\tthreads\trun\tstage\tms\n'
for s in $STREAMS; do
  bits=$VEC/$s/stream.bits
  for th in $THREADS; do
    mode="--single-thread"; pool=1
    if [ "$th" != "1" ]; then mode=""; pool=$th; fi
    RAYON_NUM_THREADS=$pool ZENJPEGAI_MODELS=$MODELS nice -n 19 "$BIN" \
      decode "$bits" /dev/null --discard --stats --repeat "$RUNS" $mode 2>&1 |
      sed -nE 's/^decode ([0-9]+): ([0-9.]+) ms.*/\1\tdecode_wall\t\2/p; /^[0-9]+\t/p' |
      awk -F '\t' -v s="$s" -v th="$th" '{printf "%s\t%s\t%s\t%s\t%s\n", s, th, $1, $2, $3}'
  done
done
