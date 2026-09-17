#!/usr/bin/env bash
# End-to-end decode time: stock reference decoder vs zenjpegai, same streams, same machine.
#   scripts/bench/decode_end_to_end.sh > benchmarks/decode_end_to_end_$(date +%F).tsv
# Needs: the reference checkout with its venv (CLAUDE.md), reference vectors, and
# `cargo build --release --features cli`. No -C target-cpu=native.
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VEC=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/../../target/release/zenjpegai
TMP=${TMPDIR:-$HOME/tmp}/zenjpegai-bench; mkdir -p "$TMP"
RUNS=${RUNS:-3}
STREAMS=${STREAMS:-"img30_simple_off_bpp050 img30_base_off_bpp012 img30_base_off_bpp050 img30_base_off_bpp100 img30_high_off_bpp050 img01_base_off_bpp050"}
NPROC=$(nproc)

echo "# commit $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null) host-cpu $(lscpu | sed -n 's/^Model name: *//p') threads $NPROC date $(date -u +%FT%TZ)"
echo "# ref = stock reference decoder, its own TOTAL (excludes model load); wall = whole process"
echo "# zen = zenjpegai decode, steady state (models loaded) = min of runs 1..; first = run 0 incl. model load"
printf 'stream\tdecoder\tthreads\trun\tdecode_ms\tprocess_wall_ms\n'
for s in $STREAMS; do
  bits=$VEC/$s/stream.bits
  for th in 1 "$NPROC"; do
    for r in $(seq 1 "$RUNS"); do
      line=$(cd "$REF" && . .venv/bin/activate && PYTHONPATH=. nice -n 19 python "$HERE/ref_decode.py" "$bits" "$TMP/ref.png" --threads "$th" 2>/dev/null | tail -1)
      total=$(sed -n 's/.*ref_total_s=\([0-9.]*\).*/\1/p' <<<"$line"); wall=$(sed -n 's/.*wall_s=\([0-9.]*\).*/\1/p' <<<"$line")
      printf '%s\treference\t%s\t%s\t%s\t%s\n' "$s" "$th" "$r" "$(awk "BEGIN{print $total*1000}")" "$(awk "BEGIN{print $wall*1000}")"
    done
  done
  for mode in "" "--single-thread"; do
    th=$NPROC; [ -n "$mode" ] && th=1
    # Whole process, one decode, PNG written: comparable to the reference's process wall time.
    start=$(date +%s%N)
    ZENJPEGAI_MODELS=$REF/models nice -n 19 "$BIN" decode "$bits" "$TMP/zen.png" $mode
    end=$(date +%s%N)
    out=$(ZENJPEGAI_MODELS=$REF/models nice -n 19 "$BIN" decode "$bits" "$TMP/zen.png" --repeat $((RUNS + 1)) $mode 2>&1)
    r=0
    while read -r ms; do
      printf '%s\tzenjpegai\t%s\t%s\t%s\t%s\n' "$s" "$th" "$r" "$ms" "$([ $r = 0 ] && echo $(( (end - start) / 1000000 )) || echo -)"
      r=$((r + 1))
    done < <(sed -n 's/^decode [0-9]*: \([0-9.]*\) ms.*/\1/p' <<<"$out")
  done
done
