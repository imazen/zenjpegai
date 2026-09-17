#!/usr/bin/env bash
# LEF / eICCI post-filter time: stock reference decoder vs zenjpegai, same planes, same machine.
#   scripts/bench/filters_lef_icci.sh > benchmarks/filters_lef_icci_$(date +%F).tsv
# Needs: the reference checkout with its venv (CLAUDE.md), vectors with a `filters_lef_icci/`
# dump (scripts/ref_vectors/make_reference_streams.sh filtertiles), and
# `cargo build --release --features cli --example prof_filters_lef_icci --bin zenjpegai`.
# No -C target-cpu=native.
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VEC=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
HERE=$(cd "$(dirname "$0")" && pwd)
TARGET=$HERE/../../target/release
TMP=${TMPDIR:-$HOME/tmp}/zenjpegai-bench; mkdir -p "$TMP"
RUNS=${RUNS:-3}
STREAMS=${STREAMS:-"img30_base_lef_bpp050 img30_base_eicci_bpp050 img30_base_on_bpp025 img01_base_eiccitiles_lef_bpp050"}
NPROC=$(nproc)

echo "# commit $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null || (cd "$HERE" && jj log -r @- --no-graph -T 'commit_id.short()')) host-cpu $(lscpu | sed -n 's/^Model name: *//p') threads $NPROC load $(cut -d' ' -f1 /proc/loadavg) date $(date -u +%FT%TZ)"
echo "# reference: time inside <Filter>.decompress during a stock decode (networks already loaded); threads=1 is what the reference does"
echo "# zenjpegai: best of 20 calls on the reference's input planes (eICCI networks cached); decode_ms = whole decode, steady state"
printf 'stream\timpl\tthreads\trun\teicci_ms\tlef_ms\tdecode_ms\n'
for s in $STREAMS; do
  bits=$VEC/$s/stream.bits
  for th in 1 "$NPROC"; do
    for r in $(seq 1 "$RUNS"); do
      line=$(cd "$REF" && . .venv/bin/activate && PYTHONPATH=. nice -n 19 python "$HERE/ref_filters_lef_icci.py" "$bits" "$TMP/ref.png" --threads "$th" 2>/dev/null | tail -1)
      get() { sed -n "s/.*$1=\([0-9.]*\).*/\1/p" <<<"$line" | awk '{print $1*1000}'; }
      printf '%s\treference\t%s\t%s\t%s\t%s\t%s\n' "$s" "$th" "$r" "$(get eicci_s)" "$(get lef_s)" "$(get ref_total_s)"
    done
  done
  prof=$(ZENJPEGAI_REF=$REF nice -n 19 "$TARGET/examples/prof_filters_lef_icci" "$VEC/$s" 20)
  for th in 1 "$NPROC"; do
    mode=""; [ "$th" = 1 ] && mode="--single-thread"
    dec=$(ZENJPEGAI_MODELS=$REF/models nice -n 19 "$TARGET/zenjpegai" decode "$bits" "$TMP/zen.png" --repeat $((RUNS + 1)) $mode 2>&1 \
        | sed -n 's/^decode [0-9]*: \([0-9.]*\) ms.*/\1/p' | tail -n +2 | sort -n | head -1)
    e=$(awk -F'\t' -v t="$th" '$1=="eICCI" && $2==t {print $3}' <<<"$prof")
    l=$(awk -F'\t' -v t="$th" '$1=="LEF" && $2==t {print $3}' <<<"$prof")
    printf '%s\tzenjpegai\t%s\tbest\t%s\t%s\t%s\n' "$s" "$th" "${e:-0}" "${l:-0}" "$dec"
  done
done
