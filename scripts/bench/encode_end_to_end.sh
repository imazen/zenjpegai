#!/usr/bin/env bash
# End-to-end encode time: stock reference encoder vs zenjpegai, same pictures, same fixed model.
#   scripts/bench/encode_end_to_end.sh > benchmarks/encode_end_to_end_$(date +%F).tsv
# Needs: the reference checkout with its venv (CLAUDE.md) and `cargo build --release --features cli`.
# No -C target-cpu=native. Both sides run one pass at a fixed (model, beta displacement): the
# reference's bitrate matcher is off, which is what zenjpegai does today.
set -euo pipefail
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
HERE=$(cd "$(dirname "$0")" && pwd)
BIN=$HERE/../../target/release/zenjpegai
TMP=${TMPDIR:-$HOME/tmp}/zenjpegai-bench; mkdir -p "$TMP"
RUNS=${RUNS:-3}
NPROC=$(nproc)
# name:image:model:beta:profile
CASES=${CASES:-"img30_sop_m1_b0:00030_TE_560x888_8bit_sRGB.png:1:0:simple \
img30_bop_m1_b0:00030_TE_560x888_8bit_sRGB.png:1:0:base \
img30_hop_m2_b0:00030_TE_560x888_8bit_sRGB.png:2:0:high \
img01_bop_m1_b0:00001_TE_2096x1400_8bit_sRGB.png:1:0:base"}

echo "# commit $(git -C "$HERE" rev-parse --short HEAD 2>/dev/null) host-cpu $(lscpu | sed -n 's/^Model name: *//p') threads $NPROC date $(date -u +%FT%TZ)"
echo "# ref = stock reference encoder, bitrate matcher off, its own TOTAL (excludes model load); wall = whole process"
echo "# zen = zenjpegai encode, steady state (models loaded) = runs 1..; first = run 0 incl. model load"
printf 'case\tencoder\tthreads\trun\tencode_ms\tprocess_wall_ms\tbytes\n'
for c in $CASES; do
  IFS=: read -r name image model beta profile <<<"$c"
  for th in 1 "$NPROC"; do
    for r in $(seq 1 "$RUNS"); do
      line=$(cd "$REF" && . .venv/bin/activate && PYTHONPATH=. nice -n 19 python "$HERE/ref_encode.py" \
             "data/test/$image" "$TMP/ref.bits" --model "$model" --beta-disp "$beta" \
             --profile "$profile" --threads "$th" 2>/dev/null | tail -1)
      total=$(sed -n 's/.*ref_total_s=\([0-9.na]*\).*/\1/p' <<<"$line")
      wall=$(sed -n 's/.*wall_s=\([0-9.]*\).*/\1/p' <<<"$line")
      bytes=$(sed -n 's/.*bytes=\([0-9]*\).*/\1/p' <<<"$line")
      printf '%s\treference\t%s\t%s\t%s\t%s\t%s\n' "$name" "$th" "$r" \
        "$(awk "BEGIN{print $total*1000}")" "$(awk "BEGIN{print $wall*1000}")" "$bytes"
    done
  done
  op=bop; [ "$profile" = simple ] && op=sop; [ "$profile" = high ] && op=hop
  for mode in "" "--single-thread"; do
    th=$NPROC; [ -n "$mode" ] && th=1
    start=$(date +%s%N)
    ZENJPEGAI_MODELS=$REF/models nice -n 19 "$BIN" encode "$REF/data/test/$image" "$TMP/zen.bits" \
      --model "$model" --beta-disp "$beta" --op "$op" $mode 2>/dev/null
    end=$(date +%s%N)
    bytes=$(stat -c %s "$TMP/zen.bits")
    out=$(ZENJPEGAI_MODELS=$REF/models nice -n 19 "$BIN" encode "$REF/data/test/$image" "$TMP/zen.bits" \
      --model "$model" --beta-disp "$beta" --op "$op" --repeat $((RUNS + 1)) $mode 2>&1)
    r=0
    while read -r ms; do
      printf '%s\tzenjpegai\t%s\t%s\t%s\t%s\t%s\n' "$name" "$th" "$r" "$ms" \
        "$([ $r = 0 ] && echo $(( (end - start) / 1000000 )) || echo -)" "$bytes"
      r=$((r + 1))
    done < <(sed -n 's/^encode [0-9]*: \([0-9.]*\) ms.*/\1/p' <<<"$out")
  done
done
