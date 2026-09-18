#!/usr/bin/env bash
# Decode reference streams with the JPEG AI reference software on the CPU and on the GPU and
# print one TSV row per (stream, device, run): the decoder's own "TOTAL" (the wall time of
# `decode_stream`, with `torch.cuda.synchronize()` before the clock stops when the device is the
# GPU, so it is not a host timer lying about async work) and the MD5 of the reconstructed image
# it prints, so CPU / GPU agreement and run-to-run determinism are visible.
#
#   scripts/bench/reference_gpu.sh [runs] | tee benchmarks/gpu_reference_$(date +%F).tsv
#
# CUDA_VISIBLE_DEVICES must be set for the GPU path: the reference forces `target_device` back to
# `cpu` when the variable is empty, and then silently reports CPU numbers
# (`Coder.setup_device_param` in `src/codec/coders/coder.py`). Its CPU path is single-threaded
# (`torch.set_num_threads(1)` in `src/reco/coders/decoder.py`).
set -uo pipefail

REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VECTORS=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
RUNS=${1:-3}
STREAMS=${2:-"img30_simple_off_bpp050 img30_base_off_bpp050 img30_high_off_bpp050 img01_base_off_bpp050"}
WORK=$(mktemp -d "${TMPDIR:-$HOME/tmp}/refgpu.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

cd "$REF" || exit 1
# shellcheck disable=SC1091
. .venv/bin/activate

printf 'stream\tdevice\trun\ttotal_s\tmd5\tlog\n'
for name in $STREAMS; do
  bits="$VECTORS/$name/stream.bits"
  [ -f "$bits" ] || { echo "missing $bits" >&2; continue; }
  cp "$bits" "$WORK/$name.bits"
  for device in cpu gpu; do
    for run in $(seq 1 "$RUNS"); do
      log="$HOME/tmp/refgpu-$name-$device-$run.log"
      env $([ "$device" = gpu ] && echo CUDA_VISIBLE_DEVICES=0) PYTHONPATH=. \
        python -m src.reco.coders.decoder "$WORK/$name.bits" "$WORK/$name.png" \
        -target_device "$device" --device "$device" >"$log" 2>&1
      total=$(sed -n 's/^TOTAL: //p' "$log" | tail -1)
      md5=$(sed -n 's/^MD5: //p' "$log" | tail -1)
      secs=$(python - "$total" <<'PY'
import sys
try:
    h, m, s = sys.argv[1].strip().split(':')
    print(f'{int(h) * 3600 + int(m) * 60 + float(s):.3f}')
except Exception:
    print('')
PY
)
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$device" "$run" "$secs" "$md5" "$log"
    done
  done
done
