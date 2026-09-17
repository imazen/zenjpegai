#!/usr/bin/env bash
# Wasm-vs-reference parity: decode every reference vector with the wasm32-wasip1 CLI under node
# and with the native CLI, and count differing 8-bit samples against the reference decoder's
# PNG (`fixed_decoder/decoded.png` for region streams, see PORTING.md) and against each other.
#   scripts/wasm/parity.sh > benchmarks/wasm_parity_$(date +%F).tsv
set -euo pipefail
cd "$(dirname "$0")/../.."
REF=${ZENJPEGAI_REF:-$HOME/work/zen/jpeg-ai-reference-software}
VEC=${ZENJPEGAI_VECTORS:-/mnt/v/output/zenjpegai/reference/vectors}
OUT=${OUT:-$HOME/tmp/zenjpegai-wasm-parity}
mkdir -p "$OUT"
RUSTFLAGS="-Ctarget-feature=+simd128" nice -n 19 cargo build -j 8 --release --target wasm32-wasip1 \
  --no-default-features --features cli --bin zenjpegai >&2
nice -n 19 cargo build -j 8 --release --features cli --bin zenjpegai >&2
WASM=target/wasm32-wasip1/release/zenjpegai.wasm
NATIVE=target/release/zenjpegai
echo "# commit $(git rev-parse --short HEAD 2>/dev/null || jj log -r @- --no-graph -T 'commit_id.short()') host $(hostname) node $(node --version)"
echo "# columns: differing samples / total / max abs diff; wasm = Wasm128 tier, unfused multiply-add"
printf 'stream\ttotal_samples\twasm_vs_ref\twasm_vs_ref_max\tnative_vs_ref\tnative_vs_ref_max\twasm_vs_native\twasm_vs_native_max\twasm_scalar_identical\n'
for dir in "$VEC"/*/; do
  name=$(basename "$dir")
  ref="$dir/decoded.png"
  [ -f "$dir/fixed_decoder/decoded.png" ] && ref="$dir/fixed_decoder/decoded.png"
  if ! node --no-warnings scripts/wasm/run_wasi.mjs $WASM decode "$dir/stream.bits" "$OUT/$name.wasm.png" --models "$REF/models" 2>"$OUT/$name.wasm.err"; then
    printf '%s\tunsupported: %s\n' "$name" "$(tr '\n' ' ' <"$OUT/$name.wasm.err" | cut -c1-120)"
    continue
  fi
  node --no-warnings scripts/wasm/run_wasi.mjs $WASM decode "$dir/stream.bits" "$OUT/$name.wasm-scalar.png" --models "$REF/models" --scalar --single-thread
  $NATIVE decode "$dir/stream.bits" "$OUT/$name.native.png" --models "$REF/models"
  same=no; cmp -s "$OUT/$name.wasm.png" "$OUT/$name.wasm-scalar.png" && same=yes
  IFS=$'\t' read -r wr total wrm _ < <(node scripts/wasm/png_diff.mjs "$OUT/$name.wasm.png" "$ref")
  IFS=$'\t' read -r nr _ nrm _ < <(node scripts/wasm/png_diff.mjs "$OUT/$name.native.png" "$ref")
  IFS=$'\t' read -r wn _ wnm _ < <(node scripts/wasm/png_diff.mjs "$OUT/$name.wasm.png" "$OUT/$name.native.png")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$total" "$wr" "$wrm" "$nr" "$nrm" "$wn" "$wnm" "$same"
done
