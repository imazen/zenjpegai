#!/usr/bin/env bash
# Build the browser wasm packages into web/dist/:
#   pkg-simd/     single-threaded SIMD128 build: runs anywhere (no SharedArrayBuffer needed)
#   pkg-threads/  rayon on Web Workers: needs a cross-origin isolated page
# Both are built with a pinned nightly and -Z build-std (threads need it; simd is faster for it).
# usage: web/scripts/build-wasm.sh [simd] [threads]      (default: both)
set -euo pipefail
cd "$(dirname "$0")/../.."
ROOT=$PWD
NIGHTLY=${ZENJPEGAI_NIGHTLY:-nightly-2026-09-02}
JOBS=${CARGO_BUILD_JOBS:-8}
TOOLS=$ROOT/web/.tools
DIST=$ROOT/web/dist
variants=("$@"); [ ${#variants[@]} -eq 0 ] && variants=(simd threads)

[ -f Cargo.lock ] || cargo generate-lockfile
[ -f Cargo.lock ] || cargo generate-lockfile
# wasm-bindgen's CLI must match the crate version in Cargo.lock exactly.
want=$(awk '/^name = "wasm-bindgen"$/ {getline; gsub(/version = |"/, ""); print; exit}' Cargo.lock)
bindgen=$(command -v wasm-bindgen || true)
[ -x "$TOOLS/bin/wasm-bindgen" ] && bindgen=$TOOLS/bin/wasm-bindgen
if [ -z "$bindgen" ] || [ "$("$bindgen" --version | awk '{print $2}')" != "$want" ]; then
  echo "installing wasm-bindgen-cli $want into $TOOLS" >&2
  if command -v cargo-binstall >/dev/null; then
    cargo binstall -y --root "$TOOLS" "wasm-bindgen-cli@$want" >&2
  else
    cargo install --locked --root "$TOOLS" wasm-bindgen-cli --version "$want" >&2
  fi
  bindgen=$TOOLS/bin/wasm-bindgen
fi
wasm_opt=$ROOT/web/node_modules/.bin/wasm-opt
[ -x "$wasm_opt" ] || { echo "run 'npm ci' in web/ first (binaryen provides wasm-opt)" >&2; exit 1; }

for v in "${variants[@]}"; do
  case $v in
    simd)
      # Nightly + build-std here too: the prebuilt std has neither simd128 nor bulk-memory, and
      # its byte-loop memcpy / memset cost a third of the decode time (measured 500 -> 371 ms).
      RUSTFLAGS="-Ctarget-feature=+simd128,+bulk-memory,+nontrapping-fptoint,+sign-ext,+mutable-globals" \
        nice -n 19 cargo +"$NIGHTLY" build -j "$JOBS" -p zenjpegai-wasm --target wasm32-unknown-unknown \
        --profile wasm-release --target-dir target/wasm-simd -Z build-std=panic_abort,std >&2
      in=target/wasm-simd/wasm32-unknown-unknown/wasm-release/zenjpegai_wasm.wasm ;;
    threads)
      # Shared memory: 1 GiB maximum (a shared memory must declare one).
      RUSTFLAGS="-Ctarget-feature=+simd128,+atomics,+bulk-memory,+mutable-globals,+nontrapping-fptoint,+sign-ext -Clink-arg=--max-memory=2147483648" \
        nice -n 19 cargo +"$NIGHTLY" build -j "$JOBS" -p zenjpegai-wasm --features threads \
        --target wasm32-unknown-unknown --profile wasm-release --target-dir target/wasm-threads \
        -Z build-std=panic_abort,std >&2
      in=target/wasm-threads/wasm32-unknown-unknown/wasm-release/zenjpegai_wasm.wasm ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
  out=$DIST/pkg-$v
  rm -rf "$out"; mkdir -p "$out"
  "$bindgen" --target web --no-typescript --out-dir "$out" --out-name zenjpegai "$in"
  raw=$(stat -c %s "$out/zenjpegai_bg.wasm")
  "$wasm_opt" -O3 "$out/zenjpegai_bg.wasm" -o "$out/zenjpegai_bg.wasm"
  opt=$(stat -c %s "$out/zenjpegai_bg.wasm")
  gz=$(gzip -9c "$out/zenjpegai_bg.wasm" | wc -c)
  br=$(node -e 'const z=require("node:zlib"),f=require("node:fs");process.stdout.write(String(z.brotliCompressSync(f.readFileSync(process.argv[1]),{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}}).length))' "$out/zenjpegai_bg.wasm")
  printf 'pkg-%s\twasm-bindgen %s bytes\twasm-opt -O3 %s bytes\tgzip -9 %s\tbrotli -q11 %s\n' "$v" "$raw" "$opt" "$gz" "$br"
done
