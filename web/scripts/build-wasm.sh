#!/usr/bin/env bash
# Build the browser wasm packages into web/dist/:
#   pkg-simd/     single-threaded SIMD128 build: runs anywhere (no SharedArrayBuffer needed)
#   pkg-threads/  rayon on Web Workers: needs a cross-origin isolated page
#   pkg-webgpu/   simd build + WebGPU synthesis (zenjpegai-gpu): async initGpu/decode/present,
#                 falls back to the CPU engine when the browser offers no non-software adapter
#   pkg-webgpu-threads/  webgpu build + rayon: on isolated pages the CPU entropy/latent stages
#                 run on the thread pool too — the serial single-threaded latent stage is the
#                 dominant cost of the single-threaded webgpu package (~120 ms of a ~250 ms
#                 1 MP decode on RTX 2080/Dawn; measured 2026-09-18)
# All are built with a pinned nightly and -Z build-std (threads need it; simd is faster for it).
# usage: web/scripts/build-wasm.sh [simd] [threads] [webgpu] [webgpu-threads]  (default: simd threads)
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
      # Shared memory: 1 GiB maximum (a shared memory must declare one), and wasm-ld needs
      # --shared-memory explicitly — +atomics plus --max-memory alone still links a NON-shared
      # memory, which is not structured-cloneable: wasm-bindgen-rayon's own worker spawner
      # (workerHelpers.no-bundler.js, `worker.postMessage({memory, ...})`) then throws
      # "Failed to execute 'postMessage' on 'Worker': #<Memory> could not be cloned" the moment
      # initThreadPool tries to spin up a second worker (caught 2026-09-17 running this build in
      # a real browser for the first time; see web/README.md).
      RUSTFLAGS="-Ctarget-feature=+simd128,+atomics,+bulk-memory,+mutable-globals,+nontrapping-fptoint,+sign-ext -Clink-arg=--max-memory=2147483648 -Clink-arg=--shared-memory -Clink-arg=--import-memory -Clink-arg=--export=__wasm_init_tls -Clink-arg=--export=__tls_size -Clink-arg=--export=__tls_align -Clink-arg=--export=__tls_base" \
        nice -n 19 cargo +"$NIGHTLY" build -j "$JOBS" -p zenjpegai-wasm --features threads \
        --target wasm32-unknown-unknown --profile wasm-release --target-dir target/wasm-threads \
        -Z build-std=panic_abort,std >&2
      in=target/wasm-threads/wasm32-unknown-unknown/wasm-release/zenjpegai_wasm.wasm ;;
    webgpu)
      # Same flags as simd: the GPU feature only adds the async exports and the wgpu
      # synthesis path; the CPU fallback in this package is the single-threaded engine.
      RUSTFLAGS="-Ctarget-feature=+simd128,+bulk-memory,+nontrapping-fptoint,+sign-ext,+mutable-globals" \
        nice -n 19 cargo +"$NIGHTLY" build -j "$JOBS" -p zenjpegai-wasm --features gpu \
        --target wasm32-unknown-unknown --profile wasm-release --target-dir target/wasm-webgpu \
        -Z build-std=panic_abort,std >&2
      in=target/wasm-webgpu/wasm32-unknown-unknown/wasm-release/zenjpegai_wasm.wasm ;;
    webgpu-threads)
      # threads flags + the gpu feature: WebGPU synthesis with the rayon CPU stages.
      RUSTFLAGS="-Ctarget-feature=+simd128,+atomics,+bulk-memory,+mutable-globals,+nontrapping-fptoint,+sign-ext -Clink-arg=--max-memory=2147483648 -Clink-arg=--shared-memory -Clink-arg=--import-memory -Clink-arg=--export=__wasm_init_tls -Clink-arg=--export=__tls_size -Clink-arg=--export=__tls_align -Clink-arg=--export=__tls_base" \
        nice -n 19 cargo +"$NIGHTLY" build -j "$JOBS" -p zenjpegai-wasm --features "gpu threads" \
        --target wasm32-unknown-unknown --profile wasm-release --target-dir target/wasm-webgpu-threads \
        -Z build-std=panic_abort,std >&2
      in=target/wasm-webgpu-threads/wasm32-unknown-unknown/wasm-release/zenjpegai_wasm.wasm ;;
    *) echo "unknown variant $v" >&2; exit 2 ;;
  esac
  out=$DIST/pkg-$v
  rm -rf "$out"; mkdir -p "$out"
  "$bindgen" --target web --no-typescript --out-dir "$out" --out-name zenjpegai "$in"
  # wasm-bindgen-rayon's spawned rayon workers init the module with the deprecated positional
  # signature (`pkg.default(module, memory)` warns once per child worker); rewrite to the
  # object form in the generated snippet.
  for helper in "$out"/snippets/wasm-bindgen-rayon-*/src/workerHelpers.no-bundler.js; do
    [ -f "$helper" ] && sed -i \
      's/pkg\.default(data\.module, data\.memory)/pkg.default({ module_or_path: data.module, memory: data.memory })/' \
      "$helper"
  done
  # Collapse the per-child JS requests: upstream, every rayon child fetches the workerHelpers
  # snippet AND the main glue (~50 requests for a 16-thread pool). pack-rayon-child.mjs
  # inlines startWorkers into the glue and emits a self-contained rayon-child.js, cloned into
  # every child through one shared blob: URL — 2 requests for the whole pool.
  if [ -d "$out/snippets" ]; then
    node "$ROOT/web/scripts/pack-rayon-child.mjs" "$out"
  fi
  raw=$(stat -c %s "$out/zenjpegai_bg.wasm")
  "$wasm_opt" -O3 "$out/zenjpegai_bg.wasm" -o "$out/zenjpegai_bg.wasm"
  opt=$(stat -c %s "$out/zenjpegai_bg.wasm")
  gz=$(gzip -9c "$out/zenjpegai_bg.wasm" | wc -c)
  br=$(node -e 'const z=require("node:zlib"),f=require("node:fs");process.stdout.write(String(z.brotliCompressSync(f.readFileSync(process.argv[1]),{params:{[z.constants.BROTLI_PARAM_QUALITY]:11}}).length))' "$out/zenjpegai_bg.wasm")
  printf 'pkg-%s\twasm-bindgen %s bytes\twasm-opt -O3 %s bytes\tgzip -9 %s\tbrotli -q11 %s\n' "$v" "$raw" "$opt" "$gz" "$br"
done
