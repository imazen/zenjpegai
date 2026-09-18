# zenjpegai task runner

default: check

check:
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings
    cargo check --no-default-features
    cargo check --no-default-features --target wasm32-unknown-unknown

# no_std + alloc only (no std, no parallel, no avx512): must build clean, no warnings.
no-std:
    cargo check --no-default-features
    cargo clippy --no-default-features -- -D warnings
    cargo check --no-default-features --target wasm32-unknown-unknown

# Upstream reference checkout (models/ + the Python oracle). Override with ZENJPEGAI_REF=...
ref := env_var_or_default("ZENJPEGAI_REF", env_var("HOME") / "work/zen/jpeg-ai-reference-software")

# `unstable-internals` (not reference-tests): every test/example that only needs internal-module
# access, no upstream checkout.
test:
    cargo test --lib --tests --examples --features unstable-internals 2>&1 | tee ~/tmp/zenjpegai-test.log

# Tests that read the upstream checkpoints / reference vectors. Missing data fails loudly.
test-ref:
    ZENJPEGAI_REF={{ref}} cargo test --lib --tests --examples --features reference-tests,zencodec 2>&1 | tee ~/tmp/zenjpegai-test-ref.log

build-release:
    ~/work/zen/scripts/run-heavy -- cargo build --release 2>&1 | tee ~/tmp/zenjpegai-build.log

# GPU backend tests: need an adapter, the checkpoints and the reference vectors (hard failures
# otherwise). Pick the adapter with ZENJPEGAI_GPU_ADAPTER=<name substring>; software rasterisers
# (llvmpipe) are refused unless ZENJPEGAI_GPU_ALLOW_SOFTWARE=1.
gpu-test:
    ZENJPEGAI_REF={{ref}} cargo test -p zenjpegai-gpu --features gpu-tests -- --test-threads 1 --nocapture 2>&1 | tee ~/tmp/zenjpegai-gpu-test.log

# The test suite on wasm32-wasip1 under node (Wasm128 tier, unfused multiply-add policy).
test-wasi:
    CARGO_TARGET_WASM32_WASIP1_RUNNER="node --no-warnings {{justfile_directory()}}/scripts/wasm/run_wasi.mjs" RUSTFLAGS="-Ctarget-feature=+simd128" ZENJPEGAI_REF={{ref}} cargo test --target wasm32-wasip1 --no-default-features --features std,reference-tests,unstable-internals --lib --tests 2>&1 | tee ~/tmp/zenjpegai-test-wasi.log

# Wasm-vs-reference-vs-native 8-bit parity table.
wasm-parity:
    scripts/wasm/parity.sh | tee benchmarks/wasm_parity_$(date +%F).tsv

# Browser build + Playwright suite: wasm packages, npm deps, demo-assets-v1 (needs `gh` auth
# against the private repo today), the servable site, then chromium+firefox+webkit tests.
# See web/README.md for the pieces and what each covers.
web-test:
    web/scripts/build-wasm.sh
    cd web && npm ci
    node web/scripts/fetch-demo-assets.mjs
    node web/scripts/build-site.mjs
    cd web && npx playwright test
