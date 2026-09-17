# zenjpegai task runner

default: check

check:
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings

# Upstream reference checkout (models/ + the Python oracle). Override with ZENJPEGAI_REF=...
ref := env_var_or_default("ZENJPEGAI_REF", env_var("HOME") / "work/zen/jpeg-ai-reference-software")

test:
    cargo test --lib --tests --examples 2>&1 | tee ~/tmp/zenjpegai-test.log

# Tests that read the upstream checkpoints / reference vectors. Missing data fails loudly.
test-ref:
    ZENJPEGAI_REF={{ref}} cargo test --lib --tests --examples --features reference-tests 2>&1 | tee ~/tmp/zenjpegai-test-ref.log

build-release:
    ~/work/zen/scripts/run-heavy -- cargo build --release 2>&1 | tee ~/tmp/zenjpegai-build.log

# GPU backend tests: need an adapter, the checkpoints and the reference vectors (hard failures
# otherwise). Pick the adapter with ZENJPEGAI_GPU_ADAPTER=<name substring>; software rasterisers
# (llvmpipe) are refused unless ZENJPEGAI_GPU_ALLOW_SOFTWARE=1.
gpu-test:
    ZENJPEGAI_REF={{ref}} cargo test -p zenjpegai-gpu --features gpu-tests -- --test-threads 1 --nocapture 2>&1 | tee ~/tmp/zenjpegai-gpu-test.log
