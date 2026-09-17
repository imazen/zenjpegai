# zenjpegai task runner

default: check

check:
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings

# Upstream reference checkout (models/ + the Python oracle). Override with ZENJPEGAI_REF=...
ref := env_var_or_default("ZENJPEGAI_REF", env_var("HOME") / "work/zen/jpeg-ai-reference-software")

test:
    cargo test --all-targets 2>&1 | tee ~/tmp/zenjpegai-test.log

# Tests that read the upstream checkpoints / reference vectors. Missing data fails loudly.
test-ref:
    ZENJPEGAI_REF={{ref}} cargo test --all-targets --features reference-tests 2>&1 | tee ~/tmp/zenjpegai-test-ref.log

build-release:
    ~/work/zen/scripts/run-heavy -- cargo build --release 2>&1 | tee ~/tmp/zenjpegai-build.log
