# zenjpegai task runner

default: check

check:
    cargo fmt --check
    cargo clippy --all-targets --all-features -- -D warnings

test:
    cargo test --all-targets 2>&1 | tee ~/tmp/zenjpegai-test.log

build-release:
    ~/work/zen/scripts/run-heavy -- cargo build --release 2>&1 | tee ~/tmp/zenjpegai-build.log
