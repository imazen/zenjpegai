# zenjpegai — project instructions

Port of the JPEG AI reference software ("VM") to safe Rust. Goal set by the user 2026-09-17:
"port https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software to Rust, using archmage and
magetypes and forbidunsafe, but make it faster".

Read `PORTING.md` first: it is the honest status table (module → upstream source → parity gate).

## Hard rules for this repo

- `#![forbid(unsafe_code)]`. SIMD only through archmage / magetypes.
- The entropy stage (container, headers, me-tANS, z decode, hyper-scale-decoder, sigma indices,
  residual symbols) is integer-only and MUST match the reference bit-exactly. Any mismatch is a bug.
- Float networks (hyper-decoder, MCM, synthesis, post-filter nets): every SIMD tier must produce
  bit-identical output to the scalar tier. Vectorise across independent lanes (output channels,
  pixels) and keep the per-element op sequence fixed, with FMA everywhere (`f32::mul_add` in
  scalar code). Never use a vector-width-dependent reduction order.
  **wasm32 exception:** WebAssembly has no deterministic FMA, so on `target_arch = "wasm32"` the
  multiply-add is unfused in every tier and in `nn::reference` (`nn::fmadd`). Wasm output is
  bit-identical across wasm tiers / engines / thread counts and differs from native by rounding
  only (numbers in `PORTING.md`). Never use `f32::mul_add` or relaxed-simd madd in wasm code paths.
- Versus PyTorch the float path cannot be bit-exact (oneDNN's summation order is not ours);
  parity there is measured (max abs error on tensors, pixel diff histogram on output) and
  recorded in `PORTING.md`. Do not loosen a recorded bound without asking the user.
- Weights (`*.pth`) and reference vectors never go in git. Upstream checkpoints are read
  directly from a models directory.
- Imazen-only imaging software in tools/tests (zenpng for PNG IO). The reference software itself
  (Python) is the differential oracle and lives outside this repo.

## Reference environment (verified 2026-09-17 on `dev`, Ubuntu, kernel 7.0, glibc refuses execstack)

Upstream clone: `~/work/zen/jpeg-ai-reference-software` (commit `b9e573f`, LFS fully pulled, 3.4 GB).

```bash
cd ~/work/zen/jpeg-ai-reference-software
uv venv --python 3.8 .venv && . .venv/bin/activate
uv pip install --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
   "torch==1.10.2+cpu" "torchvision==0.11.3+cpu" "numpy==1.19.5"
uv pip install "setuptools==59.5.0" wheel addict==2.2.1 attrs decorator==4.4.2 einops==0.5.0 GPUtil==1.4.0 \
   "opencv-python-headless==4.5.5.62" openpyxl packaging==21.3 "pandas==1.3.5" prettytable==0.7.2 \
   "protobuf==3.20.*" psutil ptflops==0.6.5 pybind11 pynvml==8.0.4 pytorch-msssim==0.2.1 "scipy==1.5.4" \
   commentjson==0.9.0 bjontegaard==1.1.0 fsspec==2022.11.0
```

Gotchas, all hit for real:

- torch 1.10.2's `libtorch_cpu.so` is marked exec-stack; modern glibc refuses to dlopen it
  ("cannot enable executable stack"). Fix: clear the `PF_X` bit of its `PT_GNU_STACK` program
  header (tiny Python ELF patch; `scripts/ref_env/clear_execstack.py`).
- The upstream Makefiles call `python3-config`, absent in a uv venv. Build the two pybind11
  extensions by hand:
  `g++ -O3 -std=c++14 -fopenmp -shared -fPIC $(python -m pybind11 --includes) ans.cpp compressor.cpp decompressor.cpp -o ../../lib_wrappers/mans/ans$(python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")`
  (same for `direct/` → `lib_wrappers/direct/ec_direct…so`).
- Scripts under `scripts/` need `PYTHONPATH=.`.
- CPU run: add `-target_device cpu`. Encoder:
  `python -m src.reco.coders.encoder IN.png OUT.bits --set_target_bpp 50 --cfg cfg/tools_off.json cfg/profiles/base.json -target_device cpu`;
  decoder: `python -m src.reco.coders.decoder IN.bits OUT.png -target_device cpu`.
- `scripts/bitstream_probe.py X.bits` prints every header field by name — the quickest syntax oracle.
- `--cfg cfg/AE/verbose.json` prints per-substream sizes + MD5 and syntax tables.

Baseline timing (560x888 test image 00030, BOP, tools off, 0.5 bpp, CPU, torch 1.10.2, 16 threads):
decoder "TOTAL" 0.22 s, process wall 1.4 s (Python start + model load dominate).

## Parallel agents (read this if you were spawned for one task)

- **Work in your own sibling jj workspace**, never in `~/work/zen/zenjpegai` itself:
  `cd ~/work/zen/zenjpegai && jj git fetch && jj workspace add --name <slug> ../zenjpegai--<slug> -r main@origin`,
  then work in `~/work/zen/zenjpegai--<slug>`. Write `.workongoing` there and refresh it.
- **Touch only the files your brief lists as yours.** Shared files (`Cargo.toml`, `src/lib.rs`,
  `src/decoder/api.rs`, `PORTING.md`, `CHANGELOG.md`, `README.md`, `justfile`) get minimal,
  additive edits (one line where possible), because other agents edit them too.
- **Land small commits often:** `jj describe -m ...`, `jj git fetch`, `jj rebase -d main@origin`
  (resolve conflicts if any), re-run the gates, `jj bookmark set main -r @`,
  `jj git push --bookmark main`, then verify with
  `git merge-base --is-ancestor $(jj log -r main --no-graph -T commit_id) origin/main`.
  After a push jj opens a new empty change by itself. Never force-push, never
  `--allow-backwards`. If you see other agents' commits, rebase onto them; never `jj op restore`
  past them.
- **Gates before every push:** `cargo fmt --check`,
  `cargo clippy --all-targets --all-features -- -D warnings`,
  `ZENJPEGAI_REF=~/work/zen/jpeg-ai-reference-software cargo test --lib --tests --all-features -- --skip tiers_and_threads`
  (run the skipped test too if you touched `src/nn` or `src/model`). Prefix heavy commands
  with `nice -n 19`, cap cargo at `-j 8`: the box is shared.
- **Parity is measured against the reference software**, never against your own expectations.
  Oracle data: `/mnt/v/output/zenjpegai/reference/vectors/<name>/` (stream, reference tensors
  dump, decoded PNG; for region streams use `fixed_decoder/`). Generate more with
  `scripts/ref_vectors/make_reference_streams.sh` (add a set; keep scripts committed). To dump
  more intermediate tensors extend `scripts/ref_vectors/dump_decode.py` additively (new names
  only) and re-run it into a **new** subdirectory of the vector (do not overwrite existing
  dumps other agents read).
- Integer / table-driven stages must match the reference exactly. Float stages: state the
  measured max abs error in `PORTING.md`; whole-picture gate is "8-bit output differs by at most
  1 in fewer than 1/5000 samples" (`tests/decode_ref.rs`). Never loosen an existing bound.
- Scratch goes to `~/tmp`, never `/tmp`. Nothing above 30 KB and no binaries into git.
- When done: update `PORTING.md` (status row + what is still missing, missing first),
  `CHANGELOG.md`, push, then `cd ~/work/zen/zenjpegai && jj workspace forget <slug>` and
  `rm -rf ~/work/zen/zenjpegai--<slug>`. If the task turns out larger than expected, land the
  part that is proven, document the rest precisely in `PORTING.md`, and say so plainly in your
  final report: an honest partial beats a claimed complete.

## Known Bugs

(none recorded yet)
