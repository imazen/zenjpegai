# Mission: port JPEG AI reference to safe, fast, bit-exact Rust

**Enduring goal.** You are continuing a long-running port of the JPEG AI reference implementation (revision `b9e573f`) into production-grade Rust in this repository. The mission has two inseparable halves that must both hold at all times:

1. **Complete parity** — every layer, operating point, codec tool, and behavior of the reference is ported and bit-exact *by default*. Nothing may remain unported. `nn::reference` is the semantic oracle; `nn::fast` is the performance path; `decode_ref`/`encode_ref`/`codec_ref` tests prove it.
2. **Maximal speed** — the fast path should be as close to hardware limits as possible without breaking parity. Exact is the default; approximations are opt-in only (`Precision::Approximate`, env knobs).

**Invariants you may never break:**

- `#![forbid(unsafe_code)]`. SIMD goes through `archmage` + `magetypes` (`#[arcane]`/`#[rite]`, `SimdF32`, `incant!`) — never platform-specific intrinsics, even temporarily (experiments must be opt-in + reverted).
- Exact behavior is the default. Any approximation (Winograd F(2,3), etc.) is gated behind `Precision` or a `ZENJPEGAI_*` env knob and never changes the default output.
- `jj` on `main`, commit in small verified units, push each landing. Never `git reset`/`git checkout --`/`git restore` over uncommitted work.
- Memory: `BTensor`/`scratch`/`border_buf` use the bounded ledger/thread-local scratch — no unbounded allocation on the hot path.

**Current state (work backward from `WORKPLAN.md` + `PORTING.md`, not this text).** The packed-tile BLIS driver (`forward_blis`, `src/nn/fast/conv.rs`) is the default dense-conv path — ~5-6 % faster encode, bit-identical. Opt-in experiments: `ZENJPEGAI_CONV_{S2_PHASE,SWEEP,OCB2,NOBLIS}`. `coz` feature + markers exist. `gpu/` has a wgpu/WGSL **decode** backend (SOP/BOP/HOP synthesis); `wasm/` decodes in-browser bit-identically. Perf evidence: the dense conv is ~64 % backend-bound, ~29 % memory + ~35 % core; weight-per-ocb L1 residency is the linchpin every reorder must preserve.

**What "keep going" means.** Do not pause at a milestone, summarize-and-stop, or ask permission to continue. A finished chunk is the start of the next. When you hit a context limit, your context is summarized but your goal persists — reload state from the repo (`git log`, `WORKPLAN.md`, `PORTING.md`), not from memory, and continue the open task. If a subtask seems infeasible: keep it in the plan, tell the user what you tried + what's blocking + the next attempt — never silently drop it. Benchmark everything before declaring a win; a theoretically-good optimization that measures slower is kept opt-in, not made default.

**Open work arc (priority):** encoder/analysis on the wgpu backend (kernels exist — `conv3x3_s2`, attention, MCM; needs `gpu/src/analysis.rs` + `gpu/src/encoder.rs` + GPU-vs-reference-encode bit-exactness), f16 shaders (B3), wasm encoder. The remaining port-coverage items live in `PORTING.md`'s table.

**First action each session:** read `WORKPLAN.md` + `PORTING.md` + `git log -5`, check for uncommitted work, claim `.workongoing`, then resume the in-progress item — not a new idea.
