#!/usr/bin/env python3
"""Write tests/vectors/gain_scaler.bin: the reference gain-unit scaler for every reachable input.

Run inside the reference venv, from the reference checkout:

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/gen_gain_scaler_vectors.py \\
        ~/work/zen/zenjpegai/tests/vectors/gain_scaler.bin

A decoder-side scaler is `scaler_from_log(gain_vector_log[c] + beta_displacement_log)`
(GainUnit._beta_displacement_log_updated, float32):
    round_half_even(exp(scaler_log * log_k / 128) * 1024) / 1024.

The reachable inputs are every gain_vector_log entry of the eight VM_common_int checkpoints
(Y/UV x 0.002/0.012/0.075/0.5, computed exactly like get_gain_vector_log) times every
beta_displacement_log the header can signal: a 12-bit field minus 2048, i.e. -2048..2047
(dec_flag reads it raw; the -1069..702 clip is encoder-side only, in enc_flag).

File layout (little-endian):
    i32 min_log, u32 n_gain, u32 n_val,
    i16[n_gain]   distinct gain_vector_log values, sorted
    u32[n_val]    f32 bits of the reference scaler for scaler_log = min_log + i
                  (the reachable sums tile [min_log, min_log + n_val) contiguously)
"""
import struct
import sys

import torch

MODELS = "models/VM_common_int"
BETAS = ["0.002", "0.012", "0.075", "0.5"]
LOG_K = 0.20036548400207613  # (ln 54.82 - ln 0.11) / 31, the GainUnit default
BETA_MIN, BETA_MAX = -(1 << 11), (1 << 11) - 1


def gain_vector_log(c: torch.Tensor) -> torch.Tensor:
    """get_gain_vector_log: the one populated column of vr_vec.c in Q7."""
    chs, n_beta = c.shape
    if ((c - 1.0).abs() < 1e-5).all():
        vec_idx = 0
    else:
        means = c.mean(-2)  # _get_min_and_max_beta
        min_n, max_n = 0, n_beta - 1
        if (means == 0).all():
            min_n = max_n = 0
        else:
            for i in range(n_beta - 1):
                if means[i] == 0 and means[i + 1] != 0:
                    min_n = i + 1
                if means[i] != 0 and means[i + 1] == 0:
                    max_n = i
        assert min_n == max_n
        vec_idx = min_n
    return (c[:, vec_idx] * 128).round().int().clamp_min(-(1 << 11))


def main():
    out = sys.argv[1]
    gains = set()
    for comp in ("Y", "UV"):
        for beta in BETAS:
            ck = torch.load(f"{MODELS}/{comp}_{beta}.pth", map_location="cpu")
            gains.update(gain_vector_log(ck["vr_vec.c"]).tolist())
    gains = sorted(gains)

    reachable = sorted({g + b for g in gains for b in range(BETA_MIN, BETA_MAX + 1)})
    min_log = reachable[0]
    assert reachable == list(range(min_log, min_log + len(reachable))), "reachable set not contiguous"

    # _beta_displacement_log_updated: int32 log tensor * python float -> f32, exp in f32,
    # round half to even at 10 fractional bits.
    t = torch.tensor(reachable, dtype=torch.int32)
    scaler = (torch.exp(t * LOG_K / 128.0) * 1024).round() / 1024

    with open(out, "wb") as f:
        f.write(struct.pack("<iII", min_log, len(gains), len(reachable)))
        f.write(struct.pack(f"<{len(gains)}h", *gains))
        f.write(scaler.numpy().tobytes())
    print(f"{out}: {len(gains)} gain values, {len(reachable)} reachable scaler_log "
          f"values in [{min_log}, {reachable[-1]}]")


if __name__ == "__main__":
    main()
