#!/usr/bin/env python3
"""Generate me-tANS parity vectors with the reference software's C++ extension.

Run inside the reference venv, from the reference checkout:

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/gen_mans_vectors.py ~/work/zen/zenjpegai/tests/vectors/mans

File format (little-endian), calls listed in DECODE order:
    "ZJMV" u32 version=1, u32 kind (0 residual, 1 z), u32 num_threads, u32 num_calls
    residual call: u32 len, u8 sigma[len], u8 mask[len], i16 values[len]
    z call:        u32 channels, u32 size_per_channel, u8 cdfs[channels*63], u8 symbols[channels*size]
    u32 num_threads, u32 thread_sizes[num_threads], u32 payload_len, u8 payload[payload_len]
"""
import json
import struct
import sys

import numpy as np

sys.path.insert(0, ".")
from src.codec.entropy_coding.lib_wrappers.mans import ans  # noqa: E402
from src.codec.entropy_coding.lib_wrappers.mans.utils import (  # noqa: E402
    get_cdf_matrix, get_decode_transitions, get_encode_transitions, get_state_maps)
import re  # noqa: E402

src = open("src/codec/entropy_coding/lib_wrappers/mans/ec_lib_mans.py").read()
pdf_r = eval(re.search(r"pdf_r=(\[\[.*?\]\s*\])\s*self\.pdf_r", src, re.S).group(1))
bounds_l = eval(re.search(r"bound_table_r=(\[.*?\])", src).group(1))
pmf = np.zeros([32, 256], dtype=np.int64)
for i in range(32):
    pmf[i, : len(pdf_r[i])] = pdf_r[i]
    pmf[i, bounds_l[i] * 2 - 1] = 1
cdf = get_cdf_matrix(pmf)
ENC = get_encode_transitions(pmf, cdf)
SM = get_state_maps(cdf)
DEC = get_decode_transitions(pmf, cdf)
BOUNDS = np.array(bounds_l, dtype=np.uint8)


def encode(kind, num_threads, calls):
    enc = ans.ANSEncoder(1 << 22, num_threads)
    enc.set_sgm_transitions(ENC.data, BOUNDS.data, SM.data)
    for c in reversed(calls):
        if kind == 0:
            sigma, mask, values = c
            enc.encode_sgm(sigma.copy(), values.copy(), mask.copy())
        else:
            cdfs, symbols = c
            enc.encode_factorize(cdfs.copy(), symbols.copy())
    total = enc.close()
    mem = np.zeros(total, dtype=np.uint8)
    enc.get_memory(mem)
    sizes = np.zeros(num_threads, dtype=np.uint32)
    enc.get_thread_sizes(sizes)
    assert int(sizes.sum()) == total
    # sanity: the reference decoder gives the input back
    dec = ans.ANSDecoder(mem, np.cumsum(sizes).astype(np.uint32))
    dec.set_sgm_transitions(DEC.data, BOUNDS.data)
    for c in calls:
        if kind == 0:
            sigma, mask, values = c
            out = np.zeros(values.shape, dtype=np.int16)
            dec.decode_sgm(sigma, out, mask)
            assert (out[mask] == values[mask]).all()
        else:
            cdfs, symbols = c
            out = np.zeros(symbols.shape, dtype=np.uint8)
            dec.decode_factorize(cdfs, out)
            assert (out == symbols).all(), (out, symbols)
    return mem, sizes


def write(path, kind, num_threads, calls):
    mem, sizes = encode(kind, num_threads, calls)
    with open(path, "wb") as f:
        f.write(b"ZJMV" + struct.pack("<IIII", 1, kind, num_threads, len(calls)))
        for c in calls:
            if kind == 0:
                sigma, mask, values = c
                f.write(struct.pack("<I", len(values)))
                f.write(sigma.tobytes() + mask.astype(np.uint8).tobytes() + values.astype("<i2").tobytes())
            else:
                cdfs, symbols = c
                f.write(struct.pack("<II", cdfs.shape[0], symbols.shape[1]))
                f.write(cdfs.tobytes() + symbols.tobytes())
        f.write(struct.pack("<I", num_threads) + sizes.astype("<u4").tobytes())
        f.write(struct.pack("<I", len(mem)) + mem.tobytes())
    print(path, "threads", num_threads, "calls", len(calls), "payload", len(mem))


def residual_call(rng, n, mask_p, wide):
    sigma = rng.integers(0, 32, n).astype(np.uint8)
    mask = rng.random(n) < mask_p
    scale = np.array([0.3 + 1.5 ** (s / 3.0) for s in sigma])
    values = np.rint(rng.normal(0, 1, n) * scale)
    if wide:  # exercise both escape widths, including the int16 extremes
        k = rng.integers(0, n, max(1, n // 16))
        values[k] = rng.choice([-32768, -32767, -129, -128, 127, 128, 200, 32767, 5000, -5000], len(k))
    return sigma, mask, np.clip(values, -32768, 32767).astype(np.int16)


def z_call(rng, channels, size, zero_mass):
    freqs = rng.integers(0 if zero_mass else 1, 2000, (channels, 63)).astype(np.int64)
    if zero_mass:
        freqs[:, rng.integers(0, 63, 20)] = 0
        freqs[:, 31] += 1
    cs = np.cumsum(freqs, axis=1)
    cdfs = ((cs * 255 + (cs[:, -1:] >> 1)) // cs[:, -1:]).astype(np.uint8)
    symbols = rng.integers(0, 63, (channels, size)).astype(np.uint8)
    return cdfs, symbols


def main(out_dir):
    rng = np.random.default_rng(20260917)
    # residual: every tail shape (len % 4), with and without masks, 1..16 threads
    for t in [1, 2, 4, 8, 16]:
        calls = [residual_call(rng, n, p, w) for n, p, w in
                 [(16 * t, 1.0, False), (4 * t * 3, 0.6, True), (1, 1.0, False), (2, 1.0, True), (3, 0.5, True),
                  (4 * t + 1, 0.9, True), (4 * t + 2, 0.9, False), (4 * t + 3, 0.3, True), (5 * t + 3, 1.0, True),
                  (65, 0.0, False), (255, 0.8, True)]]
        write(f"{out_dir}/residual_t{t}.bin", 0, t, calls)
    for t in [1, 2, 4, 8, 16]:
        calls = [z_call(rng, c, s, zm) for c, s, zm in
                 [(3, 8 * t, False), (5, 1, False), (2, 2, True), (4, 3, True), (3, 4 * t + 1, True),
                  (2, 4 * t + 2, False), (2, 4 * t + 3, True), (5, 5 * t + 3, True), (2, 253, True)]]
        write(f"{out_dir}/z_t{t}.bin", 1, t, calls)


if __name__ == "__main__":
    main(sys.argv[1])
