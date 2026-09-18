#!/usr/bin/env python3
"""Roofline classification of the per-dispatch profile TSV.

Each row: kernel key + dispatch params -> FLOPs and ideal DRAM bytes, then
achieved TFLOP/s and GB/s vs the RTX 2080's ~10 TFLOP/s f32 / ~448 GB/s.
"""
import csv, re, sys
from collections import defaultdict

PEAK_TFLOPS = 10.06   # RTX 2080 f32 (68 SM * 128 lanes * 2 FLOP * ~1.8 GHz boost)
PEAK_GBS = 448.0

def taps_avg(k, s):
    # convt: ky runs r, r+s, ... < K for r = phase in 0..s -> mean count per dim
    return sum(((k - 1 - r) // s + 1) for r in range(s)) / s

def classify(kernel, p):
    """-> (flops, bytes) ideal counts for one dispatch."""
    m = re.match(r"conv_k(\d+)x(\d+)_s(\d+)_ob(\d+)", kernel)
    if m and not kernel.startswith("convt"):
        kh, kw, s, ob = map(int, m.groups())
        ih, iw, ic4, oh, ow, oc4, _py, _px, icg4, ocg4 = p[:10]
        macs = oh * ow * ocg4 * icg4 * kh * kw * 16
        b_in = ic4 * ih * iw * 16
        b_out = oc4 * oh * ow * 16
        b_w = icg4 * ocg4 * kh * kw * 16 * 4 + oc4 * 16
        return 2 * macs, b_in + b_out + b_w
    m = re.match(r"convt_k(\d+)_s(\d+)_ob(\d+)", kernel)
    if m:
        k, s, ob = map(int, m.groups())
        ih, iw, ic4, oh, ow, oc4 = p[:6]
        t = taps_avg(k, s) ** 2
        macs = oh * ow * oc4 * ic4 * 16 * t
        return 2 * macs, (ic4 * ih * iw + oc4 * oh * ow) * 16 + ic4 * oc4 * k * k * 64
    if kernel.startswith("depthwise3x3"):
        h, w, c4 = p[:3]
        return 2 * h * w * c4 * 4 * 9, 2 * h * w * c4 * 16 + c4 * 9 * 16
    if kernel == "elu_gate":
        n, _row, n4 = p[:3]
        return n * n4 * 4 * 8, n * n4 * 16 * 3
    if kernel.startswith("pw_"):
        n = p[0]
        ins = {"pw_Relu": 1, "pw_Add": 2, "pw_Gate": 2, "pw_SigmoidMulAdd": 3}[kernel]
        return n * 4 * 4, n * 16 * (ins + 1)
    if kernel == "layer_norm":
        n, _row, _c, c4 = p[:4]
        return n * c4 * 4 * 6, n * c4 * 16 * 3
    if kernel == "gram_chunks":
        n, chunks, pairs = p[:3]
        return pairs * n * 16 * 2, pairs * chunks * 256 * 2 * 16 + pairs * chunks * 64
    if kernel == "gram_reduce":
        chunks, pairs = p[:2]
        return pairs * chunks * 16, pairs * chunks * 64 + pairs * 64
    if kernel == "attention_apply":
        n, _row, src_c4, _v, hc4, oc4 = p[:6]
        return 2 * n * oc4 * hc4 * 16, n * (src_c4 + oc4) * 16
    if kernel.startswith("pixel_shuffle_"):
        oh, ow, _oc, oc4 = p[:4]
        return oh * ow * oc4 * 4, oh * ow * oc4 * 16 * 2
    if kernel == "copy_channels":
        h, w, n4 = p[:3]
        return 0, h * w * n4 * 16 * 2
    if kernel.startswith("emit_"):
        cw, ch = p[:2]
        return cw * ch * 8, cw * ch * (16 + 4)
    if kernel == "emit_output":
        words, _row, n = p[:3]
        return n * 8, n * 4 + words * 4
    if kernel == "yuv_to_rgba":
        w, h = p[:2]
        return w * h * 16, w * h * (12 + 4)
    return 0, 0

def main(path):
    agg = defaultdict(lambda: [0.0, 0.0, 0.0, 0])  # ns, flops, bytes, count
    rows = []
    lines = open(path).read().splitlines()
    hdr = next(i for i, l in enumerate(lines) if l.startswith("kind\t"))
    for r in csv.DictReader(lines[hdr:], delimiter="\t"):
        key = (r["op"], r["width"], r["height"], r["kernel"])
        ns = float(r["ns"])
        p = [int(x) for x in r["params"].split()] if r["params"] else []
        fl, by = classify(r["kernel"], p)
        a = agg[key]
        a[0] += ns; a[1] += fl; a[2] += by; a[3] += 1
        rows.append((r, fl, by))
    cur = None
    for (op, w, h, kernel), (ns, fl, by, n) in sorted(
        agg.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], -kv[1][0])
    ):
        grp = f"{op} {w}x{h}"
        if grp != cur:
            cur = grp
            tot = sum(v[0] for k, v in agg.items() if k[0] == op and k[1] == w and k[2] == h)
            print(f"\n== {grp}  ({tot/1e6:.2f} ms summed) ==")
            print(f"{'kernel':44s} {'n':>3s} {'ms':>8s} {'GFLOP':>8s} {'GB':>7s} "
                  f"{'AI':>7s} {'TF/s':>6s} {'GB/s':>6s} {'%pk':>5s} bound")
        if ns == 0:
            continue
        tf = fl / ns / 1e3  # FLOP/ns -> TFLOP/s
        gb = by / ns        # B/ns -> GB/s
        ai = fl / by if by else float("inf")
        pct = 100 * max(tf / PEAK_TFLOPS, gb / PEAK_GBS)
        bound = "compute" if tf / PEAK_TFLOPS > gb / PEAK_GBS else "mem"
        print(f"{kernel:44s} {n:3d} {ns/1e6:8.3f} {fl/1e9:8.2f} {by/1e9:7.3f} "
              f"{ai:7.1f} {tf:6.2f} {gb:6.0f} {pct:5.1f} {bound}")

main(sys.argv[1])
