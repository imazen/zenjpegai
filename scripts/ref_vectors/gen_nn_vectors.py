#!/usr/bin/env python3
"""Tiny layer-level vectors computed by PyTorch (the reference's torch 1.10.2), for tests/nn_vectors.rs.

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/gen_nn_vectors.py ~/work/zen/zenjpegai/tests/vectors/nn

Bundle format ("ZJTB", little-endian): u32 count, then per tensor
    u16 name_len, name bytes, u8 ndim, u32 dims[ndim], f32 data[prod(dims)]
"""
import struct
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(20260917)
torch.set_num_threads(1)


def write(path, tensors):
    with open(path, "wb") as f:
        f.write(b"ZJTB" + struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            t = t.detach().to(torch.float32).contiguous()
            f.write(struct.pack("<H", len(name)) + name.encode())
            f.write(struct.pack("<B", t.dim()) + struct.pack(f"<{t.dim()}I", *t.shape))
            f.write(t.numpy().astype("<f4").tobytes())


def conv_case(out_dir, name, cin, cout, k, stride, pad, groups, bias, h, w):
    kh, kw = k if isinstance(k, tuple) else (k, k)
    x = torch.randn(1, cin, h, w)
    wt = torch.randn(cout, cin // groups, kh, kw) * 0.3
    b = torch.randn(cout) if bias else None
    y = F.conv2d(x, wt, b, stride=stride, padding=pad, groups=groups)
    t = {"x": x[0], "w": wt, "y": y[0]}
    if bias:
        t["b"] = b
    write(f"{out_dir}/{name}.bin", t)


def convt_case(out_dir, name, cin, cout, k, stride, pad, out_pad, bias, h, w):
    x = torch.randn(1, cin, h, w)
    wt = torch.randn(cin, cout, k, k) * 0.3
    b = torch.randn(cout) if bias else None
    y = F.conv_transpose2d(x, wt, b, stride=stride, padding=pad, output_padding=out_pad)
    t = {"x": x[0], "w": wt, "y": y[0]}
    if bias:
        t["b"] = b
    write(f"{out_dir}/{name}.bin", t)


def main(out_dir):
    conv_case(out_dir, "conv3x3_s1_p1_bias", 8, 6, 3, 1, 1, 1, True, 7, 9)
    conv_case(out_dir, "conv3x3_s1_p1_g4_nobias", 16, 16, 3, 1, 1, 4, False, 6, 5)
    conv_case(out_dir, "conv3x3_depthwise", 12, 12, 3, 1, 1, 12, False, 5, 7)
    conv_case(out_dir, "conv1x1_nobias", 10, 7, 1, 1, 0, 1, False, 4, 6)
    conv_case(out_dir, "conv3x3_s2_p1_bias", 5, 8, 3, 2, 1, 1, True, 9, 8)
    conv_case(out_dir, "conv2x2_s1_p0", 6, 8, 2, 1, 0, 1, False, 6, 7)
    conv_case(out_dir, "conv1x3_p01_bias", 4, 4, (1, 3), 1, (0, 1), 1, True, 5, 6)
    conv_case(out_dir, "conv3x1_p10_bias", 4, 4, (3, 1), 1, (1, 0), 1, True, 5, 6)
    convt_case(out_dir, "convt4x4_s2_p1_bias", 6, 5, 4, 2, 1, 0, True, 5, 4)
    convt_case(out_dir, "convt3x3_s2_p1_op1_bias", 5, 3, 3, 2, 1, 1, True, 4, 6)
    x = torch.randn(1, 32, 3, 5)
    write(f"{out_dir}/pixel_shuffle.bin", {"x": x[0], "y2": F.pixel_shuffle(x, 2)[0], "y4": F.pixel_shuffle(x, 4)[0]})


if __name__ == "__main__":
    main(sys.argv[1])
