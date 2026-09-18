#!/usr/bin/env python3
"""Chroma up-sampling vectors computed by PyTorch (the reference's torch 1.10.2), for
tests/nn_vectors.rs: `F.interpolate(x, size, mode="bicubic", align_corners=True)`, the call
behind the reference's `Image.to_444_` (`TensorOps.resize_tensor`).

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/gen_resize_vectors.py ~/work/zen/zenjpegai/tests/vectors/nn

Same "ZJTB" bundle format as gen_nn_vectors.py.
"""
import struct
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(20260918)
torch.set_num_threads(1)


def write(path, tensors):
    with open(path, "wb") as f:
        f.write(b"ZJTB" + struct.pack("<I", len(tensors)))
        for name, t in tensors.items():
            t = t.detach().to(torch.float32).contiguous()
            f.write(struct.pack("<H", len(name)) + name.encode())
            f.write(struct.pack("<B", t.dim()) + struct.pack(f"<{t.dim()}I", *t.shape))
            f.write(t.numpy().astype("<f4").tobytes())


def main(out_dir):
    tensors = {}
    # (in_h, in_w) -> (out_h, out_w): exact 2x, odd luma sizes (chroma = ceil(luma / 2)),
    # 4:2:2 (width only), and one-sample axes.
    for i, ((ih, iw), (oh, ow)) in enumerate(
        [((6, 5), (12, 10)), ((7, 9), (13, 17)), ((8, 4), (8, 7)), ((1, 3), (1, 6)), ((3, 1), (5, 1)), ((2, 2), (3, 4))]
    ):
        x = torch.rand(1, 2, ih, iw) * 255.0
        y = F.interpolate(x, size=(oh, ow), mode="bicubic", align_corners=True)
        tensors[f"x{i}"] = x[0]
        tensors[f"y{i}"] = y[0]
    write(f"{out_dir}/bicubic_align_corners.bin", tensors)

    # `Image.to_420_` / `to_422_`'s resampler: `mode="bilinear"`, chroma down-sampling
    # (odd sizes included), a 4:2:2 width-only case, a one-sample axis and one up-sample.
    # Single channel, as the encoder resamples one component plane at a time (a [1, 1, H, W]
    # tensor takes PyTorch's channels-last scalar tail; a wider C vectorises differently).
    tensors = {}
    for i, ((ih, iw), (oh, ow)) in enumerate(
        [((6, 5), (3, 3)), ((7, 9), (4, 5)), ((8, 4), (8, 2)), ((5, 7), (3, 4)),
         ((12, 10), (6, 5)), ((1, 7), (1, 4)), ((3, 4), (7, 9)), ((9, 6), (5, 3))]
    ):
        x = torch.rand(1, 1, ih, iw) * 255.0
        y = F.interpolate(x, size=(oh, ow), mode="bilinear", align_corners=True)
        tensors[f"x{i}"] = x[0]
        tensors[f"y{i}"] = y[0]
    write(f"{out_dir}/bilinear_align_corners.bin", tensors)


if __name__ == "__main__":
    main(sys.argv[1])
