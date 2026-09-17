#!/usr/bin/env python
"""Write raw YUV test inputs (4:2:0 / 4:2:2 / 4:4:4, 8 and 10 bit) from upstream test image 00030,
using the reference software's own Image class, so the files are exactly what its encoder expects.

Run from the reference checkout with its venv:  PYTHONPATH=. python make_yuv_inputs.py OUT_DIR
"""
import os
import sys

from src.codec.common import Image

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
img = Image.read_file("data/test/00030_TE_560x888_8bit_sRGB.png")
for name, bits in (
    ("img30_560x888_8bit_420.yuv", 8),
    ("img30_560x888_8bit_422.yuv", 8),
    ("img30_560x888_8bit_444.yuv", 8),
    ("img30_560x888_10bit_420.yuv", 10),
    ("img30_560x888_10bit_444.yuv", 10),
):
    img.clone().write_file(os.path.join(out, name), bit_depth=bits)
# An odd-sized crop: chroma plane sizes round up.
t = img.get_tensor()[:, :, :301, :203]
odd = Image.create_from_tensors(
    t[:, 0:1], t[:, 1:2], t[:, 2:3], img.data_range, bit_depth=8, format="444", color_space=img.color_space
)
odd.write_file(os.path.join(out, "img30crop_203x301_8bit_420.yuv"), bit_depth=8)
