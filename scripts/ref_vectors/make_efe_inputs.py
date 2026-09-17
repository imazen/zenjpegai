#!/usr/bin/env python3
"""Make test inputs for the EFE filter vectors with the reference software's own image IO.

    python make_efe_inputs.py IN.png OUT WIDTH HEIGHT

Crops the top-left WIDTH x HEIGHT of IN.png and writes it to OUT. OUT's name decides the format the
way the reference's reader does: `*.png` (RGB 4:4:4) or `*_WxH_8bit_420.yuv` / `_422.yuv` /
`_444.yuv` (BT.709 YCbCr, chroma resampled by `Image.to_format_`). Odd sizes are the point: the
filters replicate-pad odd planes.
"""
import sys

sys.path.insert(0, ".")
from src.codec.common import Image  # noqa: E402


def main():
    src, dst, w, h = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    img = Image.read_file(src)
    for c in "abc":
        img.set_component(c, img.get_component(c)[:, :, :h, :w].clone())
    if dst.lower().endswith(".png"):
        img.write_png(dst, bit_depth=8)
    else:
        img.write_yuv(dst, bit_depth=8)


if __name__ == "__main__":
    main()
