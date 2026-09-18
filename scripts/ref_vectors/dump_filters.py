#!/usr/bin/env python3
"""Run the reference decoder on a bitstream and dump the picture around every post-filter.

Run inside the reference venv, from the reference checkout:

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/dump_filters.py IN.bits OUT_DIR

OUT_DIR (use `<vector>/filters/`) receives `tensors.bin` + `manifest.txt` in the format of
`dump_decode.py`. Tensor names, per enabled filter `<tool>` in chain order
(`EFElinear`, `eICCI`, `EFEnonlinear`, `LEF`):

    <tool>.in.{a,b,c}    picture handed to the filter (YUV planes, range 0..255)
    <tool>.alt.{a,b,c}   second list entry handed to the filter, when it is not None
    <tool>.out.{a,b,c}   picture the filter returns
    <tool>.up.{a,b,c}    second list entry the filter returns, when it is not None

`timing.txt` gets the wall time of each filter's `decompress`. The decoder itself is untouched:
the hooks only copy tensors.

`--msssim` skips the decode entirely and instead dumps an oracle set for the encoder's
MS-SSIM port (`pytorch_msssim == 0.2.1`, `data_range = 1`): deterministic `msssim.<i>.x` /
`.y` planes at eICCI-relevant sizes (the untiled 560x888 picture, the 1024 filter tile, a
minimum-size 176x296 boundary tile, odd and near-minimum shapes), each with the reference's
`ms_sssim.<i>.val` scalar. Usage:

    python dump_filters.py --msssim OUT_DIR
"""
import argparse
import contextlib
import io
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_decode import Dumper  # noqa: E402
from src.reco.coders.decoder import RecoDecoder, def_base_parser, process_decoder  # noqa: E402
from src.codec.coders import def_decoder_parser_decorator  # noqa: E402


def dump_msssim_oracle(dump):
    """`pytorch_msssim == 0.2.1` `ms_ssim(x, y, data_range=1.)` on deterministic planes."""
    import torch
    from pytorch_msssim import ms_ssim

    torch.manual_seed(0)
    # (height, width): the untiled 560x888 picture, the 1024 filter tile, a 176x296
    # minimum-size boundary tile, then odd and near-minimum shapes.
    for i, (h, w) in enumerate(
        [(560, 888), (1024, 1024), (176, 296), (201, 301), (161, 161)]
    ):
        x = torch.rand(1, 1, h, w)
        # A plausible reconstruction: mostly the source, locally biased, mildly noisy.
        y = (x * 0.97 + 0.02 + 0.03 * torch.rand(1, 1, h, w)).clamp(0.0, 1.0)
        if i == 0:
            y = x.clone()  # identical planes: ms_ssim == 1 exactly
        dump.add(f"msssim.{i}.x", x)
        dump.add(f"msssim.{i}.y", y)
        dump.add(f"msssim.{i}.val", ms_ssim(x, y, data_range=1.0).reshape(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bits", nargs="?", default=None)
    ap.add_argument("out_dir", nargs="?", default=None)
    ap.add_argument(
        "--msssim",
        action="store_true",
        help="dump the pytorch_msssim oracle set (no decode; BITS unused)",
    )
    args = ap.parse_args()
    if args.out_dir is None:
        args.bits, args.out_dir = None, args.bits
    if args.out_dir is None or (args.bits is None and not args.msssim):
        ap.error("usage: dump_filters.py BITS OUT_DIR, or --msssim OUT_DIR")
    os.makedirs(args.out_dir, exist_ok=True)
    dump = Dumper(args.out_dir)
    if args.msssim:
        dump_msssim_oracle(dump)
        dump.close()
        print(open(os.path.join(args.out_dir, "manifest.txt")).read())
        return

    base_parser = def_base_parser()
    coder = RecoDecoder(base_parser, def_decoder_parser_decorator(base_parser))
    orig_decode_stream = coder.decode_stream

    timings = []

    def add_image(prefix, img):
        if img is None:
            return
        for c in "abc":
            dump.add(f"{prefix}.{c}", img.get_component(c).clone())

    def hook(name, tool):
        orig = tool.decompress

        def decompress(imgs, *a, **k):
            # Filters modify their input in place (EFE non-linear does): copy first.
            add_image(f"{name}.in", imgs[0])
            add_image(f"{name}.alt", imgs[1])
            t0 = time.perf_counter()
            out = orig(imgs, *a, **k)
            timings.append(f"{name} {(time.perf_counter() - t0) * 1e3:.2f} ms")
            add_image(f"{name}.out", out[0])
            add_image(f"{name}.up", out[1])
            return out

        tool.decompress = decompress

    def decode_stream(bit_fpath, rec_file, params):
        for name, tool in coder.ce.post_filters.iter_over_naming_tools():
            hook(name, tool)
        return orig_decode_stream(bit_fpath, rec_file, params)

    coder.decode_stream = decode_stream
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_decoder(coder, [args.bits, os.path.join(args.out_dir, "decoded.png"), "-target_device", "cpu"])
    with open(os.path.join(args.out_dir, "stdout.log"), "w") as f:
        f.write(log.getvalue())
    dump.close()
    # Wall time of each filter's `decompress` (single torch thread, as the decoder forces).
    with open(os.path.join(args.out_dir, "timing.txt"), "w") as f:
        f.write("\n".join(timings) + "\n")
    print(open(os.path.join(args.out_dir, "manifest.txt")).read())


if __name__ == "__main__":
    main()
