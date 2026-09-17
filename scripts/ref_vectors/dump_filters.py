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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bits")
    ap.add_argument("out_dir")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    dump = Dumper(args.out_dir)

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
