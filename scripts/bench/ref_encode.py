#!/usr/bin/env python
"""Run the stock JPEG AI reference encoder once at a fixed model and report its own TOTAL time.

Usage (from the reference checkout, venv active, PYTHONPATH=.):
    python ref_encode.py <in.png> <out.bits> --model N --beta-disp B --profile base [--threads N]
        [--tools on|off]
    python ref_encode.py <in.png> <out.bits> --bpp N --profile base [--threads N] [--tools on|off]

The bitrate matcher is off, so this is a single-pass encode at the given (model, beta
displacement) - what `zenjpegai encode --model N --beta-disp B` does. With `--bpp N` (N in
1/100 bpp) the matcher's `--set_target_bpp` path runs instead - the reference's
rate-matched encode, like `zenjpegai encode --bpp`. Prints one line:
`ref_total_s=<seconds> wall_s=<seconds> threads=<N> bytes=<n>`.
"""
import contextlib
import io
import os
import sys

import ref_common
from src.reco.coders.encoder import RecoEncoder, def_base_parser, process_encoder  # noqa: E402
import src.reco.coders.encoder as enc_mod  # noqa: E402


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    src, out = sys.argv[1], sys.argv[2]
    threads = int(arg("--threads", "1"))
    ref_common.threads(threads)
    model, beta = arg("--model", "1"), arg("--beta-disp", "0")
    profile = arg("--profile", "base")
    tools = arg("--tools", "off")
    bpp = arg("--bpp")
    base = def_base_parser()
    coder = RecoEncoder(base, enc_mod.def_encoder_parser_decorator(base))
    sys.argv = [sys.argv[0], src, out,
                "--cfg", f"cfg/tools_{tools}.json", f"cfg/profiles/{profile}.json",
                "-target_device", "cpu"]
    if bpp is not None:
        sys.argv += ["--set_target_bpp", bpp]
    else:
        sys.argv += ["-model.bitrate_matcher.enabled", "0",
                     "-model.bitrate_matcher.target_tool_idx", model,
                     "-model.bitrate_matcher.target_beta_disp_Y", beta]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_encoder(coder, None, True, None, True, True)
    total = ref_common.total_s(log.getvalue())
    print(f"ref_total_s={total:.4f} wall_s={ref_common.wall_s():.3f} threads={threads} "
          f"bytes={os.path.getsize(out)}")


if __name__ == "__main__":
    main()
