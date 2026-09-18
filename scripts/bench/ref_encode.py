#!/usr/bin/env python
"""Run the stock JPEG AI reference encoder once at a fixed model and report its own TOTAL time.

Usage (from the reference checkout, venv active, PYTHONPATH=.):
    python ref_encode.py <in.png> <out.bits> --model N --beta-disp B --profile base [--threads N]

The bitrate matcher is off, so this is a single-pass encode at the given (model, beta
displacement) - what `zenjpegai encode --model N --beta-disp B` does. Prints one line:
`ref_total_s=<seconds> wall_s=<seconds> threads=<N> bytes=<n>`.
"""
import contextlib
import io
import os
import re
import sys
import time

t0 = time.time()
import torch  # noqa: E402

sys.path.insert(0, ".")
from src.reco.coders.encoder import RecoEncoder, def_base_parser, process_encoder  # noqa: E402
import src.reco.coders.encoder as enc_mod  # noqa: E402


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def main():
    src, out = sys.argv[1], sys.argv[2]
    threads = int(arg("--threads", "1"))
    if threads != 1:
        real = torch.set_num_threads
        torch.set_num_threads = lambda _n: real(threads)
    model, beta = arg("--model", "1"), arg("--beta-disp", "0")
    profile = arg("--profile", "base")
    base = def_base_parser()
    coder = RecoEncoder(base, enc_mod.def_encoder_parser_decorator(base))
    sys.argv = [sys.argv[0], src, out,
                "--cfg", "cfg/tools_off.json", f"cfg/profiles/{profile}.json",
                "-target_device", "cpu",
                "-model.bitrate_matcher.enabled", "0",
                "-model.bitrate_matcher.target_tool_idx", model,
                "-model.bitrate_matcher.target_beta_disp_Y", beta]
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_encoder(coder, None, True, None, True, True)
    m = re.findall(r"TOTAL: (\d+):(\d+):([\d.]+)", log.getvalue())
    total = float("nan")
    if m:
        h, mi, s = m[-1]
        total = int(h) * 3600 + int(mi) * 60 + float(s)
    print(f"ref_total_s={total:.4f} wall_s={time.time() - t0:.3f} threads={threads} "
          f"bytes={os.path.getsize(out)}")


if __name__ == "__main__":
    main()
