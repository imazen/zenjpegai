#!/usr/bin/env python
"""Run the stock JPEG AI reference decoder once and report its own `TOTAL` time.

Usage (from the reference checkout, venv active, PYTHONPATH=.):
    python ref_decode.py <stream.bits> <out.png> [--threads N]

The reference pins PyTorch to one CPU thread (`torch.set_num_threads(1)` in
`RecoDecoder.decode_stream`). `--threads N` (N != 1) replaces that call's argument, to show what
the reference would do if it were allowed to thread; N = 1 is the stock behaviour.
Prints one line: `ref_total_s=<seconds> wall_s=<seconds> threads=<N>`.
"""
import contextlib
import io
import re
import sys
import time

t0 = time.time()
import torch  # noqa: E402

sys.path.insert(0, ".")
from src.reco.coders.decoder import RecoDecoder, def_base_parser, process_decoder  # noqa: E402
from src.codec.coders import def_decoder_parser_decorator  # noqa: E402


def main():
    bits, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
    if threads != 1:
        real = torch.set_num_threads
        torch.set_num_threads = lambda _n: real(threads)
    base = def_base_parser()
    coder = RecoDecoder(base, def_decoder_parser_decorator(base))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_decoder(coder, [bits, out, "-target_device", "cpu"])
    m = re.findall(r"TOTAL: (\d+):(\d+):([\d.]+)", log.getvalue())
    h, mi, s = m[-1]
    total = int(h) * 3600 + int(mi) * 60 + float(s)
    print(f"ref_total_s={total:.4f} wall_s={time.time() - t0:.3f} threads={threads}")


if __name__ == "__main__":
    main()
