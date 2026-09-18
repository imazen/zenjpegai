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
import sys

import ref_common
from src.reco.coders.decoder import RecoDecoder, def_base_parser, process_decoder  # noqa: E402
from src.codec.coders import def_decoder_parser_decorator  # noqa: E402


def main():
    bits, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
    ref_common.threads(threads)
    base = def_base_parser()
    coder = RecoDecoder(base, def_decoder_parser_decorator(base))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_decoder(coder, [bits, out, "-target_device", "cpu"])
    total = ref_common.total_s(log.getvalue())
    print(f"ref_total_s={total:.4f} wall_s={ref_common.wall_s():.3f} threads={threads}")


if __name__ == "__main__":
    main()
