#!/usr/bin/env python
"""Time the reference decoder's LEF and eICCI post-filters inside a stock decode.

Usage (from the reference checkout, venv active, PYTHONPATH=.):
    python ref_filters_lef_icci.py <stream.bits> <out.png> [--threads N]

`--threads` as in ref_decode.py (1 = what the reference does). Prints one line:
`eicci_s=<s> lef_s=<s> ref_total_s=<s> threads=<N>`; a filter the stream does not enable is 0.
The eICCI networks are loaded before the decode starts, so its time is inference only.
"""
import contextlib
import io
import sys
import time

import ref_common
from src.reco.coders.decoder import RecoDecoder, def_base_parser, process_decoder  # noqa: E402
from src.codec.coders import def_decoder_parser_decorator  # noqa: E402
from src.codec.coding_tools.filters.LEF.LEFfilter import LEF  # noqa: E402
from src.codec.coding_tools.filters.eICCI.icci_filter import EfficientICCIFilter  # noqa: E402

spent = {"eicci": 0.0, "lef": 0.0}


def timed(cls, tag):
    orig = cls.decompress

    def decompress(self, *a, **k):
        t = time.time()
        out = orig(self, *a, **k)
        spent[tag] += time.time() - t
        return out

    cls.decompress = decompress


def main():
    bits, out = sys.argv[1], sys.argv[2]
    threads = int(sys.argv[sys.argv.index("--threads") + 1]) if "--threads" in sys.argv else 1
    ref_common.threads(threads)
    timed(EfficientICCIFilter, "eicci")
    timed(LEF, "lef")
    base = def_base_parser()
    coder = RecoDecoder(base, def_decoder_parser_decorator(base))
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_decoder(coder, [bits, out, "-target_device", "cpu"])
    total = ref_common.total_s(log.getvalue())
    print(f"eicci_s={spent['eicci']:.4f} lef_s={spent['lef']:.4f} ref_total_s={total:.4f} threads={threads}")


if __name__ == "__main__":
    main()
