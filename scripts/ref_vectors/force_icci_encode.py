#!/usr/bin/env python3
"""Run the reference ENCODER with eICCI forced on a chroma-subsampled source.

    python force_icci_encode.py [--sel Y:UV] -- <encoder args...>

The reference encoder never enables eICCI for chroma-subsampled sources:
`EfficientICCIFilter.compress` returns early (`s_ver != 1 or s_hor != 1` ->
`set_enable(False)`), and for 4:2:0 sources `icci_enable_flag` is not even part of the
syntax (`auto_enableflag_detected_value` returns False there). This wrapper overrides the
*decisions only*:

  - `compress` runs with `s_ver`/`s_hor` hidden (the whole stock body, RD search
    included), and the model selection is then overwritten with `--sel` (default a
    non-empty long-list choice, so `icci_use_shortList = 0` streams exist);
  - `--flag420` additionally makes `auto_enableflag_detected_value` return `None`, so
    `icci_enable_flag` and the eICCI header are coded for a 4:2:0 source. The resulting
    stream is NOT conformant by the reference's own syntax: the stock decoder skips
    those bits and desynchronises. Only a decoder with the same patch (see
    `dump_filters_lef_icci.py --patch-icci420`) reads it.

The stream syntax itself (for 4:2:2), the weight derivation and the whole decoder are
untouched.
"""
import sys

sys.path.insert(0, ".")
import src.reco.coders.encoder as enc_mod  # noqa: E402
from src.codec.common import Image  # noqa: E402
from src.codec.coding_tools.filters.eICCI.icci_filter import EfficientICCIFilter  # noqa: E402
from src.reco.coders.encoder import RecoEncoder, process_encoder  # noqa: E402


def force_icci(sel):
    sel_y, sel_uv = [int(x) for x in sel.split(":")]
    assert 1 <= sel_y <= 10 and 1 <= sel_uv <= 10
    orig = EfficientICCIFilter.compress

    def compress(self, imgs, *a, **k):
        s_ver = self.get_owner_param("s_ver")
        s_hor = self.get_owner_param("s_hor")
        if s_ver == 1 and s_hor == 1:
            return orig(self, imgs, *a, **k)
        # Subsampled source: the stock body early-returns. Run it with the check hidden;
        # the only other place it reads s_ver/s_hor is the final to_format_, which then
        # keeps the filtered picture at 4:4:4, so convert it back below.
        real_gop = self.get_owner_param

        def gop(name, *aa, **kk):
            return 1 if name in ("s_ver", "s_hor") else real_gop(name, *aa, **kk)

        self.get_owner_param = gop
        try:
            out = orig(self, imgs, *a, **k)
        finally:
            del self.get_owner_param
        fmt = Image.get_format_from_subsampling(s_ver, s_hor)
        out[0].to_format_(fmt)
        # Force a non-empty selection on the long list: the RD search may pick [0,0,0]
        # (no plane filtered), which exercises nothing.
        self.models_idxes.use_short_list = False
        self.models_idxes.model_state_per_tile = [
            [sel_y, sel_uv, sel_uv] for _ in self.models_idxes.model_state_per_tile
        ]
        return out

    EfficientICCIFilter.compress = compress


def code_icci_flag_for_420():
    # `s_ver != 1 and s_hor != 1` -> False in the stock code: icci_enable_flag is not
    # coded. Returning None puts it (and the eICCI header) back into the TON; a decoder
    # needs the same patch to read the stream.
    def auto_enableflag_detected_value(self):
        if self.get_owner_param("s_ver") != 1 and self.get_owner_param("s_hor") != 1:
            return None
        return None

    EfficientICCIFilter.auto_enableflag_detected_value = auto_enableflag_detected_value


def main():
    sep = sys.argv.index("--")
    opts, enc_args = sys.argv[1:sep], sys.argv[sep + 1:]
    sel = "7:3"
    flag420 = False
    i = 0
    while i < len(opts):
        if opts[i] == "--sel":
            sel = opts[i + 1]
            i += 2
        elif opts[i] == "--flag420":
            flag420 = True
            i += 1
        else:
            raise SystemExit(f"unknown option {opts[i]}")
    force_icci(sel)
    if flag420:
        code_icci_flag_for_420()
    base_parser = enc_mod.def_base_parser()
    coder = RecoEncoder(base_parser, enc_mod.def_encoder_parser_decorator(base_parser))
    sys.argv = [sys.argv[0]] + enc_args
    process_encoder(coder, None, True, None, True, True)


if __name__ == "__main__":
    main()
