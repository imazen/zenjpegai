#!/usr/bin/env python3
"""Probe: can the reference software round-trip colour_transform_idx = 2 at all?

At upstream b9e573f the user-defined colour transform is dead code:
ColourTransformation.{pre,post}_processing call `Image.convert_range_(0, 1)`, but
`convert_range_` takes a single tuple argument — the encoder raises TypeError before writing a
bitstream. This script patches only that call signature (a tuple is what every other call site
passes), encodes with the transform forced on, decodes, and reports what comes out.

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/probe_colour_transform2.py \
        IN.png OUT.bits OUT.png [matrix(9 ints 0-255)] [offset(3 ints)]
"""
import sys

sys.path.insert(0, ".")
import numpy as np  # noqa: E402

# convert_range_((0, 1)) vs convert_range_(0, 1): accept both, semantics unchanged.
from src.codec.common.image import Image  # noqa: E402

_orig_convert_range_ = Image.convert_range_


def convert_range_(self, *args):
    new_range = args[0] if len(args) == 1 else tuple(args)
    return _orig_convert_range_(self, new_range)


Image.convert_range_ = convert_range_

from src.codec.coding_tools.colour_processing.colour_transformation import (  # noqa: E402
    colour_transformation as ct,
)

# Log the effective matrices so the report can quote them.
_orig_pre = ct.ColourTransformation.pre_processing
_orig_post = ct.ColourTransformation.post_processing


def pre_processing(self, img, *a, **k):
    out = _orig_pre(self, img, *a, **k)
    if self.colour_transform_idx == 2:
        print("PRE  clr_tr_matrix:\n", self.clr_tr_matrix)
        print("PRE  inv_matrix:\n", self.inv_matrix)
        print("PRE  offset:", self.clr_tr_offset)
        for c in "abc":
            p = out.get_component(c)
            print(f"PRE  coded {c}: shape {tuple(p.shape)} mean {p.mean().item():.4f}")
    return out


def post_processing(self, img, *a, **k):
    out = _orig_post(self, img, *a, **k)
    if self.colour_transform_idx == 2:
        for c in "abc":
            p = out.get_component(c)
            print(f"POST out {c}: shape {tuple(p.shape)} mean {p.mean().item():.4f}")
    return out


ct.ColourTransformation.pre_processing = pre_processing
ct.ColourTransformation.post_processing = post_processing


def main():
    src, bits, png = sys.argv[1:4]
    matrix = sys.argv[4:13] or ["255", "0", "0", "0", "255", "0", "0", "0", "255"]
    offset = sys.argv[13:16] or ["0", "0", "0"]
    extra = [
        "-colour_processing.colour_transform.colour_transform_idx", "2",
        "-colour_processing.colour_transform.colour_transform_matrix", *matrix,
        "-colour_processing.colour_transform.colour_transform_offset", *offset,
    ]

    from src.reco.coders.encoder import (
        RecoEncoder,
        def_base_parser,
        process_encoder,
    )
    from src.reco.coders.decoder import (
        RecoDecoder,
        def_base_parser as dec_base_parser,
        process_decoder,
    )
    from src.codec.coders import (
        def_encoder_parser_decorator,
        def_decoder_parser_decorator,
    )

    base = def_base_parser()
    enc = RecoEncoder(base, def_encoder_parser_decorator(base))
    process_encoder(
        enc,
        [src, bits, "--set_target_bpp", "50", "--cfg", "cfg/tools_off.json",
         "cfg/profiles/base.json", "-target_device", "cpu"] + extra,
        True, None, True,
    )

    dbase = dec_base_parser()
    dec = RecoDecoder(dbase, def_decoder_parser_decorator(dbase))
    process_decoder(dec, [bits, png, "-target_device", "cpu"])

    # Compare decoded vs source channel means: a working colour transform should reproduce
    # the source's colours, not collapse them onto one mixture.
    from PIL import Image as PILImage

    a = np.asarray(PILImage.open(src).convert("RGB"), dtype=np.float64)
    b = np.asarray(PILImage.open(png).convert("RGB"), dtype=np.float64)
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    a, b = a[:h, :w], b[:h, :w]
    print("source  channel means:", a.reshape(-1, 3).mean(0))
    print("decoded channel means:", b.reshape(-1, 3).mean(0))
    print("decoded channel std :", b.reshape(-1, 3).std(0))
    mse = ((a - b) ** 2).mean()
    print(f"MSE {mse:.1f}  PSNR {10 * np.log10(255 * 255 / mse):.2f} dB")


if __name__ == "__main__":
    main()
