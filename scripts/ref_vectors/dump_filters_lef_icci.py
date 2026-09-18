#!/usr/bin/env python3
"""Run the reference decoder on a bitstream and dump the picture around the LEF and eICCI
post-filters.

Run inside the reference venv, from the reference checkout:

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/dump_filters_lef_icci.py IN.bits VECTOR_DIR/filters_lef_icci

OUT_DIR receives `tensors.bin` + `manifest.txt` in the format of `dump_decode.py`:

    <filter>.in.{a,b,c}     YUV planes entering the filter (codec range, 0..255 for 8 bit)
    <filter>.out.{a,b,c}    YUV planes leaving it
    lef.scale_log           the tensor the LEF reads (`decisions[model]['model_y']['scale_log']`)

with <filter> in {eicci, lef}. Comment lines (`# ...`) record the eICCI tile layout, the per-tile
model selection and whether `lef.scale_log` equals the `scale_log` left in the decisions at the
end of the decode (the tensor `dump_decode.py` stores as `y.scale_log`).
"""
import argparse
import contextlib
import io
import os
import sys

import torch

sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_decode import Dumper  # noqa: E402
from src.reco.coders.decoder import RecoDecoder, def_base_parser, process_decoder  # noqa: E402
from src.codec.coders import def_decoder_parser_decorator  # noqa: E402
from src.codec.coding_tools.filters.LEF.LEFfilter import LEF  # noqa: E402
from src.codec.coding_tools.filters.eICCI.icci_filter import EfficientICCIFilter  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bits")
    ap.add_argument("out_dir")
    ap.add_argument(
        "--patch-icci420",
        action="store_true",
        help="read icci_enable_flag + the eICCI header for a 4:2:0 source, where the stock "
        "syntax omits them (the flag auto-detects to off). Only streams written by "
        "force_icci_encode.py --flag420 need this; see PORTING.md.",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.patch_icci420:
        # A forced-4:2:0 stream carries the flag; the stock decoder would skip it and
        # desynchronise on the bits that follow.
        EfficientICCIFilter.auto_enableflag_detected_value = lambda self: None
    dump = Dumper(args.out_dir)
    captured = {}
    notes = []

    def wrap(cls, tag):
        orig = cls.decompress

        def decompress(self, imgs, *a, **k):
            for c in "abc":
                captured[f"{tag}.in.{c}"] = imgs[0].get_component(c).clone()
            if tag == "lef":
                d = self.get_decision_y(k.get("decisions", dict()))
                captured["lef.scale_log"] = d.get("scale_log").clone()
                notes.append(f"# lef channel {self.reference_channel_idx} model_id {self.get_base_model_id()}")
            out = orig(self, imgs, *a, **k)
            for c in "abc":
                captured[f"{tag}.out.{c}"] = out[0].get_component(c).clone()
            if tag == "eicci":
                tm = self.tile_manager
                mi = self.models_idxes
                # Signalled (coded) indices, inverse of `model_selection = map[idx] + 1`;
                # the forced-4:2:0 stream's tool header is not readable by an unpatched
                # parser, so the test rebuilds it from this note.
                map_y, map_uv = mi.map_idx
                sig = [[map_y.index(t[0] - 1) if t[0] else 0,
                        map_uv.index(t[1] - 1) if t[1] else 0]
                       for t in mi.model_state_per_tile]
                notes.append(f"# eicci tiling enabled {int(bool(tm.is_enabled()))} "
                             f"selection {mi.model_state_per_tile} "
                             f"short_list {int(mi.use_short_list)} signalled {sig} "
                             f"base {self.get_base_model_info()}")
                # Machine-readable copy of the same data: the forced-4:2:0 stream's tool
                # header cannot be read by an unpatched parser, so the test rebuilds the
                # IcciHeader from these tensors (icci_use = selection != 0 per plane).
                captured["eicci.selection"] = torch.tensor(mi.model_state_per_tile,
                                                         dtype=torch.int32)
                captured["eicci.signalled"] = torch.tensor(sig, dtype=torch.int32)
                captured["eicci.short_list"] = torch.tensor(int(mi.use_short_list),
                                                           dtype=torch.uint8)
                for row in tm.image_tiles.tiles:
                    for t in row:
                        core, rel = tm.get_core_of_overlapping_image_tile(t)
                        notes.append(
                            f"# eicci tile {t.position.x} {t.position.y} {t.size.width} {t.size.height}"
                            f" core {core.position.x} {core.position.y} {core.size.width} {core.size.height}"
                            f" rel {rel.position.x} {rel.position.y}")
            return out

        cls.decompress = decompress

    wrap(LEF, "lef")
    wrap(EfficientICCIFilter, "eicci")

    base_parser = def_base_parser()
    coder = RecoDecoder(base_parser, def_decoder_parser_decorator(base_parser))
    orig_decode_stream = coder.decode_stream

    def decode_stream(bit_fpath, rec_file, params):
        decisions = orig_decode_stream(bit_fpath, rec_file, params)
        captured["decisions"] = decisions
        return decisions

    coder.decode_stream = decode_stream
    png = os.path.join(args.out_dir, "decoded.png")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_decoder(coder, [args.bits, png, "-target_device", "cpu"])
    os.remove(png)

    decisions = captured.pop("decisions")
    model = next(v for v in decisions.values() if isinstance(v, dict) and "model_y" in v)
    if "lef.scale_log" in captured:
        same = torch.equal(captured["lef.scale_log"], model["model_y"]["scale_log"])
        notes.append(f"# lef.scale_log equals final y.scale_log: {same}")
    dump.lines.extend(notes)
    for name in sorted(captured):
        dump.add(name, captured[name])
    dump.close()
    print(open(os.path.join(args.out_dir, "manifest.txt")).read())


if __name__ == "__main__":
    main()
