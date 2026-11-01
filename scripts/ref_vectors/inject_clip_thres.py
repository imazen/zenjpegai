#!/usr/bin/env python3
"""Inject `clip_thres` tables into a models/ copy, for `clipping_mode` (`--clipping`) testing.

`set_clip_thres` (`model_train.py`) writes `clip_thres.{layer}.{MinList,MaxList}` —
per-channel clamp bounds `feature_clipping` applies after each encoder /
hyper-encoder conv. No shipped model carries them; this writes a set that
actually clamps so `zenjpegai encode --clipping` visibly changes the stream.

    inject_clip_thres.py <models-dir> <out-dir>

<models-dir> keeps the `VM_<op>/encoder_{Y,UV}_*.pth` and
`VM_common_int/{Y,UV}_*.pth` layout. Only the 0.012 (model 1) files are
modified; everything else is copied as-is.
"""
import os
import shutil
import sys

import torch

CONV = {"E1B": "conv1", "E2B": "conv2", "E3B": "conv3", "E4B": "conv4", "E5B": "conv5"}


def bounds(c):
    # Per-channel ±(0.3..1.5) — some channels clamped hard, most within range.
    return {
        "MinList": [-(0.3 + (ch % 5) * 0.3) for ch in range(c)],
        "MaxList": [0.3 + (ch % 5) * 0.3 for ch in range(c)],
        "StaList": [0.0] * c,
    }


def inject_encoder(sd, layers):
    ct = {}
    for layer in layers:
        c = sd[CONV[layer] + ".weight"].shape[0]
        ct[layer] = bounds(c)
    sd["clip_thres"] = ct


def inject_hyper(sd):
    ct = {}
    for i in range(1, 6):
        c = sd[f"hyper_encoder.conv{i}.weight"].shape[0]
        ct[f"HE{i}"] = {
            "MinList": [-(0.5 + (ch % 3) * 0.4) for ch in range(c)],
            "MaxList": [0.5 + (ch % 3) * 0.4 for ch in range(c)],
            "StaList": [0.0] * c,
        }
    sd["hyper_encoder.clip_thres"] = ct


def main():
    src, out = sys.argv[1], sys.argv[2]
    for dirpath, _, files in os.walk(src):
        for f in files:
            if not f.endswith(".pth"):
                continue
            rel = os.path.relpath(os.path.join(dirpath, f), src)
            dst = os.path.join(out, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            sd = None
            if "VM_bop" in rel or "VM_hop" in rel or "VM_sop" in rel:
                if os.path.basename(f) == "encoder_Y_0.012.pth":
                    sd = torch.load(os.path.join(dirpath, f), map_location="cpu")
                    inject_encoder(sd, ["E1B", "E2B", "E3B", "E4B", "E5B"])
                elif os.path.basename(f) == "encoder_UV_0.012.pth":
                    sd = torch.load(os.path.join(dirpath, f), map_location="cpu")
                    inject_encoder(sd, ["E1B", "E2B", "E3B", "E5B"])
            elif "VM_common_int" in rel and os.path.basename(f) in ("Y_0.012.pth", "UV_0.012.pth"):
                sd = torch.load(os.path.join(dirpath, f), map_location="cpu")
                inject_hyper(sd)
            if sd is None:
                shutil.copy2(os.path.join(dirpath, f), dst)
            else:
                torch.save(sd, dst)
                print("injected", rel)


if __name__ == "__main__":
    main()
