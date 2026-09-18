#!/usr/bin/env python3
"""Replay the reference `EFEnonlinear.compress` standalone on a vector's `filters/` dumps
and record every encode-side decision it takes. Oracle for the EFE non-linear encoder port
(tests/encode_ref.rs, feature `reference-tests`).

Usage (from the ref checkout, venv active):

    PYTHONPATH=. python dump_efe_nonlinear.py OUT_DIR FILTERS_DIR SOURCE MODEL_ID

FILTERS_DIR: `<vector>/filters/` holding `EFEnonlinear.{in,alt,out}.{a,b,c}`
             (produced by dump_filters.py; `alt` is the EFE linear filter's up-sampled
             picture and is absent when the stream did not build it, e.g. DCTIF_only)
SOURCE:      the source image path (.png/.yuv) as the encoder saw it
MODEL_ID:    the model the stream was coded with (the PIH's `model_id`)

The real `EFEnonlinear.compress` runs unchanged on a `__new__`'d tool instance (the same
pattern as dump_efe_solves.py); hooks record the `lstsq` triples, the `integerize` calls,
the downsampled lumas, the per-tile `lumaMin`/`lumaMax`, the integerised weight codes, the
candidate and final on/off masks, the PSNR sequence (`10*log10`), and the output picture.

MKL's f32 `sgelsy` behind `torch.linalg.lstsq` is not run-to-run deterministic on these
hinge-column matrices (its rcond cut truncates the rank; the coded streams and this replay
draw different flags/masks on identical inputs). The Rust port solves in f64 instead, so
each solve is additionally recomputed as `dgelsy` in f64 and integerised through the
reference's own `integerize` into `weights_{u,v}_f64` — the deterministic oracle the tests
assert; the f32 draws are kept as `weights_{u,v}` for the divergence measurement.
"""
import sys, os
sys.path.insert(0, ".")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from torch import tensor, int64
from src.codec.common import Image
import src.codec.coding_tools.filters.EFEnonlinear.EFEnonlinear  # noqa: F401
from src.codec.coding_tools.filters.EFEnonlinear.EFEnonlinear import EFEnonlinear

# `import ... as` would bind the re-exported class (the package `__init__` shadows the module
# name); `sys.modules` is the real module whose `mean`/`log10` globals `compress` calls.
nl_mod = sys.modules["src.codec.coding_tools.filters.EFEnonlinear.EFEnonlinear"]
from dump_decode import Dumper
from dump_efe_solves import read_dump

# `core_models/CCS_SGMM` `base_model_beta` per model (`cfg/BRM/default.json`).
BASE_MODEL_BETA = [0.002, 0.012, 0.075, 0.5]


def main():
    out_dir, filters_dir, source, model_id = sys.argv[1:5]
    model_id = int(model_id)
    os.makedirs(out_dir, exist_ok=True)
    dump = Dumper(out_dir)
    rec = read_dump(filters_dir)
    for c in "abc":
        assert f"EFEnonlinear.in.{c}" in rec, f"{filters_dir} has no EFEnonlinear.in.{c}"
    ra, rb, rc = (rec[f"EFEnonlinear.in.{c}"] for c in "abc")
    alt = [rec.get(f"EFEnonlinear.alt.{c}") for c in "abc"]

    # The source image's subsampling is the filter's `d_ver`/`d_hor` (`s_ver`/`s_hor`).
    src_img = Image.read_file(source)
    s_ver, s_hor = src_img.get_chroma_subsampling()

    efe = EFEnonlinear.__new__(EFEnonlinear)
    torch.nn.Module.__init__(efe)
    # The attrs `__init__` sets (compress and the methods it calls use these).
    efe.onetwothree = tensor([0, 1, 2, 3, 4, 5, 6, 7])
    efe.wP = 16
    efe.lossModifier = 1.5
    efe.NonlinearFilter_tile_width_base = 1200
    efe.NonlinearFilter_tile_height_base = 1200
    efe.bSize = 64
    efe.hist_split_num = 8
    # `compress` reads these through the params machinery; stub it.
    efe.get_base_model_id = lambda: model_id
    efe.get_base_model_beta = lambda: BASE_MODEL_BETA[model_id]
    efe.get_owner_param = lambda name: {"s_ver": s_ver, "s_hor": s_hor}[name]

    img = Image.create_from_tensors(
        ra, rb, rc, [0.0, 255.0],
        format=src_img.format, bit_depth=src_img.bit_depth, color_space='yuv')
    alt_img = None
    if all(t is not None for t in alt):
        alt_img = Image.create_from_tensors(
            alt[0], alt[1], alt[2], [0.0, 255.0],
            format=src_img.format, bit_depth=src_img.bit_depth, color_space='yuv')
    org_img = Image.read_file(source)
    # `compress` clones `org_img_i` and runs `to_YUV_()` + `convert_range_(rec.data_range)`;
    # replicate that for the `org.*` dumps (the chroma also lands in `self.c`, dumped from the
    # encoder hook below).
    org_conv = Image.read_file(source)
    org_conv.to_YUV_()
    org_conv.convert_range_(img.data_range)

    dump.add("in.a", ra); dump.add("in.b", rb); dump.add("in.c", rc)
    if alt_img is not None:
        for c, t in zip("abc", alt):
            dump.add(f"alt.{c}", t)
    for c in "abc":
        dump.add(f"org.{c}", org_conv.get_component(c).clone())

    # --- hooks -----------------------------------------------------------
    counter = [0]
    orig_lstsq = torch.linalg.lstsq
    codes64 = []

    def integerize_f64(v):
        # `EFEnonlinear.integerize` verbatim: `a.item()` of the f32-narrowed value,
        # `round(w * 2**11) + 32767` (ties to even), clamped to [0, 2**16 - 1].
        w32 = torch.tensor(v, dtype=torch.float64).float().item()
        return int(min(max(round(w32 * 2048.0) + 32767, 0), 65535))

    def lstsq_hook(A, B, *a, **k):
        X = orig_lstsq(A, B, *a, **k)
        i = counter[0]; counter[0] += 1
        dump.add(f"solve.{i}.A", A.detach().cpu())
        dump.add(f"solve.{i}.B", B.detach().cpu())
        dump.add(f"solve.{i}.X", X.solution.detach().cpu())
        # The deterministic f64 re-solve (`dgelsy`, the same pivoted-QR family the Rust
        # port implements), integerised exactly as `compress` will do.
        X64 = orig_lstsq(A.double(), B.double()).solution
        dump.add(f"solve.{i}.X64", X64.detach().cpu())
        codes64.append([integerize_f64(v) for v in X64.reshape(-1).tolist()])
        return X

    icounter = [0]
    orig_int = EFEnonlinear.integerize

    def int_hook(self, x):
        out = orig_int(self, x)
        i = icounter[0]; icounter[0] += 1
        dump.add(f"int.{i}.in", x.detach().cpu().reshape(1))
        dump.add(f"int.{i}.out", torch.tensor([out], dtype=int64))
        return out

    dcounter = [0]
    orig_ds = EFEnonlinear.downsample

    def ds_hook(self, t):
        out = orig_ds(self, t)
        dump.add(f"ds.{dcounter[0]}", out.detach().clone()); dcounter[0] += 1
        return out

    psnrs = []
    orig_log10 = nl_mod.log10

    def log10_hook(x):
        v = orig_log10(x)
        psnrs.append(float(v))
        return v

    orig_enc = EFEnonlinear.LumaAidedAdaptiveNonlinearFilter_encoder

    def enc_hook(self, ans):
        # `self.c` / `self.luma` are live by now: the exact org planes the fit sees.
        dump.add("org.u", self.c[:, 0:1]); dump.add("org.v", self.c[:, 1:2])
        out = orig_enc(self, ans)
        dump.add("luma_min", torch.stack([x.reshape(()) for x in self.lumaMin]).to(int64))
        dump.add("luma_max", torch.stack([x.reshape(()) for x in self.lumaMax]).to(int64))
        dump.add("weights_u", torch.tensor(self.NonlinearFilterWeights_U, dtype=int64))
        dump.add("weights_v", torch.tensor(self.NonlinearFilterWeights_V, dtype=int64))
        # The f64 draws: the solves run U-then-V per tile, in tile order.
        dump.add("weights_u_f64", torch.tensor(codes64[0::2], dtype=int64).reshape(-1))
        dump.add("weights_v_f64", torch.tensor(codes64[1::2], dtype=int64).reshape(-1))
        return out

    orig_apply = EFEnonlinear.LumaAidedAdaptiveNonlinearFilter_apply

    def apply_hook(self, ans):
        out = orig_apply(self, ans)
        for c in "abc":
            dump.add(f"filt.{c}", out.get_component(c).clone())
        return out

    orig_onoff = EFEnonlinear.calculateOnOff

    def onoff_hook(self, rec2, filt2, **k):
        out = orig_onoff(self, rec2, filt2, **k)
        dump.add("mask_cand.0", self.mask1.to(int64))
        dump.add("mask_cand.1", self.mask2.to(int64))
        return out

    sw_count = [0]
    orig_switch = EFEnonlinear.apply_OnoffSwitch

    def switch_hook(self, rec2, filt2):
        out = orig_switch(self, rec2, filt2)
        for c in "abc":
            dump.add(f"sw.{sw_count[0]}.{c}", out.get_component(c).clone())
        sw_count[0] += 1
        return out

    torch.linalg.lstsq = lstsq_hook
    EFEnonlinear.integerize = int_hook
    EFEnonlinear.downsample = ds_hook
    EFEnonlinear.LumaAidedAdaptiveNonlinearFilter_encoder = enc_hook
    EFEnonlinear.LumaAidedAdaptiveNonlinearFilter_apply = apply_hook
    EFEnonlinear.calculateOnOff = onoff_hook
    EFEnonlinear.apply_OnoffSwitch = switch_hook
    nl_mod.log10 = log10_hook
    torch.set_grad_enabled(False)
    try:
        out = EFEnonlinear.compress(efe, [img, alt_img], org_img)
    finally:
        torch.linalg.lstsq = orig_lstsq
        EFEnonlinear.integerize = orig_int
        EFEnonlinear.downsample = orig_ds
        EFEnonlinear.LumaAidedAdaptiveNonlinearFilter_encoder = orig_enc
        EFEnonlinear.LumaAidedAdaptiveNonlinearFilter_apply = orig_apply
        EFEnonlinear.calculateOnOff = orig_onoff
        EFEnonlinear.apply_OnoffSwitch = orig_switch
        nl_mod.log10 = orig_log10

    # --- the decisions the TON header carries ----------------------------
    dump.add("mask.0", efe.mask1.to(int64))
    dump.add("mask.1", efe.mask2.to(int64))
    dump.add("psnr", torch.tensor(psnrs, dtype=torch.float64))
    tile_w = getattr(efe, "NonlinearFilter_tile_width", 0)
    tile_h = getattr(efe, "NonlinearFilter_tile_height", 0)
    dump.add("meta", torch.tensor([
        model_id, s_ver, s_hor, efe.bSize, tile_w, tile_h,
        len(efe.lumaMin), efe.NonLinearFilter_U_enabled, efe.NonLinearFilter_V_enabled,
        int(efe.mask1.shape[2] > 0), int(efe.mask2.shape[2] > 0),
    ], dtype=int64))
    for c in "abc":
        dump.add(f"out.{c}", out[0].get_component(c).clone())
    if len(out) > 1 and out[1] is not None:
        for c in "abc":
            dump.add(f"out.up.{c}", out[1].get_component(c).clone())
    dump.close()
    print(open(os.path.join(out_dir, "manifest.txt")).read())


if __name__ == "__main__":
    main()
