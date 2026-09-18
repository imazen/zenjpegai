#!/usr/bin/env python3
"""Re-run the reference EFElinear.SplitDecide standalone and dump every lstsq triple
plus the integerized filter codes. Oracle for `encoder::filters::efe_linear::tests`
(env vars ZENJPEGAI_EFE_DUMP / ZENJPEGAI_EFE_CVER / ZENJPEGAI_EFE_SPECS).

Usage (from the ref checkout, venv active):

    PYTHONPATH=. EFE_CVER=1 EFE_CHOR=1 python dump_efe_solves.py \
        OUT_DIR REC_MANIFEST_DIR SOURCE [specs...]

REC_MANIFEST_DIR: a vector's filters/ dir holding EFElinear.in.{a,b,c}
                  (produced by dump_filters.py)
SOURCE: the source image path (.png/.yuv) as the encoder saw it
specs: "fl:cand,fl:cand" pairs to evaluate, e.g. "4:5 up2" runs SplitDecide
       once per spec (fl = filter length, cand = index into the candidate
       table), or "search:model_id" to run the free search selection.
EFE_CVER / EFE_CHOR set the *coded* chroma subsampling (1 = 4:4:4, 2 = half);
the default is the source's own subsampling.
"""
import sys, os
sys.path.insert(0, ".")
import numpy as np
import torch
from torch import tensor
from torch.nn import Conv2d
from src.codec.common import Image
from src.codec.coding_tools.filters.EFElinear.EFElinear import EFElinear

DTYPES = {torch.float32: ("f32", "<f4"), torch.int64: ("i64", "<i8"), torch.int32: ("i32","<i4"), torch.float64: ("f64", "<f8")}

class Dumper:
    def __init__(self, out):
        self.out = out; self.lines = []; self.off = 0
        os.makedirs(out, exist_ok=True)
        self.blob = open(os.path.join(out, "tensors.bin"), "wb")
    def add(self, name, t):
        t = t.detach().cpu().contiguous()
        tag, npdt = DTYPES[t.dtype]
        raw = t.numpy().astype(npdt, copy=False).tobytes()
        dims = " ".join(str(x) for x in t.shape)
        self.lines.append(f"{name} {tag} {t.dim()} {dims} {self.off} {len(raw)}")
        self.blob.write(raw); self.off += len(raw)
    def close(self):
        self.blob.close()
        with open(os.path.join(self.out, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")

def read_dump(dir):
    out = {}
    man = open(os.path.join(dir, "manifest.txt")).read().splitlines()
    blob = open(os.path.join(dir, "tensors.bin"), "rb").read()
    for line in man:
        if line.startswith("#") or not line.strip(): continue
        f = line.split()
        nd = int(f[2]); shape = [int(x) for x in f[3:3+nd]]
        off, nb = int(f[3+nd]), int(f[4+nd])
        npdt = {"f32": "<f4", "f64": "<f8", "i64": "<i8", "i32": "<i4"}[f[1]]
        arr = np.frombuffer(blob[off:off+nb], dtype=npdt).copy()
        out[f[0]] = torch.from_numpy(arr).reshape(shape)
    return out

def main():
    out_dir, rec_dir, source = sys.argv[1], sys.argv[2], sys.argv[3]
    specs = sys.argv[4:]
    dump = Dumper(out_dir)
    rec = read_dump(rec_dir)
    ra, rb, rc = rec["EFElinear.in.a"], rec["EFElinear.in.b"], rec["EFElinear.in.c"]

    # --- the source image's YUV planes, as EFElinear.compress builds them ---
    img = Image.read_file(source)
    hh, ww = img.get_component('a').shape[-2:]
    img.to_YUV_()
    img.convert_range_([0.0, 255.0])
    s_ver, s_hor = img.get_chroma_subsampling()
    c_ver = int(os.environ.get("EFE_CVER", s_ver))
    c_hor = int(os.environ.get("EFE_CHOR", s_hor))

    efe = EFElinear.__new__(EFElinear)
    torch.nn.Module.__init__(efe)
    # the attrs __init__ / compress set, minus the params machinery
    efe.conv = [Conv2d(4,4,(k,k),padding=0,stride=1,bias=True,groups=4) for k in (1,2,3,4)]
    efe.convY = [Conv2d(4,4,(k,k),padding=0,stride=1,bias=False,groups=4) for k in (1,2,3,4)]
    efe.onetwothree = tensor([0,1,2,3,4,5,6,7])
    efe.cands = [
        [1, 0, [0, 1, 0, 1]],
        [2, 1, [0, 0.5, 0, 1], [0.5, 1, 0, 1]],
        [2, 2, [0, 1, 0, 0.5], [0, 1, 0.5, 1]],
        [3, 3, [0, 1, 0, 0.33], [0, 1, 0.33, 0.66], [0, 1, 0.66, 1]],
        [3, 4, [0, 0.33, 0, 1], [0.33, 0.66, 0, 1], [0.66, 1, 0, 1]],
        [4, 5, [0, 0.5, 0, 0.5], [0, 0.5, 0.5, 1], [0.5, 1, 0, 0.5],[0.5, 1, 0.5, 1]],
        [6, 6, [0, 0.33, 0, 0.5], [0, 0.33, 0.5, 1], [0.33, 0.66, 0, 0.5],[0.33, 0.66, 0.5, 1],[0.66, 1, 0, 0.5],[0.66, 1, 0.5, 1]],
        [6, 7, [0, 0.5, 0, 0.33], [0.5, 1, 0, 0.33], [0, 0.5, 0.33, 0.66],[0.5, 1, 0.33, 0.66],[0, 0.5, 0.66, 1],[0.5, 1, 0.66, 1]],
    ]
    efe.wP = 16
    efe.lossModifier = 0.5
    efe.fL = efe.fL_U = efe.fL_V = 3
    efe.filters = {'Y':[],'U':[],'V':[],'U2':[],'V2':[],'candY':-1,'candU':-1,'candV':-1,'mean1':0,'mean2':0}
    efe.bSize = 64
    # The DCT tensors are plain tensor() literals in __init__; copy them verbatim:
    efe.DCT_IF_4TAP = tensor([
    [[[ 0.0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]],
    [[[0,0,0,0],[-0.0625,-0.4375,0.5625,-0.0625],[0,0,0,0],[0,0,0,0]]],
    [[[0,-0.0625,0,0],[0,-0.4375,0,0],[0,0.5625,0,0],[0,-0.0625,0,0]]],
    [[[0.00390625,-0.03515625,-0.03515625,0.00390625],
      [-0.03515625,-0.68359375,0.31640625,-0.03515625],
      [-0.03515625,0.31640625,0.31640625,-0.03515625],
      [0.00390625,-0.03515625,-0.03515625,0.00390625]]]])
    efe.DCT_IF_4TAP_444 = torch.zeros(4,1,4,4)
    efe.encoder_output = []
    efe.s_ver, efe.s_hor = s_ver, s_hor
    efe.c_ver, efe.c_hor = c_ver, c_hor
    efe.scale_ver = 3 - c_ver/s_ver
    efe.scale_hor = 3 - c_hor/s_hor
    efe.cal_parameter(s_ver, s_hor)
    efe.c = torch.cat((img.get_component('b').clone(), img.get_component('c').clone()), dim=1)
    efe.luma = img.get_component('a').clone()
    efe.get_base_model_id = lambda: int(os.environ.get("EFE_MODEL", "1"))
    efe.DCTIF_only = False
    efe.doPixelUnShuffle = True

    rec_img = Image.create_from_tensors(ra, rb, rc, [0.0,255.0], format=img.format, bit_depth=8, color_space='yuv')

    # dump the inputs once
    dump.add("rec.a", ra); dump.add("rec.b", rb); dump.add("rec.c", rc)
    dump.add("org.a", efe.luma); dump.add("org.b", img.get_component('b')); dump.add("org.c", img.get_component('c'))

    # --- lstsq hook ---
    orig_lstsq = torch.linalg.lstsq
    counter = [0]
    def lstsq_hook(A, B, *a, **k):
        X = orig_lstsq(A, B, *a, **k)
        i = counter[0]; counter[0] += 1
        dump.add(f"solve.{i}.A", A.detach().cpu())
        dump.add(f"solve.{i}.B", B.detach().cpu())
        dump.add(f"solve.{i}.X", X.solution.detach().cpu())
        return X
    torch.linalg.lstsq = lstsq_hook

    orig_it = EFElinear.integerizeTensor
    icounter = [0]
    def it_hook(self, t):
        out = orig_it(self, t)
        i = icounter[0]; icounter[0] += 1
        dump.add(f"int.{i}.in", t.detach().cpu())
        dump.add(f"int.{i}.out", out.detach().cpu())
        return out
    EFElinear.integerizeTensor = it_hook

    torch.set_grad_enabled(False)
    for spec in specs:
        if spec.startswith("search:"):
            model_id = int(spec.split(":")[1])
            efe.get_base_model_id = lambda: model_id
            # replicate compress()'s selection
            if ww*hh <= 1e6:
                cands = efe.cands[0:3] if model_id < 2 else efe.cands[4:]
                filtL = [1,2] if model_id == 0 else [3,4]
            elif ww*hh <= 4e6:
                cands = efe.cands if model_id < 2 else efe.cands[4:]
                filtL = [1,2,3] if model_id == 0 else ([2,3,4] if model_id == 1 else [3,4])
            elif ww*hh <= 9e6:
                cands = efe.cands[3:] if model_id < 2 else efe.cands[6:]
                filtL = [1,2,3] if model_id == 0 else ([3,4] if model_id == 1 else [4])
            elif ww*hh < 16e6:
                cands = efe.cands[5:] if model_id < 2 else efe.cands[6:]
                filtL = [1,2,3] if model_id == 0 else [4]
            else:
                cands = efe.cands[6:]
                filtL = [3,4] if model_id == 0 else [4]
            ans, filters = EFElinear.SplitDecide(efe, rec_img, filtL, cands)
        elif spec == "up2":
            ans, filters = EFElinear.SplitDecide(efe, rec_img, [1], efe.cands[0:1])
        else:
            fl, cand = [int(x) for x in spec.split(":")]
            ans, filters = EFElinear.SplitDecide(efe, rec_img, [fl], [efe.cands[cand]])
        dump.add(f"{spec}.rec_u", ans.get_component('b'))
        dump.add(f"{spec}.rec_v", ans.get_component('c'))
        for key in ("U","V","U2","V2"):
            for i, f in enumerate(filters[key]):
                dump.add(f"{spec}.{key}.{i}", f)
        dump.add(f"{spec}.meta", torch.tensor([filters['candU'], filters['candV']], dtype=torch.int32))
        dump.add(f"{spec}.means", torch.tensor([filters['mean1'], filters['mean2']], dtype=torch.float64))
    torch.linalg.lstsq = orig_lstsq
    EFElinear.integerizeTensor = orig_it
    dump.close()
    print("\n".join(dump.lines[:40]), f"... {len(dump.lines)} tensors", sep="\n")

if __name__ == "__main__":
    main()
