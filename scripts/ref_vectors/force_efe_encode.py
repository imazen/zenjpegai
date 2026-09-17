#!/usr/bin/env python3
"""Run the reference ENCODER with the EFE filters' rate-distortion decisions overridden.

    python force_efe_encode.py [--linear U_FL:U_CAND,V_FL:V_CAND] [--nonlinear] -- <encoder args...>

The reference encoder picks the EFE linear filter length / region split and switches the EFE
non-linear filter by a rate-distortion test, and on the test pictures it nearly always lands on the
smallest choice (1x1 filter, one region, non-linear filter off). To get streams that exercise the
other decoder branches this wrapper overrides the *decisions only*:

  --linear 3:5,4:7   U gets a 3x3 filter with candidate split 5, V a 4x4 filter with split 7
                     (least-squares weights are still the encoder's own).
  --nonlinear        the EFE non-linear filter and its on/off masks are always judged worth it.

The stream syntax, the weight derivation and the whole decoder are untouched, so the streams are
ordinary conformant streams and the stock reference decoder is the oracle for them.
"""
import sys

sys.path.insert(0, ".")
import src.reco.coders.encoder as enc_mod  # noqa: E402
from src.codec.coding_tools.filters.EFElinear.EFElinear import EFElinear  # noqa: E402
from src.codec.coding_tools.filters.EFEnonlinear.EFEnonlinear import EFEnonlinear  # noqa: E402
from src.reco.coders.encoder import RecoEncoder, process_encoder  # noqa: E402


def force_linear(spec):
    (ufl, ucand), (vfl, vcand) = [tuple(int(x) for x in p.split(":")) for p in spec.split(",")]
    orig = EFElinear.SplitDecide

    def split_decide(self, img, filter_lengths, candidates):
        if list(filter_lengths) == [1] and len(candidates) == 1:
            return orig(self, img, filter_lengths, candidates)  # the up-sampling set / DCTIF_only
        saved = self.lossModifier
        self.lossModifier = -1e12  # more coefficients always "win"
        try:
            img_u, f_u = orig(self, img.clone(), [ufl], [self.cands[ucand]])
            img_v, f_v = orig(self, img.clone(), [vfl], [self.cands[vcand]])
        finally:
            self.lossModifier = saved
        assert f_u["candU"] == ucand and f_u["U"][0].shape[2] == ufl, (f_u["candU"], f_u["U"][0].shape)
        assert f_v["candV"] == vcand and f_v["V"][0].shape[2] == vfl, (f_v["candV"], f_v["V"][0].shape)
        img_u.set_component("c", img_v.get_component("c"))
        for k in ("V", "V2", "candV", "mean2"):
            f_u[k] = f_v[k]
        return img_u, f_u

    EFElinear.SplitDecide = split_decide


def force_nonlinear():
    orig = EFEnonlinear.compress

    def compress(self, *a, **k):
        self.lossModifier = -1e12
        return orig(self, *a, **k)

    EFEnonlinear.compress = compress


def main():
    sep = sys.argv.index("--")
    opts, enc_args = sys.argv[1:sep], sys.argv[sep + 1:]
    i = 0
    while i < len(opts):
        if opts[i] == "--linear":
            force_linear(opts[i + 1])
            i += 2
        elif opts[i] == "--nonlinear":
            force_nonlinear()
            i += 1
        else:
            raise SystemExit(f"unknown option {opts[i]}")
    base_parser = enc_mod.def_base_parser()
    coder = RecoEncoder(base_parser, enc_mod.def_encoder_parser_decorator(base_parser))
    sys.argv = [sys.argv[0]] + enc_args
    process_encoder(coder, None, True, None, True, True)


if __name__ == "__main__":
    main()
