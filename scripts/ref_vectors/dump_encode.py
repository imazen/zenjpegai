#!/usr/bin/env python3
"""Run the reference ENCODER and dump the tensors it committed to the bitstream.

Run inside the reference venv, from the reference checkout:

    python ~/work/zen/zenjpegai/scripts/ref_vectors/dump_encode.py OUT_DIR [--enc2] [--lef] -- <encoder args...>

With `--enc2` (use a separate OUT_DIR such as `<vector>/enc2`) the dump also holds the latent `y`,
`psi`, the cube flags and every analysis / hyper-encoder call's input and output.
With `--lef` the dump additionally holds `lef.ch_idx` (the channel the LEF's `analyze` chose,
scalar) and `lef.avg_sig` (the per-channel means it chose from), if the stream enables the LEF.

where <encoder args> are exactly what `python -m src.reco.coders.encoder` takes (input image,
output .bits, --cfg ..., overrides). OUT_DIR receives `enc_tensors.bin` + `enc_manifest.txt` in the
same format as dump_decode.py, with `<comp>.{z_hat,scale_log,skip_scale_log,residual_quant,residual}`.
Ground truth for what a decoder is supposed to recover when the reference decoder itself is in doubt.
"""
import contextlib
import io
import os
import sys

import torch

sys.path.insert(0, ".")
from src.reco.coders.encoder import RecoEncoder, process_encoder  # noqa: E402
import src.reco.coders.encoder as enc_mod  # noqa: E402

DTYPES = {
    torch.float32: ("f32", "<f4"), torch.float64: ("f64", "<f8"), torch.int8: ("i8", "i1"),
    torch.uint8: ("u8", "u1"), torch.int16: ("i16", "<i2"), torch.int32: ("i32", "<i4"),
    torch.int64: ("i64", "<i8"), torch.bool: ("bool", "u1"),
}


def install_enc2_hooks(calls):
    """`--enc2`: record what the analysis side computed, per network call, in call order.

    Names: `<net>.<call index>.{in,out}` with net one of `analysis_y`, `analysis_uv`,
    `hyper_y`, `hyper_uv`, and `mcm_y.<call>.mean<stage>` for the context model's per-stage means.
    """
    from torch.nn.modules.module import register_module_forward_hook

    counters = {}

    def hook(module, inputs, output):
        cls = type(module).__name__
        if cls.startswith("Encoder") and cls.endswith(("Prim", "Sec")):
            net = "analysis_y" if cls.endswith("Prim") else "analysis_uv"
        elif cls == "HyperEncoderBasic":
            net = "hyper_y" if inputs[0].shape[1] == 160 else "hyper_uv"
        elif cls.startswith("MCM_phase") and cls != "MCM_phase_base":
            idx = counters.get("mcm_y", 0)
            stage = int(cls[-1])
            calls.append((f"mcm_y.{idx}.mean{stage}", output.detach().clone()))
            if stage == 3:
                counters["mcm_y"] = idx + 1
            return
        else:
            return
        idx = counters.get(net, 0)
        counters[net] = idx + 1
        calls.append((f"{net}.{idx}.in", inputs[0].detach().clone()))
        calls.append((f"{net}.{idx}.out", output.detach().clone()))

    register_module_forward_hook(hook)


def install_lef_hook(calls):
    """`--lef`: record the LEF's channel choice (`LEF.analyze`) and the per-channel means of the
    luma `scale_log` it chooses from, as `lef.ch_idx` / `lef.avg_sig`. Nothing is recorded for a
    stream that does not enable the filter (`analyze` is only called from `LEF.compress`)."""
    from src.codec.coding_tools.filters.LEF.LEFfilter import LEF

    orig = LEF.analyze

    def analyze(self, decisions):
        idx = orig(self, decisions)
        scale_log = decisions.get("scale_log", None)
        calls.append(("lef.ch_idx", torch.tensor([idx], dtype=torch.int32)))
        if isinstance(scale_log, torch.Tensor):
            calls.append(("lef.avg_sig", torch.mean(scale_log.float(), dim=[2, 3]).detach().cpu()))
        return idx

    LEF.analyze = analyze


def main():
    sep = sys.argv.index("--")
    out_dir, enc_args = sys.argv[1], sys.argv[sep + 1:]
    enc2 = "--enc2" in sys.argv[2:sep]
    enc2_calls = []
    if enc2:
        install_enc2_hooks(enc2_calls)
    lef_calls = []
    if "--lef" in sys.argv[2:sep]:
        install_lef_hook(lef_calls)
    os.makedirs(out_dir, exist_ok=True)
    base_parser = enc_mod.def_base_parser()
    coder = RecoEncoder(base_parser, enc_mod.def_encoder_parser_decorator(base_parser))
    captured = {}
    orig = coder.encode_stream

    def encode_stream(params):
        captured["decisions"] = orig(params)
        return captured["decisions"]

    coder.encode_stream = encode_stream
    log = io.StringIO()
    # Mirror the real CLI (`RecoEncoderProcess.process`): with cmd_args_add=True the tool tree
    # registers its options with argparse first, which is the only path on which the reference's
    # parser copes with negative option values (e.g. a beta displacement of -100).
    sys.argv = [sys.argv[0]] + enc_args
    with contextlib.redirect_stdout(log):
        process_encoder(coder, None, True, None, True, True)
    with open(os.path.join(out_dir, "enc_stdout.log"), "w") as f:
        f.write(log.getvalue())

    model = next(v for v in captured["decisions"].values() if isinstance(v, dict) and "model_y" in v)
    lines, offset = [], 0
    with open(os.path.join(out_dir, "enc_tensors.bin"), "wb") as blob:
        for comp, key in (("y", "model_y"), ("uv", "model_uv")):
            d = model[key]
            names = ("z_hat", "scale_log", "skip_scale_log", "residual_quant", "residual")
            if enc2:
                names += ("y", "psi", "cube_flag")
            for name in names:
                t = d.get(name)
                if not isinstance(t, torch.Tensor):
                    continue
                t = t.detach().cpu().contiguous()
                tag, npdt = DTYPES[t.dtype]
                raw = t.numpy().astype(npdt, copy=False).tobytes()
                dims = " ".join(str(x) for x in t.shape)
                lines.append(f"{comp}.{name} {tag} {t.dim()} {dims} {offset} {len(raw)}")
                blob.write(raw)
                offset += len(raw)
        for name, t in enc2_calls + lef_calls:
            t = t.cpu().contiguous()
            tag, npdt = DTYPES[t.dtype]
            raw = t.numpy().astype(npdt, copy=False).tobytes()
            dims = " ".join(str(x) for x in t.shape)
            lines.append(f"{name} {tag} {t.dim()} {dims} {offset} {len(raw)}")
            blob.write(raw)
            offset += len(raw)
    with open(os.path.join(out_dir, "enc_manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
