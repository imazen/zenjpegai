#!/usr/bin/env python3
"""Run the reference ENCODER and dump the tensors it committed to the bitstream.

Run inside the reference venv, from the reference checkout:

    python ~/work/zen/zenjpegai/scripts/ref_vectors/dump_encode.py OUT_DIR -- <encoder args...>

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


def main():
    sep = sys.argv.index("--")
    out_dir, enc_args = sys.argv[1], sys.argv[sep + 1:]
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
            for name in ("z_hat", "scale_log", "skip_scale_log", "residual_quant", "residual"):
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
    with open(os.path.join(out_dir, "enc_manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
