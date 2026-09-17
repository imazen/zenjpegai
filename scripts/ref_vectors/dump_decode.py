#!/usr/bin/env python3
"""Run the reference decoder on a bitstream and dump its intermediate tensors.

Run inside the reference venv, from the reference checkout:

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/dump_decode.py IN.bits OUT_DIR [--threads N]

OUT_DIR receives:
    tensors.bin     every tensor, little-endian, C-contiguous, concatenated
    manifest.txt    one line per tensor: name dtype ndim dims... offset nbytes
    decoded.png     the reference reconstruction (what decoder.py writes)
    stdout.log      the decoder's log, control-point MD5s included

Tensor names: `<comp>.<key>` with comp in {y, uv} for the per-component decisions
(z_hat, scale_log, skip_scale_log, residual_quant, residual, psi, y_hat), then `rec.{a,b,c}` for
the synthesis output before post-filters (YUV, internal range) and `out.{a,b,c}` for the final image.
"""
import argparse
import contextlib
import io
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")
from src.reco.coders.decoder import RecoDecoder, def_base_parser, process_decoder  # noqa: E402
from src.codec.coders import def_decoder_parser_decorator  # noqa: E402

DTYPES = {
    torch.float32: ("f32", "<f4"), torch.float64: ("f64", "<f8"), torch.int8: ("i8", "i1"),
    torch.uint8: ("u8", "u1"), torch.int16: ("i16", "<i2"), torch.int32: ("i32", "<i4"),
    torch.int64: ("i64", "<i8"), torch.bool: ("bool", "u1"),
}


class Dumper:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.blob = open(os.path.join(out_dir, "tensors.bin"), "wb")
        self.lines = []
        self.offset = 0

    def add(self, name, t):
        if not isinstance(t, torch.Tensor):
            return
        t = t.detach().cpu().contiguous()
        tag, npdt = DTYPES[t.dtype]
        raw = t.numpy().astype(npdt, copy=False).tobytes()
        dims = " ".join(str(d) for d in t.shape)
        self.lines.append(f"{name} {tag} {t.dim()} {dims} {self.offset} {len(raw)}".replace("  ", " "))
        self.blob.write(raw)
        self.offset += len(raw)

    def close(self):
        self.blob.close()
        with open(os.path.join(self.out_dir, "manifest.txt"), "w") as f:
            f.write("\n".join(self.lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bits")
    ap.add_argument("out_dir")
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (the reference forces 1)")
    ap.add_argument(
        "--contiguous-masks",
        action="store_true",
        help="hand the C++ ANS decoder a contiguous skip mask, as the reference encoder does. "
        "Without it the reference decoder mis-decodes region streams (see PORTING.md).",
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.contiguous_masks:
        from src.codec.entropy_coding.lib_wrappers.mans import sgt_prob_wrapper as spw

        def decode(self, sigma, masks, name=None, entropy_model=None):
            model = self.get_model(entropy_model)
            indexes = model.build_indexes(sigma).to(dtype=torch.uint8).cpu().numpy()
            x = np.zeros(indexes.shape, dtype=np.int16)
            self.backend.decode_sgm(indexes, x, np.ascontiguousarray(masks.cpu().numpy()))
            return torch.from_numpy(x.astype(np.float32))

        spw.SgtProbWrapper.decode = decode
    dump = Dumper(args.out_dir)

    base_parser = def_base_parser()
    coder = RecoDecoder(base_parser, def_decoder_parser_decorator(base_parser))
    captured = {}

    orig_decode_stream = coder.decode_stream

    def decode_stream(bit_fpath, rec_file, params):
        ce = coder.ce
        orig_model_decompress = ce.model.decompress

        def model_decompress(decisions, *a, **k):
            img = orig_model_decompress(decisions, *a, **k)
            for c in "abc":
                captured[f"rec.{c}"] = img.get_component(c).clone()
            return img

        ce.model.decompress = model_decompress
        decisions = orig_decode_stream(bit_fpath, rec_file, params)
        if args.threads != 1:
            pass  # decode_stream already forced 1 thread; the flag is for timing runs only
        captured["decisions"] = decisions
        return decisions

    coder.decode_stream = decode_stream
    png = os.path.join(args.out_dir, "decoded.png")
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        process_decoder(coder, [args.bits, png, "-target_device", "cpu"])
    with open(os.path.join(args.out_dir, "stdout.log"), "w") as f:
        f.write(log.getvalue())

    decisions = captured["decisions"]
    model = next(v for v in decisions.values() if isinstance(v, dict) and "model_y" in v)
    dump.lines.append(f"# tool_id {model.get('tool_id')}")
    for comp, key in (("y", "model_y"), ("uv", "model_uv")):
        d = model[key]
        for name in ("z_hat", "scale_log", "skip_scale_log", "residual_quant", "residual", "psi", "y_hat"):
            if name in d:
                dump.add(f"{comp}.{name}", d[name])
    for c in "abc":
        dump.add(f"rec.{c}", captured[f"rec.{c}"])
        dump.add(f"out.{c}", coder.rec_image.get_component(c))
    dump.close()
    print(open(os.path.join(args.out_dir, "manifest.txt")).read())


if __name__ == "__main__":
    main()
