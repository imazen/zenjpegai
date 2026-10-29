#!/usr/bin/env python3
"""Run the reference ENCODER with RDLR enabled and dump every luma refinement iteration.

    cd ~/work/zen/jpeg-ai-reference-software && . .venv/bin/activate
    python ~/work/zen/zenjpegai/scripts/ref_vectors/dump_rdlr.py OUT_DIR -- <encoder args...>

<encoder args> are exactly what `python -m src.reco.coders.encoder` takes; the caller adds
`cfg/tools/RDLR.json` (enabled = 1). For every `RDLRCompY.rdlr_y_iteration` call this records:

    rdlr.<i>.y            the latent tile the optimizer refines (with overlap)
    rdlr.<i>.original    the source picture tile (luma, 0..255)
    rdlr.<i>.rate        the iteration's rate point (bpp estimate incl. z bits), scalar
    rdlr.<i>.psnr        PSNR on the tile core, scalar
    rdlr.<i>.msssim      MS-SSIM on the tile core, scalar
    rdlr.<i>.loss        the RDLR loss, scalar
    rdlr.<i>.y_grad      dLoss/dy after `loss.backward()`
    rdlr.area            the optimized part's (y0, y1, x0, x1) inside the overlap tile, i32[4]
    rdlr.<comp>.<name>   the decisions tensors `_compress_ar_scale` reads (psi, scale_log,
                         skip_scale_log, z_hat, quantizer sub-tensors, mask/skip state)

OUT_DIR receives `rdlr_tensors.bin` + `rdlr_manifest.txt` in dump_decode.py's format, plus
`stream.bits` and `encoder.log` (the RDLR stream is itself the target vector).
"""

import contextlib
import io
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dump_encode import DTYPES  # noqa: E402

from src.reco.coders.encoder import RecoEncoder, process_encoder  # noqa: E402
import src.reco.coders.encoder as enc_mod  # noqa: E402


def main():
    sep = sys.argv.index("--")
    out_dir, enc_args = sys.argv[1], sys.argv[sep + 1:]
    os.makedirs(out_dir, exist_ok=True)

    from src.codec.coding_tools.rdlr.rdlr_comp_Y import RDLRCompY  # noqa: E402
    from src.codec.coding_tools.tiling.tiling import TileManager  # noqa: E402

    # b9e573f ships RDLR broken: `TileManager.set_alignment_size` is commented out in
    # tiling.py ("only used by rdlr, do not care for now"), so `process_y` crashes on entry.
    # Restoring the commented body verbatim — the same patched-oracle pattern as
    # `dump_decode.py --contiguous-masks` — so the algorithm's own semantics are observable.
    def _set_alignment_size(self, value):
        self.alignment_size = value
        self.latent_downscale_factor_y = value

    TileManager.set_alignment_size = _set_alignment_size

    # Second stale call: `process_y` calls `setup_tiles_enc(image_shape, latent_shape,
    # minimum_tile_size=...)` but the method now also wants `latent_shape_psi` and
    # `latent_shape_z`. The tile manager's own scale factors define them (psi at x2 the
    # latent downscale, z at x4).
    _orig_setup_tiles_enc = TileManager.setup_tiles_enc

    def _setup_tiles_enc(self, image_shape, latent_shape, *args, **kwargs):
        if len(args) < 2:
            psi = (
                latent_shape[0], latent_shape[1],
                math.ceil(image_shape[2] / self.latent_downscale_factor_psi),
                math.ceil(image_shape[3] / self.latent_downscale_factor_psi),
            )
            z = (
                latent_shape[0], latent_shape[1],
                math.ceil(image_shape[2] / self.latent_downscale_factor_z),
                math.ceil(image_shape[3] / self.latent_downscale_factor_z),
            )
            args = (psi, z) + args
        return _orig_setup_tiles_enc(self, image_shape, latent_shape, *args, **kwargs)

    TileManager.setup_tiles_enc = _setup_tiles_enc

    # Third stale call: `get_core_of_overlapping_latent_tile` gained a `tile_manager_hd`
    # argument. It is consulted only under region partitioning — and the method itself
    # resolves `None` to the owner's hyper-domain manager — so a default-arg shim mirrors
    # the code's own fallback.
    _orig_core = TileManager.get_core_of_overlapping_latent_tile

    def _core(self, latent_tile, corresponding_image_tile, tile_manager_hd=None):
        return _orig_core(self, latent_tile, corresponding_image_tile, tile_manager_hd)

    TileManager.get_core_of_overlapping_latent_tile = _core

    # Fourth stale call: `get_current_rate_point` calls `bs.encode_init()`/`encode_term()`
    # on BitstreamStructure. Both methods are gone at b9e573f: `get_ec` self-initialises
    # each substream's library on first use, so init is a no-op; termination sums
    # `get_total_bits` over the structure's aemem objects (what the old method returned:
    # bits, added to the likelihood rate).
    from src.codec.bitstream_structure import BitstreamStructure  # noqa: E402

    def _bs_encode_init(self):
        # `get_ec` dispatches on coder_direction == 0 for encode-side libraries;
        # the rdlr call sites never set it (the real call sites pass
        # `coder_direction=0` in the factory kwargs). `max_compressed_size` is a
        # factory attribute `ECComposite` normally provides; give each substream a
        # 16 MiB buffer (the z substream for one tile is far below that).
        self.coder_direction = 0
        self.ae_factory.max_compressed_size = [1 << 24] * 64

    def _bs_encode_term(self):
        total = 0.0
        for lst in self.ae_mems_dict.values():
            for mem in lst:
                if mem is not None:
                    mem.encode_term()
                    total += float(mem.ae.get_total_bits().item())
        return total

    BitstreamStructure.encode_init = _bs_encode_init
    BitstreamStructure.encode_term = _bs_encode_term

    calls = []

    def put(name, t):
        t = t.detach().to(torch.float32).cpu().contiguous() if isinstance(t, torch.Tensor) \
            else torch.tensor(t, dtype=torch.float32)
        calls.append((name, t))

    # Fifth stale call: `_compress_ar_scale` now returns
    # `(resi_q, resi_dq, y_rec, cube_flag)` — cube_flag was added after RDLR was
    # written; it unpacks three. RDLR runs strictly after the main compress, so
    # shadowing the bound method on the tool's common_modules only affects RDLR's
    # own calls. Installed through get_current_rate_point so it is in place before
    # `get_ref_rate_data`'s first use.
    import src.codec.coding_tools.rdlr.rdlr_comp_Y as _rcy  # noqa: E402
    import src.codec.coding_tools.rdlr.rdlr_comp_UV as _rcuv  # noqa: E402

    def _install_cas3(tool):
        if not getattr(tool.common_modules, "_cas3_patched", False):
            cm = tool.common_modules
            _orig = cm._compress_ar_scale
            cm._compress_ar_scale = lambda y, d, _o=_orig: _o(y, d)[:3]

            # Sixth stale call: `_compress_y` was deleted; the tiled path's
            # equivalent is `compress_ar_scale_tile`'s assignments plus the
            # `encoder_skip_and_cubeflag_for_tiles` mask/cubeflag update, at
            # whole-picture scope for the single-tile case.
            def _compress_y(y, decisions, h=None, w=None, _cm=cm):
                resi_q, resi_dq, y_hat, cube_flag = _cm._compress_ar_scale_orig(y, decisions)
                decisions['residual'] = resi_dq
                decisions['residual_quant'] = resi_q
                decisions['y_hat'] = y_hat
                decisions['cube_flag'] = cube_flag
                if _cm.skip_mode.enabled:
                    _cm.skip_mode.cube_flag = cube_flag
                    _cm.skip_mode.cube_flags_full = _cm.skip_mode.convert_cubeflag_map(
                        cube_flag, y.shape)
                    _cm.skip_mode._update_mask(decisions['skip_scale_log'])
                    mask = _cm.skip_mode.mask()
                    decisions['residual_quant'][~mask] = 0
                    decisions['residual'][~mask] = 0

            cm._compress_ar_scale_orig = _orig
            cm._compress_y = _compress_y
            cm._cas3_patched = True

    _gcrp_count = [0]

    for _mod in (_rcy, _rcuv):
        _orig_gcrp = _mod.RDLRCompY.get_current_rate_point if _mod is _rcy \
            else _mod.RDLRCompUV.get_current_rate_point
        _cls = _mod.RDLRCompY if _mod is _rcy else _mod.RDLRCompUV

        def _gcrp(self, tool, *a, _o=_orig_gcrp, **kw):
            _install_cas3(tool)
            r = _o(self, tool, *a, **kw)
            n = _gcrp_count[0]
            _gcrp_count[0] += 1
            for j, tag in enumerate(("rate", "psnr", "msssim")):
                put(f"rdlr.gcrp.{n}.{tag}", float(r[j]))
            return r

        _cls.get_current_rate_point = _gcrp

    # The latent-domain optimized area (where the leaf sits inside the overlap
    # tile) is a process_y local; capture it off `tiling.assign_data`.
    import src.codec.coding_tools.rdlr.rdlr_comp_base as _rcb  # noqa: E402

    _orig_assign = _rcb.tiling.assign_data

    def _assign_data(t, area, src, *a, **kw):
        try:
            put("rdlr.lat_area", [area.position.y, area.position.x,
                                  area.size.height, area.size.width])
        except AttributeError:
            pass
        return _orig_assign(t, area, src, *a, **kw)

    _rcb.tiling.assign_data = _assign_data

    orig_y = RDLRCompY.rdlr_y_iteration

    def y_iter(self, tool, y_comp, original, decisions, optimizer, scheduler, area, ref, sup_info=None):
        i = sum(1 for n, _ in calls if n.startswith("rdlr.") and n.endswith(".loss"))
        if i == 0:
            # First call: record the whole decision context once.
            for k, v in decisions.items():
                if isinstance(v, torch.Tensor):
                    put(f"rdlr.decisions.{k}", v)
                elif isinstance(v, dict):
                    for kk, vv in v.items():
                        if isinstance(vv, torch.Tensor):
                            put(f"rdlr.decisions.{k}.{kk}", vv)
            for name, val in (
                ("sigma_out_precision", getattr(tool.common_modules.hyper_scale_decoder,
                                                "sigma_out_precision", 0)),
                ("log_k", getattr(tool.common_modules, "log_k", 0)),
                ("sigma_quant_min_log", float(np.log(getattr(
                    tool.common_modules, "sigma_quant_min", 1.0)))),
            ):
                put(f"rdlr.scalar.{name}", [float(val)])
        put(f"rdlr.{i}.y", y_comp)
        put(f"rdlr.{i}.original", original)
        try:
            put("rdlr.area", [area.position.y, area.position.x,
                              area.size.height, area.size.width])
        except AttributeError:
            put("rdlr.area", list(area))
        for j, tag in enumerate(("rate_ref", "psnr_ref", "msssim_ref")):
            put(f"rdlr.{i}.{tag}", float(ref[j]))
        # The differentiable leaf is the optimized-part tensor inside the
        # optimizer (process_y `assign_data`s it into the tile before each call,
        # keeping the graph). Capture its grad after the call's own
        # `loss.backward()` — this is the ground-truth gradient oracle.
        leaf = optimizer.param_groups[0]["params"][0]
        put(f"rdlr.{i}.optimized_part", leaf)
        r = orig_y(self, tool, y_comp, original, decisions, optimizer, scheduler, area, ref, sup_info)
        put(f"rdlr.{i}.loss", float(r))
        if leaf.grad is not None:
            put(f"rdlr.{i}.y_grad", leaf.grad)
        return r

    RDLRCompY.rdlr_y_iteration = y_iter

    base_parser = enc_mod.def_base_parser()
    coder = RecoEncoder(base_parser, enc_mod.def_encoder_parser_decorator(base_parser))
    log = io.StringIO()
    sys.argv = [sys.argv[0]] + enc_args
    with contextlib.redirect_stdout(log):
        process_encoder(coder, None, True, None, True, True)
    with open(os.path.join(out_dir, "encoder.log"), "w") as f:
        f.write(log.getvalue())

    lines, offset = [], 0
    with open(os.path.join(out_dir, "rdlr_tensors.bin"), "wb") as blob:
        for name, t in calls:
            tag, npdt = DTYPES[t.dtype]
            raw = t.numpy().astype(npdt, copy=False).tobytes()
            dims = " ".join(str(x) for x in t.shape)
            lines.append(f"{name} {tag} {t.dim()} {dims} {offset} {len(raw)}")
            blob.write(raw)
            offset += len(raw)
    with open(os.path.join(out_dir, "rdlr_manifest.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
