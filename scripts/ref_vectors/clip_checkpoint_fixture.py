#!/usr/bin/env python3
"""Write the small synthetic clipping checkpoint as hex text, without PyTorch.

Python's independent pickle and ZIP writers produce the PyTorch wire layout.
The only tensor is a synthetic one-element FloatStorage containing 0.25.
Timestamps are fixed so regenerating the text fixture is deterministic.
"""
import io
import pickle
import struct
import sys
import types
import zipfile
from collections import OrderedDict
from pathlib import Path

torch = types.ModuleType("torch")
utils = types.ModuleType("torch._utils")
sys.modules["torch"] = torch
sys.modules["torch._utils"] = utils


class FloatStorage:
    pass


FloatStorage.__module__ = "torch"
torch.FloatStorage = FloatStorage


def _rebuild_tensor_v2(storage, offset, shape, stride, requires_grad, hooks):
    return storage, offset, shape, stride


_rebuild_tensor_v2.__module__ = "torch._utils"
utils._rebuild_tensor_v2 = _rebuild_tensor_v2
storage = FloatStorage()


class Tensor:
    def __reduce__(self):
        return _rebuild_tensor_v2, (storage, 0, (1,), (1,), False, OrderedDict())


class Writer(pickle.Pickler):
    def persistent_id(self, obj):
        if obj is storage:
            return "storage", FloatStorage, "0", "cpu", 1
        return None


state = OrderedDict([
    ("conv1.weight", Tensor()),
    ("epoch", 3),
    ("clip_thres", {
        "E1B": {"MinList": [-1.5] * 160, "MaxList": [1.5] * 160},
        "HE1": {"MinList": [-2.0] * 64},
    }),
])
program = io.BytesIO()
Writer(program, protocol=2).dump(state)


class Reader(pickle.Unpickler):
    def persistent_load(self, pid):
        assert pid == ("storage", FloatStorage, "0", "cpu", 1)
        return "storage-0"


# Cross-check the writer's program with Python's independent pickle reader.
decoded = Reader(io.BytesIO(program.getvalue())).load()
assert decoded["conv1.weight"] == ("storage-0", 0, (1,), (1,))
assert decoded["clip_thres"] == state["clip_thres"]
assert decoded["epoch"] == 3
archive = io.BytesIO()
with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as z:
    for name, value in [
        ("archive/data.pkl", program.getvalue()),
        ("archive/data/0", struct.pack("<f", 0.25)),
        ("archive/version", b"3\n"),
        ("archive/byteorder", b"little"),
    ]:
        z.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), value)
with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as z:
    assert z.testzip() is None
    assert struct.unpack("<f", z.read("archive/data/0")) == (0.25,)
hex_text = archive.getvalue().hex()
Path(sys.argv[1]).write_text("\n".join(hex_text[i:i + 64] for i in range(0, len(hex_text), 64)) + "\n")
