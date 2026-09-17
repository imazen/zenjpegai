#!/usr/bin/env python3
"""Clear the PF_X bit on PT_GNU_STACK for torch 1.10.2's shared objects.

Modern glibc refuses to dlopen a library that requests an executable stack
("cannot enable executable stack as shared object requires"). The old torch
wheels the JPEG AI reference software pins are marked that way. Run this once
inside the reference venv:

    python clear_execstack.py .venv/lib/python3.8/site-packages/torch
"""
import glob
import os
import struct
import sys

PT_GNU_STACK = 0x6474E551


def fix(path: str) -> bool:
    with open(path, "r+b") as f:
        ident = f.read(16)
        if ident[:4] != b"\x7fELF" or ident[4] != 2:  # ELF64 only
            return False
        f.seek(0x20)
        (e_phoff,) = struct.unpack("<Q", f.read(8))
        f.seek(0x36)
        e_phentsize, e_phnum = struct.unpack("<HH", f.read(4))
        changed = False
        for i in range(e_phnum):
            off = e_phoff + i * e_phentsize
            f.seek(off)
            p_type, p_flags = struct.unpack("<II", f.read(8))
            if p_type == PT_GNU_STACK and (p_flags & 1):
                f.seek(off + 4)
                f.write(struct.pack("<I", p_flags & ~1))
                changed = True
        return changed


def main() -> None:
    root = sys.argv[1]
    for so in glob.glob(os.path.join(root, "lib", "*.so*")) + glob.glob(os.path.join(root, "*.so")):
        if fix(so):
            print("cleared execstack:", os.path.basename(so))


if __name__ == "__main__":
    main()
