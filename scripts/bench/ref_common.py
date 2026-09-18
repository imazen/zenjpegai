"""Shared scaffolding for the ref_*.py one-shot benchmark drivers.

Each driver runs in the reference checkout (cwd) with its venv active; python puts the
script's own directory on sys.path, so a plain `import ref_common` finds this file, and
`sys.path.insert(0, ".")` below makes the checkout's `src.` importable (the upstream
scripts' `PYTHONPATH=.` convention).
"""
import re
import sys
import time

T0 = time.time()
import torch  # noqa: E402

sys.path.insert(0, ".")


def threads(n):
    """Permit PyTorch n CPU threads: the stock decoder/encoder pin it to 1 via
    `torch.set_num_threads(1)`, so intercept the call (n = 1 keeps stock behaviour)."""
    if n != 1:
        real = torch.set_num_threads
        torch.set_num_threads = lambda _n: real(n)


def total_s(log_text):
    """The last `TOTAL: h:mm:ss.ss` line the reference logged, in seconds (nan if none)."""
    m = re.findall(r"TOTAL: (\d+):(\d+):([\d.]+)", log_text)
    if not m:
        return float("nan")
    h, mi, s = m[-1]
    return int(h) * 3600 + int(mi) * 60 + float(s)


def wall_s():
    """Whole-process wall time so far (Python start-up + model load included)."""
    return time.time() - T0
