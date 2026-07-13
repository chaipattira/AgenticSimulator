"""Parses MP-Gadget's own powerspectrum-<time>.txt output and interpolates it onto
a fixed canonical k-grid. See shenqi/libgadget/powerspectrum.cpp:91-117 for the
writer: columns are `k P N P(z=0)` in Mpc/h units; we use column 2 (raw P at this
snapshot's time), not column 4 (P(z=0) = P/D1^2 — a linear-growth extrapolation
that's redundant at our snapshots, which are already written at Time=1.0)."""
from pathlib import Path

import numpy as np


def read_powerspectrum(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    k_raw, p_raw = data[:, 0], data[:, 1]
    order = np.argsort(k_raw)
    return k_raw[order], p_raw[order]


def interpolate_to_grid(k_raw: np.ndarray, p_raw: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Log-log interpolation — P(k) is close to a power law locally, so this is far
    more accurate than linear interpolation. k_grid must lie within [k_raw.min(), k_raw.max()]
    for every legal (ngrid, box_size) choice; see the design spec for why that's guaranteed."""
    log_p = np.interp(np.log(k_grid), np.log(k_raw), np.log(p_raw))
    return np.exp(log_p)
