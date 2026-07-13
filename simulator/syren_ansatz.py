"""
SyrenAnsatz — a free, unlimited-call heuristic wrapper around syren_new/symbolic_pofk for
the MP-Gadget phase. Separate from and does NOT modify simulator/syren_wrapper.py::
SyrenSimulator, which stays exactly as-is for the syren_new-only MVP experiment.

Two syren_new-family wrappers now exist, deliberately different:
  - SyrenSimulator (simulator/syren_wrapper.py): the MVP's own simulator. Takes a
    caller-chosen `cpu_hours` and adds realization noise sigma_realization =
    sigma0/sqrt(cpu_hours) — this IS the thing being calibrated in the syren_new-only
    experiment, so its noise model mocks a resolution/volume-vs-cost tradeoff.
  - SyrenAnsatz (this file): used only as a free pre-screening heuristic ahead of a real,
    costly MP-Gadget trial. No cpu_hours argument, no realization-noise dial — just a
    single fixed cosmic-variance-style fractional noise term on top of a cheap P(k) proxy.

IMPORTANT FINDING that shapes this module — read before changing which emulator this
calls: MP-Gadget's canonical k-grid (logspace(log10(2), log10(15), 25) h/Mpc) lies entirely
outside syren_new's own validated k-range (0.01-1.0 h/Mpc). An earlier version of this
module called the *nonlinear* emulator, `symbolic_pofk.syren_new.pnl_new_emulated`, on the
MP-Gadget grid — empirically this does not just lose accuracy, it diverges catastrophically
(P(k) values as large as 1e154 at k=15 h/Mpc for WMAP9-like inputs, confirmed by direct
testing). That symbolic-regression fit is simply not defined outside its trained regime and
is unusable there. This module instead calls `symbolic_pofk.linear.plin_emulated` (the
*linear* matter power spectrum emulator from the same Bartlett et al. package family) with
its default `extrapolate=False`, which falls back to an Eisenstein & Hu 1998 fit for
k outside `[kmin, kmax]` — well-behaved (finite, positive, smoothly declining) across the
full MP-Gadget k-range, confirmed by direct testing. This is a real accuracy tradeoff (linear
theory misses nonlinear clustering and all baryonic/feedback effects — exactly the
small-scale, high-k regime the MP-Gadget phase cares about) but a *usable* free heuristic
beats an unusable one; the module docstring and the CLAUDE.md template both call this out
explicitly so the agent treats the ansatz as directional only, never quantitative.

`plin_emulated` takes `sigma8` directly (unlike `pnl_new_emulated`, which takes `As`), so
no sigma8->As unit conversion is needed here.
"""
import numpy as np
from symbolic_pofk.linear import plin_emulated


class SyrenAnsatz:
    """Free, unlimited-call heuristic P(k) proxy. No cpu_hours argument exists — there is
    no cost dial. Only a fixed fractional noise `sigma0` (cosmic-variance-style, not
    resolution-dependent) is added."""

    def __init__(self, k_vec: np.ndarray, fixed: dict, sigma0: float = 0.02):
        self.k_vec = k_vec
        self.fixed = fixed  # {"h": ..., "ns": ..., "w0": ...} — never agent-chosen
        self.sigma0 = sigma0
        self.rng = np.random.default_rng()

    def __call__(self, params: dict) -> np.ndarray:
        """params must contain om, ob, sigma8 (any other keys, e.g. the sub-grid
        parameters, are ignored — this linear-theory proxy has no notion of them, and
        w0 plays no role in the linear P(k) shape at a=1 either — only the fixed
        cosmological shape parameters om/ob/h/ns/sigma8 matter here)."""
        h, ns = self.fixed["h"], self.fixed["ns"]
        pk_true = plin_emulated(
            self.k_vec, sigma8=params["sigma8"], Om=params["om"], Ob=params["ob"],
            h=h, ns=ns, extrapolate=False,
        )
        noise = self.rng.normal(0.0, self.sigma0, size=len(self.k_vec))
        return pk_true * (1.0 + noise)
