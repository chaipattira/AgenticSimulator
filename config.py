from pathlib import Path

import numpy as np
import yaml

PARAM_KEYS = ["om", "ob", "h", "ns", "as_", "w0"]

# MP-Gadget phase: a deliberate narrowing from PARAM_KEYS — h, ns, w0 stay fixed at
# fiducial values in this phase (see config/prior_bounds_mpgadget.yaml's `fixed` block),
# as_ is replaced by sigma8 (MP-Gadget's own paramfile normalization convention), and
# three sub-grid baryonic params are added.
MPGADGET_PARAM_KEYS = [
    "om", "ob", "sigma8", "wind_energy_fraction", "wind_speed_factor", "bh_feedback_factor",
]

# Shared runs.csv schema for the MP-Gadget phase — used by both tools/get_pk_ansatz.py
# (the free pre-screening tool, cpu_hours=0.0, tool="ansatz") and
# tools/run_mpgadget_trial.py (the one paid call per iteration, tool="mpgadget_trial").
# The `tool` column lets a reader (agent or human) distinguish free scans from the one
# paid call at a glance. Defined here (not in simulator/mpgadget_wrapper.py) so the free
# ansatz tool doesn't need to import the real-simulator wrapper module at all.
MPGADGET_CSV_FIELDS = (
    ["call_idx"] + MPGADGET_PARAM_KEYS
    + ["ngrid", "box_size_kpc", "cpu_hours", "tool", "timestamp", "chi2", "notes"]
)


def load_config(project_root: Path, filename: str = "prior_bounds.yaml") -> dict:
    return yaml.safe_load((Path(project_root) / "config" / filename).read_text())


def make_k_vec(cfg: dict) -> np.ndarray:
    kv = cfg["k_vector"]
    return np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
