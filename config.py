from pathlib import Path

import numpy as np
import yaml

PARAM_KEYS = ["om", "ob", "h", "ns", "as_", "w0"]


def load_config(project_root: Path) -> dict:
    return yaml.safe_load((Path(project_root) / "config" / "prior_bounds.yaml").read_text())


def make_k_vec(cfg: dict) -> np.ndarray:
    kv = cfg["k_vector"]
    return np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
