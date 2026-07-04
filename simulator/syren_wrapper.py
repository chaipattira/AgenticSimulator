import csv
import time
from pathlib import Path

import numpy as np
from symbolic_pofk.syren_new import pnl_new_emulated


class OutOfPriorError(ValueError):
    pass


_PARAM_KEYS = ["om", "ob", "h", "ns", "as_", "w0"]
_CSV_FIELDS = ["call_idx"] + _PARAM_KEYS + ["timestamp", "chi2", "notes"]


class SyrenSimulator:
    """Wraps pnl_new_emulated with call counting and CSV logging."""

    def __init__(self, k_vec: np.ndarray, csv_path: Path, prior_bounds: dict | None = None):
        self.k_vec = k_vec
        self.csv_path = Path(csv_path)
        self.prior_bounds = prior_bounds
        if self.csv_path.exists():
            with open(self.csv_path) as f:
                self.call_count = sum(1 for _ in csv.DictReader(f))
        else:
            self.call_count = 0
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=_CSV_FIELDS).writeheader()

    def __call__(self, params: dict, chi2_fn=None, notes: str = "") -> np.ndarray:
        if self.prior_bounds is not None:
            for key, b in self.prior_bounds.items():
                if not (b["min"] <= params[key] <= b["max"]):
                    raise OutOfPriorError(f"{key}={params[key]} outside prior [{b['min']}, {b['max']}]")
        pk = pnl_new_emulated(
            self.k_vec, As=params["as_"], Om=params["om"], Ob=params["ob"],
            h=params["h"], ns=params["ns"], mnu=0.0, w0=params["w0"], wa=0.0, a=1.0,
        )
        self.call_count += 1
        chi2 = chi2_fn(pk) if chi2_fn is not None else None
        row = {"call_idx": self.call_count, **{k: params[k] for k in _PARAM_KEYS},
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
               "chi2": "" if chi2 is None else chi2, "notes": notes}
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=_CSV_FIELDS).writerow(row)
        return pk
