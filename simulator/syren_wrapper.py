import csv
import time
from pathlib import Path

import numpy as np
from symbolic_pofk.syren_new import pnl_new_emulated

from config import PARAM_KEYS


class OutOfPriorError(ValueError):
    pass


TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
CSV_FIELDS = ["call_idx"] + PARAM_KEYS + ["cpu_hours", "timestamp", "chi2", "notes"]


class SyrenSimulator:
    """
    Wraps pnl_new_emulated with call counting, cpu_hours cost tracking, and CSV logging.

    Mocks the resolution/volume-vs-cost tradeoff of a real N-body simulator: the caller
    chooses how many cpu_hours to spend on a call, and the returned pk carries synthetic
    realization noise sigma_realization = sigma0 / sqrt(cpu_hours) — the same 1/sqrt(N)
    scaling as shot noise from a finite-particle/finite-volume simulation. More cpu_hours
    buys a less noisy measurement of P(k) at that theta.
    """

    def __init__(self, k_vec: np.ndarray, csv_path: Path, prior_bounds: dict | None = None,
                 sigma0: float = 0.1):
        self.k_vec = k_vec
        self.csv_path = Path(csv_path)
        self.prior_bounds = prior_bounds
        self.sigma0 = sigma0
        self.rng = np.random.default_rng()
        if self.csv_path.exists():
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            self.call_count = len(rows)
            self.cpu_hours_total = sum(float(r["cpu_hours"]) for r in rows if r.get("cpu_hours"))
        else:
            self.call_count = 0
            self.cpu_hours_total = 0.0
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def __call__(self, params: dict, cpu_hours: float, chi2_fn=None, notes: str = "") -> tuple[np.ndarray, float | None]:
        if cpu_hours <= 0:
            raise ValueError(f"cpu_hours must be positive, got {cpu_hours}")
        if self.prior_bounds is not None:
            for key, b in self.prior_bounds.items():
                if not (b["min"] <= params[key] <= b["max"]):
                    raise OutOfPriorError(f"{key}={params[key]} outside prior [{b['min']}, {b['max']}]")

        pk_true = pnl_new_emulated(
            self.k_vec, As=params["as_"], Om=params["om"], Ob=params["ob"],
            h=params["h"], ns=params["ns"], mnu=0.0, w0=params["w0"], wa=0.0, a=1.0,
        )
        sigma_realization = (self.sigma0 / np.sqrt(cpu_hours)) * pk_true
        noise = self.rng.normal(0.0, 1.0, size=len(self.k_vec))
        pk_measured = pk_true + noise * sigma_realization

        self.call_count += 1
        self.cpu_hours_total += cpu_hours
        chi2 = chi2_fn(pk_measured, sigma_realization) if chi2_fn is not None else None
        row = {"call_idx": self.call_count, **{k: params[k] for k in PARAM_KEYS},
               "cpu_hours": cpu_hours,
               "timestamp": time.strftime(TIMESTAMP_FORMAT),
               "chi2": "" if chi2 is None else chi2, "notes": notes}
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        return pk_measured, chi2
