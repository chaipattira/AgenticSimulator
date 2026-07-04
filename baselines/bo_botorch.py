"""BoTorch GP-BO baseline. Writes runs.csv in same format as compute_chi2.py."""
import csv, time
from pathlib import Path

import numpy as np
import torch
import yaml
from botorch.fit import fit_gpytorch_mll
from botorch.models import SingleTaskGP
from botorch.acquisition import LogExpectedImprovement
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
from symbolic_pofk.syren_new import pnl_new_emulated

_PARAM_KEYS = ["om", "ob", "h", "ns", "as_", "w0"]
_CSV_FIELDS = ["call_idx"] + _PARAM_KEYS + ["timestamp", "chi2", "notes"]


def _chi2(params: dict, obs_pk: np.ndarray, k_vec: np.ndarray, sigma_frac: float) -> float:
    pk = pnl_new_emulated(k_vec, As=params["as_"], Om=params["om"], Ob=params["ob"],
                          h=params["h"], ns=params["ns"], mnu=0.0, w0=params["w0"], wa=0.0, a=1.0)
    if not np.all(np.isfinite(pk)):
        return 1e10
    sigma = sigma_frac * obs_pk
    chi2 = float(np.sum(((pk - obs_pk) / sigma) ** 2) / len(obs_pk))
    return chi2 if np.isfinite(chi2) else 1e10


def run_botorch(workdir: Path, n_calls: int = 100, seed: int = 0):
    workdir = Path(workdir)
    with open(workdir / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    bounds_cfg = cfg["parameters"]
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    sigma_frac = cfg["noise"]["sigma_frac"]
    obs_pk = np.load(workdir / "obs_pk.npy")

    bounds_lo = torch.tensor([bounds_cfg[k]["min"] for k in _PARAM_KEYS], dtype=torch.float64)
    bounds_hi = torch.tensor([bounds_cfg[k]["max"] for k in _PARAM_KEYS], dtype=torch.float64)

    def denormalize(x: torch.Tensor) -> dict:
        raw = x * (bounds_hi - bounds_lo) + bounds_lo
        return {k: raw[i].item() for i, k in enumerate(_PARAM_KEYS)}

    torch.manual_seed(seed)
    n_init = max(3, min(10, n_calls // 10))
    X_init = torch.rand(n_init, len(_PARAM_KEYS), dtype=torch.float64)
    Y_init = []
    all_rows = []

    for i, x in enumerate(X_init):
        params = denormalize(x)
        chi2 = _chi2(params, obs_pk, k_vec, sigma_frac)
        Y_init.append(-chi2)
        all_rows.append({"call_idx": i + 1, **params,
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "chi2": chi2, "notes": "bo_init"})

    X = X_init
    Y = torch.tensor(Y_init, dtype=torch.float64).unsqueeze(-1)
    unit_bounds = torch.stack([torch.zeros(len(_PARAM_KEYS), dtype=torch.float64),
                               torch.ones(len(_PARAM_KEYS), dtype=torch.float64)])

    for call_idx in range(n_init + 1, n_calls + 1):
        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)

        acq = LogExpectedImprovement(gp, best_f=Y.max())
        candidate, _ = optimize_acqf(acq, bounds=unit_bounds, q=1,
                                     num_restarts=5, raw_samples=128)
        params = denormalize(candidate.squeeze(0))
        chi2 = _chi2(params, obs_pk, k_vec, sigma_frac)

        X = torch.cat([X, candidate], dim=0)
        Y = torch.cat([Y, torch.tensor([[-chi2]], dtype=torch.float64)], dim=0)
        all_rows.append({"call_idx": call_idx, **params,
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "chi2": chi2, "notes": "bo_acq"})

    with open(workdir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
