"""Optuna TPE baseline."""
import csv, time
from pathlib import Path

import numpy as np
import optuna
import yaml
from symbolic_pofk.syren_new import pnl_new_emulated

optuna.logging.set_verbosity(optuna.logging.WARNING)

_PARAM_KEYS = ["om", "ob", "h", "ns", "as_", "w0"]
_CSV_FIELDS = ["call_idx"] + _PARAM_KEYS + ["timestamp", "chi2", "notes"]


def run_optuna(workdir: Path, n_calls: int = 100, seed: int = 0):
    workdir = Path(workdir)
    with open(workdir / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    bounds = cfg["parameters"]
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    sigma_frac = cfg["noise"]["sigma_frac"]
    obs_pk = np.load(workdir / "obs_pk.npy")
    sigma = sigma_frac * obs_pk

    all_rows = []
    call_idx = [0]

    def objective(trial):
        params = {k: trial.suggest_float(k, bounds[k]["min"], bounds[k]["max"])
                  for k in _PARAM_KEYS}
        pk = pnl_new_emulated(k_vec, As=params["as_"], Om=params["om"], Ob=params["ob"],
                              h=params["h"], ns=params["ns"], mnu=0.0, w0=params["w0"],
                              wa=0.0, a=1.0)
        chi2 = float(np.sum(((pk - obs_pk) / sigma) ** 2) / len(obs_pk))
        call_idx[0] += 1
        all_rows.append({"call_idx": call_idx[0], **params,
                         "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                         "chi2": chi2, "notes": "optuna_tpe"})
        return chi2

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_calls)

    with open(workdir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        w.writeheader()
        w.writerows(all_rows)
