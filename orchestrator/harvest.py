import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from config import PARAM_KEYS, load_config, make_k_vec
from judge.chi2 import compute_chi2
from simulator.syren_wrapper import TIMESTAMP_FORMAT


@dataclass
class RolloutResult:
    n_calls: int
    cpu_hours_total: float  # total simulated compute spent — the primary benchmark metric
    chi2_final: float      # chi2 of best_params.json (from runs.csv row)
    chi2_min: float        # minimum chi2 seen across all calls
    theta_agent: dict      # parameters from best_params.json
    converged: bool
    cpu_seconds: float     # real wall-clock time (harness runtime), NOT the mocked cpu_hours budget
    workdir: Path


def _row_chi2(row: dict) -> float:
    try:
        return float(row["chi2"]) if row["chi2"] else float("inf")
    except (ValueError, KeyError):
        return float("inf")


def _row_cpu_hours(row: dict) -> float:
    try:
        return float(row["cpu_hours"]) if row.get("cpu_hours") else 0.0
    except (ValueError, KeyError):
        return 0.0


def should_stop(workdir: Path, epsilon: float, max_cpu_hours: float) -> bool:
    """True if runs.csv shows convergence or budget exhaustion. Used by run_agent_loop
    to decide whether to spawn another iteration; independent of agent self-reporting."""
    csv_path = Path(workdir) / "runs.csv"
    if not csv_path.exists():
        return False
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return False
    chi2_min = min(_row_chi2(r) for r in rows)
    cpu_hours_total = sum(_row_cpu_hours(r) for r in rows)
    return chi2_min < epsilon or cpu_hours_total >= max_cpu_hours


def _compute_chi2_from_obs(workdir: Path, theta: dict) -> float:
    """Compute chi2 against obs_pk.npy without spending a budget call."""
    from symbolic_pofk.syren_new import pnl_new_emulated

    workdir = Path(workdir)
    cfg = load_config(workdir)
    k_vec = make_k_vec(cfg)
    sigma_frac = cfg["noise"]["sigma_frac"]

    obs_pk = np.load(workdir / "obs_pk.npy")
    pk_sim = pnl_new_emulated(
        k_vec,
        As=theta["as_"], Om=theta["om"], Ob=theta["ob"],
        h=theta["h"], ns=theta["ns"], mnu=0.0,
        w0=theta["w0"], wa=0.0, a=1.0,
    )
    sigma = sigma_frac * obs_pk
    return compute_chi2(pk_sim, obs_pk, sigma)


def harvest_rollout(workdir: Path, epsilon: float = 50.0) -> RolloutResult:
    workdir = Path(workdir)

    best_params_path = workdir / "best_params.json"
    if not best_params_path.exists():
        raise FileNotFoundError(f"best_params.json not found in {workdir}")
    theta_agent = json.loads(best_params_path.read_text())

    csv_path = workdir / "runs.csv"
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"runs.csv in {workdir} is empty")

    chi2_min = min(_row_chi2(r) for r in rows)

    # cpu_seconds: wall-clock from first to last timestamp
    try:
        t0 = datetime.strptime(rows[0]["timestamp"], TIMESTAMP_FORMAT)
        t1 = datetime.strptime(rows[-1]["timestamp"], TIMESTAMP_FORMAT)
        cpu_seconds = (t1 - t0).total_seconds()
    except (KeyError, ValueError):
        cpu_seconds = 0.0

    # chi2_final: match theta_agent back to runs.csv for an exact chi2.
    # If no match (e.g. agent used get_pk-based optimizer that leaves chi2 blank),
    # compute chi2 directly from obs_pk.npy so harvest is always accurate.
    chi2_final = float("inf")
    for row in rows:
        try:
            if all(abs(float(row[k]) - theta_agent[k]) < 1e-12 for k in PARAM_KEYS):
                chi2_final = _row_chi2(row)
                break
        except (KeyError, ValueError):
            continue

    if chi2_final == float("inf"):
        chi2_final = _compute_chi2_from_obs(workdir, theta_agent)

    chi2_min = min(chi2_min, chi2_final)
    cpu_hours_total = sum(_row_cpu_hours(r) for r in rows)

    return RolloutResult(
        n_calls=len(rows),
        cpu_hours_total=cpu_hours_total,
        chi2_final=chi2_final,
        chi2_min=chi2_min,
        theta_agent=theta_agent,
        converged=chi2_min < epsilon,
        cpu_seconds=cpu_seconds,
        workdir=workdir,
    )
