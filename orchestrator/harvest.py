import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RolloutResult:
    n_calls: int
    chi2_final: float      # chi2 of best_params.json (from runs.csv row)
    chi2_min: float        # minimum chi2 seen across all calls
    theta_agent: dict      # parameters from best_params.json
    converged: bool
    cpu_seconds: float     # wall-clock time from first to last runs.csv timestamp
    workdir: Path


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

    def _chi2(row: dict) -> float:
        try:
            return float(row["chi2"]) if row["chi2"] else float("inf")
        except (ValueError, KeyError):
            return float("inf")

    chi2_min = min(_chi2(r) for r in rows)

    # cpu_seconds: wall-clock from first to last timestamp
    fmt = "%Y-%m-%dT%H:%M:%S"
    try:
        t0 = datetime.strptime(rows[0]["timestamp"], fmt)
        t1 = datetime.strptime(rows[-1]["timestamp"], fmt)
        cpu_seconds = (t1 - t0).total_seconds()
    except (KeyError, ValueError):
        cpu_seconds = 0.0

    # chi2_final: the chi2 of the committed best_params.json, found by matching
    # theta_agent back to runs.csv. Fall back to chi2_min if no exact match.
    param_keys = ["om", "ob", "h", "ns", "as_", "w0"]
    chi2_final = chi2_min
    for row in rows:
        try:
            if all(abs(float(row[k]) - theta_agent[k]) < 1e-12 for k in param_keys):
                chi2_final = _chi2(row)
                break
        except (KeyError, ValueError):
            continue

    return RolloutResult(
        n_calls=len(rows),
        chi2_final=chi2_final,
        chi2_min=chi2_min,
        theta_agent=theta_agent,
        converged=chi2_min < epsilon,
        cpu_seconds=cpu_seconds,
        workdir=workdir,
    )
