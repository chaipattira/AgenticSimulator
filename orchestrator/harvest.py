import csv
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class RolloutResult:
    n_calls: int
    chi2_final: float
    chi2_min: float
    theta_agent: dict
    converged: bool
    cpu_seconds: float
    workdir: Path


def harvest_rollout(workdir: Path, epsilon: float = 50.0) -> RolloutResult:
    csv_path = Path(workdir) / "runs.csv"
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        raise ValueError(f"runs.csv in {workdir} is empty")

    best_row = min(rows, key=lambda r: float(r["chi2"]) if r["chi2"] else float("inf"))
    last_row = rows[-1]

    param_keys = ["om", "ob", "h", "ns", "as_", "w0"]
    theta_agent = {k: float(best_row[k]) for k in param_keys}
    chi2_min = float(best_row["chi2"])
    chi2_final = float(last_row["chi2"]) if last_row["chi2"] else chi2_min

    try:
        fmt = "%Y-%m-%dT%H:%M:%S"
        cpu_seconds = (datetime.strptime(last_row["timestamp"], fmt) -
                       datetime.strptime(rows[0]["timestamp"], fmt)).total_seconds()
    except (KeyError, ValueError):
        cpu_seconds = 0.0

    return RolloutResult(
        n_calls=len(rows),
        chi2_final=chi2_final,
        chi2_min=chi2_min,
        theta_agent=theta_agent,
        converged=chi2_min < epsilon,
        cpu_seconds=cpu_seconds,
        workdir=Path(workdir),
    )
