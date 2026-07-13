#!/usr/bin/env python
"""
get_pk_ansatz.py — FREE, UNLIMITED-CALL pre-screening tool for the MP-Gadget phase.

There is no --cpu_hours flag: there is no cost dial. Calls to this tool do NOT count
toward the cpu_hours budget in runs.csv (each row is logged with cpu_hours=0.0,
tool="ansatz") — call it as many times as useful for cheap pre-screening BEFORE
committing to the one paid run_mpgadget_trial.py call each iteration is allowed.

CAVEAT (see simulator/syren_ansatz.py's module docstring for the full story): this proxies
a LINEAR-theory-only, sub-grid-blind heuristic. It has no notion of nonlinear clustering,
baryonic feedback, or the wind/black-hole parameters at all — varying those leaves its
output unchanged. Use it for qualitative cosmological trend sense only (does raising sigma8
raise power, roughly how much), never as a quantitative substitute for a real MP-Gadget
measurement.

Expects in CWD: obs_pk.npy, config/prior_bounds_mpgadget.yaml.
Appends to runs.csv (created on first call, header shared with run_mpgadget_trial.py).
Outputs: JSON with keys k, pk, obs_pk, residual_frac, cpu_hours_spent (always 0.0),
cpu_hours_total, tool.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import MPGADGET_CSV_FIELDS, MPGADGET_PARAM_KEYS, make_k_vec
from simulator.syren_ansatz import SyrenAnsatz


def _load_or_init_csv(csv_path: Path) -> list[dict]:
    if csv_path.exists():
        return list(csv.DictReader(open(csv_path)))
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=MPGADGET_CSV_FIELDS).writeheader()
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--notes", default="ansatz")
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds_mpgadget.yaml") as f:
        cfg = yaml.safe_load(f)
    k_vec = make_k_vec(cfg)
    sigma0 = cfg["noise"]["ansatz_sigma0"]
    obs_pk = np.load(cwd / "obs_pk.npy")

    params = json.loads(args.params)
    ansatz = SyrenAnsatz(k_vec=k_vec, fixed=cfg["fixed"], sigma0=sigma0)
    pk = ansatz(params)

    csv_path = cwd / "runs.csv"
    rows = _load_or_init_csv(csv_path)
    call_idx = len(rows) + 1
    cpu_hours_total = sum(float(r["cpu_hours"]) for r in rows if r.get("cpu_hours"))

    row = {
        "call_idx": call_idx,
        **{k: params.get(k, "") for k in MPGADGET_PARAM_KEYS},
        "ngrid": "", "box_size_kpc": "",
        "cpu_hours": 0.0, "tool": "ansatz",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chi2": "", "notes": args.notes,
    }
    with open(csv_path, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=MPGADGET_CSV_FIELDS).writerow(row)

    print(json.dumps({
        "k": k_vec.tolist(),
        "pk": pk.tolist(),
        "obs_pk": obs_pk.tolist(),
        "residual_frac": ((pk - obs_pk) / obs_pk).tolist(),
        "cpu_hours_spent": 0.0,
        "cpu_hours_total": cpu_hours_total,  # unaffected by this call
        "tool": "ansatz",
    }))


if __name__ == "__main__":
    main()
