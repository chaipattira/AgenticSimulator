#!/usr/bin/env python
"""
get_pk.py — shape diagnostic tool. Counts toward cpu_hours budget and appends to runs.csv.
Outputs: JSON with keys k, pk, obs_pk, residual_frac, cpu_hours_spent, cpu_hours_total

--cpu_hours controls the resolution/volume tradeoff: more cpu_hours -> less realization
noise on this call's pk (sigma_realization = sigma0 / sqrt(cpu_hours)).
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import make_k_vec
from simulator.syren_wrapper import SyrenSimulator, OutOfPriorError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--cpu_hours", type=float, required=True,
                    help="Compute budget for this call; more cpu_hours -> less realization noise")
    p.add_argument("--notes", default="get_pk")
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    k_vec = make_k_vec(cfg)
    sigma0 = cfg["noise"]["sigma0_realization"]
    obs_pk = np.load(cwd / "obs_pk.npy")

    params = json.loads(args.params)
    sim = SyrenSimulator(k_vec=k_vec, csv_path=cwd / "runs.csv",
                         prior_bounds=cfg["parameters"], sigma0=sigma0)
    try:
        pk, _ = sim(params, cpu_hours=args.cpu_hours, chi2_fn=None, notes=args.notes)
    except OutOfPriorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "k": k_vec.tolist(),
        "pk": pk.tolist(),
        "obs_pk": obs_pk.tolist(),
        "residual_frac": ((pk - obs_pk) / obs_pk).tolist(),
        "cpu_hours_spent": args.cpu_hours,
        "cpu_hours_total": sim.cpu_hours_total,
    }))


if __name__ == "__main__":
    main()
