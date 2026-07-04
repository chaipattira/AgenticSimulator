#!/usr/bin/env python
"""
get_pk.py — shape diagnostic tool. Counts toward call budget and appends to runs.csv.
Outputs: JSON with keys k, pk, obs_pk, residual_frac
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from simulator.syren_wrapper import SyrenSimulator, OutOfPriorError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--notes", default="get_pk")
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    obs_pk = np.load(cwd / "obs_pk.npy")

    params = json.loads(args.params)
    sim = SyrenSimulator(k_vec=k_vec, csv_path=cwd / "runs.csv",
                         prior_bounds=cfg["parameters"])
    try:
        pk = sim(params, chi2_fn=None, notes=args.notes)
    except OutOfPriorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps({
        "k": k_vec.tolist(),
        "pk": pk.tolist(),
        "obs_pk": obs_pk.tolist(),
        "residual_frac": ((pk - obs_pk) / obs_pk).tolist(),
    }))


if __name__ == "__main__":
    main()
