#!/usr/bin/env python
"""
compute_chi2.py — bash-callable agent tool.
Expects in CWD: obs_pk.npy, config/prior_bounds.yaml
Appends to runs.csv (created on first call). Outputs: chi2=<value>  call_idx=<N>
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
    p.add_argument("--notes", default="")
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    sigma_frac = cfg["noise"]["sigma_frac"]
    obs_pk = np.load(cwd / "obs_pk.npy")
    sigma = sigma_frac * obs_pk
    chi2_fn = lambda pk: float(np.sum(((pk - obs_pk) / sigma) ** 2) / len(obs_pk))

    params = json.loads(args.params)
    sim = SyrenSimulator(k_vec=k_vec, csv_path=cwd / "runs.csv",
                         prior_bounds=cfg["parameters"])
    try:
        pk = sim(params, chi2_fn=chi2_fn, notes=args.notes)
    except OutOfPriorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"chi2={chi2_fn(pk):.6f}  call_idx={sim.call_count}")


if __name__ == "__main__":
    main()
