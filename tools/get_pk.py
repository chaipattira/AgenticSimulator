#!/usr/bin/env python
"""
get_pk.py — shape diagnostic tool. Does NOT count toward call budget or append to runs.csv.
Outputs: JSON with keys k, pk, obs_pk, residual_frac
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml
from symbolic_pofk.syren_new import pnl_new_emulated


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    bounds = cfg["parameters"]
    obs_pk = np.load(cwd / "obs_pk.npy")

    params = json.loads(args.params)
    for key, b in bounds.items():
        if not (b["min"] <= params[key] <= b["max"]):
            print(f"ERROR: {key}={params[key]} outside prior [{b['min']}, {b['max']}]", file=sys.stderr)
            sys.exit(1)

    pk = pnl_new_emulated(k_vec, As=params["as_"], Om=params["om"], Ob=params["ob"],
                          h=params["h"], ns=params["ns"], mnu=0.0, w0=params["w0"], wa=0.0, a=1.0)
    print(json.dumps({
        "k": k_vec.tolist(),
        "pk": pk.tolist(),
        "obs_pk": obs_pk.tolist(),
        "residual_frac": ((pk - obs_pk) / obs_pk).tolist(),
    }))


if __name__ == "__main__":
    main()
