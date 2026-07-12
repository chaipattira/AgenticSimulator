#!/usr/bin/env python
"""
compute_chi2.py — bash-callable agent tool.
Expects in CWD: obs_pk.npy, config/prior_bounds.yaml
Appends to runs.csv (created on first call). Outputs: chi2=<value>  call_idx=<N>  cpu_hours_spent=<v>  cpu_hours_total=<v>

--cpu_hours controls the resolution/volume tradeoff: more cpu_hours -> less realization
noise on this call's pk (sigma_realization = sigma0 / sqrt(cpu_hours)), at the cost of
counting more against the budget.
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import make_k_vec
from judge.chi2 import compute_chi2
from simulator.syren_wrapper import SyrenSimulator, OutOfPriorError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--cpu_hours", type=float, required=True,
                    help="Compute budget for this call; more cpu_hours -> less realization noise")
    p.add_argument("--notes", default="")
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    k_vec = make_k_vec(cfg)
    sigma_frac = cfg["noise"]["sigma_frac"]
    sigma0 = cfg["noise"]["sigma0_realization"]
    obs_pk = np.load(cwd / "obs_pk.npy")
    sigma_obs = sigma_frac * obs_pk

    def chi2_fn(pk_measured, sigma_realization):
        sigma_eff = np.sqrt(sigma_obs ** 2 + sigma_realization ** 2)
        return compute_chi2(pk_measured, obs_pk, sigma_eff)

    params = json.loads(args.params)
    sim = SyrenSimulator(k_vec=k_vec, csv_path=cwd / "runs.csv",
                         prior_bounds=cfg["parameters"], sigma0=sigma0)
    try:
        pk, chi2 = sim(params, cpu_hours=args.cpu_hours, chi2_fn=chi2_fn, notes=args.notes)
    except OutOfPriorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"chi2={chi2:.6f}  call_idx={sim.call_count}  "
          f"cpu_hours_spent={args.cpu_hours}  cpu_hours_total={sim.cpu_hours_total:.4f}")


if __name__ == "__main__":
    main()
