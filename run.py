#!/usr/bin/env python
"""Single-shot entry point: draw a random theta_fid, set up results/agent/, run the agent, print summary."""
import json
from pathlib import Path

import numpy as np
import yaml
from symbolic_pofk.syren_new import pnl_new_emulated

from judge.oracle import Oracle
from orchestrator.run_agent import setup_workdir, run_agent
from orchestrator.harvest import harvest_rollout


def _draw_valid_theta(rng, bounds, k_vec):
    for _ in range(100):
        theta = {k: float(rng.uniform(bounds[k]["min"], bounds[k]["max"]))
                 for k in ["om", "ob", "h", "ns", "as_", "w0"]}
        pk = pnl_new_emulated(k_vec, As=theta["as_"], Om=theta["om"], Ob=theta["ob"],
                              h=theta["h"], ns=theta["ns"], mnu=0.0, w0=theta["w0"], wa=0.0, a=1.0)
        if np.all(np.isfinite(pk)) and np.all(pk > 0) and np.all(pk < 1e10):
            return theta
    raise RuntimeError("Could not draw a valid theta_fid in 100 attempts")


def main():
    project_root = Path(__file__).parent
    cfg = yaml.safe_load(open(project_root / "config" / "prior_bounds.yaml"))
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    bounds, sigma_frac, epsilon = cfg["parameters"], cfg["noise"]["sigma_frac"], cfg["chi2"]["epsilon"]

    rng = np.random.default_rng()
    theta_fid = _draw_valid_theta(rng, bounds, k_vec)
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac,
                    seed=int(rng.integers(0, 2**31)))

    workdir = setup_workdir(project_root / "results" / "agent", oracle, project_root)
    print(f"workdir: {workdir}")
    run_agent(workdir, project_root)

    result = harvest_rollout(workdir, epsilon=epsilon)
    chi2_oracle = oracle.score(result.theta_agent)
    print(f"n_calls={result.n_calls}  chi2_min={result.chi2_min:.4f}  "
          f"chi2_oracle={chi2_oracle:.4f}  converged={result.converged}")
    print(f"theta={json.dumps(result.theta_agent)}")


if __name__ == "__main__":
    main()
