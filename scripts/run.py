#!/usr/bin/env python
"""Single-shot entry point: draw a random theta_fid, set up results/agent/, run the agent loop, print summary."""
import json
from pathlib import Path

import numpy as np

from config import load_config, make_k_vec
from judge.oracle import Oracle, draw_valid_theta_fid
from orchestrator.run_agent import setup_workdir, run_agent_loop
from orchestrator.harvest import harvest_rollout


def main():
    project_root = Path(__file__).parent.parent
    cfg = load_config(project_root)
    k_vec = make_k_vec(cfg)
    bounds, sigma_frac, epsilon = cfg["parameters"], cfg["noise"]["sigma_frac"], cfg["chi2"]["epsilon"]
    max_cpu_hours = cfg["budget"]["max_cpu_hours"]

    rng = np.random.default_rng()
    theta_fid = draw_valid_theta_fid(rng, bounds, k_vec)
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac,
                    seed=int(rng.integers(0, 2**31)))

    workdir = setup_workdir(project_root / "results" / "agent", oracle, project_root)
    print(f"workdir: {workdir}")
    run_agent_loop(workdir, project_root, max_cpu_hours=max_cpu_hours, epsilon=epsilon)

    result = harvest_rollout(workdir, epsilon=epsilon)
    chi2_oracle = oracle.score(result.theta_agent)
    print(f"n_calls={result.n_calls}  cpu_hours_total={result.cpu_hours_total:.4f}  "
          f"chi2_min={result.chi2_min:.4f}  chi2_oracle={chi2_oracle:.4f}  converged={result.converged}")
    print(f"theta={json.dumps(result.theta_agent)}")


if __name__ == "__main__":
    main()
