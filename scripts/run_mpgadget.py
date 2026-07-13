#!/usr/bin/env python
"""
Single-shot entry point for the MP-Gadget phase: draw a random theta_fid over the 6
MP-Gadget tunable params, build a real-simulator MPGadgetOracle, set up results/agent_mpgadget/,
run the agent loop with the mpgadget backend, print summary.

Parallel to scripts/run.py, but every simulator call here — the oracle's ground-truth run,
its final scoring run, and every agent trial — is a real SLURM round trip, not a
milliseconds-fast in-process call. Budget accordingly before running this.
"""
import json
from pathlib import Path

import numpy as np

from config import MPGADGET_PARAM_KEYS, default_shenqi_root, load_config, make_k_vec
from judge.oracle import MPGadgetOracle, draw_valid_theta_fid_mpgadget
from orchestrator.harvest import harvest_rollout
from orchestrator.run_agent import run_agent_loop, setup_workdir
from simulator.mpgadget_wrapper import MPGadgetSimulator

# See docs/superpowers/specs/2026-07-13-mpgadget-agent-integration-design.md section 4 for
# why this is much larger than scripts/run.py's implicit 600s default: a real SLURM job pair
# can take up to ~60 minutes of walltime plus unbounded queue wait on a shared partition.
# Provisional, not empirically measured.
_MPGADGET_ITERATION_TIMEOUT = 7200


def main():
    project_root = Path(__file__).parent.parent
    cfg = load_config(project_root, filename="prior_bounds_mpgadget.yaml")
    k_vec = make_k_vec(cfg)
    bounds, fixed = cfg["parameters"], cfg["fixed"]
    sigma_frac, epsilon = cfg["noise"]["sigma_frac"], cfg["chi2"]["epsilon"]
    max_cpu_hours = cfg["budget"]["max_cpu_hours"]

    rng = np.random.default_rng()
    theta_fid = draw_valid_theta_fid_mpgadget(rng, bounds, fixed, k_vec)

    out_dir = project_root / "results" / "agent_mpgadget"
    mpgadget_sim = MPGadgetSimulator(
        shenqi_root=default_shenqi_root(project_root),
        csv_path=out_dir / "oracle_runs.csv",  # the ORACLE's own real-run log — never the agent's runs.csv
    )
    oracle = MPGadgetOracle(theta_fid=theta_fid, sigma_frac=sigma_frac, mpgadget_sim=mpgadget_sim,
                            ground_truth_workdir=out_dir / "ground_truth", seed=int(rng.integers(0, 2**31)))

    workdir = setup_workdir(out_dir, oracle, project_root, backend="mpgadget")
    print(f"workdir: {workdir}")
    run_agent_loop(workdir, project_root, max_cpu_hours=max_cpu_hours, epsilon=epsilon,
                   iteration_timeout=_MPGADGET_ITERATION_TIMEOUT)

    result = harvest_rollout(workdir, epsilon=epsilon, param_keys=MPGADGET_PARAM_KEYS)
    chi2_oracle = oracle.score(result.theta_agent, workdir=out_dir / "score_run")
    print(f"n_calls={result.n_calls}  cpu_hours_total={result.cpu_hours_total:.4f}  "
          f"chi2_min={result.chi2_min:.4f}  chi2_oracle={chi2_oracle:.4f}  converged={result.converged}")
    print(f"theta={json.dumps(result.theta_agent)}")


if __name__ == "__main__":
    main()
