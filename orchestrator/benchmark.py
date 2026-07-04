"""
benchmark.py — run all methods on N rollouts and collect results.

Usage:
    python orchestrator/benchmark.py --n-rollouts 5 --agent-calls 100 --seed 0
"""
import argparse, csv, json
from pathlib import Path

import numpy as np
import yaml

from judge.oracle import Oracle
from orchestrator.run_agent import setup_workdir, run_agent
from orchestrator.harvest import harvest_rollout
from baselines.bo_botorch import run_botorch
from baselines.bo_optuna import run_optuna
from baselines.random_search import run_random_search


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-rollouts", type=int, default=5)
    p.add_argument("--agent-calls", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("benchmark_results"))
    args = p.parse_args()

    project_root = Path(__file__).parent.parent
    with open(project_root / "config" / "prior_bounds.yaml") as f:
        cfg = yaml.safe_load(f)
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])
    sigma_frac = cfg["noise"]["sigma_frac"]
    epsilon = cfg["chi2"]["epsilon"]
    bounds = cfg["parameters"]

    rng = np.random.default_rng(args.seed)
    summary_rows = []

    for rollout_idx in range(args.n_rollouts):
        rollout_seed = int(rng.integers(0, 2**31))
        theta_fid = {k: float(rng.uniform(bounds[k]["min"], bounds[k]["max"]))
                     for k in ["om", "ob", "h", "ns", "as_", "w0"]}
        oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec,
                        sigma_frac=sigma_frac, seed=rollout_seed)
        rollout_dir = args.out_dir / f"rollout_{rollout_idx:03d}"

        for method in ["agent", "random", "botorch", "optuna"]:
            workdir = setup_workdir(rollout_dir / method, oracle, project_root)

            if method == "agent":
                run_agent(workdir, project_root)
            elif method == "random":
                run_random_search(workdir, n_calls=args.agent_calls, seed=rollout_seed)
            elif method == "botorch":
                run_botorch(workdir, n_calls=args.agent_calls, seed=rollout_seed)
            elif method == "optuna":
                run_optuna(workdir, n_calls=args.agent_calls, seed=rollout_seed)

            result = harvest_rollout(workdir, epsilon=epsilon)
            chi2_oracle = oracle.score(result.theta_agent)
            summary_rows.append({
                "rollout_idx": rollout_idx,
                "method": method,
                "n_calls": result.n_calls,
                "chi2_min": result.chi2_min,
                "chi2_oracle": chi2_oracle,
                "converged": result.converged,
                "cpu_seconds": result.cpu_seconds,
                "theta_agent": json.dumps(result.theta_agent),
                "theta_fid": json.dumps(theta_fid),
            })
            print(f"[rollout {rollout_idx}] {method}: n_calls={result.n_calls} "
                  f"chi2_min={result.chi2_min:.2f} converged={result.converged}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.out_dir / "benchmark_summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
