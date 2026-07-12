#!/usr/bin/env python
"""
Run the Optuna TPE baseline against the same theta_fid/obs_pk pairs the
agent benchmark in results/benchmark/summary.jsonl was scored on.

Results are saved to results/baselines/optuna/summary.jsonl.
Each run gets its own workdir: results/baselines/optuna/run_{seed:05d}/

Usage:
    python scripts/run_optuna_baseline.py
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from baselines.optuna_baseline import run_one_optuna


def _load_seed_theta_fid(summary_path: Path) -> dict:
    """seed -> most-recent theta_fid, since summary.jsonl is append-only."""
    seed_to_theta = {}
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            seed_to_theta[rec["seed"]] = rec["theta_fid"]
    return seed_to_theta


def main():
    project_root = Path(__file__).parent.parent

    agent_summary_path = project_root / "results" / "benchmark" / "summary.jsonl"
    seed_to_theta = _load_seed_theta_fid(agent_summary_path)

    out_dir = project_root / "results" / "baselines" / "optuna"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    seeds = sorted(seed_to_theta)
    print(f"Launching {len(seeds)} Optuna studies (seeds={seeds})")

    records = []
    with ProcessPoolExecutor(max_workers=len(seeds)) as pool:
        futures = {
            pool.submit(run_one_optuna, seed, seed_to_theta[seed], project_root): seed
            for seed in seeds
        }
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                rec = fut.result()
                records.append(rec)
                with open(summary_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
                print(f"[seed={seed}] n_calls={rec['n_calls']}  chi2_min={rec['chi2_min']:.4f}  "
                      f"converged={rec['converged']}", flush=True)
            except Exception as e:
                print(f"[seed={seed}] FAILED: {e}", file=sys.stderr)

    if records:
        converged = [r for r in records if r["converged"]]
        print(f"\n=== Optuna baseline summary ===")
        print(f"Runs: {len(records)}  Converged: {len(converged)}/{len(records)}")
        print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
