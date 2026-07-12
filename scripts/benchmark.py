#!/usr/bin/env python
"""
Run N parallel agent rollouts with independent theta_fid draws.
Results (including theta_fid) are saved to results/benchmark/summary.jsonl.
Each run gets its own workdir: results/benchmark/run_{seed:05d}/

Usage:
    python benchmark.py [--n 10] [--seeds 42 43 44 ...]
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import load_config, make_k_vec
from judge.oracle import Oracle, draw_valid_theta_fid
from orchestrator.run_agent import setup_workdir, run_agent_loop
from orchestrator.harvest import harvest_rollout


def run_one(seed: int, project_root: Path, epsilon: float) -> dict:
    project_root = Path(project_root)
    cfg = load_config(project_root)
    k_vec = make_k_vec(cfg)
    bounds, sigma_frac = cfg["parameters"], cfg["noise"]["sigma_frac"]
    max_cpu_hours = cfg["budget"]["max_cpu_hours"]

    rng = np.random.default_rng(seed)
    theta_fid = draw_valid_theta_fid(rng, bounds, k_vec)
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac,
                    seed=int(rng.integers(0, 2**31)))

    workdir = project_root / "results" / "benchmark" / f"run_{seed:05d}"
    setup_workdir(workdir, oracle, project_root)
    run_agent_loop(workdir, project_root, max_cpu_hours=max_cpu_hours, epsilon=epsilon)

    result = harvest_rollout(workdir, epsilon=epsilon)
    chi2_oracle = oracle.score(result.theta_agent)

    record = {
        "seed": seed,
        "theta_fid": theta_fid,
        "theta_agent": result.theta_agent,
        "n_calls": result.n_calls,
        "cpu_hours_total": result.cpu_hours_total,
        "chi2_min": result.chi2_min,
        "chi2_oracle": chi2_oracle,
        "converged": result.converged,
        "cpu_seconds": result.cpu_seconds,
    }
    print(f"[seed={seed}] n_calls={result.n_calls}  cpu_hours_total={result.cpu_hours_total:.4f}  "
          f"chi2_min={result.chi2_min:.4f}  chi2_oracle={chi2_oracle:.4f}  converged={result.converged}",
          flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of runs")
    parser.add_argument("--seeds", type=int, nargs="*", help="Explicit seed list (overrides --n)")
    parser.add_argument("--workers", type=int, default=None, help="Max parallel workers (default: N)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cfg = load_config(project_root)
    epsilon = cfg["chi2"]["epsilon"]

    seeds = args.seeds if args.seeds else list(range(args.n))
    max_workers = args.workers or len(seeds)

    out_dir = project_root / "results" / "benchmark"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    print(f"Launching {len(seeds)} runs (seeds={seeds}) with max_workers={max_workers}")

    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, s, project_root, epsilon): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                rec = fut.result()
                records.append(rec)
                with open(summary_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[seed={seed}] FAILED: {e}", file=sys.stderr)

    # Final summary
    if records:
        converged = [r for r in records if r["converged"]]
        cpu_hours = [r["cpu_hours_total"] for r in converged]
        print(f"\n=== Benchmark summary ===")
        print(f"Runs: {len(records)}  Converged: {len(converged)}/{len(records)}")
        if cpu_hours:
            print(f"cpu_hours to convergence — mean: {np.mean(cpu_hours):.2f}  "
                  f"median: {np.median(cpu_hours):.2f}  min: {min(cpu_hours):.2f}  max: {max(cpu_hours):.2f}")
        print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
