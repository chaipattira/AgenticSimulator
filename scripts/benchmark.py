#!/usr/bin/env python
"""
Run N agent rollouts (default 1) against a chosen simulator backend, with independent
theta_fid draws. Results (including theta_fid) are saved to
results/benchmark/<backend>/summary.jsonl. Each run gets its own workdir:
results/benchmark/<backend>/run_{seed:05d}/

Usage:
    python scripts/benchmark.py --backend syren_new [--n 10] [--seeds 42 43 44]
    python scripts/benchmark.py --backend mpgadget [--n 1]

For "mpgadget", every simulator call — the oracle's ground-truth run, its final scoring
run, and every agent trial — is a real SLURM round trip, not a milliseconds-fast in-process
call. Budget accordingly. Parallel rollouts default to sequential (--workers 1) for this
backend, since Anvil enforces a per-user concurrent SLURM job submission limit that
independent parallel rollouts can trip; override with --workers if you know your quota
allows more.
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import MPGADGET_PARAM_KEYS, default_shenqi_root, load_config, make_k_vec
from judge.oracle import MPGadgetOracle, Oracle, draw_valid_theta_fid, draw_valid_theta_fid_mpgadget
from orchestrator.harvest import harvest_rollout
from orchestrator.run_agent import run_agent_loop, setup_workdir
from simulator.mpgadget_wrapper import MPGadgetSimulator

_MPGADGET_ITERATION_TIMEOUT = 7200  # real SLURM job pair can take a long time plus queue wait
_CONFIG_FILENAME = {"syren_new": "prior_bounds.yaml", "mpgadget": "prior_bounds_mpgadget.yaml"}


def run_one(seed: int, project_root: Path, backend: str) -> dict:
    project_root = Path(project_root)
    cfg = load_config(project_root, filename=_CONFIG_FILENAME[backend])
    k_vec = make_k_vec(cfg)
    epsilon = cfg["chi2"]["epsilon"]
    max_cpu_hours = cfg["budget"]["max_cpu_hours"]
    sigma_frac = cfg["noise"]["sigma_frac"]

    rng = np.random.default_rng(seed)
    workdir = project_root / "results" / "benchmark" / backend / f"run_{seed:05d}"

    if backend == "syren_new":
        theta_fid = draw_valid_theta_fid(rng, cfg["parameters"], k_vec)
        oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac,
                        seed=int(rng.integers(0, 2**31)))
        setup_workdir(workdir, oracle, project_root, backend="syren_new")
        run_agent_loop(workdir, project_root, max_cpu_hours=max_cpu_hours, epsilon=epsilon)
        result = harvest_rollout(workdir, epsilon=epsilon)
        chi2_oracle = oracle.score(result.theta_agent)
    else:  # mpgadget
        theta_fid = draw_valid_theta_fid_mpgadget(rng, cfg["parameters"], cfg["fixed"], k_vec)
        mpgadget_sim = MPGadgetSimulator(
            shenqi_root=default_shenqi_root(project_root), csv_path=workdir / "oracle_runs.csv",
        )
        oracle = MPGadgetOracle(theta_fid=theta_fid, sigma_frac=sigma_frac, mpgadget_sim=mpgadget_sim,
                                ground_truth_workdir=workdir / "ground_truth",
                                seed=int(rng.integers(0, 2**31)))
        setup_workdir(workdir, oracle, project_root, backend="mpgadget")
        run_agent_loop(workdir, project_root, max_cpu_hours=max_cpu_hours, epsilon=epsilon,
                       iteration_timeout=_MPGADGET_ITERATION_TIMEOUT)
        result = harvest_rollout(workdir, epsilon=epsilon, param_keys=MPGADGET_PARAM_KEYS)
        chi2_oracle = oracle.score(result.theta_agent, workdir=workdir / "score_run")

    record = {
        "backend": backend, "seed": seed, "theta_fid": theta_fid, "theta_agent": result.theta_agent,
        "n_calls": result.n_calls, "cpu_hours_total": result.cpu_hours_total,
        "chi2_min": result.chi2_min, "chi2_oracle": chi2_oracle,
        "converged": result.converged, "cpu_seconds": result.cpu_seconds,
    }
    print(f"[{backend} seed={seed}] n_calls={result.n_calls}  cpu_hours_total={result.cpu_hours_total:.4f}  "
          f"chi2_min={result.chi2_min:.4f}  chi2_oracle={chi2_oracle:.4f}  converged={result.converged}",
          flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["syren_new", "mpgadget"], default="syren_new",
                        help="Which simulator to calibrate against")
    parser.add_argument("--n", type=int, default=1, help="Number of rollouts")
    parser.add_argument("--seeds", type=int, nargs="*", help="Explicit seed list (overrides --n)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: N for syren_new, 1 for mpgadget — "
                             "Anvil's per-user concurrent SLURM job limit makes unrestricted "
                             "parallelism risky for real-cluster rollouts)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    seeds = args.seeds if args.seeds else list(range(args.n))
    if args.workers is not None:
        max_workers = args.workers
    else:
        max_workers = len(seeds) if args.backend == "syren_new" else 1

    out_dir = project_root / "results" / "benchmark" / args.backend
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.jsonl"

    print(f"Launching {len(seeds)} '{args.backend}' rollout(s) (seeds={seeds}) with max_workers={max_workers}")

    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_one, s, project_root, args.backend): s for s in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                rec = fut.result()
                records.append(rec)
                with open(summary_path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception as e:
                print(f"[seed={seed}] FAILED: {e}", file=sys.stderr)

    if records:
        converged = [r for r in records if r["converged"]]
        cpu_hours = [r["cpu_hours_total"] for r in converged]
        print(f"\n=== Benchmark summary ({args.backend}) ===")
        print(f"Runs: {len(records)}  Converged: {len(converged)}/{len(records)}")
        if cpu_hours:
            print(f"cpu_hours to convergence — mean: {np.mean(cpu_hours):.2f}  "
                  f"median: {np.median(cpu_hours):.2f}  min: {min(cpu_hours):.2f}  max: {max(cpu_hours):.2f}")
        print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
