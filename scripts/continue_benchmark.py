#!/usr/bin/env python
"""
Continue non-converged benchmark runs.

Each run_{seed:05d}/ workdir is left intact (runs.csv, journal.md, best_params.json,
obs_pk.npy all preserved). The agent resumes via the compaction-recovery prompt.

The oracle is reconstructed deterministically from the seed so chi2_oracle can be
scored at the end.

Usage:
    # Continue all non-converged runs found in results/benchmark/
    python continue_benchmark.py

    # Continue specific seeds
    python continue_benchmark.py --seeds 0 3 4 6 7 8 9

    # Limit parallelism
    python continue_benchmark.py --workers 4
"""
import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from symbolic_pofk.syren_new import pnl_new_emulated

from config import PARAM_KEYS, load_config, make_k_vec
from judge.oracle import Oracle
from orchestrator.run_agent import continue_agent
from orchestrator.harvest import harvest_rollout


def _draw_valid_theta(rng, bounds, k_vec):
    for _ in range(100):
        theta = {k: float(rng.uniform(bounds[k]["min"], bounds[k]["max"])) for k in PARAM_KEYS}
        pk = pnl_new_emulated(
            k_vec, As=theta["as_"], Om=theta["om"], Ob=theta["ob"],
            h=theta["h"], ns=theta["ns"], mnu=0.0, w0=theta["w0"], wa=0.0, a=1.0,
        )
        if np.all(np.isfinite(pk)) and np.all(pk > 0) and np.all(pk < 1e10):
            return theta
    raise RuntimeError("Could not draw valid theta_fid in 100 attempts")


def _reconstruct_oracle(seed: int, cfg: dict, k_vec: np.ndarray) -> Oracle:
    """Reconstruct the oracle for a given seed deterministically."""
    bounds, sigma_frac = cfg["parameters"], cfg["noise"]["sigma_frac"]
    rng = np.random.default_rng(seed)
    theta_fid = _draw_valid_theta(rng, bounds, k_vec)
    oracle_seed = int(rng.integers(0, 2**31))
    return Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=oracle_seed)


def continue_one(seed: int, project_root: Path, epsilon: float) -> dict:
    project_root = Path(project_root)
    cfg = load_config(project_root)
    k_vec = make_k_vec(cfg)

    workdir = project_root / "results" / "benchmark" / f"run_{seed:05d}"
    if not workdir.exists():
        raise FileNotFoundError(f"Workdir does not exist: {workdir}")

    oracle = _reconstruct_oracle(seed, cfg, k_vec)
    continue_agent(workdir, project_root)

    result = harvest_rollout(workdir, epsilon=epsilon)
    chi2_oracle = oracle.score(result.theta_agent)

    record = {
        "seed": seed,
        "theta_fid": oracle._theta_fid,
        "theta_agent": result.theta_agent,
        "n_calls": result.n_calls,
        "chi2_min": result.chi2_min,
        "chi2_oracle": chi2_oracle,
        "converged": result.converged,
        "cpu_seconds": result.cpu_seconds,
    }
    print(
        f"[seed={seed}] n_calls={result.n_calls}  chi2_min={result.chi2_min:.4f}  "
        f"chi2_oracle={chi2_oracle:.4f}  converged={result.converged}",
        flush=True,
    )
    return record


def _non_converged_seeds(benchmark_dir: Path, epsilon: float) -> list[int]:
    """Return seeds whose run dirs exist but haven't converged."""
    import csv as csv_mod

    seeds = []
    for d in sorted(benchmark_dir.glob("run_?????")):
        try:
            seed = int(d.name.split("_")[1])
        except (IndexError, ValueError):
            continue
        csv_path = d / "runs.csv"
        if not csv_path.exists():
            continue
        rows = list(csv_mod.DictReader(open(csv_path)))
        chi2_vals = [float(r["chi2"]) for r in rows if r.get("chi2", "").strip()]
        if not chi2_vals or min(chi2_vals) >= epsilon:
            seeds.append(seed)
    return seeds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="*",
                        help="Seeds to continue (default: all non-converged)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel workers (default: number of seeds)")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    cfg = load_config(project_root)
    epsilon = cfg["chi2"]["epsilon"]

    benchmark_dir = project_root / "results" / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    seeds = args.seeds if args.seeds is not None else _non_converged_seeds(benchmark_dir, epsilon)
    if not seeds:
        print("No non-converged runs found.")
        return

    max_workers = args.workers or len(seeds)
    print(f"Continuing {len(seeds)} run(s) (seeds={seeds}) with max_workers={max_workers}")

    summary_path = benchmark_dir / "summary.jsonl"
    records = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(continue_one, s, project_root, epsilon): s for s in seeds}
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
        calls = [r["n_calls"] for r in converged]
        print(f"\n=== Continuation summary ===")
        print(f"Runs: {len(records)}  Converged: {len(converged)}/{len(records)}")
        if calls:
            print(
                f"Calls to convergence — mean: {np.mean(calls):.1f}  "
                f"median: {np.median(calls):.1f}  min: {min(calls)}  max: {max(calls)}"
            )
        print(f"Results appended to {summary_path}")


if __name__ == "__main__":
    main()
