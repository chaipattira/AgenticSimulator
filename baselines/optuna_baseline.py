"""
Optuna TPE baseline — run against the same (theta_fid, obs_pk noise) pairs
the agent was scored on, for a directly paired sample-efficiency comparison.

Spends a fixed cpu_hours per trial (CPU_HOURS_PER_TRIAL): vanilla TPE has no notion of
a variable-fidelity resolution/volume dial the way the agent does, so this keeps the
baseline simple while remaining comparable on the same cpu_hours budget.
"""
import json
import shutil
import time
from pathlib import Path

import numpy as np
import optuna
from optuna.samplers import TPESampler

from config import PARAM_KEYS, load_config, make_k_vec
from judge.chi2 import compute_chi2
from judge.oracle import Oracle, draw_valid_theta_fid
from simulator.syren_wrapper import SyrenSimulator

SENTINEL_CHI2 = 1e12
CPU_HOURS_PER_TRIAL = 1.0


def run_one_optuna(seed: int, theta_fid: dict, project_root: Path) -> dict:
    """
    Run one Optuna TPE study against the theta_fid/obs_pk the agent for this
    seed was scored on. Reconstructs the Oracle deterministically from `seed`
    and asserts the reconstruction is bit-identical to what's on disk from
    the agent's run before proceeding.
    """
    project_root = Path(project_root)
    cfg = load_config(project_root)
    k_vec = make_k_vec(cfg)
    bounds = cfg["parameters"]
    sigma_frac = cfg["noise"]["sigma_frac"]
    epsilon = cfg["chi2"]["epsilon"]
    max_cpu_hours = cfg["budget"]["max_cpu_hours"]

    rng = np.random.default_rng(seed)
    reconstructed_theta_fid = draw_valid_theta_fid(rng, bounds, k_vec)
    if reconstructed_theta_fid != theta_fid:
        raise RuntimeError(
            f"Reconstructed theta_fid for seed={seed} does not match "
            f"results/benchmark/summary.jsonl — cannot guarantee a paired "
            f"comparison with the agent run. "
            f"Reconstructed: {reconstructed_theta_fid}  Stored: {theta_fid}"
        )
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac,
                     seed=int(rng.integers(0, 2**31)))

    workdir = project_root / "results" / "baselines" / "optuna" / f"run_{seed:05d}"
    workdir.mkdir(parents=True, exist_ok=True)
    obs_path = workdir / "obs_pk.npy"
    oracle.generate_obs(obs_path)

    stored_obs_path = project_root / "results" / "benchmark" / f"run_{seed:05d}" / "obs_pk.npy"
    if not stored_obs_path.exists():
        raise FileNotFoundError(
            f"No stored obs_pk.npy at {stored_obs_path} — cannot verify the "
            f"paired-comparison invariant for seed={seed}."
        )
    regen_obs = np.load(obs_path)
    stored_obs = np.load(stored_obs_path)
    if not np.array_equal(regen_obs, stored_obs):
        raise RuntimeError(
            f"Reconstructed obs_pk.npy for seed={seed} does not bit-match "
            f"{stored_obs_path} — config/prior_bounds.yaml (e.g. sigma_frac) "
            f"may have changed since the agent run; cannot guarantee a paired "
            f"comparison."
        )

    cfg_dst = workdir / "config"
    cfg_dst.mkdir(exist_ok=True)
    shutil.copy(project_root / "config" / "prior_bounds.yaml", cfg_dst)

    sim = SyrenSimulator(k_vec=k_vec, csv_path=workdir / "runs.csv", prior_bounds=bounds)
    sigma_obs = sigma_frac * regen_obs

    def objective(trial: optuna.Trial) -> float:
        params = {k: trial.suggest_float(k, bounds[k]["min"], bounds[k]["max"]) for k in PARAM_KEYS}
        try:
            _, chi2 = sim(params, cpu_hours=CPU_HOURS_PER_TRIAL,
                         chi2_fn=lambda pk, sigma_real: compute_chi2(
                             pk, regen_obs, np.sqrt(sigma_obs ** 2 + sigma_real ** 2)))
        except Exception:
            return SENTINEL_CHI2
        return chi2

    def stop_at_convergence_or_budget(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if trial.value is not None and trial.value < epsilon:
            study.stop()
        elif sim.cpu_hours_total >= max_cpu_hours:
            study.stop()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=seed))

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=100_000, callbacks=[stop_at_convergence_or_budget])
    wall_seconds = time.perf_counter() - t0

    theta_best = study.best_params
    (workdir / "best_params.json").write_text(json.dumps(theta_best))

    chi2_min = study.best_value
    return {
        "seed": seed,
        "theta_fid": theta_fid,
        "theta_agent": theta_best,
        "n_calls": sim.call_count,
        "cpu_hours_total": sim.cpu_hours_total,
        "chi2_min": chi2_min,
        "converged": chi2_min < epsilon,
        "wall_seconds": wall_seconds,
    }
