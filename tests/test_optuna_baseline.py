import csv
import json
from pathlib import Path

import numpy as np
import pytest

from baselines.optuna_baseline import run_one_optuna, CPU_HOURS_PER_TRIAL
from config import PARAM_KEYS, load_config, make_k_vec
from judge.oracle import Oracle, draw_valid_theta_fid

_PRIOR_BOUNDS_TEMPLATE = """\
parameters:
  om:   {{min: 0.24,   max: 0.40}}
  ob:   {{min: 0.04,   max: 0.06}}
  h:    {{min: 0.61,   max: 0.73}}
  ns:   {{min: 0.92,   max: 1.00}}
  as_:  {{min: 1.7e-9, max: 2.5e-9}}
  w0:   {{min: -1.3,   max: -0.7}}
fiducial_wmap9:
  om: 0.281
  ob: 0.046
  h: 0.697
  ns: 0.971
  as_: 2.1e-9
  w0: -1.0
k_vector:
  logspace_start: -2.0
  logspace_end: 0.0
  n_points: 50
noise:
  sigma_frac: 0.02
  sigma0_realization: 0.1
chi2:
  epsilon: {epsilon}
budget:
  max_cpu_hours: {max_cpu_hours}
"""


def _setup_project(tmp_path: Path, seed: int, epsilon: float, max_cpu_hours: float) -> tuple[Path, dict]:
    """Build a fake project_root with config + a stored agent obs_pk.npy for `seed`."""
    project_root = tmp_path
    cfg_dir = project_root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    # PyYAML only recognizes exponential notation with an explicit decimal
    # point (e.g. "1.0e+30"), not Python's default repr ("1e+30", parsed as
    # a plain string) -- format explicitly to avoid that footgun.
    (cfg_dir / "prior_bounds.yaml").write_text(
        _PRIOR_BOUNDS_TEMPLATE.format(epsilon=f"{float(epsilon):.6e}", max_cpu_hours=max_cpu_hours)
    )

    cfg = load_config(project_root)
    k_vec = make_k_vec(cfg)
    bounds = cfg["parameters"]
    sigma_frac = cfg["noise"]["sigma_frac"]

    rng = np.random.default_rng(seed)
    theta_fid = draw_valid_theta_fid(rng, bounds, k_vec)
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac,
                     seed=int(rng.integers(0, 2**31)))

    stored_dir = project_root / "results" / "benchmark" / f"run_{seed:05d}"
    stored_dir.mkdir(parents=True, exist_ok=True)
    oracle.generate_obs(stored_dir / "obs_pk.npy")

    return project_root, theta_fid


def test_run_produces_expected_outputs(tmp_path):
    seed = 1
    project_root, theta_fid = _setup_project(tmp_path, seed, epsilon=1e30, max_cpu_hours=5.0)

    record = run_one_optuna(seed, theta_fid, project_root)

    workdir = project_root / "results" / "baselines" / "optuna" / f"run_{seed:05d}"
    rows = list(csv.DictReader(open(workdir / "runs.csv")))
    assert len(rows) == record["n_calls"]

    best_params = json.loads((workdir / "best_params.json").read_text())
    assert set(best_params.keys()) == set(PARAM_KEYS)

    for key in ("seed", "theta_fid", "theta_agent", "n_calls", "chi2_min", "converged", "wall_seconds"):
        assert key in record


def test_early_stop_before_budget_exhausted(tmp_path):
    seed = 2
    max_cpu_hours = 50.0
    # epsilon set far above any achievable chi2 (even the sentinel for a
    # simulator failure), so the very first trial must trigger study.stop().
    project_root, theta_fid = _setup_project(tmp_path, seed, epsilon=1e30, max_cpu_hours=max_cpu_hours)

    record = run_one_optuna(seed, theta_fid, project_root)

    assert record["cpu_hours_total"] < max_cpu_hours
    assert record["converged"] is True


def test_survives_forced_trial_failure(tmp_path, monkeypatch):
    seed = 3
    max_cpu_hours = 5.0
    # epsilon unreachable (chi2 is always >= 0), so the study always runs until
    # cpu_hours_total reaches the budget, regardless of any individual trial's outcome.
    project_root, theta_fid = _setup_project(tmp_path, seed, epsilon=-1.0, max_cpu_hours=max_cpu_hours)

    import simulator.syren_wrapper as syren_wrapper
    real_pnl = syren_wrapper.pnl_new_emulated
    calls = {"n": 0}

    def flaky_pnl(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated simulator failure")
        return real_pnl(*args, **kwargs)

    monkeypatch.setattr(syren_wrapper, "pnl_new_emulated", flaky_pnl)

    record = run_one_optuna(seed, theta_fid, project_root)

    # One Optuna trial's simulator call failed before SyrenSimulator incremented
    # cpu_hours_total -> the failed attempt doesn't count against budget, so the
    # study keeps retrying until exactly max_cpu_hours worth of successful calls.
    assert record["n_calls"] == int(max_cpu_hours / CPU_HOURS_PER_TRIAL)
    assert record["converged"] is False

    workdir = project_root / "results" / "baselines" / "optuna" / f"run_{seed:05d}"
    rows = list(csv.DictReader(open(workdir / "runs.csv")))
    assert len(rows) == record["n_calls"]

    best_params = json.loads((workdir / "best_params.json").read_text())
    assert set(best_params.keys()) == set(PARAM_KEYS)


def test_theta_fid_mismatch_raises(tmp_path):
    seed = 4
    project_root, theta_fid = _setup_project(tmp_path, seed, epsilon=1e30, max_cpu_hours=3.0)
    wrong_theta_fid = {**theta_fid, "om": theta_fid["om"] + 0.01}

    with pytest.raises(RuntimeError, match="does not match"):
        run_one_optuna(seed, wrong_theta_fid, project_root)
