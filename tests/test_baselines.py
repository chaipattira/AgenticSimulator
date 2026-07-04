import csv
import shutil
from pathlib import Path

import pytest

from judge.oracle import Oracle
from baselines.bo_botorch import run_botorch
from baselines.bo_optuna import run_optuna
from baselines.random_search import run_random_search


def _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac, seed):
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=seed)
    oracle.generate_obs(tmp_path / "obs_pk.npy")
    (tmp_path / "config").mkdir()
    shutil.copy(Path(__file__).parent.parent / "config" / "prior_bounds.yaml", tmp_path / "config")
    return tmp_path


def test_botorch_produces_runs_csv(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac, seed=0)
    run_botorch(workdir=wd, n_calls=10, seed=0)
    rows = list(csv.DictReader(open(wd / "runs.csv")))
    assert len(rows) == 10
    assert all(float(r["chi2"]) >= 0 for r in rows)


def test_optuna_produces_runs_csv(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac, seed=1)
    run_optuna(workdir=wd, n_calls=10, seed=1)
    rows = list(csv.DictReader(open(wd / "runs.csv")))
    assert len(rows) == 10


def test_random_search_produces_runs_csv(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac, seed=3)
    run_random_search(workdir=wd, n_calls=10, seed=3)
    rows = list(csv.DictReader(open(wd / "runs.csv")))
    assert len(rows) == 10
    assert all(float(r["chi2"]) >= 0 for r in rows)
