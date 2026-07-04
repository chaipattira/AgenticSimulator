import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from simulator.syren_wrapper import SyrenSimulator, OutOfPriorError

THETA = {"om": 0.281, "ob": 0.046, "h": 0.697, "ns": 0.971, "as_": 2.1e-9, "w0": -1.0}
K_VEC = np.logspace(-2, 0, 50)
PYTHON = sys.executable
TOOLS_DIR = Path(__file__).parent.parent / "tools"


def test_returns_pk_array(tmp_path):
    sim = SyrenSimulator(k_vec=K_VEC, csv_path=tmp_path / "runs.csv")
    pk = sim(THETA)
    assert isinstance(pk, np.ndarray)
    assert pk.shape == (50,)
    assert np.all(pk > 0)
    assert np.all(np.isfinite(pk))


def test_call_counter_increments(tmp_path):
    sim = SyrenSimulator(k_vec=K_VEC, csv_path=tmp_path / "runs.csv")
    assert sim.call_count == 0
    sim(THETA)
    assert sim.call_count == 1
    sim(THETA)
    assert sim.call_count == 2


def test_logs_to_csv(tmp_path, prior_bounds):
    csv_path = tmp_path / "runs.csv"
    sim = SyrenSimulator(k_vec=K_VEC, csv_path=csv_path, prior_bounds=prior_bounds)
    sim(THETA)
    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1
    assert rows[0]["call_idx"] == "1"
    assert float(rows[0]["om"]) == pytest.approx(0.281)


def test_out_of_prior_raises(tmp_path, prior_bounds):
    sim = SyrenSimulator(k_vec=K_VEC, csv_path=tmp_path / "runs.csv", prior_bounds=prior_bounds)
    with pytest.raises(OutOfPriorError):
        sim({**THETA, "om": 9.99})


# --- CLI tool tests ---

def _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac):
    import shutil, yaml
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    oracle.generate_obs(tmp_path / "obs_pk.npy")
    cfg_dst = tmp_path / "config"
    cfg_dst.mkdir()
    shutil.copy(Path(__file__).parent.parent / "config" / "prior_bounds.yaml", cfg_dst)
    return tmp_path


def test_compute_chi2_prints_value(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac)
    params_json = json.dumps({"om": 0.3, "ob": 0.05, "h": 0.7, "ns": 0.97, "as_": 2.1e-9, "w0": -1.0})
    result = subprocess.run(
        [PYTHON, str(TOOLS_DIR / "compute_chi2.py"), "--params", params_json],
        capture_output=True, text=True, cwd=wd,
    )
    assert result.returncode == 0, result.stderr
    assert "chi2=" in result.stdout
    chi2_val = float(result.stdout.split("chi2=")[1].split()[0])
    assert chi2_val >= 0


def test_compute_chi2_appends_csv(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac)
    params_json = json.dumps({"om": 0.3, "ob": 0.05, "h": 0.7, "ns": 0.97, "as_": 2.1e-9, "w0": -1.0})
    for _ in range(2):
        subprocess.run([PYTHON, str(TOOLS_DIR / "compute_chi2.py"), "--params", params_json],
                       cwd=wd, capture_output=True)
    rows = list(csv.DictReader(open(wd / "runs.csv")))
    assert len(rows) == 2
    assert rows[0]["call_idx"] == "1"
    assert rows[1]["call_idx"] == "2"


def test_get_pk_returns_json(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac)
    params_json = json.dumps({"om": 0.3, "ob": 0.05, "h": 0.7, "ns": 0.97, "as_": 2.1e-9, "w0": -1.0})
    result = subprocess.run(
        [PYTHON, str(TOOLS_DIR / "get_pk.py"), "--params", params_json],
        capture_output=True, text=True, cwd=wd,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "k" in data and "pk" in data
    assert len(data["k"]) == 50
    assert all(v > 0 for v in data["pk"])


def test_compute_chi2_out_of_bounds_fails(tmp_path, k_vec, theta_fid, sigma_frac):
    wd = _setup_workdir(tmp_path, k_vec, theta_fid, sigma_frac)
    bad = json.dumps({"om": 9.9, "ob": 0.05, "h": 0.7, "ns": 0.97, "as_": 2.1e-9, "w0": -1.0})
    result = subprocess.run(
        [PYTHON, str(TOOLS_DIR / "compute_chi2.py"), "--params", bad],
        capture_output=True, text=True, cwd=wd,
    )
    assert result.returncode != 0
    assert "error" in result.stderr.lower() or "prior" in result.stderr.lower()
