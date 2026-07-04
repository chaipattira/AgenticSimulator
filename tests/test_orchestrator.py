import csv
import json
from pathlib import Path

import pytest

from orchestrator.harvest import harvest_rollout, RolloutResult
from orchestrator.run_agent import setup_workdir


def test_setup_workdir_contains_required_files(tmp_path, k_vec, theta_fid, sigma_frac):
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(
        base=tmp_path / "rollout_0",
        oracle=oracle,
        project_root=Path(__file__).parent.parent,
    )
    assert (workdir / "obs_pk.npy").exists()
    assert (workdir / "config" / "prior_bounds.yaml").exists()
    assert not (workdir / "theta_fid.json").exists()
    assert not (workdir / "theta_fid.yaml").exists()
    settings = json.loads((workdir / ".claude" / "settings.json").read_text())
    assert settings["allowedPaths"] == [str(workdir)]


def test_harvest_parses_runs_csv(tmp_path):
    csv_path = tmp_path / "runs.csv"
    fields = ["call_idx", "om", "ob", "h", "ns", "as_", "w0", "timestamp", "chi2", "notes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"call_idx": 1, "om": 0.28, "ob": 0.046, "h": 0.7,
                    "ns": 0.97, "as_": 2.1e-9, "w0": -1.0,
                    "timestamp": "2026-07-01T00:00:00", "chi2": 145.3, "notes": ""})
        w.writerow({"call_idx": 2, "om": 0.30, "ob": 0.046, "h": 0.7,
                    "ns": 0.97, "as_": 2.1e-9, "w0": -1.0,
                    "timestamp": "2026-07-01T00:00:01", "chi2": 98.2, "notes": ""})

    result = harvest_rollout(tmp_path)
    assert result.n_calls == 2
    assert result.chi2_final == pytest.approx(98.2)
    assert result.theta_agent["om"] == pytest.approx(0.30)
    assert result.converged is False


def test_harvest_marks_converged(tmp_path):
    csv_path = tmp_path / "runs.csv"
    fields = ["call_idx", "om", "ob", "h", "ns", "as_", "w0", "timestamp", "chi2", "notes"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerow({"call_idx": 1, "om": 0.281, "ob": 0.046, "h": 0.697,
                    "ns": 0.971, "as_": 2.1e-9, "w0": -1.0,
                    "timestamp": "2026-07-01T00:00:00", "chi2": 12.3, "notes": ""})
    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.converged is True
