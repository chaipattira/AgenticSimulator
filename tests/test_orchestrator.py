import csv
import json
from pathlib import Path

import pytest

from config import PARAM_KEYS
from orchestrator.harvest import harvest_rollout, RolloutResult
from orchestrator.run_agent import setup_workdir
from simulator.syren_wrapper import CSV_FIELDS

_PROJECT_ROOT = Path(__file__).parent.parent

_THETA_A = {"om": 0.28, "ob": 0.046, "h": 0.70, "ns": 0.97, "as_": 2.1e-9, "w0": -1.0}
_THETA_B = {"om": 0.30, "ob": 0.046, "h": 0.70, "ns": 0.97, "as_": 2.1e-9, "w0": -1.0}


# ---------------------------------------------------------------------------
# setup_workdir
# ---------------------------------------------------------------------------

def test_setup_workdir_contains_required_files(tmp_path, k_vec, theta_fid, sigma_frac):
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(base=tmp_path / "rollout_0", oracle=oracle, project_root=_PROJECT_ROOT)

    assert (workdir / "obs_pk.npy").exists()
    assert (workdir / "config" / "prior_bounds.yaml").exists()
    assert (workdir / "CLAUDE.md").exists()
    assert not (workdir / "theta_fid.json").exists()
    assert not (workdir / "theta_fid.yaml").exists()

    settings = json.loads((workdir / ".claude" / "settings.json").read_text())
    allowed = settings["allowedPaths"]
    assert str(workdir.resolve()) in allowed
    # syren_new source should be readable by the agent
    assert any("symbolic_pofk" in p for p in allowed)


def test_claude_md_contains_journal_instructions(tmp_path, k_vec, theta_fid, sigma_frac):
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(base=tmp_path / "rollout_0", oracle=oracle, project_root=_PROJECT_ROOT)

    content = (workdir / "CLAUDE.md").read_text()
    assert "journal" in content.lower()
    assert "best_params.json" in content
    assert "compaction" in content.lower()


# ---------------------------------------------------------------------------
# harvest_rollout
# ---------------------------------------------------------------------------

def _write_runs_csv(workdir: Path, rows: list[dict]) -> None:
    with open(workdir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_harvest_reads_best_params_json(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "timestamp": "2026-07-04T10:00:00", "chi2": 145.3, "notes": ""},
        {"call_idx": 2, **_THETA_B, "timestamp": "2026-07-04T10:00:01", "chi2": 98.2, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0)

    assert result.theta_agent == pytest.approx(_THETA_B)
    assert result.n_calls == 2
    assert result.chi2_min == pytest.approx(98.2)
    assert result.converged is False


def test_harvest_marks_converged(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "timestamp": "2026-07-04T10:00:00", "chi2": 12.3, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_A))

    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.converged is True


def test_harvest_raises_if_best_params_missing(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "timestamp": "2026-07-04T10:00:00", "chi2": 80.0, "notes": ""},
    ])
    with pytest.raises(FileNotFoundError, match="best_params.json"):
        harvest_rollout(tmp_path)


def test_harvest_cpu_seconds(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "timestamp": "2026-07-04T10:00:00", "chi2": 100.0, "notes": ""},
        {"call_idx": 2, **_THETA_B, "timestamp": "2026-07-04T10:00:30", "chi2": 60.0, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.cpu_seconds == pytest.approx(30.0)


def test_harvest_skips_blank_chi2_rows(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "timestamp": "2026-07-04T10:00:00", "chi2": "", "notes": "get_pk"},
        {"call_idx": 2, **_THETA_B, "timestamp": "2026-07-04T10:00:01", "chi2": 80.0, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.chi2_min == pytest.approx(80.0)
    assert result.n_calls == 2
