import csv
import json
from pathlib import Path

import pytest

from config import MPGADGET_CSV_FIELDS, MPGADGET_PARAM_KEYS, PARAM_KEYS
from orchestrator.harvest import harvest_rollout, should_stop, RolloutResult
from orchestrator.run_agent import setup_workdir, run_agent_loop
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
    assert "index" in content.lower()
    assert "cpu_hours" in content
    assert "TOOLS_PATH" not in content  # substituted with the real tools dir


# ---------------------------------------------------------------------------
# setup_workdir — backend="mpgadget"
# ---------------------------------------------------------------------------

def test_setup_workdir_mpgadget_backend_contains_required_files(tmp_path, k_vec, theta_fid, sigma_frac):
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(base=tmp_path / "rollout_mpg", oracle=oracle,
                            project_root=_PROJECT_ROOT, backend="mpgadget")

    assert (workdir / "obs_pk.npy").exists()
    assert (workdir / "config" / "prior_bounds_mpgadget.yaml").exists()
    assert not (workdir / "config" / "prior_bounds.yaml").exists()
    assert (workdir / "CLAUDE.md").exists()
    assert not (workdir / "theta_fid.json").exists()

    content = (workdir / "CLAUDE.md").read_text()
    assert "TOOLS_PATH" not in content
    assert "ansatz" in content.lower()
    assert "exactly one" in content.lower()
    assert "logspace(log10(2), log10(15), 25)" in content


def test_setup_workdir_mpgadget_backend_allows_shenqi_when_present(tmp_path, k_vec, theta_fid, sigma_frac, monkeypatch):
    from judge.oracle import Oracle
    fake_shenqi = tmp_path / "fake_shenqi"
    fake_shenqi.mkdir()
    monkeypatch.setenv("MPGADGET_SHENQI_ROOT", str(fake_shenqi))

    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(base=tmp_path / "rollout_mpg2", oracle=oracle,
                            project_root=_PROJECT_ROOT, backend="mpgadget")

    settings = json.loads((workdir / ".claude" / "settings.json").read_text())
    allowed = settings["allowedPaths"]
    assert str(fake_shenqi.resolve()) in allowed
    assert not any("symbolic_pofk" in p for p in allowed)


def test_setup_workdir_mpgadget_backend_omits_shenqi_when_absent(tmp_path, k_vec, theta_fid, sigma_frac, monkeypatch):
    from judge.oracle import Oracle
    monkeypatch.setenv("MPGADGET_SHENQI_ROOT", str(tmp_path / "does_not_exist"))

    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(base=tmp_path / "rollout_mpg3", oracle=oracle,
                            project_root=_PROJECT_ROOT, backend="mpgadget")

    settings = json.loads((workdir / ".claude" / "settings.json").read_text())
    allowed = settings["allowedPaths"]
    assert str(workdir.resolve()) in allowed  # workdir itself always present
    assert not any("does_not_exist" in p for p in allowed)


def test_setup_workdir_unknown_backend_raises(tmp_path, k_vec, theta_fid, sigma_frac):
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    with pytest.raises(ValueError):
        setup_workdir(base=tmp_path / "rollout_bad", oracle=oracle,
                      project_root=_PROJECT_ROOT, backend="not_a_real_backend")


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
        {"call_idx": 1, **_THETA_A, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:00", "chi2": 145.3, "notes": ""},
        {"call_idx": 2, **_THETA_B, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:01", "chi2": 98.2, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0)

    assert result.theta_agent == pytest.approx(_THETA_B)
    assert result.n_calls == 2
    assert result.cpu_hours_total == pytest.approx(2.0)
    assert result.chi2_min == pytest.approx(98.2)
    assert result.converged is False


def test_harvest_marks_converged(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:00", "chi2": 12.3, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_A))

    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.converged is True


def test_harvest_raises_if_best_params_missing(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:00", "chi2": 80.0, "notes": ""},
    ])
    with pytest.raises(FileNotFoundError, match="best_params.json"):
        harvest_rollout(tmp_path)


def test_harvest_cpu_seconds(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:00", "chi2": 100.0, "notes": ""},
        {"call_idx": 2, **_THETA_B, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:30", "chi2": 60.0, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.cpu_seconds == pytest.approx(30.0)


def test_harvest_skips_blank_chi2_rows(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "cpu_hours": 0.5, "timestamp": "2026-07-04T10:00:00", "chi2": "", "notes": "get_pk"},
        {"call_idx": 2, **_THETA_B, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:01", "chi2": 80.0, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0)
    assert result.chi2_min == pytest.approx(80.0)
    assert result.n_calls == 2
    assert result.cpu_hours_total == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# harvest_rollout — param_keys="mpgadget" schema (no h/ns/as_/w0 columns at all)
# ---------------------------------------------------------------------------

_MPG_THETA_A = {"om": 0.28, "ob": 0.046, "sigma8": 0.80,
                "wind_energy_fraction": 1.0, "wind_speed_factor": 3.7, "bh_feedback_factor": 0.05}
_MPG_THETA_B = {"om": 0.30, "ob": 0.046, "sigma8": 0.82,
                "wind_energy_fraction": 1.0, "wind_speed_factor": 3.7, "bh_feedback_factor": 0.05}


def _write_mpgadget_runs_csv(workdir: Path, rows: list) -> None:
    with open(workdir / "runs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MPGADGET_CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def test_harvest_mpgadget_schema_matches_row_without_syren_columns(tmp_path):
    """The default param_keys (syren_new PARAM_KEYS) would KeyError on this schema —
    passing param_keys=MPGADGET_PARAM_KEYS must match correctly instead."""
    _write_mpgadget_runs_csv(tmp_path, [
        {"call_idx": 1, **_MPG_THETA_A, "ngrid": 56, "box_size_kpc": 6000, "cpu_hours": 3.0,
         "tool": "mpgadget_trial", "timestamp": "2026-07-13T10:00:00", "chi2": 120.0, "notes": ""},
        {"call_idx": 2, **_MPG_THETA_B, "ngrid": 56, "box_size_kpc": 6000, "cpu_hours": 3.2,
         "tool": "mpgadget_trial", "timestamp": "2026-07-13T10:00:01", "chi2": 45.0, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_MPG_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0, param_keys=MPGADGET_PARAM_KEYS)

    assert result.theta_agent == pytest.approx(_MPG_THETA_B)
    assert result.chi2_final == pytest.approx(45.0)
    assert result.chi2_min == pytest.approx(45.0)
    assert result.converged is True
    assert result.cpu_hours_total == pytest.approx(6.2)


def test_harvest_mpgadget_ansatz_rows_do_not_count_toward_cpu_hours(tmp_path):
    _write_mpgadget_runs_csv(tmp_path, [
        {"call_idx": 1, **_MPG_THETA_A, "ngrid": "", "box_size_kpc": "", "cpu_hours": 0.0,
         "tool": "ansatz", "timestamp": "2026-07-13T10:00:00", "chi2": "", "notes": "scan"},
        {"call_idx": 2, **_MPG_THETA_B, "ngrid": 56, "box_size_kpc": 6000, "cpu_hours": 3.2,
         "tool": "mpgadget_trial", "timestamp": "2026-07-13T10:00:01", "chi2": 45.0, "notes": ""},
    ])
    (tmp_path / "best_params.json").write_text(json.dumps(_MPG_THETA_B))

    result = harvest_rollout(tmp_path, epsilon=50.0, param_keys=MPGADGET_PARAM_KEYS)

    assert result.n_calls == 2  # ansatz calls still count as calls in the audit trail
    assert result.cpu_hours_total == pytest.approx(3.2)  # but not toward cpu_hours


def test_harvest_mpgadget_unmatched_row_falls_back_to_chi2_min_not_syren_recompute(tmp_path):
    """No free way exists to recompute an MP-Gadget chi2 from obs_pk.npy (see
    MPGadgetOracle's docstring) — an unmatched theta_agent must fall back to chi2_min,
    not attempt the syren_new-specific pnl_new_emulated-based recompute (which would
    KeyError on an MPGADGET_PARAM_KEYS-shaped theta_agent anyway)."""
    _write_mpgadget_runs_csv(tmp_path, [
        {"call_idx": 1, **_MPG_THETA_A, "ngrid": 56, "box_size_kpc": 6000, "cpu_hours": 3.0,
         "tool": "mpgadget_trial", "timestamp": "2026-07-13T10:00:00", "chi2": 120.0, "notes": ""},
    ])
    # best_params.json deliberately does not match any runs.csv row
    (tmp_path / "best_params.json").write_text(json.dumps({**_MPG_THETA_A, "om": 0.999}))

    result = harvest_rollout(tmp_path, epsilon=50.0, param_keys=MPGADGET_PARAM_KEYS)
    assert result.chi2_final == pytest.approx(120.0)  # fell back to chi2_min, no crash


def test_should_stop_mpgadget_schema_ignores_ansatz_cpu_hours(tmp_path):
    _write_mpgadget_runs_csv(tmp_path, [
        {"call_idx": 1, **_MPG_THETA_A, "ngrid": "", "box_size_kpc": "", "cpu_hours": 0.0,
         "tool": "ansatz", "timestamp": "2026-07-13T10:00:00", "chi2": "", "notes": ""},
    ])
    assert should_stop(tmp_path, epsilon=50.0, max_cpu_hours=0.5) is False


# ---------------------------------------------------------------------------
# should_stop
# ---------------------------------------------------------------------------

def test_should_stop_false_when_no_runs_csv(tmp_path):
    assert should_stop(tmp_path, epsilon=50.0, max_cpu_hours=10.0) is False


def test_should_stop_true_on_convergence(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:00", "chi2": 12.0, "notes": ""},
    ])
    assert should_stop(tmp_path, epsilon=50.0, max_cpu_hours=10.0) is True


def test_should_stop_true_on_budget_exhausted(tmp_path):
    rows = [
        {"call_idx": i, **_THETA_A, "cpu_hours": 1.0, "timestamp": f"2026-07-04T10:00:{i:02d}", "chi2": 999.0, "notes": ""}
        for i in range(1, 4)
    ]
    _write_runs_csv(tmp_path, rows)
    assert should_stop(tmp_path, epsilon=50.0, max_cpu_hours=3.0) is True


def test_should_stop_false_when_neither(tmp_path):
    _write_runs_csv(tmp_path, [
        {"call_idx": 1, **_THETA_A, "cpu_hours": 1.0, "timestamp": "2026-07-04T10:00:00", "chi2": 999.0, "notes": ""},
    ])
    assert should_stop(tmp_path, epsilon=50.0, max_cpu_hours=10.0) is False


# ---------------------------------------------------------------------------
# run_agent_loop
# ---------------------------------------------------------------------------

def test_run_agent_loop_stops_on_convergence(tmp_path, monkeypatch):
    def fake_invoke(workdir, prompt, project_root, timeout_seconds):
        csv_path = Path(workdir) / "runs.csv"
        existing = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
        idx = len(existing) + 1
        chi2 = 10.0 if idx == 3 else 999.0
        _write_runs_csv(workdir, existing + [
            {"call_idx": idx, **_THETA_A, "cpu_hours": 1.0, "timestamp": f"2026-07-04T10:00:{idx:02d}", "chi2": chi2, "notes": ""}
        ])
        return 0

    monkeypatch.setattr("orchestrator.run_agent._invoke_claude", fake_invoke)
    n = run_agent_loop(tmp_path, project_root=tmp_path, max_cpu_hours=100.0, epsilon=50.0)
    assert n == 3


def test_run_agent_loop_stops_on_budget(tmp_path, monkeypatch):
    def fake_invoke(workdir, prompt, project_root, timeout_seconds):
        csv_path = Path(workdir) / "runs.csv"
        existing = list(csv.DictReader(open(csv_path))) if csv_path.exists() else []
        idx = len(existing) + 1
        _write_runs_csv(workdir, existing + [
            {"call_idx": idx, **_THETA_A, "cpu_hours": 1.0, "timestamp": f"2026-07-04T10:00:{idx:02d}", "chi2": 999.0, "notes": ""}
        ])
        return 0

    monkeypatch.setattr("orchestrator.run_agent._invoke_claude", fake_invoke)
    n = run_agent_loop(tmp_path, project_root=tmp_path, max_cpu_hours=2.0, epsilon=1e-9)
    assert n == 2


def test_run_agent_loop_stops_on_consecutive_failures(tmp_path, monkeypatch):
    def fake_invoke(workdir, prompt, project_root, timeout_seconds):
        return 1

    monkeypatch.setattr("orchestrator.run_agent._invoke_claude", fake_invoke)
    n = run_agent_loop(tmp_path, project_root=tmp_path, max_cpu_hours=100.0, epsilon=50.0,
                        max_consecutive_failures=3)
    assert n == 3


def test_run_agent_loop_respects_max_iterations_cap(tmp_path, monkeypatch):
    def fake_invoke(workdir, prompt, project_root, timeout_seconds):
        return 0  # never writes runs.csv -> should_stop always False

    monkeypatch.setattr("orchestrator.run_agent._invoke_claude", fake_invoke)
    n = run_agent_loop(tmp_path, project_root=tmp_path, max_cpu_hours=100.0, epsilon=50.0,
                        max_iterations=5)
    assert n == 5
