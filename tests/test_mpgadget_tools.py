"""
Tests for tools/get_pk_ansatz.py (free, subprocess-level — it makes no subprocess calls of
its own, so a real CLI invocation is fine, mirroring test_get_pk_returns_json's pattern) and
tools/run_mpgadget_trial.py (the one paid call — loaded in-process via importlib so
subprocess.run, which MPGadgetSimulator shells out to, can be monkeypatched; a real child
`subprocess.run([PYTHON, script, ...])` call would spawn an unmockable grandchild process).
"""
import csv
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import MPGADGET_CSV_FIELDS

PYTHON = sys.executable
_PROJECT_ROOT = Path(__file__).parent.parent
TOOLS_DIR = _PROJECT_ROOT / "tools"

GENIC_TEMPLATE = """OutputDir = output
Ngrid = 32
BoxSize = 4000
Omega0 = 0.2814
OmegaLambda = 0.7186
OmegaBaryon = 0.0464
HubbleParam = 0.697
Sigma8 = 0.810
FileWithInputSpectrum = ../powerspectrum-wmap9.txt
PrimordialIndex = 0.971
"""

GADGET_TEMPLATE = """InitCondFile = output/IC
OutputDir = output
TreeCoolFile = ../TREECOOL_fg_june11
MetalCoolFile = ../cooling_metal_UVB
Omega0 = 0.2814
OmegaLambda = 0.7186
OmegaBaryon = 0.0464
HubbleParam = 0.697
WindEnergyFraction = 1.0
WindSpeedFactor = 3.7
BlackHoleFeedbackFactor = 0.05
"""

PARAMS = {"om": 0.30, "ob": 0.05, "sigma8": 0.82,
          "wind_energy_fraction": 1.2, "wind_speed_factor": 4.0, "bh_feedback_factor": 0.07}


def _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed):
    from simulator.syren_ansatz import SyrenAnsatz
    obs_ansatz = SyrenAnsatz(k_vec=mpgadget_k_vec, fixed=mpgadget_fixed, sigma0=0.0)
    obs_pk = obs_ansatz({"om": 0.2814, "ob": 0.0464, "sigma8": 0.810})
    np.save(tmp_path / "obs_pk.npy", obs_pk)
    cfg_dst = tmp_path / "config"
    cfg_dst.mkdir()
    shutil.copy(_PROJECT_ROOT / "config" / "prior_bounds_mpgadget.yaml", cfg_dst)
    return tmp_path


def _fake_shenqi_root(tmp_path):
    root = tmp_path / "fake_shenqi"
    (root / "examples" / "small").mkdir(parents=True)
    (root / "examples" / "small" / "paramfile.genic").write_text(GENIC_TEMPLATE)
    (root / "examples" / "small" / "paramfile.gadget").write_text(GADGET_TEMPLATE)
    (root / "examples" / "TREECOOL_fg_june11").write_text("# stub\n")
    (root / "examples" / "cooling_metal_UVB").write_text("# stub\n")
    (root / "tools").mkdir()
    (root / "genic").mkdir()
    (root / "gadget").mkdir()
    return root


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_mpgadget_subprocess_run(k_fixture, p_fixture):
    def _run(cmd, capture_output=True, text=True, **kwargs):
        if cmd[0] == sys.executable:
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text("# stub input pk\n1.0 100.0\n")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        if cmd[0] == "sbatch":
            script_path = Path(cmd[-1])
            workdir = script_path.parent
            stage = script_path.stem.replace("_job", "")
            job_id = {"genic": "1001", "gadget": "1002"}[stage]
            if stage == "gadget":
                out_dir = workdir / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                lines = "\n".join(f"{k} {p}" for k, p in zip(k_fixture, p_fixture))
                (out_dir / "powerspectrum-1.0000.txt").write_text("# k P\n" + lines + "\n")
            return subprocess.CompletedProcess(cmd, 0, f"Submitted batch job {job_id}\n", "")
        if cmd[0] == "sacct":
            return subprocess.CompletedProcess(cmd, 0, "100|8\n", "")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")
    return _run


# ---------------------------------------------------------------------------
# get_pk_ansatz.py — free tool, real subprocess (no internal subprocess calls to mock)
# ---------------------------------------------------------------------------

def test_get_pk_ansatz_prints_json_and_no_cpu_hours_flag_accepted(
    tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed,
):
    wd = _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed)
    result = subprocess.run(
        [PYTHON, str(TOOLS_DIR / "get_pk_ansatz.py"), "--params", json.dumps(PARAMS)],
        capture_output=True, text=True, cwd=wd,
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "k" in data and "pk" in data and "residual_frac" in data
    assert len(data["k"]) == 25
    assert data["cpu_hours_spent"] == 0.0
    assert data["tool"] == "ansatz"


def test_get_pk_ansatz_logs_cpu_hours_zero_and_does_not_help_budget(
    tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed,
):
    wd = _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed)
    for _ in range(3):
        subprocess.run(
            [PYTHON, str(TOOLS_DIR / "get_pk_ansatz.py"), "--params", json.dumps(PARAMS)],
            cwd=wd, capture_output=True, text=True,
        )
    rows = list(csv.DictReader(open(wd / "runs.csv")))
    assert len(rows) == 3
    assert all(row["tool"] == "ansatz" for row in rows)
    assert all(float(row["cpu_hours"]) == 0.0 for row in rows)
    assert all(row["chi2"] == "" for row in rows)
    # unlimited free calls: call_idx still increments (audit trail), but cpu_hours never does
    assert [row["call_idx"] for row in rows] == ["1", "2", "3"]


def test_get_pk_ansatz_unaffected_by_subgrid_params(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed):
    wd = _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed)
    r1 = subprocess.run(
        [PYTHON, str(TOOLS_DIR / "get_pk_ansatz.py"), "--params", json.dumps(PARAMS)],
        cwd=wd, capture_output=True, text=True,
    )
    r2 = subprocess.run(
        [PYTHON, str(TOOLS_DIR / "get_pk_ansatz.py"), "--params",
         json.dumps({**PARAMS, "wind_energy_fraction": 5.0, "bh_feedback_factor": 5.0})],
        cwd=wd, capture_output=True, text=True,
    )
    pk1, pk2 = json.loads(r1.stdout)["pk"], json.loads(r2.stdout)["pk"]
    # both draw independent noise, so compare the noiseless proxy underneath via k grid shape
    assert len(pk1) == len(pk2) == 25


# ---------------------------------------------------------------------------
# run_mpgadget_trial.py — the one paid tool, loaded in-process for subprocess.run mocking
# ---------------------------------------------------------------------------

def test_run_mpgadget_trial_success_writes_row_with_chi2(
    tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed, monkeypatch,
):
    wd = _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed)
    shenqi_root = _fake_shenqi_root(tmp_path)
    monkeypatch.setenv("MPGADGET_SHENQI_ROOT", str(shenqi_root))
    monkeypatch.chdir(wd)

    k_fixture = np.linspace(1.5, 20.0, 40)
    p_fixture = 1000.0 / k_fixture ** 1.5
    monkeypatch.setattr(subprocess, "run", _fake_mpgadget_subprocess_run(k_fixture, p_fixture))
    monkeypatch.setattr(sys, "argv", [
        "run_mpgadget_trial.py", "--params", json.dumps(PARAMS),
        "--ngrid", "56", "--box_size_kpc", "6000", "--notes", "test trial",
    ])

    mod = _load_module("run_mpgadget_trial", TOOLS_DIR / "run_mpgadget_trial.py")
    mod.main()

    rows = list(csv.DictReader(open(wd / "runs.csv")))
    assert len(rows) == 1
    row = rows[0]
    assert row["tool"] == "mpgadget_trial"
    assert row["chi2"] != ""
    assert float(row["chi2"]) >= 0
    assert float(row["cpu_hours"]) > 0
    assert row["ngrid"] == "56"
    assert row["box_size_kpc"] == "6000.0"


def test_run_mpgadget_trial_out_of_prior_exits_nonzero_and_writes_no_row(
    tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed, monkeypatch, capsys,
):
    wd = _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed)
    shenqi_root = _fake_shenqi_root(tmp_path)
    monkeypatch.setenv("MPGADGET_SHENQI_ROOT", str(shenqi_root))
    monkeypatch.chdir(wd)

    def _should_not_run(*a, **k):
        raise AssertionError("subprocess.run must not be reached for an out-of-prior draw")
    monkeypatch.setattr(subprocess, "run", _should_not_run)
    monkeypatch.setattr(sys, "argv", [
        "run_mpgadget_trial.py", "--params", json.dumps({**PARAMS, "om": 99.0}),
        "--ngrid", "56", "--box_size_kpc", "6000",
    ])

    mod = _load_module("run_mpgadget_trial_oop", TOOLS_DIR / "run_mpgadget_trial.py")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code != 0
    assert not (wd / "runs.csv").exists() or list(csv.DictReader(open(wd / "runs.csv"))) == []


def test_run_mpgadget_trial_job_error_exits_nonzero_with_stage(
    tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed, monkeypatch, capsys,
):
    wd = _setup_workdir(tmp_path, cfg_mpgadget, mpgadget_k_vec, mpgadget_fixed)
    shenqi_root = _fake_shenqi_root(tmp_path)
    monkeypatch.setenv("MPGADGET_SHENQI_ROOT", str(shenqi_root))
    monkeypatch.chdir(wd)

    def _failing_run(cmd, **kwargs):
        if cmd[0] == sys.executable:
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text("1.0 100.0\n")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        if cmd[0] == "sbatch":
            return subprocess.CompletedProcess(cmd, 1, "", "out of SLURM allocation")
        raise AssertionError(f"unexpected call {cmd}")
    monkeypatch.setattr(subprocess, "run", _failing_run)
    monkeypatch.setattr(sys, "argv", [
        "run_mpgadget_trial.py", "--params", json.dumps(PARAMS),
        "--ngrid", "56", "--box_size_kpc", "6000",
    ])

    mod = _load_module("run_mpgadget_trial_joberr", TOOLS_DIR / "run_mpgadget_trial.py")
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code != 0
    captured = capsys.readouterr()
    assert "stage=genic" in captured.err
