"""
Unit tests for the PLACEHOLDER simulator/mpgadget_wrapper.py (see that file's module
docstring — sub-project 1 owns the real implementation). Mocks subprocess.run (the
sbatch/sacct/make_class_power.py calls) throughout, per the project's convention of
mocking the expensive boundary for unit tests. No real SLURM job is ever submitted here.
"""
import csv
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from config import MPGADGET_CSV_FIELDS
from simulator.mpgadget_wrapper import (
    CANONICAL_K_VEC,
    MPGadgetJobError,
    MPGadgetSimulator,
    OutOfPriorError,
    _override_paramfile,
)

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
WindModel = ofjt10,isotropic
"""

PARAMS = {
    "om": 0.30, "ob": 0.05, "sigma8": 0.82,
    "wind_energy_fraction": 1.2, "wind_speed_factor": 4.0, "bh_feedback_factor": 0.07,
}


@pytest.fixture
def shenqi_root(tmp_path):
    root = tmp_path / "shenqi"
    (root / "examples" / "small").mkdir(parents=True)
    (root / "examples" / "small" / "paramfile.genic").write_text(GENIC_TEMPLATE)
    (root / "examples" / "small" / "paramfile.gadget").write_text(GADGET_TEMPLATE)
    (root / "examples" / "TREECOOL_fg_june11").write_text("# stub\n")
    (root / "examples" / "cooling_metal_UVB").write_text("# stub\n")
    (root / "tools").mkdir()
    (root / "tools" / "make_class_power.py").write_text("# stub — always mocked via subprocess.run\n")
    (root / "genic").mkdir()
    (root / "gadget").mkdir()
    return root


def _fake_run(job_ids: dict, k_fixture=None, p_fixture=None, sbatch_rc: dict | None = None,
              sacct_stdout: str = "100|8\n", write_powerspectrum: bool = True):
    """Builds a subprocess.run stand-in covering all three stages this wrapper shells out to."""
    sbatch_rc = sbatch_rc or {}

    def _run(cmd, capture_output=True, text=True, **kwargs):
        if cmd[0] == sys.executable:
            out_path = Path(cmd[cmd.index("--out") + 1])
            out_path.write_text("# input pk\n1.0 100.0\n")
            return subprocess.CompletedProcess(cmd, 0, "ok", "")
        if cmd[0] == "sbatch":
            script_path = Path(cmd[-1])
            workdir = script_path.parent
            stage = script_path.stem.replace("_job", "")
            rc = sbatch_rc.get(stage, 0)
            if rc != 0:
                return subprocess.CompletedProcess(cmd, rc, "", f"SLURM error at stage {stage}")
            if stage == "gadget" and write_powerspectrum:
                out_dir = workdir / "output"
                out_dir.mkdir(parents=True, exist_ok=True)
                lines = "\n".join(f"{k} {p}" for k, p in zip(k_fixture, p_fixture))
                (out_dir / "powerspectrum-1.0000.txt").write_text("# k P N Pz0\n" + lines + "\n")
            return subprocess.CompletedProcess(cmd, 0, f"Submitted batch job {job_ids[stage]}\n", "")
        if cmd[0] == "sacct":
            return subprocess.CompletedProcess(cmd, 0, sacct_stdout, "")
        raise AssertionError(f"unexpected subprocess.run call: {cmd}")

    return _run


# ---------------------------------------------------------------------------
# _override_paramfile (pure function)
# ---------------------------------------------------------------------------

def test_override_paramfile_rewrites_matched_keys_only():
    out = _override_paramfile(GENIC_TEMPLATE.splitlines(), {"Omega0": "0.31", "Ngrid": "56"})
    assert "Omega0 = 0.31" in out
    assert "Ngrid = 56" in out
    assert "Sigma8 = 0.810" in out  # untouched, byte-for-byte


def test_override_paramfile_raises_on_unknown_key():
    with pytest.raises(KeyError):
        _override_paramfile(GENIC_TEMPLATE.splitlines(), {"NotAKey": "1"})


def test_override_paramfile_preserves_multivalue_fields():
    out = _override_paramfile(GADGET_TEMPLATE.splitlines(), {"Omega0": "0.31"})
    assert "WindModel = ofjt10,isotropic" in out


# ---------------------------------------------------------------------------
# paramfile writing / OmegaLambda consistency
# ---------------------------------------------------------------------------

def test_write_paramfiles_omega_lambda_consistency(tmp_path, shenqi_root):
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv")
    workdir = tmp_path / "trial"
    workdir.mkdir()
    genic_path, gadget_path = sim._write_paramfiles(
        PARAMS, ngrid=56, box_size_kpc=6000, workdir=workdir, input_pk_path=tmp_path / "in_pk.txt"
    )
    genic_text, gadget_text = genic_path.read_text(), gadget_path.read_text()
    expected_lambda = f"OmegaLambda = {1.0 - PARAMS['om']}"
    assert expected_lambda in genic_text
    assert expected_lambda in gadget_text
    assert "Ngrid = 56" in genic_text
    assert "BoxSize = 6000" in genic_text
    assert "Sigma8 = 0.82" in genic_text
    assert "WindSpeedFactor = 4.0" in gadget_text
    assert "BlackHoleFeedbackFactor = 0.07" in gadget_text
    # h/ns are never overridden by this wrapper — fixed values pass through unchanged
    assert "HubbleParam = 0.697" in genic_text
    assert "HubbleParam = 0.697" in gadget_text


# ---------------------------------------------------------------------------
# prior bounds enforcement
# ---------------------------------------------------------------------------

def test_prior_bounds_enforced_before_any_subprocess(tmp_path, shenqi_root, monkeypatch):
    bounds = {"parameters": {"om": {"min": 0.2, "max": 0.4}},
              "resolution": {"ngrid": {"min": 48, "max": 64}}}
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv", prior_bounds=bounds)

    def _should_not_run(*a, **k):
        raise AssertionError("subprocess.run must not be called on an out-of-prior draw")

    monkeypatch.setattr(subprocess, "run", _should_not_run)
    with pytest.raises(OutOfPriorError):
        sim({**PARAMS, "om": 0.99}, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial")


def test_resolution_bounds_enforced(tmp_path, shenqi_root, monkeypatch):
    bounds = {"parameters": {}, "resolution": {"box_size_kpc": {"min": 4000, "max": 8000}}}
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv", prior_bounds=bounds)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    with pytest.raises(OutOfPriorError):
        sim(PARAMS, ngrid=56, box_size_kpc=99999, workdir=tmp_path / "trial")


# ---------------------------------------------------------------------------
# full mocked pipeline: cpu_hours arithmetic + k-grid interpolation
# ---------------------------------------------------------------------------

def test_full_pipeline_mocked_returns_pk_and_cpu_hours(tmp_path, shenqi_root, monkeypatch):
    k_fixture = np.linspace(1.5, 20.0, 40)
    p_fixture = 1000.0 / k_fixture ** 1.5
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run({"genic": "1001", "gadget": "1002"}, k_fixture=k_fixture, p_fixture=p_fixture),
    )

    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv")
    pk, cpu_hours = sim(PARAMS, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial")

    assert pk.shape == CANONICAL_K_VEC.shape == (25,)
    assert np.all(np.isfinite(pk))
    assert np.all(pk > 0)

    # 2 jobs x (ElapsedRaw=100s * NCPUS=8) / 3600
    expected_cpu_hours = 2 * (100 * 8) / 3600
    assert cpu_hours == pytest.approx(expected_cpu_hours, rel=1e-9)
    assert sim.call_count == 1
    assert sim.cpu_hours_total == pytest.approx(expected_cpu_hours, rel=1e-9)

    expected_log_pk = np.interp(np.log(CANONICAL_K_VEC), np.log(k_fixture), np.log(p_fixture))
    np.testing.assert_allclose(np.log(pk), expected_log_pk, rtol=1e-10)


def test_call_count_and_cpu_hours_accumulate_across_calls(tmp_path, shenqi_root, monkeypatch):
    k_fixture = np.linspace(1.5, 20.0, 40)
    p_fixture = 1000.0 / k_fixture ** 1.5
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run({"genic": "1001", "gadget": "1002"}, k_fixture=k_fixture, p_fixture=p_fixture),
    )
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv")
    sim(PARAMS, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial1")
    sim(PARAMS, ngrid=48, box_size_kpc=4000, workdir=tmp_path / "trial2")
    assert sim.call_count == 2
    assert sim.cpu_hours_total == pytest.approx(2 * 2 * (100 * 8) / 3600, rel=1e-9)


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------

def test_genic_failure_raises_job_error_and_skips_gadget(tmp_path, shenqi_root, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run({"genic": "1001", "gadget": "1002"}, sbatch_rc={"genic": 1}),
    )
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv")
    with pytest.raises(MPGadgetJobError) as exc:
        sim(PARAMS, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial")
    assert exc.value.stage == "genic"
    assert "SLURM error" in exc.value.stderr_tail
    assert sim.call_count == 0  # a failed trial never increments call_count


def test_missing_powerspectrum_output_raises_job_error(tmp_path, shenqi_root, monkeypatch):
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run({"genic": "1001", "gadget": "1002"}, write_powerspectrum=False),
    )
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv")
    with pytest.raises(MPGadgetJobError) as exc:
        sim(PARAMS, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial")
    assert exc.value.stage == "parse_output"


def test_make_class_power_failure_raises_before_slurm(tmp_path, shenqi_root, monkeypatch):
    def _run(cmd, **kwargs):
        if cmd[0] == sys.executable:
            return subprocess.CompletedProcess(cmd, 1, "", "unphysical cosmology")
        raise AssertionError("sbatch must not be reached if make_class_power failed")

    monkeypatch.setattr(subprocess, "run", _run)
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=tmp_path / "runs.csv")
    with pytest.raises(MPGadgetJobError) as exc:
        sim(PARAMS, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial")
    assert exc.value.stage == "make_class_power"


# ---------------------------------------------------------------------------
# csv_path resumption (no CSV row is written by this class itself — see module docstring)
# ---------------------------------------------------------------------------

def test_resumes_call_count_and_cpu_hours_from_existing_csv(tmp_path):
    csv_path = tmp_path / "runs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MPGADGET_CSV_FIELDS)
        w.writeheader()
        w.writerow({
            "call_idx": 1, "om": 0.3, "ob": 0.05, "sigma8": 0.8,
            "wind_energy_fraction": 1.0, "wind_speed_factor": 3.7, "bh_feedback_factor": 0.05,
            "ngrid": 56, "box_size_kpc": 6000, "cpu_hours": 2.5, "tool": "mpgadget_trial",
            "timestamp": "2026-07-13T00:00:00", "chi2": 40.0, "notes": "",
        })
    sim = MPGadgetSimulator(shenqi_root=Path("/nonexistent"), csv_path=csv_path)
    assert sim.call_count == 1
    assert sim.cpu_hours_total == pytest.approx(2.5)


def test_does_not_write_its_own_csv_row(tmp_path, shenqi_root, monkeypatch):
    """CSV writing (with chi2 already known) is the caller's job — see module docstring."""
    k_fixture = np.linspace(1.5, 20.0, 40)
    p_fixture = 1000.0 / k_fixture ** 1.5
    monkeypatch.setattr(
        subprocess, "run",
        _fake_run({"genic": "1001", "gadget": "1002"}, k_fixture=k_fixture, p_fixture=p_fixture),
    )
    csv_path = tmp_path / "runs.csv"
    sim = MPGadgetSimulator(shenqi_root=shenqi_root, csv_path=csv_path)
    sim(PARAMS, ngrid=56, box_size_kpc=6000, workdir=tmp_path / "trial")
    rows = list(csv.DictReader(open(csv_path)))
    assert rows == []  # header only — this class never appends a row itself
