import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from config import default_shenqi_root
from simulator.mpgadget_wrapper import MPGadgetSimulator, MPGadgetJobError
from simulator.syren_wrapper import OutOfPriorError

_PROJECT_ROOT = Path(__file__).parent.parent
# default_shenqi_root honors MPGADGET_SHENQI_ROOT when set, falling back to
# <project_root>/shenqi otherwise (identical to the old hardcoded path in every
# environment where shenqi/ actually lives at the project root) — see config.py's
# docstring. Needed so these tests also run inside a git worktree, which does not carry
# gitignored directories like shenqi/ from the main checkout.
_SHENQI_ROOT = default_shenqi_root(_PROJECT_ROOT)

PARAMS = {
    "om": 0.2814, "ob": 0.0464, "sigma8": 0.81,
    "wind_energy_fraction": 1.0, "wind_speed_factor": 3.7, "bh_feedback_factor": 0.05,
}


def _prior_bounds():
    return yaml.safe_load((_PROJECT_ROOT / "config" / "prior_bounds_mpgadget.yaml").read_text())


def _seed_fake_powerspectrum(workdir: Path):
    out = workdir / "output"
    out.mkdir(parents=True, exist_ok=True)
    (out / "powerspectrum-1.0000.txt").write_text(
        "# in Mpc/h Units \n# D1 = 1 \n# k P N P(z=0)\n"
        + "\n".join(f"{k:.4f} {k**-2:.6f} 100 {k**-2:.6f}" for k in np.linspace(1.6, 24.0, 40))
        + "\n"
    )


def _patch_subprocess(monkeypatch, job_id="999"):
    def fake_run(cmd, capture_output=True, text=True, **kwargs):
        from unittest.mock import MagicMock
        result = MagicMock()
        result.returncode = 0
        result.stderr = ""
        if cmd[0] == "sbatch":
            result.stdout = f"Submitted batch job {job_id}\n"
        elif cmd[0] == "sacct":
            result.stdout = f"{job_id}|60|16\n"
        else:
            result.stdout = ""
        return result
    monkeypatch.setattr("simulator.slurm.subprocess.run", fake_run)
    monkeypatch.setattr("simulator.mpgadget_wrapper.subprocess.run", fake_run)


def test_call_returns_pk_and_cpu_hours(tmp_path, monkeypatch):
    _patch_subprocess(monkeypatch)
    workdir = tmp_path / "trial_0"
    workdir.mkdir()
    _seed_fake_powerspectrum(workdir)

    sim = MPGadgetSimulator(shenqi_root=_SHENQI_ROOT, csv_path=tmp_path / "runs.csv",
                             prior_bounds=_prior_bounds())
    pk, cpu_hours = sim(PARAMS, ngrid=48, box_size_kpc=4000, workdir=workdir)

    assert pk.shape == (25,)
    assert np.all(pk > 0) and np.all(np.isfinite(pk))
    # two jobs (genic + gadget), each 60s * 16 cpus / 3600 = 0.2667h -> total 0.5333h
    assert cpu_hours == pytest.approx(2 * 60 * 16 / 3600, rel=1e-6)


def test_call_writes_consistent_omega_lambda(tmp_path, monkeypatch):
    _patch_subprocess(monkeypatch)
    workdir = tmp_path / "trial_0"
    workdir.mkdir()
    _seed_fake_powerspectrum(workdir)

    sim = MPGadgetSimulator(shenqi_root=_SHENQI_ROOT, csv_path=tmp_path / "runs.csv",
                             prior_bounds=_prior_bounds())
    sim(PARAMS, ngrid=48, box_size_kpc=4000, workdir=workdir)

    genic_text = (workdir / "paramfile.genic").read_text()
    gadget_text = (workdir / "paramfile.gadget").read_text()
    expected_lambda = 1.0 - PARAMS["om"]
    assert f"OmegaLambda = {expected_lambda}" in genic_text
    assert f"OmegaLambda = {expected_lambda}" in gadget_text
    assert f"Omega0 = {PARAMS['om']}" in genic_text
    assert f"Omega0 = {PARAMS['om']}" in gadget_text


def test_call_logs_to_csv(tmp_path, monkeypatch):
    _patch_subprocess(monkeypatch)
    workdir = tmp_path / "trial_0"
    workdir.mkdir()
    _seed_fake_powerspectrum(workdir)
    csv_path = tmp_path / "runs.csv"

    sim = MPGadgetSimulator(shenqi_root=_SHENQI_ROOT, csv_path=csv_path, prior_bounds=_prior_bounds())
    sim(PARAMS, ngrid=48, box_size_kpc=4000, workdir=workdir, notes="test trial")

    rows = list(csv.DictReader(open(csv_path)))
    assert len(rows) == 1
    assert rows[0]["call_idx"] == "1"
    assert float(rows[0]["om"]) == pytest.approx(PARAMS["om"])
    assert int(rows[0]["ngrid"]) == 48
    assert rows[0]["notes"] == "test trial"


def test_out_of_prior_params_raises(tmp_path, monkeypatch):
    _patch_subprocess(monkeypatch)
    workdir = tmp_path / "trial_0"
    workdir.mkdir()
    sim = MPGadgetSimulator(shenqi_root=_SHENQI_ROOT, csv_path=tmp_path / "runs.csv",
                             prior_bounds=_prior_bounds())
    with pytest.raises(OutOfPriorError):
        sim({**PARAMS, "om": 99.0}, ngrid=48, box_size_kpc=4000, workdir=workdir)


def test_out_of_prior_resolution_raises(tmp_path, monkeypatch):
    _patch_subprocess(monkeypatch)
    workdir = tmp_path / "trial_0"
    workdir.mkdir()
    sim = MPGadgetSimulator(shenqi_root=_SHENQI_ROOT, csv_path=tmp_path / "runs.csv",
                             prior_bounds=_prior_bounds())
    with pytest.raises(OutOfPriorError):
        sim(PARAMS, ngrid=9999, box_size_kpc=4000, workdir=workdir)


def test_missing_powerspectrum_file_raises(tmp_path, monkeypatch):
    _patch_subprocess(monkeypatch)
    workdir = tmp_path / "trial_0"
    workdir.mkdir()
    # deliberately do NOT seed the fixture powerspectrum file
    sim = MPGadgetSimulator(shenqi_root=_SHENQI_ROOT, csv_path=tmp_path / "runs.csv",
                             prior_bounds=_prior_bounds())
    with pytest.raises(MPGadgetJobError):
        sim(PARAMS, ngrid=48, box_size_kpc=4000, workdir=workdir)
