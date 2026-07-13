"""
Tests for judge/oracle.py::MPGadgetOracle and draw_valid_theta_fid_mpgadget. The real
MPGadgetSimulator is monkeypatched throughout — no real SLURM job is ever submitted here,
per the project convention of mocking the expensive boundary for unit tests.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from config import MPGADGET_PARAM_KEYS
from judge.oracle import MPGadgetOracle, draw_valid_theta_fid_mpgadget

THETA_FID = {"om": 0.30, "ob": 0.05, "sigma8": 0.82,
             "wind_energy_fraction": 1.1, "wind_speed_factor": 3.8, "bh_feedback_factor": 0.06}


class _FakeMPGadgetSim:
    """Records every call; returns a deterministic pk shaped like the real k-grid."""

    def __init__(self, k_vec, cpu_hours=1.0):
        self.k_vec = k_vec
        self.cpu_hours = cpu_hours
        self.calls = []

    def __call__(self, params, ngrid, box_size_kpc, workdir, notes=""):
        self.calls.append({"params": dict(params), "ngrid": ngrid,
                            "box_size_kpc": box_size_kpc, "workdir": Path(workdir), "notes": notes})
        # Deterministic pk purely as a function of om (so score() can distinguish good/bad fits)
        pk = self.k_vec ** -1.5 * (1.0 + params["om"])
        return pk, self.cpu_hours


@pytest.fixture
def fake_k_vec():
    return np.logspace(np.log10(2.0), np.log10(15.0), 25)


# ---------------------------------------------------------------------------
# generate_obs
# ---------------------------------------------------------------------------

def test_generate_obs_calls_real_sim_exactly_once_at_fixed_resolution(tmp_path, fake_k_vec):
    sim = _FakeMPGadgetSim(fake_k_vec)
    oracle = MPGadgetOracle(theta_fid=THETA_FID, sigma_frac=0.02, mpgadget_sim=sim,
                            ground_truth_workdir=tmp_path / "ground_truth", seed=0)
    oracle.generate_obs(tmp_path / "obs_pk.npy")

    assert len(sim.calls) == 1
    call = sim.calls[0]
    assert call["ngrid"] == MPGadgetOracle.GROUND_TRUTH_NGRID == 32
    assert call["box_size_kpc"] == MPGadgetOracle.GROUND_TRUTH_BOX_SIZE_KPC == 4000
    assert call["params"] == THETA_FID


def test_generate_obs_writes_npy_with_noise(tmp_path, fake_k_vec):
    sim = _FakeMPGadgetSim(fake_k_vec)
    oracle = MPGadgetOracle(theta_fid=THETA_FID, sigma_frac=0.02, mpgadget_sim=sim,
                            ground_truth_workdir=tmp_path / "ground_truth", seed=0)
    out = tmp_path / "obs_pk.npy"
    oracle.generate_obs(out)

    assert out.exists()
    obs = np.load(out)
    pk_true = fake_k_vec ** -1.5 * (1.0 + THETA_FID["om"])
    assert not np.allclose(obs, pk_true)  # noise was added
    assert np.all(obs > 0)


def test_theta_fid_never_written_to_disk(tmp_path, fake_k_vec):
    sim = _FakeMPGadgetSim(fake_k_vec)
    oracle = MPGadgetOracle(theta_fid=THETA_FID, sigma_frac=0.02, mpgadget_sim=sim,
                            ground_truth_workdir=tmp_path / "ground_truth", seed=0)
    oracle.generate_obs(tmp_path / "obs_pk.npy")

    for f in tmp_path.rglob("*"):
        if f.is_file() and f.suffix in (".json", ".yaml", ".txt"):
            content = f.read_text()
            for key, val in THETA_FID.items():
                assert str(val) not in content, f"theta_fid leaked into {f}"


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------

def test_score_calls_real_sim_exactly_once_more(tmp_path, fake_k_vec):
    sim = _FakeMPGadgetSim(fake_k_vec)
    oracle = MPGadgetOracle(theta_fid=THETA_FID, sigma_frac=0.02, mpgadget_sim=sim,
                            ground_truth_workdir=tmp_path / "ground_truth", seed=0)
    oracle.generate_obs(tmp_path / "obs_pk.npy")
    assert len(sim.calls) == 1

    proposal = {**THETA_FID, "om": 0.32}
    chi2 = oracle.score(proposal, workdir=tmp_path / "score_run")

    assert len(sim.calls) == 2
    assert sim.calls[1]["params"] == proposal
    assert sim.calls[1]["ngrid"] == MPGadgetOracle.GROUND_TRUTH_NGRID
    assert chi2 >= 0


def test_score_at_theta_fid_lower_than_far_perturbation(tmp_path, fake_k_vec):
    sim = _FakeMPGadgetSim(fake_k_vec)
    oracle = MPGadgetOracle(theta_fid=THETA_FID, sigma_frac=0.0, mpgadget_sim=sim,
                            ground_truth_workdir=tmp_path / "ground_truth", seed=0)
    oracle.generate_obs(tmp_path / "obs_pk.npy")

    chi2_exact = oracle.score(THETA_FID, workdir=tmp_path / "run_exact")
    chi2_far = oracle.score({**THETA_FID, "om": THETA_FID["om"] + 0.15}, workdir=tmp_path / "run_far")

    assert chi2_exact == pytest.approx(0.0, abs=1e-9)
    assert chi2_far > chi2_exact


def test_score_before_generate_obs_raises(tmp_path, fake_k_vec):
    sim = _FakeMPGadgetSim(fake_k_vec)
    oracle = MPGadgetOracle(theta_fid=THETA_FID, sigma_frac=0.02, mpgadget_sim=sim,
                            ground_truth_workdir=tmp_path / "ground_truth", seed=0)
    with pytest.raises(RuntimeError):
        oracle.score(THETA_FID, workdir=tmp_path / "run")


# ---------------------------------------------------------------------------
# draw_valid_theta_fid_mpgadget — uses the free ansatz, never the real sim
# ---------------------------------------------------------------------------

def test_draw_valid_theta_fid_mpgadget_never_calls_real_sim(fake_k_vec, mpgadget_bounds, mpgadget_fixed):
    rng = np.random.default_rng(0)
    theta = draw_valid_theta_fid_mpgadget(rng, mpgadget_bounds, mpgadget_fixed, fake_k_vec)
    assert set(theta.keys()) == set(MPGADGET_PARAM_KEYS)
    for key in MPGADGET_PARAM_KEYS:
        b = mpgadget_bounds[key]
        assert b["min"] <= theta[key] <= b["max"]


def test_draw_valid_theta_fid_mpgadget_is_deterministic_given_seed(fake_k_vec, mpgadget_bounds, mpgadget_fixed):
    theta_a = draw_valid_theta_fid_mpgadget(np.random.default_rng(42), mpgadget_bounds, mpgadget_fixed, fake_k_vec)
    theta_b = draw_valid_theta_fid_mpgadget(np.random.default_rng(42), mpgadget_bounds, mpgadget_fixed, fake_k_vec)
    assert theta_a == theta_b
