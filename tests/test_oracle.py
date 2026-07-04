import numpy as np
import pytest
from pathlib import Path
from judge.oracle import Oracle


def perturb(theta, frac):
    return {k: v * (1 + frac) for k, v in theta.items()}


def test_score_zero_at_theta_fid(k_vec, theta_fid):
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=0.0, seed=42)
    assert oracle.score(theta_fid) == pytest.approx(0.0, abs=1e-6)


def test_score_increases_with_perturbation(k_vec, theta_fid):
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=0.0, seed=42)
    chi2_near = oracle.score(perturb(theta_fid, 0.05))
    chi2_far = oracle.score(perturb(theta_fid, 0.20))
    assert chi2_far > chi2_near > 0


def test_generate_obs_writes_npy(k_vec, theta_fid, sigma_frac, tmp_path):
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=42)
    out = tmp_path / "obs_pk.npy"
    oracle.generate_obs(out)
    assert out.exists()
    obs = np.load(out)
    assert obs.shape == (len(k_vec),)
    assert np.all(obs > 0)


def test_obs_has_noise(k_vec, theta_fid):
    oracle_clean = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=0.0, seed=42)
    oracle_noisy = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=0.02, seed=42)
    assert not np.allclose(oracle_clean._pk_obs, oracle_noisy._pk_obs)
    frac_residual = np.abs((oracle_noisy._pk_obs - oracle_clean._pk_obs) / oracle_clean._pk_obs)
    assert frac_residual.mean() < 0.10


def test_chi2_formula_uses_sigma(k_vec, theta_fid):
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=0.02, seed=42)
    oracle2 = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=0.04, seed=42)
    proposal = perturb(theta_fid, 0.1)
    assert oracle.score(proposal) > oracle2.score(proposal)


def test_oracle_does_not_expose_theta_fid(tmp_path, k_vec, theta_fid, sigma_frac):
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    oracle.generate_obs(tmp_path / "obs_pk.npy")
    file_names = [f.name for f in tmp_path.iterdir()]
    assert "theta_fid.json" not in file_names
    assert "theta_fid.yaml" not in file_names
    if sigma_frac > 0:
        from symbolic_pofk.syren_new import pnl_new_emulated
        pk_true = pnl_new_emulated(k_vec, As=theta_fid["as_"], Om=theta_fid["om"],
                                   Ob=theta_fid["ob"], h=theta_fid["h"], ns=theta_fid["ns"],
                                   mnu=0.0, w0=theta_fid["w0"], wa=0.0, a=1.0)
        obs = np.load(tmp_path / "obs_pk.npy")
        assert not np.allclose(obs, pk_true)
