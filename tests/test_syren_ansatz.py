import inspect

import numpy as np
import pytest

from simulator.syren_ansatz import SyrenAnsatz

PARAMS = {"om": 0.30, "ob": 0.05, "sigma8": 0.82,
          "wind_energy_fraction": 1.2, "wind_speed_factor": 4.0, "bh_feedback_factor": 0.07}


def test_no_cpu_hours_argument_exists():
    """SyrenAnsatz has no cost dial at all — confirm __call__'s signature has no
    cpu_hours parameter (unlike SyrenSimulator.__call__)."""
    sig = inspect.signature(SyrenAnsatz.__call__)
    assert "cpu_hours" not in sig.parameters


def test_returns_finite_positive_pk(mpgadget_k_vec, mpgadget_fixed):
    ansatz = SyrenAnsatz(k_vec=mpgadget_k_vec, fixed=mpgadget_fixed, sigma0=0.02)
    pk = ansatz(PARAMS)
    assert pk.shape == mpgadget_k_vec.shape
    assert np.all(np.isfinite(pk))
    assert np.all(pk > 0)


def test_ignores_subgrid_params(mpgadget_k_vec, mpgadget_fixed):
    """syren_new has no notion of wind/BH feedback — varying them must not change output."""
    ansatz = SyrenAnsatz(k_vec=mpgadget_k_vec, fixed=mpgadget_fixed, sigma0=0.0)  # no noise
    pk_a = ansatz(PARAMS)
    pk_b = ansatz({**PARAMS, "wind_energy_fraction": 99.0, "bh_feedback_factor": 99.0})
    np.testing.assert_allclose(pk_a, pk_b)


def test_noise_is_fixed_fractional_not_cpu_hours_dependent(mpgadget_k_vec, mpgadget_fixed):
    """Unlike SyrenSimulator, repeated calls at the same sigma0 should have the same noise
    scale regardless of anything resembling a resolution/cost choice (there is none)."""
    ansatz = SyrenAnsatz(k_vec=mpgadget_k_vec, fixed=mpgadget_fixed, sigma0=0.1)
    draws = np.array([ansatz(PARAMS) for _ in range(300)])
    frac_std = (draws.std(axis=0) / draws.mean(axis=0))
    # should be close to sigma0 (0.1), not shrink with anything — no dial to shrink it with
    assert frac_std.mean() == pytest.approx(0.1, rel=0.25)


def test_zero_noise_is_deterministic(mpgadget_k_vec, mpgadget_fixed):
    ansatz = SyrenAnsatz(k_vec=mpgadget_k_vec, fixed=mpgadget_fixed, sigma0=0.0)
    pk1 = ansatz(PARAMS)
    pk2 = ansatz(PARAMS)
    np.testing.assert_allclose(pk1, pk2)


def test_pk_is_smoothly_decreasing_not_extrapolation_blowup(mpgadget_k_vec, mpgadget_fixed):
    """Regression guard for the finding documented in this module's docstring: an earlier
    version called the *nonlinear* emulator on this k-grid and got runaway values as large
    as 1e154. The linear-theory proxy used now must stay numerically sane (roughly
    power-law declining, nowhere near that scale) across the whole MP-Gadget k-range."""
    ansatz = SyrenAnsatz(k_vec=mpgadget_k_vec, fixed=mpgadget_fixed, sigma0=0.0)
    pk = ansatz(PARAMS)
    assert np.all(pk < 1e6)
    assert np.all(np.diff(pk) < 0)  # monotonically declining across this k-range
