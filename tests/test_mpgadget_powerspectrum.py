import numpy as np
import pytest

from simulator.mpgadget_powerspectrum import read_powerspectrum, interpolate_to_grid


def test_read_powerspectrum_skips_comments(tmp_path):
    f = tmp_path / "powerspectrum-1.0000.txt"
    f.write_text(
        "# in Mpc/h Units \n"
        "# D1 = 1 \n"
        "# k P N P(z=0)\n"
        "1.5 100.0 10 100.0\n"
        "3.0 50.0 20 50.0\n"
        "10.0 5.0 30 5.0\n"
    )
    k, p = read_powerspectrum(f)
    assert list(k) == [1.5, 3.0, 10.0]
    assert list(p) == [100.0, 50.0, 5.0]


def test_read_powerspectrum_sorts_by_k(tmp_path):
    f = tmp_path / "powerspectrum-1.0000.txt"
    f.write_text("10.0 5.0 30 5.0\n1.5 100.0 10 100.0\n3.0 50.0 20 50.0\n")
    k, p = read_powerspectrum(f)
    assert list(k) == [1.5, 3.0, 10.0]


def test_interpolate_to_grid_is_log_log():
    # P(k) = k^-2 exactly; log-log interpolation of a power law is exact
    k_raw = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0])
    p_raw = k_raw ** -2
    k_grid = np.array([2.0, 5.0, 15.0])
    pk = interpolate_to_grid(k_raw, p_raw, k_grid)
    assert pk == pytest.approx(k_grid ** -2, rel=1e-6)


def test_interpolate_to_grid_within_bounds_no_extrapolation_warning():
    k_raw = np.linspace(1.5, 25.0, 40)
    p_raw = np.ones_like(k_raw)
    k_grid = np.geomspace(2.0, 15.0, 25)
    pk = interpolate_to_grid(k_raw, p_raw, k_grid)
    assert np.all(np.isfinite(pk))
    assert np.all(pk > 0)
