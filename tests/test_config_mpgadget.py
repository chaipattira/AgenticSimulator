from pathlib import Path

import pytest
import yaml

from config import MPGADGET_PARAM_KEYS, make_k_vec

_PROJECT_ROOT = Path(__file__).parent.parent


def test_mpgadget_param_keys():
    assert MPGADGET_PARAM_KEYS == [
        "om", "ob", "sigma8",
        "wind_energy_fraction", "wind_speed_factor", "bh_feedback_factor",
    ]


def test_prior_bounds_mpgadget_loads():
    cfg = yaml.safe_load((_PROJECT_ROOT / "config" / "prior_bounds_mpgadget.yaml").read_text())
    assert set(cfg["parameters"].keys()) == set(MPGADGET_PARAM_KEYS)
    assert cfg["resolution"]["ngrid"]["min"] == 48
    assert cfg["resolution"]["ngrid"]["max"] == 64
    assert cfg["resolution"]["box_size_kpc"]["min"] == 4000
    assert cfg["resolution"]["box_size_kpc"]["max"] == 8000
    k_vec = make_k_vec(cfg)
    assert len(k_vec) == 25
    assert k_vec[0] == pytest.approx(2.0, rel=1e-3)
    assert k_vec[-1] == pytest.approx(15.0, rel=1e-3)
