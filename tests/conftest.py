import pytest
from pathlib import Path

from config import load_config, make_k_vec

_PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def cfg():
    return load_config(_PROJECT_ROOT)


@pytest.fixture(scope="session")
def prior_bounds(cfg):
    return cfg["parameters"]


@pytest.fixture(scope="session")
def k_vec(cfg):
    return make_k_vec(cfg)


@pytest.fixture(scope="session")
def theta_fid(cfg):
    return cfg["fiducial_wmap9"]


@pytest.fixture(scope="session")
def sigma_frac(cfg):
    return cfg["noise"]["sigma_frac"]
