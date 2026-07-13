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


@pytest.fixture(scope="session")
def cfg_mpgadget():
    return load_config(_PROJECT_ROOT, filename="prior_bounds_mpgadget.yaml")


@pytest.fixture(scope="session")
def mpgadget_bounds(cfg_mpgadget):
    return cfg_mpgadget["parameters"]


@pytest.fixture(scope="session")
def mpgadget_resolution_bounds(cfg_mpgadget):
    return cfg_mpgadget["resolution"]


@pytest.fixture(scope="session")
def mpgadget_k_vec(cfg_mpgadget):
    return make_k_vec(cfg_mpgadget)


@pytest.fixture(scope="session")
def mpgadget_fixed(cfg_mpgadget):
    return cfg_mpgadget["fixed"]


@pytest.fixture(scope="session")
def mpgadget_fiducial(cfg_mpgadget):
    return cfg_mpgadget["fiducial"]
