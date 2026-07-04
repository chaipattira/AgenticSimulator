import numpy as np
import pytest
import yaml
from pathlib import Path

CONFIG = Path(__file__).parent.parent / "config" / "prior_bounds.yaml"


@pytest.fixture(scope="session")
def cfg():
    with open(CONFIG) as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def prior_bounds(cfg):
    return cfg["parameters"]


@pytest.fixture(scope="session")
def k_vec(cfg):
    kv = cfg["k_vector"]
    return np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])


@pytest.fixture(scope="session")
def theta_fid(cfg):
    return cfg["fiducial_wmap9"]


@pytest.fixture(scope="session")
def sigma_frac(cfg):
    return cfg["noise"]["sigma_frac"]
