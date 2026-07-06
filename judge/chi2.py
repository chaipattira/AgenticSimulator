import numpy as np


def compute_chi2(pk: np.ndarray, obs_pk: np.ndarray, sigma: np.ndarray) -> float:
    """Mean squared standardised residual (reduced chi-squared)."""
    return float(np.mean(((pk - obs_pk) / sigma) ** 2))
