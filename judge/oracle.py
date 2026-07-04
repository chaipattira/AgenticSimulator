from pathlib import Path

import numpy as np
from symbolic_pofk.syren_new import pnl_new_emulated


def _eval_pk(k_vec: np.ndarray, params: dict) -> np.ndarray:
    return pnl_new_emulated(
        k_vec, As=params["as_"], Om=params["om"], Ob=params["ob"],
        h=params["h"], ns=params["ns"], mnu=0.0, w0=params["w0"], wa=0.0, a=1.0,
    )


class Oracle:
    """
    Holds theta_fid in memory. Generates obs_pk = P(theta_fid) * (1 + epsilon)
    where epsilon ~ N(0, sigma_frac). Computes chi2 against obs_pk.

    theta_fid is NEVER written to disk by this class.
    """

    def __init__(self, theta_fid: dict, k_vec: np.ndarray, sigma_frac: float, seed: int):
        self._theta_fid = theta_fid
        self._k_vec = k_vec
        self._sigma_frac = sigma_frac
        rng = np.random.default_rng(seed)
        pk_true = _eval_pk(k_vec, theta_fid)
        noise = rng.normal(0.0, sigma_frac, size=len(k_vec))
        self._pk_obs = pk_true * (1.0 + noise)
        self._sigma_k = sigma_frac * self._pk_obs

    def generate_obs(self, out_path: Path) -> None:
        np.save(out_path, self._pk_obs)

    def score(self, params: dict) -> float:
        pk_proposal = _eval_pk(self._k_vec, params)
        sigma = self._sigma_k if self._sigma_frac > 0 else np.ones_like(self._pk_obs)
        return float(np.sum(((pk_proposal - self._pk_obs) / sigma) ** 2) / len(self._k_vec))


def generate_obs_cli():
    import argparse, json, yaml
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--sigma-frac", type=float, default=0.02)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--theta-json", type=str, required=True)
    p.add_argument("--config", type=Path, default=Path("config/prior_bounds.yaml"))
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    kv = cfg["k_vector"]
    k_vec = np.logspace(kv["logspace_start"], kv["logspace_end"], kv["n_points"])

    oracle = Oracle(theta_fid=json.loads(args.theta_json), k_vec=k_vec,
                    sigma_frac=args.sigma_frac, seed=args.seed)
    oracle.generate_obs(Path(args.out))
    print(f"obs_pk.npy written to {args.out}")


if __name__ == "__main__":
    generate_obs_cli()
