from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from symbolic_pofk.syren_new import pnl_new_emulated

from config import MPGADGET_PARAM_KEYS, PARAM_KEYS, make_k_vec
from judge.chi2 import compute_chi2

if TYPE_CHECKING:
    from simulator.mpgadget_wrapper import MPGadgetSimulator


def draw_valid_theta_fid(rng: np.random.Generator, bounds: dict, k_vec: np.ndarray) -> dict:
    """Draw a theta_fid uniformly from bounds, rejecting draws that produce a
    non-finite or unphysical P(k). Deterministic given rng's state."""
    for _ in range(100):
        theta = {k: float(rng.uniform(bounds[k]["min"], bounds[k]["max"])) for k in PARAM_KEYS}
        pk = pnl_new_emulated(
            k_vec, As=theta["as_"], Om=theta["om"], Ob=theta["ob"],
            h=theta["h"], ns=theta["ns"], mnu=0.0, w0=theta["w0"], wa=0.0, a=1.0,
        )
        if np.all(np.isfinite(pk)) and np.all(pk > 0) and np.all(pk < 1e10):
            return theta
    raise RuntimeError("Could not draw a valid theta_fid in 100 attempts")


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
        return compute_chi2(pk_proposal, self._pk_obs, sigma)


def draw_valid_theta_fid_mpgadget(rng: np.random.Generator, bounds: dict, fixed: dict,
                                   k_vec: np.ndarray) -> dict:
    """Draw a theta_fid over the 6 MP-Gadget tunable params, rejecting draws that produce a
    non-finite/non-positive/blown-up P(k) — same rejection-sampling shape as
    draw_valid_theta_fid, but validity is checked via the FREE SyrenAnsatz proxy, never a
    real MP-Gadget run (spending real SLURM budget just to validate a random draw would be
    wasteful; approximation is fine for this coarse a check — see design doc section 7)."""
    from simulator.syren_ansatz import SyrenAnsatz

    ansatz = SyrenAnsatz(k_vec=k_vec, fixed=fixed, sigma0=0.0)
    for _ in range(100):
        theta = {k: float(rng.uniform(bounds[k]["min"], bounds[k]["max"])) for k in MPGADGET_PARAM_KEYS}
        pk = ansatz(theta)
        if np.all(np.isfinite(pk)) and np.all(pk > 0) and np.all(pk < 1e10):
            return theta
    raise RuntimeError("Could not draw a valid theta_fid_mpgadget in 100 attempts")


class MPGadgetOracle:
    """
    The MP-Gadget phase's Judge. Holds theta_fid (the 6 MP-Gadget tunable params) in
    memory only — NEVER written to disk by this class, same invariant as Oracle.

    Unlike Oracle (free — just re-evaluates pnl_new_emulated), both generate_obs() and
    score() here are REAL, COSTLY SLURM round trips through the caller-supplied
    MPGadgetSimulator: one to build the ground-truth obs_pk.npy, one more (at rollout end)
    to score the agent's final answer in data space. This is a genuine, non-trivial SLURM
    allocation cost per rollout setup, on top of whatever the agent itself spends during
    the rollout — see design doc section 7's flagged open risk. Both real runs use the
    FIXED examples/small resolution (ngrid=32, box_size_kpc=4000) — never agent-selectable,
    matching sub-project 1 spec's statement that the ground-truth run uses examples/small's
    defaults unmodified.
    """

    GROUND_TRUTH_NGRID = 32
    GROUND_TRUTH_BOX_SIZE_KPC = 4000

    def __init__(self, theta_fid: dict, sigma_frac: float, mpgadget_sim: "MPGadgetSimulator",
                 ground_truth_workdir: Path, seed: int):
        self._theta_fid = theta_fid
        self._sigma_frac = sigma_frac
        self._mpgadget_sim = mpgadget_sim
        self._ground_truth_workdir = Path(ground_truth_workdir)
        self._rng = np.random.default_rng(seed)
        self._pk_obs = None
        self._sigma_k = None

    def generate_obs(self, out_path: Path) -> None:
        """ONE real MP-Gadget trial at theta_fid, at the fixed ground-truth resolution."""
        pk_true, _cpu_hours = self._mpgadget_sim(
            self._theta_fid, ngrid=self.GROUND_TRUTH_NGRID, box_size_kpc=self.GROUND_TRUTH_BOX_SIZE_KPC,
            workdir=self._ground_truth_workdir, notes="ground_truth",
        )
        noise = self._rng.normal(0.0, self._sigma_frac, size=len(pk_true))
        self._pk_obs = pk_true * (1.0 + noise)
        self._sigma_k = self._sigma_frac * self._pk_obs
        np.save(out_path, self._pk_obs)

    def score(self, params: dict, workdir: Path) -> float:
        """ONE MORE real MP-Gadget trial, at the caller's proposed params (same fixed
        ground-truth resolution), for an apples-to-apples data-space chi2 comparison.
        There is no free/trustworthy way to do this — SyrenAnsatz is explicitly not
        trustworthy for chi2-bearing comparisons (see simulator/syren_ansatz.py)."""
        if self._pk_obs is None:
            raise RuntimeError("generate_obs() must be called before score()")
        pk_proposal, _cpu_hours = self._mpgadget_sim(
            params, ngrid=self.GROUND_TRUTH_NGRID, box_size_kpc=self.GROUND_TRUTH_BOX_SIZE_KPC,
            workdir=Path(workdir), notes="oracle_score",
        )
        sigma = self._sigma_k if self._sigma_frac > 0 else np.ones_like(self._pk_obs)
        return compute_chi2(pk_proposal, self._pk_obs, sigma)


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
    k_vec = make_k_vec(cfg)

    oracle = Oracle(theta_fid=json.loads(args.theta_json), k_vec=k_vec,
                    sigma_frac=args.sigma_frac, seed=args.seed)
    oracle.generate_obs(Path(args.out))
    print(f"obs_pk.npy written to {args.out}")


if __name__ == "__main__":
    generate_obs_cli()
