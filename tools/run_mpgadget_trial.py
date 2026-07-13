#!/usr/bin/env python
"""
run_mpgadget_trial.py — the ONE paid, cpu_hours-costing call allowed per iteration in the
MP-Gadget phase. Exactly one call per iteration — same rule as compute_chi2.py/get_pk.py in
the syren_new MVP. Unlike get_pk_ansatz.py (free, unlimited, no SLURM), every call here
submits a real MP-GenIC + MP-Gadget SLURM job pair via MPGadgetSimulator and its cpu_hours
(MEASURED from sacct afterward, not chosen by the agent) is added to the budget in runs.csv.

The agent instead chooses --ngrid and --box_size_kpc (the resolution/volume dial) — bigger
ngrid / smaller box_size_kpc buys finer resolution (more particles, more cpu_hours, wider
measured k-range); bigger box_size_kpc lowers the fundamental-mode k at the cost of
per-particle resolution.

Expects in CWD: obs_pk.npy, config/prior_bounds_mpgadget.yaml.
Appends to runs.csv (created on first call if absent, header shared with get_pk_ansatz.py).
Outputs:
chi2=<value>  call_idx=<N>  cpu_hours_spent=<v>  cpu_hours_total=<v>  ngrid=<v>  box_size_kpc=<v>
On failure (out-of-prior params/resolution, or any pipeline stage failing): prints an ERROR
line to stderr and exits non-zero. No CSV row is written for a failed trial.

NOTE on MPGadgetSimulator's own csv_path: sub-project 1's MPGadgetSimulator writes its own
CSV row internally, in its own schema (CSV_FIELDS — no chi2 or tool column; see
simulator/mpgadget_wrapper.py). That schema is NOT the agent-facing runs.csv schema this
tool and get_pk_ansatz.py share (MPGADGET_CSV_FIELDS, which has chi2 and tool columns for
the journal/harvest.py to read). Passing runs.csv as MPGadgetSimulator's csv_path would
silently corrupt it — csv.DictWriter appends rows positionally by fieldnames, and rows
written under two different fieldname lists to the same file misalign under whichever
header was written first. This tool therefore gives MPGadgetSimulator a separate, internal
bookkeeping file (mpgadget_sim_internal.csv) and writes the one authoritative runs.csv row
itself, once chi2 is known — mirroring how tools/compute_chi2.py writes SyrenSimulator's
row only once chi2 is known.
"""
import argparse, csv, json, sys, time
from pathlib import Path

import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from config import MPGADGET_CSV_FIELDS, MPGADGET_PARAM_KEYS, default_shenqi_root
from simulator.mpgadget_wrapper import MPGadgetJobError, MPGadgetSimulator
from simulator.slurm import SlurmJobError
from simulator.syren_wrapper import OutOfPriorError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", required=True)
    p.add_argument("--ngrid", type=int, required=True)
    p.add_argument("--box_size_kpc", type=float, required=True)
    p.add_argument("--notes", default="")
    args = p.parse_args()

    cwd = Path.cwd()
    with open(cwd / "config" / "prior_bounds_mpgadget.yaml") as f:
        cfg = yaml.safe_load(f)
    sigma_frac = cfg["noise"]["sigma_frac"]
    obs_pk = np.load(cwd / "obs_pk.npy")
    # No synthetic realization-noise term to add here (unlike compute_chi2.py's
    # sigma_realization): the resolution/precision tradeoff for a real MP-Gadget run is
    # real physics baked into the measurement itself, not a dial layered on top. See
    # design doc section 4.
    sigma_eff = sigma_frac * obs_pk

    params = json.loads(args.params)
    # See module docstring: MPGadgetSimulator's own csv_path is an internal bookkeeping
    # file, deliberately NOT runs.csv (schema mismatch would corrupt it).
    sim = MPGadgetSimulator(shenqi_root=default_shenqi_root(_ROOT),
                            csv_path=cwd / "mpgadget_sim_internal.csv", prior_bounds=cfg)

    trial_workdir = cwd / "mpgadget_runs" / f"trial_{sim.call_count + 1:04d}"
    try:
        pk, cpu_hours = sim(params, ngrid=args.ngrid, box_size_kpc=args.box_size_kpc,
                            workdir=trial_workdir, notes=args.notes)
    except OutOfPriorError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
    except MPGadgetJobError as e:
        print(f"ERROR: stage={e.stage}  detail={e.detail}", file=sys.stderr)
        sys.exit(1)
    except SlurmJobError as e:
        # genic/gadget submission failures propagate as SlurmJobError, not MPGadgetJobError
        # — simulator/slurm.py's submit_and_wait raises it directly, uncaught by
        # MPGadgetSimulator.__call__. Both must be handled here.
        print(f"ERROR: stage={e.stage}  stderr_tail={e.stderr_tail}", file=sys.stderr)
        sys.exit(1)

    chi2 = float(np.mean(((pk - obs_pk) / sigma_eff) ** 2))
    call_idx = sim.call_count

    row = {
        "call_idx": call_idx,
        **{k: params[k] for k in MPGADGET_PARAM_KEYS},
        "ngrid": args.ngrid, "box_size_kpc": args.box_size_kpc,
        "cpu_hours": cpu_hours, "tool": "mpgadget_trial",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chi2": chi2, "notes": args.notes,
    }
    runs_csv_path = cwd / "runs.csv"
    write_header = not runs_csv_path.exists()
    with open(runs_csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MPGADGET_CSV_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"chi2={chi2:.6f}  call_idx={call_idx}  cpu_hours_spent={cpu_hours:.4f}  "
          f"cpu_hours_total={sim.cpu_hours_total:.4f}  ngrid={args.ngrid}  box_size_kpc={args.box_size_kpc}")


if __name__ == "__main__":
    main()
