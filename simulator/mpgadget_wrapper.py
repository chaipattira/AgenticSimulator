"""Wraps a real MP-Gadget (shenqi fork) run: writes paramfiles, generates the trial's
input power spectrum via CLASS, submits MP-GenIC and MP-Gadget as one blocking SLURM job
(both mpirun calls sequential in one allocation — see _write_job_script), parses the
resulting P(k), and interpolates it onto the canonical MP-Gadget k-grid.

See docs/superpowers/specs/2026-07-13-mpgadget-wrapper-design.md for the full design.
No agent involvement here — this proves (params, ngrid, box_size) -> (pk, cpu_hours)
works against a real Anvil SLURM run.
"""
import csv
import subprocess
import time
from pathlib import Path

import numpy as np

from config import MPGADGET_PARAM_KEYS
from simulator.mpgadget_paramfile import override_paramfile
from simulator.mpgadget_powerspectrum import interpolate_to_grid, read_powerspectrum
from simulator.slurm import submit_and_wait
from simulator.syren_wrapper import OutOfPriorError

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"
CSV_FIELDS = ["call_idx"] + MPGADGET_PARAM_KEYS + ["ngrid", "box_size_kpc", "cpu_hours", "timestamp", "notes"]

_MODULES = "gcc/11.2.0 openmpi/4.0.6 gsl/2.4 fftw/3.3.8 boost/1.74.0"


class MPGadgetJobError(Exception):
    def __init__(self, stage: str, detail: str):
        self.stage = stage
        self.detail = detail
        super().__init__(f"MP-Gadget pipeline failed at stage '{stage}': {detail[-2000:]}")


def _canonical_k_grid() -> np.ndarray:
    return np.logspace(np.log10(2.0), np.log10(15.0), 25)


class MPGadgetSimulator:
    def __init__(self, shenqi_root: Path, csv_path: Path, prior_bounds: dict | None = None,
                 slurm_account: str = "phy240043", partition: str = "shared", nodes: int = 1,
                 ntasks: int = 8, cpus_per_task: int = 2, walltime: str = "00:30:00"):
        self.shenqi_root = Path(shenqi_root)
        self.csv_path = Path(csv_path)
        self.prior_bounds = prior_bounds
        self.slurm_account = slurm_account
        self.partition = partition
        self.nodes = nodes
        self.ntasks = ntasks
        self.cpus_per_task = cpus_per_task
        self.walltime = walltime

        if self.csv_path.exists():
            rows = list(csv.DictReader(open(self.csv_path)))
            self.call_count = len(rows)
            self.cpu_hours_total = sum(float(r["cpu_hours"]) for r in rows if r.get("cpu_hours"))
        else:
            self.call_count = 0
            self.cpu_hours_total = 0.0
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=CSV_FIELDS).writeheader()

    def _check_bounds(self, params: dict, ngrid: int, box_size_kpc: float) -> None:
        if self.prior_bounds is None:
            return
        for key, b in self.prior_bounds["parameters"].items():
            if not (b["min"] <= params[key] <= b["max"]):
                raise OutOfPriorError(f"{key}={params[key]} outside prior [{b['min']}, {b['max']}]")
        res = self.prior_bounds["resolution"]
        if not (res["ngrid"]["min"] <= ngrid <= res["ngrid"]["max"]):
            raise OutOfPriorError(f"ngrid={ngrid} outside prior [{res['ngrid']['min']}, {res['ngrid']['max']}]")
        if not (res["box_size_kpc"]["min"] <= box_size_kpc <= res["box_size_kpc"]["max"]):
            raise OutOfPriorError(
                f"box_size_kpc={box_size_kpc} outside prior "
                f"[{res['box_size_kpc']['min']}, {res['box_size_kpc']['max']}]"
            )

    def __call__(self, params: dict, ngrid: int, box_size_kpc: float, workdir: Path,
                 notes: str = "") -> tuple[np.ndarray, float]:
        self._check_bounds(params, ngrid, box_size_kpc)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        output_dir = workdir / "output"
        output_dir.mkdir(exist_ok=True)

        omega_lambda = 1.0 - params["om"]
        example_dir = self.shenqi_root / "examples" / "small"

        genic_path = workdir / "paramfile.genic"
        genic_lines = override_paramfile(
            (example_dir / "paramfile.genic").read_text().splitlines(),
            {
                "OutputDir": str(output_dir), "FileBase": "IC",
                "Ngrid": str(ngrid), "BoxSize": str(box_size_kpc),
                "Omega0": str(params["om"]), "OmegaBaryon": str(params["ob"]),
                "OmegaLambda": str(omega_lambda), "Sigma8": str(params["sigma8"]),
                "FileWithInputSpectrum": "input_pk.dat",
            },
        )
        genic_path.write_text("\n".join(genic_lines) + "\n")

        gadget_path = workdir / "paramfile.gadget"
        gadget_lines = override_paramfile(
            (example_dir / "paramfile.gadget").read_text().splitlines(),
            {
                "InitCondFile": str(output_dir / "IC"), "OutputDir": str(output_dir),
                "Omega0": str(params["om"]), "OmegaBaryon": str(params["ob"]),
                "OmegaLambda": str(omega_lambda),
                "WindEnergyFraction": str(params["wind_energy_fraction"]),
                "WindSpeedFactor": str(params["wind_speed_factor"]),
                "BlackHoleFeedbackFactor": str(params["bh_feedback_factor"]),
                "TreeCoolFile": str(self.shenqi_root / "examples" / "TREECOOL_fg_june11"),
                "MetalCoolFile": str(self.shenqi_root / "examples" / "cooling_metal_UVB"),
            },
        )
        gadget_path.write_text("\n".join(gadget_lines) + "\n")

        self._run_make_class_power(genic_path)

        cpu_hours = submit_and_wait(
            self._write_job_script(workdir, genic_path, gadget_path), stage="mpgadget",
            log_path=workdir / "mpgadget.slurm.log",
        )

        pk_path = output_dir / "powerspectrum-1.0000.txt"
        if not pk_path.exists():
            raise MPGadgetJobError("mpgadget", f"{pk_path} not found after job completion")
        k_raw, p_raw = read_powerspectrum(pk_path)
        k_grid = _canonical_k_grid()
        pk = interpolate_to_grid(k_raw, p_raw, k_grid)

        self.call_count += 1
        self.cpu_hours_total += cpu_hours
        self._log_csv(params, ngrid, box_size_kpc, cpu_hours, notes)
        return pk, cpu_hours

    def _run_make_class_power(self, genic_path: Path) -> None:
        script = self.shenqi_root / "tools" / "make_class_power.py"
        result = subprocess.run(
            ["python", str(script), str(genic_path)], capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise MPGadgetJobError("make_class_power", result.stderr)

    def _write_job_script(self, workdir: Path, genic_paramfile: Path, gadget_paramfile: Path) -> Path:
        """One SLURM job runs both MP-GenIC and MP-Gadget sequentially in the same
        allocation — matches shenqi/examples/small/run.sh's own pattern (mpirun genic,
        then mpirun gadget, `|| exit 1` between them). Halves real job submissions per
        evaluation versus submitting genic and gadget as two separate sbatch --wait
        calls: one queue wait instead of two, one sacct query instead of two, and half
        the exposure to Anvil's per-user concurrent-job (QOS) submit limit.

        --nodes={self.nodes} pins the job to a fixed node count (default 1) rather than
        leaving placement to SLURM: an identical config was observed to take 9 minutes on
        one node but time out at 30 minutes when the scheduler split it across two
        partially-loaded nodes instead (extra MPI communication overhead across the tree/
        domain-decomposition pattern) — trading a possibly longer queue wait for
        predictable run time once started."""
        genic_binary = self.shenqi_root / "genic" / "MP-GenIC"
        gadget_binary = self.shenqi_root / "gadget" / "MP-Gadget"
        script_path = workdir / "mpgadget.slurm.sh"
        log_path = workdir / "mpgadget.slurm.log"
        script_path.write_text(f"""#!/bin/bash
#SBATCH --account={self.slurm_account}
#SBATCH --partition={self.partition}
#SBATCH --nodes={self.nodes}
#SBATCH --ntasks={self.ntasks}
#SBATCH --cpus-per-task={self.cpus_per_task}
#SBATCH --time={self.walltime}
#SBATCH --job-name=mpgadget-trial
#SBATCH --output={log_path}
#SBATCH --error={log_path}

module purge
module load {_MODULES}
export OMP_NUM_THREADS={self.cpus_per_task}

cd {workdir}
mpirun -np {self.ntasks} {genic_binary} {genic_paramfile} || exit 1
mpirun -np {self.ntasks} {gadget_binary} {gadget_paramfile} || exit 1
""")
        return script_path

    def _log_csv(self, params: dict, ngrid: int, box_size_kpc: float, cpu_hours: float, notes: str) -> None:
        row = {
            "call_idx": self.call_count, **{k: params[k] for k in MPGADGET_PARAM_KEYS},
            "ngrid": ngrid, "box_size_kpc": box_size_kpc, "cpu_hours": cpu_hours,
            "timestamp": time.strftime(TIMESTAMP_FORMAT), "notes": notes,
        }
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
