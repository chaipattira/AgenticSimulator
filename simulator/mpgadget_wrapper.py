"""
PLACEHOLDER — sub-project 1 (docs/superpowers/specs/2026-07-13-mpgadget-wrapper-design.md)
owns the canonical implementation of this file. This is sub-project 2's interim stand-in,
built strictly to that spec's documented interface contract, so the agent-facing tools/tests
in sub-project 2 (docs/superpowers/specs/2026-07-13-mpgadget-agent-integration-design.md) have
something real to import before sub-project 1 lands on `main`. DO NOT extend this file's real
SLURM/paramfile logic beyond what's needed to unblock sub-project 2's own tests — when
sub-project 1's real implementation lands on main, replace this file with theirs (git will
conflict on this exact path; take theirs, then re-run this branch's test suite to confirm
nothing in sub-project 2 relied on placeholder-only behavior).

MPGadgetSimulator: writes paramfile.genic/paramfile.gadget from shenqi/examples/small's
templates, generates the trial's input power spectrum via shenqi/tools/make_class_power.py,
submits MP-GenIC then MP-Gadget via `sbatch --wait`, parses the z=0 powerspectrum output,
interpolates it onto the caller-supplied canonical k-grid, and measures cpu_hours from sacct.
"""
import csv
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from config import MPGADGET_CSV_FIELDS

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S"

# The MP-Gadget phase's canonical k-grid — MUST match config/prior_bounds_mpgadget.yaml's
# k_vector block (logspace_start=log10(2), logspace_end=log10(15), n_points=25). Hardcoded
# here (not a __call__ argument) because the documented interface contract
# (docs/superpowers/specs/2026-07-13-mpgadget-wrapper-design.md) fixes __call__'s signature
# as (params, ngrid, box_size_kpc, workdir, notes="") -> (pk, cpu_hours) — no k_vec argument.
CANONICAL_K_VEC = np.logspace(np.log10(2.0), np.log10(15.0), 25)


class OutOfPriorError(ValueError):
    pass


class MPGadgetJobError(Exception):
    """Raised when any stage of the pipeline fails. `stage` identifies which one;
    `stderr_tail` carries the last portion of that stage's stderr for diagnosis."""

    def __init__(self, stage: str, stderr_tail: str):
        self.stage = stage
        self.stderr_tail = stderr_tail
        super().__init__(f"MP-Gadget job failed at stage={stage!r}: {stderr_tail}")


def _parse_paramfile_key(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#") or "=" not in stripped:
        return None
    return line.split("=", 1)[0].strip()


def _override_paramfile(template_lines: list[str], overrides: dict[str, str]) -> list[str]:
    """Rewrite matched `Key = Value` lines (comment stripped); pass through everything
    else byte-for-byte, including multi-value fields this wrapper never overrides."""
    remaining = dict(overrides)
    out = []
    for line in template_lines:
        key = _parse_paramfile_key(line)
        if key is not None and key in remaining:
            out.append(f"{key} = {remaining.pop(key)}")
        else:
            out.append(line)
    if remaining:
        raise KeyError(f"paramfile override key(s) not found in template: {sorted(remaining)}")
    return "\n".join(out) + "\n"


class MPGadgetSimulator:
    """Wraps a real MP-Gadget genic+gadget run with prior-bounds enforcement, cpu_hours
    tracking (from sacct), and CSV logging — structurally parallel to SyrenSimulator, but
    each call is a multi-stage SLURM pipeline rather than an in-process function call."""

    def __init__(self, shenqi_root: Path, csv_path: Path, prior_bounds: dict | None = None,
                 slurm_account: str = "phy240043", partition: str = "shared",
                 ntasks: int = 8, cpus_per_task: int = 2, walltime: str = "00:30:00"):
        self.shenqi_root = Path(shenqi_root)
        self.csv_path = Path(csv_path)
        self.prior_bounds = prior_bounds
        self.slurm_account = slurm_account
        self.partition = partition
        self.ntasks = ntasks
        self.cpus_per_task = cpus_per_task
        self.walltime = walltime

        # csv_path is used only to resume call_count/cpu_hours_total bookkeeping across
        # process restarts (each agent iteration is a fresh process) — it is NOT written to
        # by this class. The one authoritative CSV row per call (including chi2, which this
        # class cannot compute — it only returns pk) is written by the caller
        # (tools/run_mpgadget_trial.py), in a single atomic write, mirroring how
        # tools/compute_chi2.py writes SyrenSimulator's row only once chi2 is known.
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.csv_path.exists():
            with open(self.csv_path) as f:
                rows = list(csv.DictReader(f))
            self.call_count = len(rows)
            self.cpu_hours_total = sum(float(r["cpu_hours"]) for r in rows if r.get("cpu_hours"))
        else:
            self.call_count = 0
            self.cpu_hours_total = 0.0
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=MPGADGET_CSV_FIELDS).writeheader()

    # -- prior bounds -----------------------------------------------------------------

    def _check_bounds(self, params: dict, ngrid: int, box_size_kpc: float) -> None:
        if self.prior_bounds is None:
            return
        param_bounds = self.prior_bounds.get("parameters", {})
        for key, b in param_bounds.items():
            if key in params and not (b["min"] <= params[key] <= b["max"]):
                raise OutOfPriorError(f"{key}={params[key]} outside prior [{b['min']}, {b['max']}]")
        res_bounds = self.prior_bounds.get("resolution", {})
        if "ngrid" in res_bounds:
            b = res_bounds["ngrid"]
            if not (b["min"] <= ngrid <= b["max"]):
                raise OutOfPriorError(f"ngrid={ngrid} outside prior [{b['min']}, {b['max']}]")
        if "box_size_kpc" in res_bounds:
            b = res_bounds["box_size_kpc"]
            if not (b["min"] <= box_size_kpc <= b["max"]):
                raise OutOfPriorError(
                    f"box_size_kpc={box_size_kpc} outside prior [{b['min']}, {b['max']}]"
                )

    # -- paramfile writing --------------------------------------------------------------

    def _write_paramfiles(self, params: dict, ngrid: int, box_size_kpc: float,
                           workdir: Path, input_pk_path: Path) -> tuple[Path, Path]:
        omega_lambda = 1.0 - params["om"]

        genic_template = (self.shenqi_root / "examples" / "small" / "paramfile.genic").read_text().splitlines()
        genic_overrides = {
            "Ngrid": str(ngrid),
            "BoxSize": str(box_size_kpc),
            "Omega0": str(params["om"]),
            "OmegaLambda": str(omega_lambda),
            "OmegaBaryon": str(params["ob"]),
            "Sigma8": str(params["sigma8"]),
            "OutputDir": str(workdir / "output"),
            "FileWithInputSpectrum": str(input_pk_path),
        }
        genic_path = workdir / "paramfile.genic"
        genic_path.write_text(_override_paramfile(genic_template, genic_overrides))

        gadget_template = (self.shenqi_root / "examples" / "small" / "paramfile.gadget").read_text().splitlines()
        gadget_overrides = {
            "Omega0": str(params["om"]),
            "OmegaLambda": str(omega_lambda),
            "OmegaBaryon": str(params["ob"]),
            "WindEnergyFraction": str(params["wind_energy_fraction"]),
            "WindSpeedFactor": str(params["wind_speed_factor"]),
            "BlackHoleFeedbackFactor": str(params["bh_feedback_factor"]),
            "InitCondFile": str(workdir / "output" / "IC"),
            "OutputDir": str(workdir / "output"),
            "TreeCoolFile": str((self.shenqi_root / "examples" / "TREECOOL_fg_june11").resolve()),
            "MetalCoolFile": str((self.shenqi_root / "examples" / "cooling_metal_UVB").resolve()),
        }
        gadget_path = workdir / "paramfile.gadget"
        gadget_path.write_text(_override_paramfile(gadget_template, gadget_overrides))

        return genic_path, gadget_path

    # -- input power spectrum (local, CPU-only, no SLURM) --------------------------------

    def _generate_input_power_spectrum(self, params: dict, workdir: Path) -> Path:
        out_path = workdir / "input_powerspectrum.txt"
        script = self.shenqi_root / "tools" / "make_class_power.py"
        result = subprocess.run(
            [sys.executable, str(script), "--om", str(params["om"]), "--ob", str(params["ob"]),
             "--sigma8", str(params["sigma8"]), "--out", str(out_path)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise MPGadgetJobError("make_class_power", result.stderr[-2000:])
        return out_path

    # -- SLURM submission -----------------------------------------------------------------

    def _write_job_script(self, workdir: Path, stage: str, binary: str, paramfile: Path) -> Path:
        script_path = workdir / f"{stage}_job.sh"
        script_path.write_text(
            "#!/bin/bash\n"
            f"#SBATCH --account={self.slurm_account}\n"
            f"#SBATCH --partition={self.partition}\n"
            f"#SBATCH --ntasks={self.ntasks}\n"
            f"#SBATCH --cpus-per-task={self.cpus_per_task}\n"
            f"#SBATCH --time={self.walltime}\n"
            f"#SBATCH --job-name={stage}\n"
            f"export OMP_NUM_THREADS={self.cpus_per_task}\n"
            f"mpirun -np {self.ntasks} {binary} {paramfile}\n"
        )
        return script_path

    def _sbatch_wait(self, stage: str, script_path: Path) -> str:
        result = subprocess.run(
            ["sbatch", "--wait", str(script_path)], capture_output=True, text=True,
        )
        match = re.search(r"Submitted batch job (\d+)", result.stdout)
        if result.returncode != 0 or match is None:
            raise MPGadgetJobError(stage, result.stderr[-2000:])
        return match.group(1)

    def _sacct_elapsed_ncpus(self, job_id: str, attempts: int = 3, delay: float = 2.0) -> float:
        """sacct can lag briefly behind sbatch --wait returning; retry a few times."""
        for attempt in range(attempts):
            result = subprocess.run(
                ["sacct", "-j", job_id, "--format=ElapsedRaw,NCPUS", "--noheader", "-P"],
                capture_output=True, text=True,
            )
            rows = [line.split("|") for line in result.stdout.strip().splitlines() if line.strip()]
            if result.returncode == 0 and rows:
                total_core_seconds = sum(float(r[0]) * float(r[1]) for r in rows if len(r) == 2)
                return total_core_seconds / 3600.0
            if attempt < attempts - 1:
                time.sleep(delay)
        raise MPGadgetJobError("sacct", result.stderr[-2000:] if result.stderr else "no sacct rows returned")

    # -- output parsing --------------------------------------------------------------------

    def _parse_and_interpolate(self, workdir: Path) -> np.ndarray:
        pk_path = workdir / "output" / "powerspectrum-1.0000.txt"
        if not pk_path.exists():
            raise MPGadgetJobError("parse_output", f"missing {pk_path}")
        k_raw, p_raw = [], []
        for line in pk_path.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            parts = stripped.split()
            k_raw.append(float(parts[0]))
            p_raw.append(float(parts[1]))
        k_raw, p_raw = np.array(k_raw), np.array(p_raw)
        if len(k_raw) == 0:
            raise MPGadgetJobError("parse_output", f"{pk_path} contained no data rows")
        # k-grid is chosen to always sit strictly inside every allowed run's [k_f, k_Ny] —
        # this is interpolation, never extrapolation, in log-k/log-P space.
        log_pk = np.interp(np.log(CANONICAL_K_VEC), np.log(k_raw), np.log(p_raw))
        return np.exp(log_pk)

    # -- main entry point -------------------------------------------------------------------

    def __call__(self, params: dict, ngrid: int, box_size_kpc: float, workdir: Path,
                  notes: str = "") -> tuple[np.ndarray, float]:
        self._check_bounds(params, ngrid, box_size_kpc)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)

        input_pk_path = self._generate_input_power_spectrum(params, workdir)
        genic_path, gadget_path = self._write_paramfiles(params, ngrid, box_size_kpc, workdir, input_pk_path)

        genic_script = self._write_job_script(workdir, "genic", str(self.shenqi_root / "genic" / "MP-GenIC"), genic_path)
        genic_job_id = self._sbatch_wait("genic", genic_script)

        gadget_script = self._write_job_script(workdir, "gadget", str(self.shenqi_root / "gadget" / "MP-Gadget"), gadget_path)
        gadget_job_id = self._sbatch_wait("gadget", gadget_script)

        pk = self._parse_and_interpolate(workdir)
        cpu_hours = self._sacct_elapsed_ncpus(genic_job_id) + self._sacct_elapsed_ncpus(gadget_job_id)

        self.call_count += 1
        self.cpu_hours_total += cpu_hours
        return pk, cpu_hours
