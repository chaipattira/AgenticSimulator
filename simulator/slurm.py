"""Generic SLURM submission helper: submit a script with `sbatch --wait` (blocks the
calling process until the job finishes and returns the job's own exit code — this
directly implements sequential, blocking execution with no custom polling loop),
then measure cpu_hours from `sacct`. `sacct`'s accounting data can lag slightly even
after `sbatch --wait` returns, so it's queried with a brief retry."""
import re
import subprocess
import time
from pathlib import Path

_JOB_ID_RE = re.compile(r"Submitted batch job (\d+)")
_SACCT_RETRIES = 5
_SACCT_RETRY_DELAY_S = 2


class SlurmJobError(Exception):
    def __init__(self, stage: str, stderr_tail: str):
        self.stage = stage
        self.stderr_tail = stderr_tail
        super().__init__(f"SLURM job failed at stage '{stage}': {stderr_tail[-2000:]}")


def submit_and_wait(script_path: Path, stage: str = "", log_path: Path | None = None) -> float:
    """Submit script_path via `sbatch --wait`. Returns cpu_hours = ElapsedRaw * NCPUS / 3600
    for the job (excluding .batch/.extern sub-steps). Raises SlurmJobError on non-zero exit."""
    result = subprocess.run(
        ["sbatch", "--wait", str(script_path)], capture_output=True, text=True,
    )
    match = _JOB_ID_RE.search(result.stdout)
    job_id = match.group(1) if match else None

    if result.returncode != 0:
        tail = result.stderr
        if log_path and Path(log_path).exists():
            tail = Path(log_path).read_text()
        raise SlurmJobError(stage or script_path.name, tail)

    if job_id is None:
        raise SlurmJobError(stage or script_path.name, f"could not parse job id from: {result.stdout}")

    return _cpu_hours_from_sacct(job_id)


def _cpu_hours_from_sacct(job_id: str) -> float:
    for attempt in range(_SACCT_RETRIES):
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=JobID,ElapsedRaw,NCPUS", "-n", "-P"],
            capture_output=True, text=True,
        )
        for line in result.stdout.strip().splitlines():
            parts = line.split("|")
            if len(parts) == 3 and parts[0] == job_id:
                elapsed_raw, ncpus = float(parts[1]), float(parts[2])
                return elapsed_raw * ncpus / 3600.0
        if attempt < _SACCT_RETRIES - 1:
            time.sleep(_SACCT_RETRY_DELAY_S)
    raise SlurmJobError("sacct", f"no accounting data found for job {job_id} after {_SACCT_RETRIES} attempts")
