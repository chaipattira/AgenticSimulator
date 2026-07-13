from unittest.mock import patch, MagicMock

import pytest

from simulator.slurm import submit_and_wait, SlurmJobError


def _mock_run(sbatch_returncode=0, sbatch_stdout="Submitted batch job 12345\n",
              sacct_stdout="12345|125|16\n"):
    def _run(cmd, capture_output, text, **kwargs):
        result = MagicMock()
        if cmd[0] == "sbatch":
            result.returncode = sbatch_returncode
            result.stdout = sbatch_stdout
            result.stderr = ""
        elif cmd[0] == "sacct":
            result.returncode = 0
            result.stdout = sacct_stdout
            result.stderr = ""
        return result
    return _run


def test_submit_and_wait_returns_cpu_hours(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    with patch("simulator.slurm.subprocess.run", side_effect=_mock_run()):
        cpu_hours = submit_and_wait(script)
    # ElapsedRaw=125s, NCPUS=16 -> 125*16/3600 hours
    assert cpu_hours == pytest.approx(125 * 16 / 3600, rel=1e-6)


def test_submit_and_wait_raises_on_job_failure(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\nexit 1\n")
    log = tmp_path / "job.log"
    log.write_text("some slurm error output\n")
    with patch("simulator.slurm.subprocess.run", side_effect=_mock_run(sbatch_returncode=1)):
        with pytest.raises(SlurmJobError):
            submit_and_wait(script, log_path=log)


def test_submit_and_wait_parses_correct_sacct_row(tmp_path):
    script = tmp_path / "job.sh"
    script.write_text("#!/bin/bash\necho hi\n")
    # sacct returns extra .batch/.extern rows; must pick the exact jobid row
    sacct_stdout = "12345|125|16\n12345.batch|120|16\n12345.extern|125|16\n"
    with patch("simulator.slurm.subprocess.run", side_effect=_mock_run(sacct_stdout=sacct_stdout)):
        cpu_hours = submit_and_wait(script)
    assert cpu_hours == pytest.approx(125 * 16 / 3600, rel=1e-6)
