import json
from pathlib import Path

from orchestrator.run_agent import setup_workdir


def test_setup_workdir_contains_required_files(tmp_path, k_vec, theta_fid, sigma_frac):
    from judge.oracle import Oracle
    oracle = Oracle(theta_fid=theta_fid, k_vec=k_vec, sigma_frac=sigma_frac, seed=0)
    workdir = setup_workdir(
        base=tmp_path / "rollout_0",
        oracle=oracle,
        project_root=Path(__file__).parent.parent,
    )
    assert (workdir / "obs_pk.npy").exists()
    assert (workdir / "config" / "prior_bounds.yaml").exists()
    assert not (workdir / "theta_fid.json").exists()
    assert not (workdir / "theta_fid.yaml").exists()
    settings = json.loads((workdir / ".claude" / "settings.json").read_text())
    assert str(workdir.resolve()) in settings["allowedPaths"]
