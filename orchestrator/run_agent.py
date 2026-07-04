import json, shutil, subprocess
from pathlib import Path

from judge.oracle import Oracle


def setup_workdir(base: Path, oracle: Oracle, project_root: Path) -> Path:
    """
    Create a fresh agent workdir with obs_pk.npy and config.
    theta_fid is NEVER written here — only oracle.generate_obs() output.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    oracle.generate_obs(base / "obs_pk.npy")
    (base / "config").mkdir(exist_ok=True)
    shutil.copy(project_root / "config" / "prior_bounds.yaml", base / "config")

    # Restrict the agent to its own workdir — prevents reading other rollouts or project root.
    # Claude Code enforces allowedPaths at the permission layer, so even bash tool calls
    # outside this path are blocked regardless of what program.md says.
    (base / ".claude").mkdir()
    (base / ".claude" / "settings.json").write_text(
        json.dumps({"allowedPaths": [str(base)]})
    )

    return base


def run_agent(workdir: Path, project_root: Path, timeout_seconds: int = 3600) -> int:
    """
    Invoke claude --non-interactive in workdir with program.md as the prompt.
    Returns the exit code.
    """
    program_md = (project_root / "program.md").read_text()
    tools_path = str(project_root / "tools")
    prompt = program_md.replace("/path/to/tools", tools_path)

    result = subprocess.run(
        ["claude", "--print", "--dangerously-skip-permissions", prompt],
        cwd=workdir,
        timeout=timeout_seconds,
        text=True,
    )
    return result.returncode
