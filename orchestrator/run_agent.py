import json, os, shutil, subprocess
from pathlib import Path

from judge.oracle import Oracle


_CLAUDE_MD = """\
# Operating instructions

## Journal

Every iteration gets a subsection in `journal.md` with these five headings:

```
## Iteration N
**Goal:** what you are trying to learn from this call
**Hypothesis:** what you expect to happen and why
**Method:** the exact parameters you will test
**Analysis:** what the result tells you — compare to obs_pk shape, explain residuals
**Next steps:** what you will try next and why
```

Include failures. If it is not in the journal, it did not happen.

## best_params.json

After every call to `compute_chi2.py`, update `best_params.json` with your current best parameters (lowest chi2 so far):

```json
{"om": ..., "ob": ..., "h": ..., "ns": ..., "as_": ..., "w0": ...}
```

## Stopping

Stop when either:
- chi2 < ε (write "CALIBRATION COMPLETE" in the journal, update best_params.json, stop)
- You judge that further calls will not improve the calibration

Do not keep calling just to explore. Your goal is to reach a good calibration with as few simulator calls as possible.

## Compaction recovery

If your context is reset, do not restart from scratch:
1. Read `runs.csv` — find the row with the lowest chi2, that is your current best
2. Read the last few entries in `journal.md` to recover your reasoning state
3. Update `best_params.json` with the best row from `runs.csv` if it is missing
4. Continue from where you left off
"""


def _find_claude() -> str:
    if path := shutil.which("claude"):
        return path
    for p in Path.home().glob(".vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude"):
        if p.is_file():
            return str(p)
    raise FileNotFoundError("claude CLI not found — add it to PATH or install the Claude Code VSCode extension")


def setup_workdir(base: Path, oracle: Oracle, project_root: Path) -> Path:
    """
    Create a fresh agent workdir with obs_pk.npy, config, and CLAUDE.md.
    theta_fid is NEVER written here — only oracle.generate_obs() output.
    """
    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    oracle.generate_obs(base / "obs_pk.npy")
    (base / "config").mkdir(exist_ok=True)
    shutil.copy(project_root / "config" / "prior_bounds.yaml", base / "config")

    # Durable operating instructions — survive context compaction
    (base / "CLAUDE.md").write_text(_CLAUDE_MD)

    # Restrict the agent to its own workdir and the shared tools directory
    (base / ".claude").mkdir(exist_ok=True)
    (base / ".claude" / "settings.json").write_text(
        json.dumps({"allowedPaths": [str(base.resolve()), str((project_root / "tools").resolve())]})
    )

    return base


def run_agent(workdir: Path, project_root: Path, timeout_seconds: int = 3600) -> int:
    """
    Invoke claude --print in workdir with program.md as the task prompt.
    Returns the exit code.
    """
    tools_path = str(project_root / "tools")
    prompt = (project_root / "program.md").read_text().replace("TOOLS_PATH", tools_path)

    venv_bin = project_root / ".venv" / "bin"
    path = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")
    env = {**os.environ, "PATH": path}

    log_path = Path(workdir) / "agent.log"
    with open(log_path, "w") as log_file:
        result = subprocess.run(
            [_find_claude(), "--print", "--dangerously-skip-permissions", prompt],
            cwd=workdir,
            timeout=timeout_seconds,
            text=True,
            stdout=log_file,
            stderr=log_file,
            env=env,
        )
    return result.returncode
