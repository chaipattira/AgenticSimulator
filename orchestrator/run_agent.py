import importlib.util
import json, os, shutil, subprocess
from pathlib import Path

from judge.oracle import Oracle


def _syren_source_dir() -> Path | None:
    """Return the symbolic_pofk package directory, or None if not importable."""
    spec = importlib.util.find_spec("symbolic_pofk")
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return None


_CLAUDE_MD = """\
You are an expert cosmological simulator and Bayesian inference specialist. Your task is to
recover the cosmological parameters that generated the observed matter power spectrum in
`obs_pk.npy` using as few simulator calls as possible.

# Non-negotiable Rules

These rules apply at every step, including after context compaction.

## Iteration loop

Every simulator call follows this sequence — no exceptions:
1. Write the journal entry header **before** calling the tool (Goal + Hypothesis + Method)
2. Run `compute_chi2.py` or `get_pk.py`
3. Complete the journal entry (Result + Analysis + Next steps)
4. Update `best_params.json` if this call produced the lowest chi2 so far

## Journal format

Every iteration in `journal.md`:

```
## Iteration N
**Goal:** what you are trying to learn from this call
**Hypothesis:** what you predict will happen and why
**Method:** exact parameters — {"om":…, "ob":…, "h":…, "ns":…, "as_":…, "w0":…}
**Result:** chi2=<value>  call_idx=<N>
**Analysis:** what the result tells you. include failures and/or note which parameters to move and in which direction
**Next steps:** what you will try next and why
```

Remember: If it is not in the journal, it did not happen.

## Tool calling

Every call must have a hypothesis. Do not call speculatively.

All simulator evaluations must go through `compute_chi2.py` or `get_pk.py`. The `symbolic_pofk/` source is read-only.
You can inspect it to understand how parameters affect the power spectrum but NEVER call from `symbolic_pofk/` directly!

## Analysis scripts

Save every Python analysis script to a `.py` file in the workdir before running it.
If it is not on disk, it did not happen.

## best_params.json

Always holds your current lowest-chi2 parameters. Update after every call:

```json
{"om": ..., "ob": ..., "h": ..., "ns": ..., "as_": ..., "w0": ...}
```

## Stopping

Stop when any of these is true:
- **chi2 < ε** — write "CALIBRATION COMPLETE" as the Analysis in the final iteration, update `best_params.json`, stop. (ε is at key `chi2.epsilon` in `config/prior_bounds.yaml`)
- **Budget exhausted** — `call_idx` has reached the value at key `budget.max_calls` in `config/prior_bounds.yaml`
- **Converged** — last several calls improved chi2 by < 1% and you are below 2×ε

## Compaction recovery

If your context is reset mid-run:
1. Read `runs.csv` — find the row with the lowest chi2; that is your current best
2. Read the last several entries in `journal.md` to recover your reasoning
3. Verify `best_params.json` matches the best `runs.csv` row; update it if not
4. Continue from where you left off — do not restart from prior midpoints
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

    # Restrict the agent to its own workdir, the shared tools directory, and the
    # syren_new source (read-only reference; enforced by instruction, not filesystem perms)
    allowed = [str(base.resolve()), str((project_root / "tools").resolve())]
    syren_dir = _syren_source_dir()
    if syren_dir:
        allowed.append(str(syren_dir.resolve()))
    (base / ".claude").mkdir(exist_ok=True)
    (base / ".claude" / "settings.json").write_text(json.dumps({"allowedPaths": allowed}))

    return base


_RESUME_PROMPT = """\
# Cosmological Parameter Calibration — Resume

You are resuming a calibration run that is **not yet converged**.

Your working directory already contains `runs.csv`, `journal.md`, `best_params.json`,
and `obs_pk.npy` from your previous session.

## Tools

```bash
python TOOLS_PATH/compute_chi2.py --params '{{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}}' --notes "reasoning"
```
→ prints `chi2=<value>  call_idx=<N>`, appends a row to `runs.csv`.

```bash
python TOOLS_PATH/get_pk.py --params '{{"om":0.3,"ob":0.046,"h":0.7,"ns":0.97,"as_":2.1e-9,"w0":-1.0}}'
```
→ prints JSON with keys `k`, `pk`, `obs_pk`, `residual_frac`. Also appends to `runs.csv`.

## Resume now

Follow the compaction recovery procedure in `CLAUDE.md`:
1. Read `runs.csv` — the row with the lowest chi2 is your current best.
2. Read the last several entries in `journal.md` to recover your reasoning.
3. Verify `best_params.json` matches the best `runs.csv` row; update it if not.
4. Continue from where you left off — do not restart from scratch.
"""


def _invoke_claude(
    workdir: Path, prompt: str, project_root: Path,
    timeout_seconds: int = 3600, append_log: bool = False,
) -> int:
    venv_bin = project_root / ".venv" / "bin"
    env = {**os.environ, "PATH": str(venv_bin) + os.pathsep + os.environ.get("PATH", "")}
    log_path = Path(workdir) / "agent.log"
    with open(log_path, "a" if append_log else "w") as log_file:
        result = subprocess.run(
            [_find_claude(), "--print", "--dangerously-skip-permissions", prompt],
            cwd=workdir, timeout=timeout_seconds,
            text=True, stdout=log_file, stderr=log_file, env=env,
        )
    return result.returncode


def run_agent(workdir: Path, project_root: Path, timeout_seconds: int = 3600) -> int:
    """Invoke claude --print in workdir with program.md as the task prompt."""
    tools_path = str(project_root / "tools")
    prompt = (project_root / "program.md").read_text().replace("TOOLS_PATH", tools_path)
    return _invoke_claude(workdir, prompt, project_root, timeout_seconds)


def continue_agent(workdir: Path, project_root: Path, timeout_seconds: int = 3600) -> int:
    """
    Resume an existing (non-converged) run in workdir.
    Uses the compaction-recovery prompt; appends to agent.log rather than overwriting.
    Does NOT call setup_workdir() — all existing files are preserved.
    """
    tools_path = str(project_root / "tools")
    prompt = _RESUME_PROMPT.replace("TOOLS_PATH", tools_path)
    return _invoke_claude(workdir, prompt, project_root, timeout_seconds, append_log=True)
