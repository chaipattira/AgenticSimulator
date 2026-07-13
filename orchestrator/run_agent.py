import importlib.util
import json, os, shutil, subprocess
from pathlib import Path

from config import default_shenqi_root
from judge.oracle import Oracle
from orchestrator.harvest import should_stop


def _syren_source_dir(project_root: Path) -> Path | None:
    """Return the symbolic_pofk package directory, or None if not importable."""
    spec = importlib.util.find_spec("symbolic_pofk")
    if spec and spec.submodule_search_locations:
        return Path(list(spec.submodule_search_locations)[0])
    return None


def _shenqi_source_dir(project_root: Path) -> Path | None:
    """Return shenqi/'s root directory, or None if not present on disk. shenqi/ is
    gitignored (a large external C codebase — see CLAUDE.md), so this mirrors
    _syren_source_dir's "return None if genuinely absent" shape rather than assuming it's
    always there."""
    d = default_shenqi_root(project_root)
    return d if d.exists() else None


# Backend registry: which CLAUDE.md template, prior-bounds config filename, and extra
# read-only reference source directory setup_workdir wires up for a given simulator
# backend. run_agent_loop itself has no notion of "backend" at all — it only spawns
# `claude --print` and reads runs.csv, exactly as the PRD's "designed to swap cleanly to
# an HPC/SLURM simulator backend later" goal requires.
_BACKENDS = {
    "syren_new": {
        "template": Path(__file__).parent / "templates" / "agent_claude_syren_new.md",
        "config_filename": "prior_bounds.yaml",
        "extra_source_dir": _syren_source_dir,
    },
    "mpgadget": {
        "template": Path(__file__).parent / "templates" / "agent_claude_mpgadget.md",
        "config_filename": "prior_bounds_mpgadget.yaml",
        "extra_source_dir": _shenqi_source_dir,
    },
}


_ITERATION_PROMPT = "Begin your next iteration now. Follow the procedure in CLAUDE.md."


def _find_claude() -> str:
    if path := shutil.which("claude"):
        return path
    for p in Path.home().glob(".vscode-server/extensions/anthropic.claude-code-*/resources/native-binary/claude"):
        if p.is_file():
            return str(p)
    raise FileNotFoundError("claude CLI not found — add it to PATH or install the Claude Code VSCode extension")


def setup_workdir(base: Path, oracle: Oracle, project_root: Path, backend: str = "syren_new") -> Path:
    """
    Create a fresh agent workdir with obs_pk.npy, config, and CLAUDE.md.
    theta_fid is NEVER written here — only oracle.generate_obs() output.

    `backend` selects which CLAUDE.md template, prior-bounds config, and read-only
    reference source directory to wire up (see _BACKENDS above) — "syren_new" (default,
    unchanged behavior) or "mpgadget".
    """
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {sorted(_BACKENDS)}")
    backend_cfg = _BACKENDS[backend]

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)

    oracle.generate_obs(base / "obs_pk.npy")
    (base / "config").mkdir(exist_ok=True)
    shutil.copy(project_root / "config" / backend_cfg["config_filename"], base / "config")

    # Durable operating instructions — every iteration is a fresh process, so this
    # is the only context each invocation gets (auto-loaded by Claude Code from cwd).
    tools_path = str((project_root / "tools").resolve())
    claude_md = backend_cfg["template"].read_text().replace("TOOLS_PATH", tools_path)
    (base / "CLAUDE.md").write_text(claude_md)

    # Restrict the agent to its own workdir, the shared tools directory, and the
    # backend's read-only reference source (symbolic_pofk for syren_new, shenqi/ for
    # mpgadget; enforced by instruction, not filesystem perms)
    allowed = [str(base.resolve()), str((project_root / "tools").resolve())]
    extra_dir = backend_cfg["extra_source_dir"](project_root)
    if extra_dir:
        allowed.append(str(extra_dir.resolve()))
    (base / ".claude").mkdir(exist_ok=True)
    (base / ".claude" / "settings.json").write_text(json.dumps({"allowedPaths": allowed}))

    return base


def _invoke_claude(workdir: Path, prompt: str, project_root: Path, timeout_seconds: int) -> int:
    """Spawn one fresh `claude --print` process in workdir. Appends to agent.log."""
    venv_bin = project_root / ".venv" / "bin"
    env = {**os.environ, "PATH": str(venv_bin) + os.pathsep + os.environ.get("PATH", "")}
    log_path = Path(workdir) / "agent.log"
    with open(log_path, "a") as log_file:
        result = subprocess.run(
            [_find_claude(), "--print", "--dangerously-skip-permissions", prompt],
            cwd=workdir, timeout=timeout_seconds,
            text=True, stdout=log_file, stderr=log_file, env=env,
        )
    return result.returncode


def run_agent_loop(
    workdir: Path,
    project_root: Path,
    max_cpu_hours: float,
    epsilon: float,
    max_iterations: int = 50,
    iteration_timeout: int = 600,
    max_consecutive_failures: int = 3,
) -> int:
    """
    Run the calibration loop: spawn one fresh `claude --print` process per iteration.
    Each iteration reads state from disk (runs.csv, journal.md, best_params.json) and
    makes exactly one simulator call, at a cpu_hours cost of its own choosing.

    Stops when runs.csv shows convergence or budget exhaustion (should_stop), when
    max_iterations is reached (a safety cap independent of cpu_hours, since a call can
    cost an arbitrarily small amount), or when max_consecutive_failures worth of claude
    invocations exit non-zero in a row.

    Returns the number of iterations run.
    """
    workdir = Path(workdir)
    consecutive_failures = 0

    for i in range(1, max_iterations + 1):
        rc = _invoke_claude(workdir, _ITERATION_PROMPT, project_root, timeout_seconds=iteration_timeout)
        consecutive_failures = 0 if rc == 0 else consecutive_failures + 1
        if consecutive_failures >= max_consecutive_failures:
            break
        if should_stop(workdir, epsilon, max_cpu_hours):
            break

    return i
