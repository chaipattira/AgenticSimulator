# Compatibility shim: the standalone PyPI `validate` package (1.0.1) is
# Python-2-only (long-integer literals, bare-comma except clauses) and its
# sdist's own bundled configobj.py stub breaks under modern build isolation,
# so it cannot be pip-installed on Python 3. The DiffSK `configobj` fork
# (installed instead, see the `mpgadget` extra in pyproject.toml) already
# vendors a Python-3-compatible validate module at configobj.validate — this
# shim re-exports it under the top-level `validate` name that
# shenqi/tools/make_class_power.py imports.
#
# This file is not itself imported by anything in this repo. After
# `pip install -e ".[dev,mpgadget]"`, copy its contents to
# <venv>/lib/pythonX.Y/site-packages/validate.py — pip has no mechanism to
# install a bare top-level shim module as a side effect of installing
# `configobj`, so this one step stays manual.
from configobj.validate import *  # noqa: F401,F403
from configobj.validate import Validator  # noqa: F401
