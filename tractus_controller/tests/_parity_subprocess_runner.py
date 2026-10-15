"""Subprocess runner for parity / fp64_oracle tests (controller side).

ARCHITECTURE.md §10 + infra/001 require that any test which sets
``jax_enable_x64=True`` runs in its own process so it cannot flip the
default config of any other test. This module is the runner.

Usage from the tractus_controller root::

    .venv/bin/python -m tests._parity_subprocess_runner [extra pytest args]

The runner sets ``TRACTUS_PARITY_SUBPROCESS=1`` before invoking pytest so
``conftest.py`` knows to *include* the parity-marked tests instead of
skipping them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def main(argv: list[str]) -> int:
    env = dict(os.environ)
    env["TRACTUS_PARITY_SUBPROCESS"] = "1"
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "-m",
        "parity or fp64_oracle",
        # Override the default-slice addopts that excludes parity.
        "-o",
        "addopts=-ra",
        *argv,
    ]
    return subprocess.call(cmd, cwd=str(REPO), env=env)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
