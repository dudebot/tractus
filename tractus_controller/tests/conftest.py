"""Skip markers for tests that need optional deps (espeak-ng, torch)
plus parity / fp64_oracle subprocess-isolation guard (infra/001).

Parity / fp64_oracle tests opt in to ``jax_enable_x64=True`` and must
run in a child process driven by ``tests/_parity_subprocess_runner.py``
so they cannot flip the default config of any other test in the run.
"""

from __future__ import annotations

import os
import shutil

import pytest


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401, PLC0415
    except Exception:  # also catches OSError on partial / broken installs
        return False
    return True


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_espeak = pytest.mark.skip(reason="espeak-ng not on PATH")
    skip_torch = pytest.mark.skip(reason="torch not installed (CPU is fine: pip install torch)")
    has_espeak = shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None
    has_torch = _torch_available()

    in_subprocess = os.environ.get("TRACTUS_PARITY_SUBPROCESS") == "1"
    skip_parity = pytest.mark.skip(
        reason=(
            "parity/fp64 tests must run in an isolated subprocess "
            "(see tests/_parity_subprocess_runner.py). "
            "Use `pytest -m parity` only via the runner; the default "
            "slice excludes them."
        )
    )

    for item in items:
        if "espeak" in item.keywords and not has_espeak:
            item.add_marker(skip_espeak)
        if "torch" in item.keywords and not has_torch:
            item.add_marker(skip_torch)
        if not in_subprocess and ("parity" in item.keywords or "fp64_oracle" in item.keywords):
            item.add_marker(skip_parity)
