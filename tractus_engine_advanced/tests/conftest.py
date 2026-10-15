"""Test-suite scaffolding for tractus_engine_advanced.

Two responsibilities:

1. **fp32-default invariant.** The default-slice (`pytest -q`) must run with
   ``jax.config.x64_enabled is False`` (see infra/001 + ARCHITECTURE.md §10).
   Tests carrying ``@pytest.mark.parity`` or ``@pytest.mark.fp64_oracle``
   *opt in* to fp64 — but only inside an isolated subprocess so they cannot
   flip the global ``jax_enable_x64`` for any later test in the run.

2. **Subprocess-isolation guard for parity.** If a test that would need
   ``jax_enable_x64=True`` is somehow collected in the in-process default
   run, this conftest skips it with a pointer to the runner that drives it
   in a child process. The runner contract lives in
   ``tests/_parity_subprocess_runner.py``.

These guards live here so they apply to every engine-side test file without
boilerplate at the top.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items):  # noqa: ARG001
    """Auto-skip parity / fp64_oracle tests in the in-process default loop.

    They must be invoked via the parity subprocess runner so that
    ``jax_enable_x64=True`` cannot leak into any other test's process.
    The runner sets ``TRACTUS_PARITY_SUBPROCESS=1`` before importing pytest.
    """

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
        keywords = item.keywords
        if not in_subprocess and ("parity" in keywords or "fp64_oracle" in keywords):
            item.add_marker(skip_parity)
