"""fp32-default assertion on the controller side (ARCHITECTURE.md §10, infra/001).

The default-slice ``pytest -q`` must run with ``jax_enable_x64 is False``.
Mirrors the engine-side test: this catches a controller-side import that
would flip the default to fp64.

Note: ``tractus_controller.copy_synth.tea_backend`` *does* call
``jax.config.update("jax_enable_x64", True)`` — but only inside its own
function bodies, not at module import time. If a future refactor moves
that update to module top-level, this test fires.
"""

from __future__ import annotations

import pytest

# We attempt to import jax. If the jax extra isn't installed in this venv,
# the fp32-default assertion is moot — skip rather than fail.
jax = pytest.importorskip("jax")


@pytest.mark.fp32_default
def test_jax_x64_disabled_by_default() -> None:
    assert jax.config.x64_enabled is False, (
        "jax_enable_x64 is True in the default test slice. The fp32 audit "
        "(correctness/001) requires fp32-default. Parity tests opt in via "
        "tests/_parity_subprocess_runner.py and must not contaminate the "
        "default config."
    )


@pytest.mark.fp32_default
def test_default_array_dtype_is_fp32() -> None:
    import jax.numpy as jnp

    assert jnp.array([1.0]).dtype == jnp.float32
