"""fp32-default assertion (ARCHITECTURE.md §10, infra/001).

The default-slice ``pytest -q`` must run with ``jax_enable_x64 is False``.
The fp32 audit (``correctness/001``) is what enforces fp32 throughout the
TEA forward path; this test is the pin against any future drift back to
fp64-by-default.

Parity / fp64_oracle tests opt in via the subprocess runner — they cannot
flip the default for the rest of the suite.
"""

from __future__ import annotations

import pytest


@pytest.mark.fp32_default
def test_jax_x64_disabled_by_default() -> None:
    """Default test process must NOT have x64 enabled.

    If this fails, something imported during test setup flipped
    ``jax_enable_x64`` and the default slice is no longer fp32-default.
    Common culprit: a module-level ``jax.config.update("jax_enable_x64", True)``
    in production code (which should be guarded behind a parity-only entrypoint).
    """

    import jax

    assert jax.config.x64_enabled is False, (
        "jax_enable_x64 is True in the default test slice. The fp32 audit "
        "(correctness/001) requires fp32-default. Parity tests opt in via "
        "tests/_parity_subprocess_runner.py and must not contaminate the "
        "default config."
    )


@pytest.mark.fp32_default
def test_default_array_dtype_is_fp32() -> None:
    """A fresh ``jnp.array([1.0])`` lands as fp32 under the default config."""

    import jax.numpy as jnp

    assert jnp.array([1.0]).dtype == jnp.float32
