"""TEA precision policy.

ARCHITECTURE.md §7: fp32 is the production default for TEA forward.
fp64 is parity/debug only, opt-in.

Mechanism: read ``TEA_PRECISION`` env var at import time. Default
``fp32``. ``fp64`` opts back in (and enables ``jax_enable_x64``).

This module MUST be imported before any other tractus_engine_advanced
module that creates jnp arrays — it owns ``jax.config.update`` for
the x64 flag, which is illegal to set after the first JAX
computation. To preserve that ordering, the package's
``__init__.py`` imports this module first.

Public API:

* ``DTYPE`` — jnp dtype for solver state, audio, geometry.
* ``NP_DTYPE`` — numpy dtype mirror, for buffers fed to JAX.
* ``IS_FP64`` — bool; True iff ``TEA_PRECISION=fp64``.
* ``zero(shape=())`` — convenience constructor.

Tests opting back into fp64 should set the env var **before**
importing the package, ideally in a subprocess (the flag is
process-global). See ``conftest.py`` for the marker convention.
"""
from __future__ import annotations

import os

import jax
import jax.numpy as jnp
import numpy as np


def _resolve() -> tuple[bool, "jnp.dtype", "np.dtype"]:
    raw = os.environ.get("TEA_PRECISION", "fp32").strip().lower()
    if raw in ("fp64", "float64", "double", "f64"):
        # Must precede any JAX computation. Calling update() repeatedly
        # with the same value is a no-op, so re-import is safe.
        jax.config.update("jax_enable_x64", True)
        return True, jnp.float64, np.float64
    if raw not in ("fp32", "float32", "single", "f32", ""):
        raise ValueError(
            f"TEA_PRECISION={raw!r} unrecognised; expected fp32 or fp64."
        )
    return False, jnp.float32, np.float32


IS_FP64, DTYPE, NP_DTYPE = _resolve()


def zero(shape: tuple[int, ...] = ()) -> jnp.ndarray:
    """Zero-filled jnp array at the configured precision."""
    return jnp.zeros(shape, dtype=DTYPE)
