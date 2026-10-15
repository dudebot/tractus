"""tractus_engine_advanced (TEA) — differentiable transmission-line vocal-tract simulator.

The first import below pins the precision policy (fp32 default,
``TEA_PRECISION=fp64`` to opt back into fp64). Must run before any
other module that touches ``jnp``.
"""
from __future__ import annotations

from tractus_engine_advanced import precision as _precision  # noqa: F401
from tractus_engine_advanced import smoothing as _smoothing  # noqa: F401
from tractus_engine_advanced.smoothing import Smoothing  # noqa: F401
from tractus_engine_advanced import frication as _frication  # noqa: F401
from tractus_engine_advanced.frication import Frication  # noqa: F401

__all__ = ["precision", "smoothing", "Smoothing", "frication", "Frication"]
