"""DELIBERATELY VIOLATES doc-coherence regression lint (infra/002 fixture).

Hits multiple stale-framing patterns at once.
"""

import jax  # noqa: F401
import jax.numpy as jnp

# four-module framing — superseded by 5-module split in ARCH §2.
ARCH_SUMMARY = "the four-module decomposition"

# fp64 default outside the precision-policy module.
jax.config.update("jax_enable_x64", True)
ZERO = jnp.zeros((4,), dtype=jnp.float64)

def ges_to_tract(ges):  # noqa: D401
    raise NotImplementedError
