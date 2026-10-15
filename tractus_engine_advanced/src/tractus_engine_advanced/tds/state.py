"""TdsState — the per-sample carry through `lax.scan` over TDS steps.

Per-section / per-branch arrays:
  - section (93, 5): pressure, pressureRate, wallCurrent, wallCurrentRate, wallCurrentRate2
  - branch  (97, 3): magnitude, magnitudeRate, noiseMagnitude
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from tractus_engine_advanced.precision import DTYPE
from tractus_engine_advanced.tds.tube_topology import NUM_BRANCH_CURRENTS, NUM_SECTIONS

# Field-index conventions inside each section / branch row.
PRESSURE = 0
PRESSURE_RATE = 1
WALL_CURRENT = 2
WALL_CURRENT_RATE = 3
WALL_CURRENT_RATE2 = 4
NUM_SECTION_FIELDS = 5

MAGNITUDE = 0
MAGNITUDE_RATE = 1
NOISE_MAGNITUDE = 2
NUM_BRANCH_FIELDS = 3

STATE_DOUBLES = NUM_SECTIONS * NUM_SECTION_FIELDS + NUM_BRANCH_CURRENTS * NUM_BRANCH_FIELDS


class TdsState(eqx.Module):
    section: Float[Array, "93 5"]
    branch: Float[Array, "97 3"]

    @classmethod
    def zeros(cls) -> "TdsState":
        return cls(
            section=jnp.zeros((NUM_SECTIONS, NUM_SECTION_FIELDS), dtype=DTYPE),
            branch=jnp.zeros((NUM_BRANCH_CURRENTS, NUM_BRANCH_FIELDS), dtype=DTYPE),
        )

    @classmethod
    def from_flat(cls, flat: np.ndarray | Array) -> "TdsState":
        """Inverse of `to_flat`. Layout matches a C-side flat buffer."""
        flat = jnp.asarray(flat, dtype=DTYPE)
        if flat.shape != (STATE_DOUBLES,):
            raise ValueError(f"expected shape ({STATE_DOUBLES},), got {flat.shape}")
        sec_end = NUM_SECTIONS * NUM_SECTION_FIELDS
        return cls(
            section=flat[:sec_end].reshape(NUM_SECTIONS, NUM_SECTION_FIELDS),
            branch=flat[sec_end:].reshape(NUM_BRANCH_CURRENTS, NUM_BRANCH_FIELDS),
        )

    def to_flat(self) -> Array:
        """Serialize to a (756,) buffer matching the canonical flat layout."""
        return jnp.concatenate([self.section.reshape(-1), self.branch.reshape(-1)])
