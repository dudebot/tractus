"""correctness/004b acceptance: audible-frication MNA stability + bit-equality.

The 004a frication module is mathematically correct but its MNA
``D += d·source_amp`` injection slot was unstable above ~0.5 cm³/s
(per-sample integrator divergence). 004b moves the injection out of
``prepare_time_step`` (where it entered the matrix RHS) and into the
post-solve ``update_variables`` net-flow term, decoupling matrix-solve
stability from noise amplitude.

This test file pins the contract:

* **bit-equality** at ``monopole_source_amp = 0`` (zeros): identical to
  the legacy ``None``-source path (which itself is bit-equal to pre-004
  code per 004a's regression).
* **MNA stability** at ``monopole_source_amp ∈ {1, 5, 15} cm³/s``: no
  NaN/Inf in the per-sample state after a 100-sample driven run.
* **gradient finiteness** at full physiological drive (15 cm³/s): the
  ``∂L/∂source_amp`` jacobian is finite — required so the body model's
  frication output is end-to-end trainable.
"""

from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tractus_engine_advanced.tds.constants import TIMESTEP_S
from tractus_engine_advanced.tds.mna import TubeGeometry
from tractus_engine_advanced.tds.state import TdsState
from tractus_engine_advanced.tds.step import step_one
from tractus_engine_advanced.tds.tube_topology import (
    NUM_SECTIONS,
    TOPOLOGY,
)


def _default_geom() -> TubeGeometry:
    """A non-degenerate 93-section geometry: uniform 1 cm sections, 2 cm² area."""
    L = np.full(NUM_SECTIONS, 1.0, dtype=np.float32)
    A = np.full(NUM_SECTIONS, 2.0, dtype=np.float32)
    Mw = np.full(NUM_SECTIONS, 1.5, dtype=np.float32)
    Bw = np.full(NUM_SECTIONS, 8000.0, dtype=np.float32)
    Kw = np.full(NUM_SECTIONS, 4.0e6, dtype=np.float32)
    return TubeGeometry.from_arrays(L, A, Mw, Bw, Kw)


def _drive(monopole_source_amp, n_samples: int = 100, psub: float = 8000.0):
    """Run ``n_samples`` of TDS with the given per-section monopole source.

    Source is a constant DC drive (worst-case for stability) to surface
    integrator divergence quickly.
    """
    geom = _default_geom()
    state = TdsState.zeros()

    def body(carry, _):
        st = carry
        out = step_one(
            st, geom, TOPOLOGY,
            subglottal_pressure_dPa=psub,
            monopole_source_amp=monopole_source_amp,
        )
        return out.state, out.section_pressure_dPa

    final_state, traj = jax.lax.scan(body, state, jnp.arange(n_samples))
    return final_state, traj


def test_bit_equality_zero_source_vs_legacy_none() -> None:
    """``monopole_source_amp = zeros(93)`` equals ``monopole_source_amp = None``.

    The 004a contract: when there's no noise drive the per-sample step
    is bit-identical to the pre-frication forward. 004b's relocation
    of the injection point preserves this — both paths must share
    every bit of the resulting pressure trajectory.
    """
    zeros = jnp.zeros((NUM_SECTIONS,), dtype=jnp.float32)
    final_a, traj_a = _drive(zeros, n_samples=50)
    final_b, traj_b = _drive(None, n_samples=50)
    diff = float(jnp.max(jnp.abs(traj_a - traj_b)))
    assert diff == 0.0, f"bit-equality broken at zero source: max abs diff {diff:g}"


@pytest.mark.parametrize("amp", [1.0, 5.0, 15.0])
def test_mna_solve_stable_at_physiological_drive(amp: float) -> None:
    """Per-sample MNA solve stays finite at physiological turbulence
    amplitudes (Stevens 1971: typical fricative monopole 5-15 cm³/s)."""
    src = jnp.full((NUM_SECTIONS,), amp, dtype=jnp.float32)
    final, traj = _drive(src, n_samples=100)
    assert jnp.all(jnp.isfinite(traj)), (
        f"non-finite pressure at source_amp={amp} cm³/s: "
        f"any-nan={bool(jnp.any(jnp.isnan(traj)))} "
        f"any-inf={bool(jnp.any(jnp.isinf(traj)))}"
    )
    assert jnp.all(jnp.isfinite(final.section)), (
        f"non-finite final state at {amp} cm³/s"
    )


def test_grad_finite_at_full_drive() -> None:
    """Backward through ``step_one`` finite at 15 cm³/s drive.

    The body-model's frication output flows here as the gradient path
    on ``monopole_source_amp``. Required for end-to-end trainability.
    """
    src = jnp.full((NUM_SECTIONS,), 15.0, dtype=jnp.float32)

    def loss(s):
        _, traj = _drive(s, n_samples=20)
        return jnp.sum(traj ** 2)

    g = jax.grad(loss)(src)
    g_np = np.asarray(g)
    assert np.all(np.isfinite(g_np)), (
        f"non-finite grad at full drive: any-nan={bool(np.any(np.isnan(g_np)))} "
        f"any-inf={bool(np.any(np.isinf(g_np)))}"
    )
