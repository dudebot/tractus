"""Soft-surrogate configuration for differentiable forward gates.

correctness/003 (M6): the existing TEA forward exposes several hard kinks
(``jnp.maximum``, integer-segment selection, abrupt boolean masks). Hard
mode is the bit-equal fp64 parity target. Soft mode replaces each kink
with a τ-controlled smooth surrogate so ``jax.grad`` of the chained loss
sees a Lipschitz-bounded gradient field.

This is a **test-fixture-level config**, not a public CLI knob. Callers
opt in by passing ``smoothing=Smoothing.default_soft()`` to forward
helpers; the default everywhere else stays ``Smoothing.hard()``. Public
exposure of this knob (CLI, R1 NN inference, planner) is a separate
task once M6 stability lands.

Per-stage τ knobs
-----------------

* ``tau_relu``         — smoothed ReLU on (trans_pressure) and on the
  glottal lateral coordinate ``end_x·t + disp·sin_pi_t``. Surrogate:
  ``softplus(x/τ)·τ`` ≈ max(x, 0). At τ → 0 reduces to ``relu``.
* ``tau_segment``      — smooth piecewise-linear segment selection in
  ``_piecewise_linear_warp``. Surrogate: softmax over per-segment
  matches with temperature τ. At τ → 0 reduces to argmax-then-pick.
* ``tau_blend``        — smooth blend between the in-second-pulse and
  outside-second-pulse displacements. Surrogate: sigmoid((phase - p0)/τ)
  on a logistic centered at the boundary. At τ → 0 reduces to
  ``jnp.where(in_second_pulse, reduced, normal)``.
* ``tau_area_clamp``   — soft area-floor clamp in ``TubeGeometry``.
  Surrogate: ``softplus((A - MIN)/τ)·τ + MIN`` ≈ max(A, MIN). At τ → 0
  reduces to ``jnp.maximum(A, MIN_AREA_CM2)``.
* ``tau_bernoulli``    — smooth Bernoulli-loss direction gate in
  ``_bernoulli_chain_loss`` / ``_glottal_entrance_loss`` /
  ``_glottal_transition_loss``. Surrogate: sigmoid on (u/τ) and on
  (A_prev - A_self)/τ. At τ → 0 reduces to the hard step.

Default soft τ values are conservative (1e-2..1e-1 in normalized units)
so the soft forward stays within numerical noise of the hard forward at
typical operating points but produces finite gradients across the kinks.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Smoothing:
    """Per-stage τ knobs for the smooth-surrogate forward path.

    Use ``Smoothing.hard()`` for parity tests (exactly equivalent to the
    pre-M6 forward — every surrogate falls back to its hard form), and
    ``Smoothing.default_soft()`` for training (default τ values that
    keep the gradient field Lipschitz across kinks while staying within
    numerical noise of the hard forward at typical operating points).

    Attributes
    ----------
    enabled
        Master switch. ``False`` → hard mode regardless of τ values.
    tau_relu
        Softplus temperature on smoothed ReLU sites (trans_pressure,
        glottal lateral coordinate).
    tau_segment
        Softmax temperature for piecewise-linear segment selection.
    tau_blend
        Sigmoid temperature for in-second-pulse / outside blend.
    tau_area_clamp
        Softplus temperature for the MIN_AREA_CM2 floor in
        ``TubeGeometry.from_arrays``. In cm² units.
    tau_bernoulli
        Sigmoid temperature for direction / area-comparison gates in
        the Bernoulli-loss helpers.
    """

    enabled: bool = False
    tau_relu: float = 0.0
    tau_segment: float = 0.0
    tau_blend: float = 0.0
    tau_area_clamp: float = 0.0
    tau_bernoulli: float = 0.0

    @classmethod
    def hard(cls) -> "Smoothing":
        """All surrogates disabled; bit-equal to the pre-M6 forward."""
        return cls(enabled=False)

    @classmethod
    def default_soft(cls) -> "Smoothing":
        """Default training-time soft surrogates.

        The τ values were picked so the soft-mode forward output stays
        within ~1e-3 RMS-relative of the hard-mode output on the
        natasha-shape fixture and the correctness corpus while
        smoothing every kink in the gradient path. They are not a
        decision point — change with care and re-run the M6 sweep.
        """
        return cls(
            enabled=True,
            # Pressure / lateral-displacement softplus. trans_pressure is in
            # dPa (~1e3..1e4 typical); 10 dPa is below dynamic range and
            # well above the 1e-12 NaN-grad guard from correctness/006.
            # Lateral coord is in cm (length·t ≈ a few mm); 1e-3 cm = 10 µm.
            tau_relu=1e-3,
            # Piecewise-linear warp segments span π/2 in input (~1.57);
            # τ=0.05 keeps the softmax tightly focused on the active segment.
            tau_segment=0.05,
            # Phase-blend boundary at 1.5π / 3.5π; phase is in radians
            # over [0, 4π]. τ=0.05 rad keeps the blend region < 1° wide.
            tau_blend=0.05,
            # Area-clamp at MIN_AREA_CM2 = 1e-4 cm²; τ=1e-4 makes the soft
            # floor differentiable across the boundary without changing
            # the floor location measurably.
            tau_area_clamp=1e-4,
            # Bernoulli direction gates on flow magnitude (cm³/s; typical
            # 1e2..1e4) and area diffs (cm²). τ=1.0 gives a smooth
            # ~1 cm³/s boundary on direction, well below typical flow.
            tau_bernoulli=1.0,
        )


# ---------------------------------------------------------------------------
# Surrogate primitives. Pure functions over jax arrays + a Smoothing config.
# Kept module-local so call sites can ``from smoothing import soft_relu``
# without a deeper import.
# ---------------------------------------------------------------------------


def soft_relu(x, tau: float):
    """Smoothed ReLU: ``softplus(x/τ)·τ`` ≈ max(x, 0).

    At ``τ → 0+`` reduces to ``relu(x)`` (and is bit-stable to within
    softplus tail precision for ``x/τ > 30`` where softplus ≈ identity).
    Zero ``τ`` falls back to hard ReLU.
    """
    import jax.numpy as jnp
    if tau <= 0.0:
        return jnp.maximum(x, 0.0)
    # softplus(z) is well-defined for all z; multiply back by τ.
    import jax.nn as jnn
    return jnn.softplus(x / tau) * tau


def soft_max_floor(x, floor: float, tau: float):
    """Smooth ``max(x, floor)`` via a softplus shifted by ``floor``."""
    import jax.numpy as jnp
    if tau <= 0.0:
        return jnp.maximum(x, floor)
    import jax.nn as jnn
    return jnn.softplus((x - floor) / tau) * tau + floor


def soft_step(x, tau: float):
    """Smooth Heaviside: ``sigmoid(x/τ)`` ∈ (0, 1).

    At ``τ → 0+`` reduces to ``x > 0``. Used for direction gates.
    """
    import jax.numpy as jnp
    if tau <= 0.0:
        return (x > 0.0).astype(jnp.float32)
    import jax.nn as jnn
    return jnn.sigmoid(x / tau)


def soft_segment_pick(values, weights, tau: float):
    """Soft segment selection via softmax over per-segment match weights.

    ``weights`` is a non-negative vector that is 1 on the active segment
    and 0 elsewhere in hard mode (``in_segment.astype(float)``).
    Returns the weighted combination of ``values`` (one per segment).

    At ``τ → 0+`` reduces to ``values[argmax(weights)]``; for the
    PWL-warp use case where weights are exactly one-hot, this is bit-equal
    to the hard pick. For non-one-hot weights (boundary ties), the
    softmax temperature τ controls how sharply the active segment is
    selected.

    Implemented as ``sum(softmax(weights/τ) * values)``. ``tau<=0``
    falls back to hard pick (taking the first match).
    """
    import jax.numpy as jnp
    if tau <= 0.0:
        first_match = jnp.argmax(weights.astype(jnp.int32))
        return values[first_match]
    import jax.nn as jnn
    soft_w = jnn.softmax(weights / tau)
    return jnp.sum(soft_w * values)
