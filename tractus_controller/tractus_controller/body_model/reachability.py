"""Hard / soft reachability clamps + neural-field barrier priors.

Per `ARCHITECTURE.md` §2.3 + §5.3 (cheat-channel discipline):

- Areas ≥ MIN_AREA, ≤ MAX_AREA
- Lengths ≥ MIN_LENGTH
- Glottis state in physiological range (per-channel min/max)
- Subglottal pressure non-negative
- Frication gain non-negative; calibration color residual energy-capped

The neural-field body model (`neural_field.py`, forthcoming) uses
these bounds in two ways:

1. **Soft barrier** during training — differentiable saturation that
   preserves gradient near the bound but flattens past it. Used as a
   penalty term in the training loss (clamp-occupancy gate target:
   < 5 % of `(T × 93)` cells at the area boundary).
2. **Hard clamp** at eval — plain ``jnp.clip``. Pathological field
   shapes are blocked from reaching TEA.

The clamps are inert until the training loop wires them up; this
module only declares the bounds and the barrier functions.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


# Physical bounds in cgs / physical units.
MIN_AREA_CM2: float = 1e-4
MAX_AREA_CM2: float = 12.0       # generous upper bound for tract sections
MIN_LENGTH_CM: float = 0.05
MAX_LENGTH_CM: float = 2.5
MIN_F0_HZ: float = 40.0
MAX_F0_HZ: float = 600.0
MIN_LUNG_PA: float = 0.0
MAX_LUNG_PA: float = 25000.0
MIN_NASAL: float = 0.0
MAX_NASAL: float = 1.0

# Cap for frication_color (calibration residual). Per ARCH this cannot
# create frication where physical conditions don't support it; a small
# scale factor multiplied against the physically-derived gain is the v1
# story. For v0 we expose it as a free output but cap its magnitude.
FRICATION_COLOR_CAP: float = 0.25


@dataclass(frozen=True)
class ReachabilityBounds:
    """Per-field clamp bounds. All values in physical units."""

    min_area_cm2: float = MIN_AREA_CM2
    max_area_cm2: float = MAX_AREA_CM2
    min_length_cm: float = MIN_LENGTH_CM
    max_length_cm: float = MAX_LENGTH_CM
    min_f0_hz: float = MIN_F0_HZ
    max_f0_hz: float = MAX_F0_HZ
    min_lung_pa: float = MIN_LUNG_PA
    max_lung_pa: float = MAX_LUNG_PA
    min_nasal: float = MIN_NASAL
    max_nasal: float = MAX_NASAL
    frication_color_cap: float = FRICATION_COLOR_CAP


DEFAULT_REACHABILITY = ReachabilityBounds()


def _soft_min(x: jnp.ndarray, lo: float, scale: float = 0.05) -> jnp.ndarray:
    """Soft floor: smooth approximation of ``max(x, lo)`` via softplus.

    Uses :func:`jax.nn.softplus` for numerical stability. The naive
    ``log1p(exp(z))`` overflows for ``z > ~88`` in fp32; that fires
    immediately during training, where raw decoder outputs can be in
    the tens before any learning shapes them. ``jax.nn.softplus`` is
    the stable form (``max(x, 0) + log(1 + exp(-|x|))``) and matches
    the naive form to fp32 precision in the well-defined regime.
    """
    return lo + jax.nn.softplus((x - lo) / scale) * scale


def _soft_clamp(x: jnp.ndarray, lo: float, hi: float, scale: float = 0.05) -> jnp.ndarray:
    """Smooth saturation toward [lo, hi]; gradient survives near the
    interior but flattens past the bounds.
    """
    width = hi - lo
    sigma = jnp.tanh((x - 0.5 * (lo + hi)) / max(scale * width, 1e-6))
    return 0.5 * (lo + hi) + 0.5 * width * sigma


def apply_soft_clamp(
    *,
    tube_length_cm: jnp.ndarray,
    tube_area_cm2: jnp.ndarray,
    glottis_state: jnp.ndarray,
    nasal_coupling: jnp.ndarray,
    subglottal_press: jnp.ndarray,
    frication_gain: jnp.ndarray,
    frication_color: jnp.ndarray,
    bounds: ReachabilityBounds = DEFAULT_REACHABILITY,
) -> dict:
    """Differentiable soft clamps. Returns a dict keyed by field name."""
    L = _soft_clamp(tube_length_cm, bounds.min_length_cm, bounds.max_length_cm)
    A = _soft_clamp(tube_area_cm2, bounds.min_area_cm2, bounds.max_area_cm2)
    # Glottis state: F0 is column 0, others are dimensionless / physical
    # — we clamp per-column with looser bounds.
    g = glottis_state
    f0 = _soft_clamp(g[..., 0:1], bounds.min_f0_hz, bounds.max_f0_hz)
    pr = _soft_clamp(g[..., 1:2], bounds.min_lung_pa, bounds.max_lung_pa)
    rest = _soft_clamp(g[..., 2:], -10.0, 10.0)
    g_clamped = jnp.concatenate([f0, pr, rest], axis=-1)
    nas = _soft_clamp(nasal_coupling, bounds.min_nasal, bounds.max_nasal)
    sub = _soft_clamp(subglottal_press, bounds.min_lung_pa, bounds.max_lung_pa)
    fg = _soft_min(frication_gain, 0.0)  # non-negative, no upper
    fc = _soft_clamp(frication_color, -bounds.frication_color_cap, bounds.frication_color_cap)
    return {
        "tube_length_cm": L,
        "tube_area_cm2": A,
        "glottis_state": g_clamped,
        "nasal_coupling": nas,
        "subglottal_press": sub,
        "frication_gain": fg,
        "frication_color": fc,
    }


def apply_hard_clamp(
    *,
    tube_length_cm: jnp.ndarray,
    tube_area_cm2: jnp.ndarray,
    glottis_state: jnp.ndarray,
    nasal_coupling: jnp.ndarray,
    subglottal_press: jnp.ndarray,
    frication_gain: jnp.ndarray,
    frication_color: jnp.ndarray,
    bounds: ReachabilityBounds = DEFAULT_REACHABILITY,
) -> dict:
    """Plain ``jnp.clip`` clamps — for evaluation, not training."""
    L = jnp.clip(tube_length_cm, bounds.min_length_cm, bounds.max_length_cm)
    A = jnp.clip(tube_area_cm2, bounds.min_area_cm2, bounds.max_area_cm2)
    g = glottis_state
    f0 = jnp.clip(g[..., 0:1], bounds.min_f0_hz, bounds.max_f0_hz)
    pr = jnp.clip(g[..., 1:2], bounds.min_lung_pa, bounds.max_lung_pa)
    rest = jnp.clip(g[..., 2:], -10.0, 10.0)
    g_clamped = jnp.concatenate([f0, pr, rest], axis=-1)
    nas = jnp.clip(nasal_coupling, bounds.min_nasal, bounds.max_nasal)
    sub = jnp.clip(subglottal_press, bounds.min_lung_pa, bounds.max_lung_pa)
    fg = jnp.maximum(frication_gain, 0.0)
    fc = jnp.clip(
        frication_color,
        -bounds.frication_color_cap,
        bounds.frication_color_cap,
    )
    return {
        "tube_length_cm": L,
        "tube_area_cm2": A,
        "glottis_state": g_clamped,
        "nasal_coupling": nas,
        "subglottal_press": sub,
        "frication_gain": fg,
        "frication_color": fc,
    }
