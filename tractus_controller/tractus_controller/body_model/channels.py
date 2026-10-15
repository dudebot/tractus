"""Gesture channel inventory + renderer-interface field-name schema.

Pinned here so the planner / controller / smoke pipeline have a
stable contract. The body model (`neural_field.py`, forthcoming)
samples a continuous field at the field names listed here; this
schema names the per-frame outputs the renderer expects, but does
NOT imply they are free decoder heads — most fields are sampled
from `A_θ(s, t)` or derived deterministically from constriction
geometry. See `ARCHITECTURE.md` §2.3.

Gesture-channel inventory (D_gesture = 20)
------------------------------------------

Per ARCH §2.2 ("~12 articulator + 6-8 source = ~20 channels"). The
12 articulator channels match the smoke pipeline's `CHANNEL_BY_NAME`
where they overlap; we fill in the architecturally-named-but-not-
yet-in-smoke channels (e.g. ``larynx_height``, ``pharyngeal_width``,
``tongue_tip_side``) with v0 placeholders. They have neutral defaults
and are preserved end-to-end, so a future controller that emits them
won't need to renegotiate the schema.

Channels are physical-units (cm, Hz, dPa, dimensionless 0..1):

   articulator (12):
     jaw_open, tongue_body_x, tongue_body_y, tongue_tip_x, tongue_tip_y,
     lip_protrusion, lip_aperture, velum_opening, tongue_tip_side,
     tongue_side, larynx_height, pharyngeal_width
   source (8):
     F0, F0_jitter, lung_pressure, glottal_adduction,
     vocal_fold_tension, breathiness, creak_gate, pressed_voice_gate

Renderer-interface schema (per-frame field names)
-------------------------------------------------

Sampled from / derived alongside the neural field:
- tube_area_cm2      : (T, 93)   sampled from A_θ(s, t)
- tube_length_cm     : (T, 93)   mostly fixed; one global scale + lip extension
- glottis_state      : (T, 11)   source controls — F0, PR, XB, …
- nasal_coupling     : (T,)      scalar low-rank in v0
- subglottal_press   : (T,)
- frication_gain     : (T, 93)   physically derived from constriction
                                 geometry — NOT a free decoder head
- frication_color    : (T, 93)   small learned residual, energy-capped

Anatomy-static (broadcast across T from anatomy_vec, NOT free per-frame):
- wall_mass_static       : (93,) = 93
- wall_resistance_static : (93,) = 93
- wall_stiffness_static  : (93,) = 93

Static total: 279.

Optional (default OFF; see ARCH cheat-channel discipline):
- wall_dynamic_residual  : (T, 93, 3) — disabled in v0; the
  ``RendererInterface`` returns ``None`` for it.

Total contracted fields per frame ≈ 664; matches the ARCH §2.3
"~500-600 fields, mostly derived" envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np


# ---------------------------------------------------------------------------
# Gesture-channel inventory (D_gesture = 20)
# ---------------------------------------------------------------------------

# Order is part of the contract. Do NOT reorder — downstream corpus shards
# are written in this column order and a permutation invalidates them.
GESTURE_CHANNELS: tuple[str, ...] = (
    # Articulator (12)
    "jaw_open",
    "tongue_body_x",
    "tongue_body_y",
    "tongue_tip_x",
    "tongue_tip_y",
    "lip_protrusion",
    "lip_aperture",
    "velum_opening",
    "tongue_tip_side",
    "tongue_side",
    "larynx_height",
    "pharyngeal_width",
    # Source (8)
    "F0",
    "F0_jitter",
    "lung_pressure",
    "glottal_adduction",
    "vocal_fold_tension",
    "breathiness",
    "creak_gate",
    "pressed_voice_gate",
)

N_GESTURE_CHANNELS: int = len(GESTURE_CHANNELS)
assert N_GESTURE_CHANNELS == 20

# Default neutral values for each channel — used by the interpolator when
# no GestureEvent is active. Units match each channel: cm, Hz, dPa, etc.
GESTURE_NEUTRAL: dict[str, float] = {
    "jaw_open": 0.0,
    "tongue_body_x": 0.0,
    "tongue_body_y": 0.0,
    "tongue_tip_x": 0.0,
    "tongue_tip_y": 0.0,
    "lip_protrusion": 0.0,
    "lip_aperture": 1.0,        # cm; relaxed open
    "velum_opening": 0.0,        # closed (oral)
    "tongue_tip_side": 0.0,
    "tongue_side": 0.0,
    "larynx_height": 0.0,
    "pharyngeal_width": 1.0,     # cm
    "F0": 110.0,                 # Hz; mid-male relaxed
    "F0_jitter": 0.0,
    "lung_pressure": 8000.0,     # dPa; quiet voiced rest
    "glottal_adduction": 0.5,
    "vocal_fold_tension": 0.5,
    "breathiness": 0.0,
    "creak_gate": 0.0,
    "pressed_voice_gate": 0.0,
}


def gesture_neutral_vec() -> np.ndarray:
    """Return the (D_gesture,) neutral baseline as a numpy array."""
    return np.asarray(
        [GESTURE_NEUTRAL[c] for c in GESTURE_CHANNELS],
        dtype=np.float32,
    )


_CHANNEL_TO_INDEX: dict[str, int] = {
    name: i for i, name in enumerate(GESTURE_CHANNELS)
}


def gesture_channel_index(name: str) -> int:
    """Look up the column index of a gesture channel by name."""
    if name not in _CHANNEL_TO_INDEX:
        raise KeyError(
            f"unknown gesture channel {name!r}; valid: {sorted(_CHANNEL_TO_INDEX)}"
        )
    return _CHANNEL_TO_INDEX[name]


# ---------------------------------------------------------------------------
# Renderer-interface schema
# ---------------------------------------------------------------------------

N_TUBE_SECTIONS = 93
N_GLOTTIS_STATE = 11

# Field name → per-frame (or static) shape suffix. None = static.
RENDERER_INTERFACE_FIELDS: dict[str, tuple[int, ...]] = {
    "tube_length_cm": (N_TUBE_SECTIONS,),
    "tube_area_cm2": (N_TUBE_SECTIONS,),
    "glottis_state": (N_GLOTTIS_STATE,),
    "nasal_coupling": (),
    "subglottal_press": (),
    "frication_gain": (N_TUBE_SECTIONS,),
    "frication_color": (N_TUBE_SECTIONS,),
    # Anatomy-static (broadcast per-T at renderer time, but body model
    # emits them as (93,) static fields):
    "wall_mass_static": (N_TUBE_SECTIONS,),
    "wall_resistance_static": (N_TUBE_SECTIONS,),
    "wall_stiffness_static": (N_TUBE_SECTIONS,),
}

# Free per-frame fields (controller-driven). The first 5 are owned by
# the body-model decoder; frication_gain/color are wired through but
# have an additional physical-derivation layer in v1+ (this v0 just
# emits them as small-magnitude calibration outputs).
PER_FRAME_FREE_FIELDS: tuple[str, ...] = (
    "tube_length_cm",
    "tube_area_cm2",
    "glottis_state",
    "nasal_coupling",
    "subglottal_press",
    "frication_gain",
    "frication_color",
)

# Anatomy-static fields (NOT free per-frame).
STATIC_FIELDS: tuple[str, ...] = (
    "wall_mass_static",
    "wall_resistance_static",
    "wall_stiffness_static",
)


def per_frame_free_dim() -> int:
    """Sum of per-frame free output widths (used to size the decoder head)."""
    return int(
        sum(
            int(np.prod(RENDERER_INTERFACE_FIELDS[f]))
            for f in PER_FRAME_FREE_FIELDS
        )
    )


def static_field_dim() -> int:
    """Sum of static field widths (decoded from anatomy alone)."""
    return int(
        sum(
            int(np.prod(RENDERER_INTERFACE_FIELDS[f])) for f in STATIC_FIELDS
        )
    )


@dataclass(frozen=True)
class RendererInterface:
    """Structured renderer-interface output. Each per-frame field has a
    leading T axis; static fields are 1-D over the 93 tube sections.

    ``wall_dynamic_residual`` is reserved for a future ablation; v0
    leaves it at ``None``. Adding it without explicit ablation evidence
    is a cheat-channel violation per ARCH §2.3.
    """

    tube_length_cm: jnp.ndarray            # (T, 93)
    tube_area_cm2: jnp.ndarray             # (T, 93)
    glottis_state: jnp.ndarray             # (T, 11)
    nasal_coupling: jnp.ndarray            # (T,)
    subglottal_press: jnp.ndarray          # (T,)
    frication_gain: jnp.ndarray            # (T, 93)
    frication_color: jnp.ndarray           # (T, 93)
    wall_mass_static: jnp.ndarray          # (93,)
    wall_resistance_static: jnp.ndarray    # (93,)
    wall_stiffness_static: jnp.ndarray     # (93,)
    wall_dynamic_residual: Optional[jnp.ndarray] = None  # (T, 93, 3) or None

    def num_frames(self) -> int:
        return int(self.tube_length_cm.shape[-2])

    def tree_flatten(self):
        names = [f.name for f in fields(self)]
        return tuple(getattr(self, n) for n in names), tuple(names)

    @classmethod
    def tree_unflatten(cls, aux, children):
        return cls(**dict(zip(aux, children)))

    def to_flat_per_frame(self) -> jnp.ndarray:
        """Flatten per-frame free fields into a single (T, F) array.

        Field ordering matches :data:`PER_FRAME_FREE_FIELDS`. Useful for
        loss computation that wants a single (T, F) tensor.
        """
        cols = []
        for name in PER_FRAME_FREE_FIELDS:
            x = getattr(self, name)
            if x.ndim == 1:
                # (T,) -> (T, 1)
                cols.append(x[:, None])
            else:
                cols.append(x)
        return jnp.concatenate(cols, axis=-1)


jax.tree_util.register_pytree_node_class(RendererInterface)
