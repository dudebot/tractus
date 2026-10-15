"""Body model (Module 3) — gesture composition + speaker-conditioned
neural-field area function `A_θ(s, t)`.

See `ARCHITECTURE.md` §2.3. The body model maps a 50 Hz gesture
trajectory + a small speaker vector to a continuous log-area field
over normalized tract location ``s`` and time ``t``. The field is
sampled at TEA's 93 sections per frame at the renderer boundary.

This package owns:

- :mod:`channels` — the 20-channel gesture inventory (12 articulator
  + 8 source) and the renderer-interface field-name schema. Pinned
  here so the planner / controller / smoke pipeline rely on a stable
  contract.
- :mod:`interp` — the 50 Hz frame interpolator that composes
  asynchronous overlapping `GestureEvent`s into `(T, 20)` at frame
  centers. Interpolator contract is load-bearing per ARCH §2.2.
- :mod:`reachability` — soft + hard clamps on areas, lengths,
  glottis state, subglottal pressure, frication. Used as
  neural-field barrier / smoothness priors during training and as
  hard clamps at eval.
- :mod:`anchors` — Story 1996/2008 area-function loader. Held-out
  vowel sanity gate, NOT primary supervision target.
- :mod:`neural_field` — the JAX neural-field body model. **Forthcoming
  in active task 003.** Replaces the deleted conv-MLP body model
  (now at `legacy/body_model_conv_mlp.py`).

Important non-circularity rule (ARCH §2.3 cheat-channel discipline).
This package may NEVER import
`tractus_engine_advanced.pipeline.synthesize_dynamic_tube`. Training
loops consume short-window TEA gradients via the
`tractus_controller.copy_synth.tea_backend` interface, which gates
the surrogate-vs-TEA call site at training time.
"""

from __future__ import annotations

from .channels import (
    GESTURE_CHANNELS,
    N_GESTURE_CHANNELS,
    RENDERER_INTERFACE_FIELDS,
    RendererInterface,
    gesture_channel_index,
)
from .interp import compose_gesture_trajectory, FRAME_RATE_HZ
from .reachability import (
    ReachabilityBounds,
    DEFAULT_REACHABILITY,
    apply_soft_clamp,
    apply_hard_clamp,
)

__all__ = [
    "GESTURE_CHANNELS",
    "N_GESTURE_CHANNELS",
    "RENDERER_INTERFACE_FIELDS",
    "RendererInterface",
    "gesture_channel_index",
    "compose_gesture_trajectory",
    "FRAME_RATE_HZ",
    "ReachabilityBounds",
    "DEFAULT_REACHABILITY",
    "apply_soft_clamp",
    "apply_hard_clamp",
]
