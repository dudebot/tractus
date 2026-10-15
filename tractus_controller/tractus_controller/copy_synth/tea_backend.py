"""Pure-JAX synthesis backend for the copy-synth fitter.

Drop-in replacement for the legacy ctypes path. Composes the 93-section
tube as:

    trachea (0..22)             — speaker default (frozen npz)
    glottis (23..24)            — TEA's GeometricGlottis at runtime
    pharynx-mouth (25..64)      — R1 NN tract→tube adapter (jittable)
    nose (65..83)               — speaker default
    fossa (84..88)              — speaker default
    sinus (89..92)              — speaker default

then drives ``tractus_engine_advanced.synthesize_dynamic_tube`` with
``glottis_traj=trajectory[:, :11]`` (G3a path). All inputs and the full
pipeline are JAX arrays — no numpy bridge per gradient step.

The frozen default-tube fixture lives at
``outputs/fixtures/default_tube.npz`` and is loaded once via
``adapters.r1_inference.load_default_tube``.

Subglottal-pressure schedule comes from the per-frame glottis trajectory
(column 1, PR) through TEA's G3a glottis path; no scalar override is
needed.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# tractus_engine_advanced source-path helper (unchanged from the legacy version).
# ---------------------------------------------------------------------------


def _ensure_tea_on_path() -> None:
    """Add the sibling tractus_engine_advanced package to sys.path.

    The controller's venv doesn't depend on TEA at import time (it's an
    optional backend). Locate TEA's ``src/`` relative to the workspace
    root and prepend if needed. Idempotent.
    """
    if os.environ.get("TRACTUS_ENGINE_ADVANCED_SRC"):
        candidate = Path(os.environ["TRACTUS_ENGINE_ADVANCED_SRC"]).resolve()
    else:
        from ..paths import workspace_root
        candidate = (workspace_root() / "tractus_engine_advanced" / "src").resolve()
    if not candidate.exists():
        raise RuntimeError(
            f"tractus_engine_advanced source dir not found at {candidate}; "
            "set TRACTUS_ENGINE_ADVANCED_SRC env var or install TEA."
        )
    s = str(candidate)
    if s not in sys.path:
        sys.path.insert(0, s)


# ---------------------------------------------------------------------------
# Cached adapter resources (NN params + default tube).
# ---------------------------------------------------------------------------


@lru_cache(maxsize=2)
def _cached_adapter(checkpoint_path: str | None = None) -> tuple:
    """Load (params, cfg, norm, section_weight, default_tube) once per process."""
    from ..adapters.r1_inference import load_default_tube, load_r1_checkpoint

    params, cfg, norm, section_weight = load_r1_checkpoint(checkpoint_path)
    default = load_default_tube()
    return params, cfg, norm, section_weight, default


def clear_cache() -> None:
    """Invalidate the adapter cache. Call when reloading a different checkpoint."""
    _cached_adapter.cache_clear()


# ---------------------------------------------------------------------------
# Pure-JAX forward.
# ---------------------------------------------------------------------------


def synthesize_via_tea_jax(
    trajectory,                # JAX array (T, 30) — cols 0:11 glottis, 11:30 tract
    *,
    samples_per_frame: int,
    checkpoint_path: str | None = None,
    frication_inputs: dict | None = None,
):
    """End-to-end JAX synthesis: trajectory -> R1 NN -> compose -> TEA -> audio.

    Returns a JAX array of length ``(T - 1) * samples_per_frame`` (matches
    the legacy ctypes contract so the caller's downstream feature
    pipeline works unchanged). Vmappable / jittable / differentiable —
    no numpy or ctypes calls inside the graph.

    correctness/004: ``frication_inputs`` (optional) bundles the
    per-frame frication-module inputs:

    * ``frication`` — :class:`tractus_engine_advanced.Frication` config
      (master switch + place-class gains + color cap + thresholds).
    * ``place_index[T, 93]`` int32, defaulting to PLACE_NONE.
    * ``obstacle_proxy[T, 93]`` float, defaulting to zeros.
    * ``voicing[T]`` float ∈ [0, 1], defaulting to zeros.
    * ``frication_color[T, 93]`` float (optional learned residual).
    * ``noise_key`` jax.random.key (defaults to key(0)).

    None or empty dict ⇒ frication disabled (legacy pre-fric forward).
    """
    _ensure_tea_on_path()
    # NOTE: TEA owns the precision policy via tractus_engine_advanced.precision,
    # which is loaded by the package __init__ from the env var TEA_PRECISION
    # (default fp32). Importing TEA here pins it; do NOT call
    # jax.config.update("jax_enable_x64") from outside that policy module.
    import jax  # noqa: E402
    import jax.numpy as jnp  # noqa: E402
    from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: E402
    from tractus_engine_advanced.precision import DTYPE as TEA_DTYPE  # noqa: E402

    from ..adapters.r1_inference import compose_93_section_jax, tract_to_tube_jax

    params, cfg, norm, _section_weight, default = _cached_adapter(checkpoint_path)

    traj = jnp.asarray(trajectory)
    glottis_traj = traj[:, :11]
    tract_traj = traj[:, 11:]

    # NN: pharynx-mouth (T, 40) length + area, fp64.
    pm_length, pm_area = jax.vmap(
        lambda x: tract_to_tube_jax(params, cfg, norm, x)
    )(tract_traj)

    # Glottis sections: TEA's G3a path computes glottis section areas
    # internally from glottis_traj. Pass placeholder zeros at sections
    # 23-24; compose_93_section_jax fills them, then synthesize_dynamic_tube
    # overrides via glottis_traj at runtime.
    T = tract_traj.shape[0]
    gl_length = jnp.zeros((T, 2), dtype=TEA_DTYPE)
    gl_area = jnp.zeros((T, 2), dtype=TEA_DTYPE)

    L, A = jax.vmap(
        lambda pmL, pmA, glL, glA: compose_93_section_jax(
            default["length_cm"], default["area_cm2"], pmL, pmA, glL, glA
        )
    )(pm_length, pm_area, gl_length, gl_area)

    Mw = jnp.broadcast_to(default["wall_mass_cgs"][None, :], (T, 93))
    Bw = jnp.broadcast_to(default["wall_resistance_cgs"][None, :], (T, 93))
    Kw = jnp.broadcast_to(default["wall_stiffness_cgs"][None, :], (T, 93))

    fr_kwargs: dict = {}
    if frication_inputs:
        fr_kwargs = {
            "frication": frication_inputs.get("frication"),
            "place_index": frication_inputs.get("place_index"),
            "obstacle_proxy": frication_inputs.get("obstacle_proxy"),
            "voicing": frication_inputs.get("voicing"),
            "frication_color": frication_inputs.get("frication_color"),
            "noise_key": frication_inputs.get("noise_key"),
        }
        # Drop unset entries so synthesize_dynamic_tube uses its own defaults.
        fr_kwargs = {k: v for k, v in fr_kwargs.items() if v is not None}

    audio, _, _, _ = synthesize_dynamic_tube(
        length_cm=L,
        area_cm2=A,
        wall_mass=Mw,
        wall_resistance=Bw,
        wall_stiffness=Kw,
        samples_per_frame=int(samples_per_frame),
        glottis_traj=glottis_traj,
        **fr_kwargs,
    )
    expected = (T - 1) * int(samples_per_frame)
    return audio[:expected]


def synthesize_via_tea(
    trajectory: np.ndarray,
    frame_step_samples: int,
    *,
    checkpoint_path: str | None = None,
    frication_inputs: dict | None = None,
) -> np.ndarray:
    """Numpy-friendly wrapper around :func:`synthesize_via_tea_jax`.

    Same shape contract as the legacy ctypes synth path so existing
    callers don't need to change their input/output handling.
    """
    if trajectory.ndim != 2 or trajectory.shape[1] != 30:
        raise ValueError(f"trajectory must be (T, 30); got {trajectory.shape}")
    if trajectory.shape[0] < 2:
        raise ValueError(f"trajectory needs >=2 frames; got {trajectory.shape[0]}")
    audio = synthesize_via_tea_jax(
        trajectory,
        samples_per_frame=int(frame_step_samples),
        checkpoint_path=checkpoint_path,
        frication_inputs=frication_inputs,
    )
    return np.asarray(audio, dtype=np.float32)
