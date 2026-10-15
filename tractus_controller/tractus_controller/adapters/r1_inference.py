"""Inference helpers for the R1 NN adapter.

Pure-JAX, no I/O on the hot path. Importable by W3's tea_backend
replacement.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .r1_nn import R1Config, R1Norm, forward


# Mirror tea_backend.py — same section indices.
_LOWER_GLOTTIS_SECTION = 23
_UPPER_GLOTTIS_SECTION = 24
_FIRST_PHARYNX_SECTION = 25
_LAST_MOUTH_SECTION = 64  # inclusive
_NUM_PHARYNX_MOUTH = 40

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[2]
    / "outputs" / "checkpoints" / "r1_nn" / "r1_v1.pkl"
)


def load_r1_checkpoint(
    path: str | Path | None = None,
) -> tuple[dict, R1Config, R1Norm, jnp.ndarray]:
    """Load (params, cfg, norm, section_weight) from a saved checkpoint.

    Returns JAX arrays. ``section_weight`` is the (40,) loss-weight mask
    used during training; downstream code can reuse it for diagnostic
    weighting of test errors.
    """
    p = Path(path) if path is not None else DEFAULT_CHECKPOINT
    with open(p, "rb") as f:
        blob = pickle.load(f)
    cfg = blob["cfg"]
    norm = blob["norm"]
    params = blob["params"]
    section_weight = jnp.asarray(blob["section_weight"], dtype=jnp.float32)
    return params, cfg, norm, section_weight


def tract_to_tube_jax(
    params: dict,
    cfg: R1Config,
    norm: R1Norm,
    tract_params: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Vmappable. ``(T, 19)`` -> ``((T, 40) length_cm, (T, 40) area_cm2)``.

    Outputs cast to TEA's configured precision (fp32 default; fp64 if
    ``TEA_PRECISION=fp64``) so downstream TEA composition stays in its
    native precision.
    """
    from tractus_engine_advanced.precision import DTYPE as _TEA_DTYPE
    out = forward(params, tract_params, cfg, norm)
    return out["length_cm"].astype(_TEA_DTYPE), out["area_cm2"].astype(_TEA_DTYPE)


def compose_93_section_jax(
    default_length: jnp.ndarray,   # (93,)
    default_area: jnp.ndarray,     # (93,)
    pm_length: jnp.ndarray,        # (40,)
    pm_area: jnp.ndarray,          # (40,)
    gl_length: jnp.ndarray,        # (2,)
    gl_area: jnp.ndarray,          # (2,)
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Pure-JAX 93-section composer (mirrors ``tea_backend._compose_93_section``).

    No Python branching; vmappable over a leading T axis on
    ``pm_length, pm_area, gl_length, gl_area``. ``default_*`` is shared
    across frames.
    """
    from tractus_engine_advanced.precision import DTYPE as _TEA_DTYPE
    L = default_length.astype(_TEA_DTYPE)
    A = default_area.astype(_TEA_DTYPE)
    pmL = pm_length.astype(_TEA_DTYPE)
    pmA = pm_area.astype(_TEA_DTYPE)
    glL = gl_length.astype(_TEA_DTYPE)
    glA = gl_area.astype(_TEA_DTYPE)

    glottis_idx = jnp.array([_LOWER_GLOTTIS_SECTION, _UPPER_GLOTTIS_SECTION])
    pm_idx = jnp.arange(_FIRST_PHARYNX_SECTION, _LAST_MOUTH_SECTION + 1)

    L = L.at[glottis_idx].set(glL)
    A = A.at[glottis_idx].set(glA)
    L = L.at[pm_idx].set(pmL)
    A = A.at[pm_idx].set(pmA)
    return L, A


def load_default_tube(path: str | Path | None = None) -> dict:
    """Load the default-tube fixture as JAX arrays.

    Returns a dict with keys ``length_cm`` (93,), ``area_cm2`` (93,),
    ``wall_mass_cgs``, ``wall_resistance_cgs``, ``wall_stiffness_cgs``.
    """
    if path is None:
        base = Path(__file__).resolve().parents[2] / "outputs" / "fixtures"
        path = base / "default_tube.npz"
        if not path.exists():
            raise FileNotFoundError(f"no default tube fixture at {path}")
    from tractus_engine_advanced.precision import DTYPE as _TEA_DTYPE
    d = np.load(path, allow_pickle=True)
    return {
        "length_cm": jnp.asarray(d["length_cm"], dtype=_TEA_DTYPE),
        "area_cm2": jnp.asarray(d["area_cm2"], dtype=_TEA_DTYPE),
        "wall_mass_cgs": jnp.asarray(d["wall_mass_cgs"], dtype=_TEA_DTYPE),
        "wall_resistance_cgs": jnp.asarray(d["wall_resistance_cgs"], dtype=_TEA_DTYPE),
        "wall_stiffness_cgs": jnp.asarray(d["wall_stiffness_cgs"], dtype=_TEA_DTYPE),
    }
