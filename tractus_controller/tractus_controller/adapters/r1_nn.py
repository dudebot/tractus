"""R1 NN adapter — differentiable JAX emulator of ``the legacy C++ tract→tube map``.

Maps a 19-dim renderer tract-param vector to a 40-section pharynx-mouth tube
(length, area, articulator class, plus three scalar geometry helpers).

Architecture
------------
Single MLP backbone 19 -> H -> H -> H -> shared trunk, then per-head
linear readouts:

    length_cm:                softplus + min_length_cm           (40,)
    area_cm2:                 softplus + min_area_cm2             (40,)
    articulator_logits:                                           (40, 4)
    incisor_pos_cm,
    tongue_tip_side_elev,
    velum_opening_cm2:        plain linear                        (3,)

Inputs are normalized to roughly [-1, 1] via per-channel min/max
(``R1Norm``).

The module is vmappable over a leading batch axis and JIT-friendly.
fp32 by default to keep the trained weights small; the inference helper
in ``r1_inference`` casts to fp64 at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax
import jax.numpy as jnp


N_TRACT_PARAMS = 19
N_SECTIONS = 40
N_ARTICULATOR_CLASSES_DEFAULT = 4
N_SCALAR_HEADS = 3  # incisor, tongue_tip_side_elev, velum_opening


# ---------------------------------------------------------------------------
# Config + normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R1Config:
    """Static configuration for the R1 NN.

    Lives outside the param pytree so JIT can specialize on it via
    ``static_argnames``.
    """

    hidden_sizes: tuple[int, ...] = (512, 512, 512)
    activation: str = "gelu"
    min_area_cm2: float = 0.01     # smallest tube area we will emit
    min_length_cm: float = 0.05    # smallest tube section length
    n_articulator_classes: int = N_ARTICULATOR_CLASSES_DEFAULT
    dtype: str = "float32"

    def jax_dtype(self):
        return jnp.float64 if self.dtype == "float64" else jnp.float32


@dataclass(frozen=True)
class R1Norm:
    """Per-channel min/max normalizer.

    ``normalized = (x - mid) / half_range``
    where ``mid = (max + min) / 2`` and ``half_range = (max - min) / 2``.
    Half-ranges are floored at a tiny epsilon to avoid divide-by-zero on
    degenerate channels.
    """

    tract_min: jnp.ndarray  # (19,)
    tract_max: jnp.ndarray  # (19,)

    def mid(self) -> jnp.ndarray:
        return (self.tract_max + self.tract_min) * 0.5

    def half_range(self) -> jnp.ndarray:
        hr = (self.tract_max - self.tract_min) * 0.5
        return jnp.where(hr < 1e-6, 1.0, hr)

    def normalize(self, x: jnp.ndarray) -> jnp.ndarray:
        # Broadcast over leading batch dims.
        return (x - self.mid()) / self.half_range()

    @classmethod
    def from_speaker(cls, snapshot) -> "R1Norm":
        return cls(
            tract_min=jnp.asarray(snapshot.tract_param_min, dtype=jnp.float32),
            tract_max=jnp.asarray(snapshot.tract_param_max, dtype=jnp.float32),
        )

    @classmethod
    def from_arrays(cls, tract_min, tract_max) -> "R1Norm":
        return cls(
            tract_min=jnp.asarray(tract_min, dtype=jnp.float32),
            tract_max=jnp.asarray(tract_max, dtype=jnp.float32),
        )


# ---------------------------------------------------------------------------
# Param init
# ---------------------------------------------------------------------------


def _glorot(key, shape, dtype):
    fan_in, fan_out = shape[-2], shape[-1]
    scale = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)


def _zeros(shape, dtype):
    return jnp.zeros(shape, dtype=dtype)


def init_params(key, cfg: R1Config) -> dict:
    """Initialize the full pytree of MLP weights (Glorot init).

    Returns a dict with the layout:
        {
          'trunk': [{'W': (in,out), 'b': (out,)}, ...],
          'head_length':       {'W': (H, 40),  'b': (40,)},
          'head_area':         {'W': (H, 40),  'b': (40,)},
          'head_articulator':  {'W': (H, 40*K),'b': (40*K,)},
          'head_scalar':       {'W': (H, 3),   'b': (3,)},
        }
    """
    dtype = cfg.jax_dtype()
    keys = list(jax.random.split(key, 4 + len(cfg.hidden_sizes)))
    layer_sizes = (N_TRACT_PARAMS,) + tuple(cfg.hidden_sizes)
    trunk = []
    for i in range(len(cfg.hidden_sizes)):
        k = keys[i]
        trunk.append({
            "W": _glorot(k, (layer_sizes[i], layer_sizes[i + 1]), dtype),
            "b": _zeros((layer_sizes[i + 1],), dtype),
        })
    H = cfg.hidden_sizes[-1]
    K = cfg.n_articulator_classes
    return {
        "trunk": trunk,
        "head_length": {
            "W": _glorot(keys[-4], (H, N_SECTIONS), dtype),
            "b": _zeros((N_SECTIONS,), dtype),
        },
        "head_area": {
            "W": _glorot(keys[-3], (H, N_SECTIONS), dtype),
            "b": _zeros((N_SECTIONS,), dtype),
        },
        "head_articulator": {
            "W": _glorot(keys[-2], (H, N_SECTIONS * K), dtype),
            "b": _zeros((N_SECTIONS * K,), dtype),
        },
        "head_scalar": {
            "W": _glorot(keys[-1], (H, N_SCALAR_HEADS), dtype),
            "b": _zeros((N_SCALAR_HEADS,), dtype),
        },
    }


# ---------------------------------------------------------------------------
# Forward
# ---------------------------------------------------------------------------


def _activation(x: jnp.ndarray, name: str) -> jnp.ndarray:
    if name == "gelu":
        return jax.nn.gelu(x)
    if name == "relu":
        return jax.nn.relu(x)
    if name == "silu":
        return jax.nn.silu(x)
    raise ValueError(f"unknown activation: {name}")


def forward(
    params: dict,
    tract_params: jnp.ndarray,
    cfg: R1Config,
    norm: R1Norm,
) -> dict:
    """Differentiable forward pass.

    ``tract_params`` shape is ``(..., 19)``; the function broadcasts over
    any leading batch dims.

    Returns a dict with keys:
      * length_cm:           (..., 40)  softplus + min_length_cm
      * area_cm2:            (..., 40)  softplus + min_area_cm2
      * articulator_logits:  (..., 40, K)
      * incisor_pos_cm:      (...,)
      * tongue_tip_side_elev:(...,)
      * velum_opening_cm2:   (...,)     can be negative; no softplus
    """
    dtype = cfg.jax_dtype()
    x = jnp.asarray(tract_params, dtype=dtype)
    h = norm.normalize(x).astype(dtype)
    for layer in params["trunk"]:
        h = _activation(h @ layer["W"] + layer["b"], cfg.activation)

    # Length head: softplus → strictly positive, then add min.
    raw_L = h @ params["head_length"]["W"] + params["head_length"]["b"]
    length_cm = jax.nn.softplus(raw_L) + jnp.asarray(cfg.min_length_cm, dtype=dtype)

    # Area head: softplus + min. Avoid exp() — softplus is bounded below
    # for very negative pre-activations, no gradient blow-up at clamps.
    raw_A = h @ params["head_area"]["W"] + params["head_area"]["b"]
    area_cm2 = jax.nn.softplus(raw_A) + jnp.asarray(cfg.min_area_cm2, dtype=dtype)

    # Articulator head: K logits per section. Reshape (..., 40*K) -> (..., 40, K).
    raw_C = h @ params["head_articulator"]["W"] + params["head_articulator"]["b"]
    K = cfg.n_articulator_classes
    out_shape = raw_C.shape[:-1] + (N_SECTIONS, K)
    articulator_logits = raw_C.reshape(out_shape)

    # Scalar head: incisor, tts3, velum.
    raw_S = h @ params["head_scalar"]["W"] + params["head_scalar"]["b"]
    incisor = raw_S[..., 0]
    tts3 = raw_S[..., 1]
    velum = raw_S[..., 2]

    return {
        "length_cm": length_cm,
        "area_cm2": area_cm2,
        "articulator_logits": articulator_logits,
        "incisor_pos_cm": incisor,
        "tongue_tip_side_elev": tts3,
        "velum_opening_cm2": velum,
    }
