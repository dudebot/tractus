"""BodyModel — JAX neural network mapping `(gesture_traj, anatomy)` to
the renderer interface (Module 3 v0).

Architecture (≤ 5 M params)
---------------------------

Inputs:
- ``gesture_traj`` : (T, D_gesture=20)  — 50 Hz composed gestures
- ``anatomy_vec``  : (D_anatomy=64,)   — speaker static (Module 5 placeholder)

Trunk:
1. Anatomy encoder: (64) → MLP(128, 128) → ``anat_h`` (128,).
2. Per-frame conditioning: concat gesture[t] (20) + anat_h (128) → 148.
3. Temporal encoder: 3 × (Conv1D kernel=5, hidden=192, GELU, residual)
   on (T, 148). Output: (T, 192).
4. Body-latent head: Linear (192 → D_body=64) → ``body_latent`` (T, 64).
5. Decoder: per-frame MLP body_latent (64) + anat_h (128) → 192 →
   Linear(192 → 384) → GELU → Linear(384 → F=385) → reshape into the
   renderer-interface fields. Soft clamps are applied to bring outputs
   into physical range.
6. Static head: anat_h (128) → MLP(128) → 3 × 93 = 279 (wall mass /
   resistance / stiffness).

Param count (computed in :func:`BodyModel.from_default`):
   anat_enc:    64*128 + 128 + 128*128 + 128                   ≈   25K
   temporal:    3 × (148*5*192 + 192 + 192*5*192 + 192 + 192*5*192 + 192)  (residual conv triplet)
                actually one block: in_proj 148*192, then conv 5*192*192*2
   final:        ~1.5M
   decoder:     192*384 + 384 + 384*385 + 385                  ≈   222K
   static:      128*256 + 256 + 256*279 + 279                  ≈   105K

Empirical total: ~ 1.2-1.5 M params (verified by sanity test).

Notes
-----

- D_body=64 and D_anatomy=64 are placeholders. inversion/006 picks
  D_body_effective; once it lands, change :data:`D_BODY_PLACEHOLDER` and
  rebuild the network. No data-format changes needed.
- Module deliberately does NOT depend on ``tractus_engine_advanced``.
  Specifically: do not import ``synthesize_dynamic_tube``. This is a
  load-bearing non-circularity guarantee (ARCH §445-456).
- Anatomy embedding is a v0 placeholder. When Module 5 ships, the
  anatomy encoder's first layer can be retrained or fine-tuned without
  changing the body-model API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .channels import (
    GESTURE_CHANNELS,
    N_GESTURE_CHANNELS,
    N_GLOTTIS_STATE,
    N_TUBE_SECTIONS,
    PER_FRAME_FREE_FIELDS,
    RENDERER_INTERFACE_FIELDS,
    RendererInterface,
    STATIC_FIELDS,
    per_frame_free_dim,
    static_field_dim,
)
from .reachability import (
    DEFAULT_REACHABILITY,
    ReachabilityBounds,
    apply_hard_clamp,
    apply_soft_clamp,
)


# ---------------------------------------------------------------------------
# Placeholder dimensions — to be set by inversion/006 + Module 5
# ---------------------------------------------------------------------------

#: Tentative body-state latent dim until inversion/006 closes. Per the
#: task spec, the architecture step can proceed in parallel, so we pick
#: a reasonable middle of the 30-80 range. Document any change in the
#: inversion/006 acceptance gate.
D_BODY_PLACEHOLDER: int = 64

#: Tentative anatomy-vector dim. Module 5 (speaker embedding) hasn't
#: closed; 64 is a generic compact embedding size.
D_ANATOMY_PLACEHOLDER: int = 64


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BodyModelConfig:
    """Static configuration for the body model."""

    d_gesture: int = N_GESTURE_CHANNELS         # 20
    d_anatomy: int = D_ANATOMY_PLACEHOLDER       # 64 placeholder
    d_body: int = D_BODY_PLACEHOLDER             # 64 placeholder
    anat_hidden: int = 128
    temporal_hidden: int = 192
    temporal_kernel: int = 5
    temporal_blocks: int = 3
    decoder_hidden: int = 384
    static_hidden: int = 256
    activation: str = "gelu"

    def per_frame_free_dim(self) -> int:
        return per_frame_free_dim()

    def static_field_dim(self) -> int:
        return static_field_dim()


# ---------------------------------------------------------------------------
# Param init helpers
# ---------------------------------------------------------------------------


def _glorot(key, shape, dtype=jnp.float32):
    fan_in, fan_out = shape[-2], shape[-1]
    scale = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(key, shape, dtype=dtype, minval=-scale, maxval=scale)


def _zeros(shape, dtype=jnp.float32):
    return jnp.zeros(shape, dtype=dtype)


def _conv_kernel(key, kernel, in_ch, out_ch, dtype=jnp.float32):
    """1D convolution kernel, shape (kernel, in_ch, out_ch) in JAX
    convention (for ``lax.conv_general_dilated``)."""
    fan_in = kernel * in_ch
    fan_out = kernel * out_ch
    scale = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(
        key, (kernel, in_ch, out_ch), dtype=dtype, minval=-scale, maxval=scale,
    )


def _activation(x: jnp.ndarray, name: str = "gelu") -> jnp.ndarray:
    if name == "gelu":
        return jax.nn.gelu(x)
    if name == "relu":
        return jax.nn.relu(x)
    if name == "silu":
        return jax.nn.silu(x)
    raise ValueError(f"unknown activation: {name}")


def init_params(key: jax.Array, cfg: BodyModelConfig) -> dict:
    keys = list(jax.random.split(key, 16))
    ki = iter(keys)

    # Anatomy encoder: D_anatomy → anat_hidden → anat_hidden
    anat_enc = [
        {"W": _glorot(next(ki), (cfg.d_anatomy, cfg.anat_hidden)),
         "b": _zeros((cfg.anat_hidden,))},
        {"W": _glorot(next(ki), (cfg.anat_hidden, cfg.anat_hidden)),
         "b": _zeros((cfg.anat_hidden,))},
    ]

    # Temporal encoder: input projection (gesture + anat_h), then conv blocks.
    in_proj = {
        "W": _glorot(next(ki), (cfg.d_gesture + cfg.anat_hidden, cfg.temporal_hidden)),
        "b": _zeros((cfg.temporal_hidden,)),
    }
    temporal_blocks = []
    for _ in range(cfg.temporal_blocks):
        temporal_blocks.append({
            "conv1": _conv_kernel(next(ki), cfg.temporal_kernel,
                                  cfg.temporal_hidden, cfg.temporal_hidden),
            "b1": _zeros((cfg.temporal_hidden,)),
            "conv2": _conv_kernel(next(ki), cfg.temporal_kernel,
                                  cfg.temporal_hidden, cfg.temporal_hidden),
            "b2": _zeros((cfg.temporal_hidden,)),
        })

    # Body-latent head: temporal_hidden → D_body
    latent_head = {
        "W": _glorot(next(ki), (cfg.temporal_hidden, cfg.d_body)),
        "b": _zeros((cfg.d_body,)),
    }

    # Decoder: (D_body + anat_hidden) → decoder_hidden → per_frame_free_dim
    F = cfg.per_frame_free_dim()
    decoder = [
        {"W": _glorot(next(ki), (cfg.d_body + cfg.anat_hidden, cfg.decoder_hidden)),
         "b": _zeros((cfg.decoder_hidden,))},
        {"W": _glorot(next(ki), (cfg.decoder_hidden, F)),
         "b": _zeros((F,))},
    ]

    # Static head: anat_hidden → static_hidden → static_field_dim
    Fs = cfg.static_field_dim()
    static_head = [
        {"W": _glorot(next(ki), (cfg.anat_hidden, cfg.static_hidden)),
         "b": _zeros((cfg.static_hidden,))},
        {"W": _glorot(next(ki), (cfg.static_hidden, Fs)),
         "b": _zeros((Fs,))},
    ]

    return {
        "anat_enc": anat_enc,
        "in_proj": in_proj,
        "temporal_blocks": temporal_blocks,
        "latent_head": latent_head,
        "decoder": decoder,
        "static_head": static_head,
    }


def count_params(params: dict) -> int:
    """Recursively count scalars in a params pytree."""
    total = 0
    if isinstance(params, dict):
        for v in params.values():
            total += count_params(v)
    elif isinstance(params, (list, tuple)):
        for v in params:
            total += count_params(v)
    elif hasattr(params, "size"):
        total += int(params.size)
    return total


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------


def _conv1d_same(x: jnp.ndarray, kernel: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    """Causal-padded same-length 1D conv. ``x`` shape (T, C_in); kernel
    (K, C_in, C_out); bias (C_out,). Returns (T, C_out)."""
    # JAX conv expects (N, C, T) with kernel (C_out, C_in, K). Reshape.
    K = kernel.shape[0]
    pad_left = (K - 1) // 2
    pad_right = K - 1 - pad_left
    x_nct = x.T[None, :, :]  # (1, C_in, T)
    k_oik = jnp.transpose(kernel, (2, 1, 0))  # (C_out, C_in, K)
    y = jax.lax.conv_general_dilated(
        x_nct,
        k_oik,
        window_strides=(1,),
        padding=((pad_left, pad_right),),
        dimension_numbers=("NCH", "OIH", "NCH"),
    )  # (1, C_out, T)
    return y[0].T + bias  # (T, C_out)


def _decode_per_frame_flat(
    flat: jnp.ndarray,  # (T, F)
) -> dict:
    """Slice the flat per-frame decoder output into per-field arrays."""
    fields: dict = {}
    cursor = 0
    for name in PER_FRAME_FREE_FIELDS:
        shape_suffix = RENDERER_INTERFACE_FIELDS[name]
        width = int(np.prod(shape_suffix)) if shape_suffix else 1
        slab = flat[..., cursor : cursor + width]
        if not shape_suffix:
            # scalar per frame: shape (T, 1) -> (T,)
            slab = slab[..., 0]
        elif len(shape_suffix) == 1:
            # already (T, width)
            pass
        else:
            slab = slab.reshape(slab.shape[:-1] + tuple(shape_suffix))
        fields[name] = slab
        cursor += width
    return fields


def _decode_static_flat(flat: jnp.ndarray) -> dict:
    """Slice the static decoder output into per-field arrays."""
    fields: dict = {}
    cursor = 0
    for name in STATIC_FIELDS:
        shape_suffix = RENDERER_INTERFACE_FIELDS[name]
        width = int(np.prod(shape_suffix)) if shape_suffix else 1
        slab = flat[cursor : cursor + width]
        if len(shape_suffix) == 1:
            pass
        else:
            slab = slab.reshape(shape_suffix)
        fields[name] = slab
        cursor += width
    return fields


def _add_neutral_bias(decoded: dict) -> dict:
    """Shift raw decoder outputs into a sensible physical baseline so
    softplus/clamp doesn't have to drag everything across many decades.

    Uses speaker-default ballparks: ~1 cm length, ~1 cm² area, F0 110 Hz,
    PR 8000 dPa, etc. Pre-clamp values still vary freely; this just
    centers the sigmoid/clamp.
    """
    d = dict(decoded)
    d["tube_length_cm"] = decoded["tube_length_cm"] + 1.0
    d["tube_area_cm2"] = decoded["tube_area_cm2"] + 1.0
    # glottis_state (T, 11): bias F0 to 110, PR to 8000.
    g = decoded["glottis_state"]
    bias = jnp.array(
        [110.0, 8000.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=g.dtype,
    )
    d["glottis_state"] = g + bias
    d["nasal_coupling"] = decoded["nasal_coupling"] + 0.0
    d["subglottal_press"] = decoded["subglottal_press"] + 8000.0
    return d


def _add_static_neutral_bias(decoded_static: dict) -> dict:
    """Bias static wall params toward a reasonable speaker default
    (mass ~ 0.1 g, resistance ~ 1.0, stiffness ~ 1e5 cgs)."""
    d = dict(decoded_static)
    d["wall_mass_static"] = decoded_static["wall_mass_static"] + 0.1
    d["wall_resistance_static"] = decoded_static["wall_resistance_static"] + 1.0
    d["wall_stiffness_static"] = decoded_static["wall_stiffness_static"] + 1.0e5
    return d


def forward(
    params: dict,
    gesture_traj: jnp.ndarray,   # (T, D_gesture)
    anatomy_vec: jnp.ndarray,    # (D_anatomy,)
    cfg: BodyModelConfig,
    bounds: ReachabilityBounds = DEFAULT_REACHABILITY,
    *,
    use_hard_clamp: bool = False,
) -> RendererInterface:
    """Differentiable forward pass.

    Returns a :class:`RendererInterface` with all per-frame and static
    fields filled. Soft clamps are applied by default; pass
    ``use_hard_clamp=True`` for evaluation.
    """
    # --- anatomy encoder
    h = anatomy_vec
    for layer in params["anat_enc"]:
        h = _activation(h @ layer["W"] + layer["b"], cfg.activation)
    anat_h = h  # (anat_hidden,)

    # --- per-frame conditioning
    T = gesture_traj.shape[-2]
    cond = jnp.concatenate(
        [gesture_traj, jnp.broadcast_to(anat_h, (T, anat_h.shape[-1]))],
        axis=-1,
    )  # (T, D_gesture + anat_hidden)

    # --- input projection to temporal_hidden
    x = cond @ params["in_proj"]["W"] + params["in_proj"]["b"]
    x = _activation(x, cfg.activation)  # (T, temporal_hidden)

    # --- temporal encoder: residual conv blocks
    for blk in params["temporal_blocks"]:
        h1 = _conv1d_same(x, blk["conv1"], blk["b1"])
        h1 = _activation(h1, cfg.activation)
        h2 = _conv1d_same(h1, blk["conv2"], blk["b2"])
        x = _activation(x + h2, cfg.activation)

    # --- body-latent
    body_latent = x @ params["latent_head"]["W"] + params["latent_head"]["b"]
    # (T, d_body)

    # --- decoder per frame: concat body_latent + anat_h
    dec_in = jnp.concatenate(
        [body_latent, jnp.broadcast_to(anat_h, (T, anat_h.shape[-1]))],
        axis=-1,
    )
    h = _activation(
        dec_in @ params["decoder"][0]["W"] + params["decoder"][0]["b"],
        cfg.activation,
    )
    flat = h @ params["decoder"][1]["W"] + params["decoder"][1]["b"]
    decoded = _decode_per_frame_flat(flat)
    decoded = _add_neutral_bias(decoded)

    # --- static wall params from anatomy alone
    sh = _activation(
        anat_h @ params["static_head"][0]["W"] + params["static_head"][0]["b"],
        cfg.activation,
    )
    static_flat = sh @ params["static_head"][1]["W"] + params["static_head"][1]["b"]
    static_decoded = _decode_static_flat(static_flat)
    static_decoded = _add_static_neutral_bias(static_decoded)

    # --- clamps
    clamp_fn = apply_hard_clamp if use_hard_clamp else apply_soft_clamp
    clamped = clamp_fn(
        tube_length_cm=decoded["tube_length_cm"],
        tube_area_cm2=decoded["tube_area_cm2"],
        glottis_state=decoded["glottis_state"],
        nasal_coupling=decoded["nasal_coupling"],
        subglottal_press=decoded["subglottal_press"],
        frication_gain=decoded["frication_gain"],
        frication_color=decoded["frication_color"],
        bounds=bounds,
    )

    return RendererInterface(
        tube_length_cm=clamped["tube_length_cm"],
        tube_area_cm2=clamped["tube_area_cm2"],
        glottis_state=clamped["glottis_state"],
        nasal_coupling=clamped["nasal_coupling"],
        subglottal_press=clamped["subglottal_press"],
        frication_gain=clamped["frication_gain"],
        frication_color=clamped["frication_color"],
        wall_mass_static=jnp.maximum(static_decoded["wall_mass_static"], 1e-4),
        wall_resistance_static=jnp.maximum(static_decoded["wall_resistance_static"], 1e-4),
        wall_stiffness_static=jnp.maximum(static_decoded["wall_stiffness_static"], 1e-4),
        wall_dynamic_residual=None,  # default OFF (cheat-channel discipline)
    )


# ---------------------------------------------------------------------------
# High-level wrapper
# ---------------------------------------------------------------------------


@dataclass
class BodyModel:
    """Instantiable body model.

    Construct via :meth:`from_default` for a freshly-initialized model
    with the placeholder dims, or via :meth:`__init__` if you already
    have params + cfg from disk.
    """

    params: dict
    cfg: BodyModelConfig
    bounds: ReachabilityBounds = DEFAULT_REACHABILITY

    @classmethod
    def from_default(cls, seed: int = 0) -> "BodyModel":
        """Construct a freshly-initialized BodyModel with placeholder dims."""
        cfg = BodyModelConfig()
        key = jax.random.PRNGKey(seed)
        params = init_params(key, cfg)
        return cls(params=params, cfg=cfg)

    def num_params(self) -> int:
        return count_params(self.params)

    def __call__(
        self,
        gesture_traj: jnp.ndarray,
        anatomy_vec: jnp.ndarray,
        *,
        use_hard_clamp: bool = False,
    ) -> RendererInterface:
        return forward(
            self.params,
            jnp.asarray(gesture_traj, dtype=jnp.float32),
            jnp.asarray(anatomy_vec, dtype=jnp.float32),
            self.cfg,
            self.bounds,
            use_hard_clamp=use_hard_clamp,
        )

    def batched_forward(
        self,
        gesture_batch: jnp.ndarray,    # (B, T, D_gesture)
        anatomy_batch: jnp.ndarray,    # (B, D_anatomy)
        *,
        use_hard_clamp: bool = False,
    ) -> RendererInterface:
        """vmapped forward over a leading batch axis."""
        fn = partial(
            forward, self.params,
            cfg=self.cfg, bounds=self.bounds,
            use_hard_clamp=use_hard_clamp,
        )
        return jax.vmap(fn, in_axes=(0, 0))(gesture_batch, anatomy_batch)
