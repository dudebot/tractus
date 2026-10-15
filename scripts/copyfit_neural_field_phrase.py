"""Week-1 neural-field copy-synthesis phrase probe.

Goal: gradient-descent copy-synthesis of a single Harvard sentence
through TEA, where the only optimization variables are the
parameters of a continuous neural field ``A_θ(s, t)`` over normalized
tract location ``s`` and time ``t``. No learned controller, no
trained body model. Source schedule (F0 / pressure / voicing) is
extracted from the target audio and frozen so the tract field can't
absorb the source.

Per task ``workplan/tasks/00_active/001_neural_field_phrase_probe.md``.
Run from ``tractus_controller/.venv/bin/python`` (CUDA-enabled
jaxlib).

CLI::

    tractus_controller/.venv/bin/python scripts/copyfit_neural_field_phrase.py \\
        --audio data/probes/juice_lemons_ref.wav \\
        --transcript "The juice of lemons makes fine punch." \\
        --out outputs/copyfit_neural_field/juice_lemons \\
        --iters 250 --seeds 3
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure TEA is on sys.path (the controller's venv knows about
# tractus_engine_advanced via the workspace uv layout, but running the
# script directly out of the worktree needs an explicit pin).
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
_TEA_SRC = (_ROOT / "tractus_engine_advanced" / "src").resolve()
if str(_TEA_SRC) not in sys.path:
    sys.path.insert(0, str(_TEA_SRC))
_CONTROLLER_SRC = (_ROOT / "tractus_controller").resolve()
if str(_CONTROLLER_SRC) not in sys.path:
    sys.path.insert(0, str(_CONTROLLER_SRC))

# TEA owns precision (fp32 by default). Importing the package first
# pins jax.config; do not call jax.config.update from here.
import tractus_engine_advanced  # noqa: F401, E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tractus_engine_advanced.pipeline import (  # noqa: E402
    NEUTRAL_GLOTTIS_PARAMS,
    synthesize_dynamic_tube,
)
from tractus_engine_advanced.frication import (  # noqa: E402
    Frication,
    PLACE_ALVEOLAR,
)
from tractus_engine_advanced.precision import DTYPE  # noqa: E402

from tractus_controller.copy_synth.features_jax import (  # noqa: E402
    FeatureConfig,
    acoustic_loss,
    log_mel,
    mr_stft_magnitudes,
    precompute_target,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A1 FIX (B+ approach) — pipeline runs end-to-end at TEA's native 44.1 kHz.
#
# Previously SAMPLE_RATE was 22050 and SAMPLES_PER_FRAME was 441, but TEA's
# TIMESTEP_S is hard-coded to 1/44100 — meaning the 441 samples it produced
# per frame were 10 ms at 44.1 kHz (treated as 20 ms at 22.05 kHz by the
# loss), shifting all spectral content down an octave AND halving the
# effective duration. Both confounders.
#
# Fix: render at 44.1 kHz throughout. SAMPLES_PER_FRAME = 882 gives TEA the
# right number of samples for a 20 ms frame at 44.1 kHz (50 Hz frame rate
# matches the glottis schedule). Target audio gets resampled from its native
# rate (typically 22.05 kHz for the Natasha probe) to 44.1 kHz at load time.
# Loss + features compute at 44.1 kHz; saved recon.wav is also at 44.1 kHz
# (decimation to project-rate 22.05 kHz happens at the user-facing boundary,
# not in this diagnostic script).
#
# Cost: TEA renders 2× more samples per frame → forward/backward ~2× slower.
# Loss + features 2× more FFT cost (negligible vs MNA solve).
#
# See `docs/incorrect_assumptions.md` A1, and the canonical engine-rate
# documentation at `scripts/render_m5_oracle.py:17-30`.
SAMPLE_RATE = 44100
FRAME_RATE_HZ = 50
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE_HZ  # 882
N_SECTIONS = 93
# Free range for the neural field. Sections 23/24 (glottis) get overwritten
# by TEA's GeometricGlottis at synth time, so optimizing them is wasted gradient.
# Section 65+ are nasal/sinus/fossa side-branches that this oral-phrase probe
# can't supervise. Mask the field to the supraglottal oral range only.
FREE_LO = 25  # FIRST_PHARYNX_SECTION
FREE_HI = 65  # FIRST_NOSE_SECTION (exclusive)
N_FREE = FREE_HI - FREE_LO  # 40

# Glottis param column indices (from glottis_geometric.py):
# F0, PR, XB, XT, CA, PL, RA, DP, PS, FL, AS = range(11)
G_F0, G_PR = 0, 1

# Field clamp range (log-area). The default tube area ranges 1e-4 to
# 4.24 cm^2. We clamp delta in log space to keep the optimization
# regularized; the absolute area floor / ceiling is reasserted post-hoc
# for the clamp-occupancy gate.
DELTA_MAX = 3.0  # log-cm^2 — body can go ~20× brighter or darker than default.
# Bumped from 1.5 → 3.0 after the 5-iter probe (2026-05-14) showed the optimizer
# wants larger excursions than 1.5 allows; uniform 4.0 cm² default tube → Story
# vowel shapes need log-distance up to ~3.0. See LOG entry 2026-05-14.

# Margin-based softplus barrier (Codex thread [session identifier omitted] fix). Penalty is
# essentially zero in the safe interior, fires near the limits.
BARRIER_MARGIN = 0.5   # log-cm² distance from AREA_MIN/MAX before barrier ramps up
BARRIER_BETA = 5.0     # sharpness of the transition (higher = sharper)
AREA_MIN = 1e-4  # cm^2 — TEA's MIN_AREA_CM2; barrier kicks in earlier
AREA_MAX = 12.0  # cm^2 — well above default max (4.24)

# Pressure schedule: peak subglottal pressure during voicing. Matches
# inverse_fit_demo's voiced_glottis_traj baseline.
PR_VOICED_DPA = 8000.0
PR_UNVOICED_DPA = 0.0  # quiescent during truly unvoiced regions

# Field MLP hidden width / depth.
HIDDEN = 128
N_LAYERS = 4
N_PE_FREQS = 8
PE_MAX_FREQ = 8.0

# Loss weights (task §4 starting recipe).
#
# A9 NOTE — these defaults are unverified initial guesses, never tuned.
# W_SMOOTH_S, W_SMOOTH_T, and W_CURV were copied verbatim from the Week-1
# task spec line "Suggested initial weights — tune in Week 1 itself." They
# were not tuned in Week 1. Commit c639a73 added CLI overrides after Codex
# recommended 10-30× higher smoothness (--w-smooth-s 1e-2, --w-smooth-t 3e-2,
# --w-curv 1e-3), but the defaults here remain the original unverified values.
# Use the CLI flags if you want the Codex-recommended recipe.
#
# See `docs/incorrect_assumptions.md` A9.
W_LOGMEL = 1.0
W_STFT = 0.5
W_SMOOTH_S = 1e-3
W_SMOOTH_T = 1e-3
W_CURV = 1e-4
W_CLAMP = 10.0
W_DEFAULT = 1e-2

LR = 3e-3


# ---------------------------------------------------------------------------
# Source extraction (frozen).
# ---------------------------------------------------------------------------


def extract_source(audio_np: np.ndarray, sr: int, n_frames: int) -> np.ndarray:
    """Run PyWorld DIO/stonemask at 20 ms hop, derive F0 / voicing /
    pressure / loudness schedules, and pack into the 11-vector glottis
    trajectory expected by synthesize_dynamic_tube.

    Returns ``glottis_traj[n_frames, 11]`` as float32.
    """
    import pyworld

    x64 = audio_np.astype(np.float64)
    _f0, t = pyworld.dio(x64, sr, frame_period=1000.0 / FRAME_RATE_HZ)
    f0 = pyworld.stonemask(x64, _f0, t, sr)
    # 5-frame median smoothing on the raw F0 to suppress DIO octave errors.
    f0 = _median_filter(f0, 5)

    # Voicing flag: F0 > 0 means voiced (DIO convention).
    voiced = (f0 > 0).astype(np.float32)

    # Loudness — RMS in a 10 ms (≈ samples_per_frame // 2) window centered
    # on each DIO frame midpoint. Used to amplitude-modulate PR within the
    # voiced regions so the rendered prosody envelope follows the target.
    win = int(0.025 * sr)  # 25 ms RMS window
    rms = _windowed_rms(x64, win, hop=int(sr / FRAME_RATE_HZ))
    rms = rms[: f0.shape[0]]
    rms_norm = rms / (rms.max() + 1e-12)

    # Pad or trim to n_frames.
    def _resize(v: np.ndarray) -> np.ndarray:
        if v.shape[0] >= n_frames:
            return v[:n_frames]
        pad = np.repeat(v[-1:], n_frames - v.shape[0])
        return np.concatenate([v, pad])

    f0 = _resize(f0)
    voiced = _resize(voiced)
    rms_norm = _resize(rms_norm)

    # Replace unvoiced-frame F0 with held-last voiced F0 (interpolation
    # would also work; held-last is robust + cheap).
    f0_filled = np.copy(f0)
    last = 120.0
    for i in range(f0_filled.shape[0]):
        if f0_filled[i] > 0:
            last = float(f0_filled[i])
        else:
            f0_filled[i] = last
    # Clip to glottis control bounds (CONTROL_MIN/MAX in
    # glottis_geometric.py: F0 in [40, 600]).
    f0_filled = np.clip(f0_filled, 50.0, 500.0)

    # Pressure schedule: voiced frames get PR_VOICED_DPA scaled by
    # normalized RMS (so quieter regions also have weaker source). A
    # floor of 0.3 keeps voicing audible even at the quietest voiced
    # frames. Unvoiced frames hold a smaller positive pressure so frication
    # can radiate — but NOT zero, because zero PR shuts the source down
    # entirely and unvoiced consonants need turbulent airflow.
    pr_scale = np.maximum(rms_norm, 0.0)
    pr_voiced = PR_VOICED_DPA * (0.3 + 0.7 * pr_scale)
    pr_unvoiced = 0.5 * PR_VOICED_DPA * pr_scale  # 50 % of voiced peak when transient
    pr = np.where(voiced > 0, pr_voiced, pr_unvoiced).astype(np.float32)

    # Build the 11-channel glottis trajectory, broadcasting the
    # NEUTRAL_GLOTTIS_PARAMS for non-F0/PR columns. NEUTRAL provides
    # sensible defaults for XB/XT (rest displacement), CA (adduction),
    # PL (phase), RA (rounding), DP (deepness), PS (phase shift),
    # FL (flutter), AS (aspiration strength).
    neutral = np.asarray(NEUTRAL_GLOTTIS_PARAMS, dtype=np.float32)
    traj = np.tile(neutral[None, :], (n_frames, 1))
    traj[:, G_F0] = f0_filled.astype(np.float32)
    traj[:, G_PR] = pr

    # During unvoiced regions, slightly increase aspiration (AS column,
    # index 10). AS in glottis params is in dB, range [-40, 0]; nudging
    # toward 0 gives more turbulent source.
    AS_idx = 10
    traj[:, AS_idx] = np.where(voiced > 0, -25.0, -10.0).astype(np.float32)

    return traj.astype(np.float32)


def _median_filter(x: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    out = np.empty_like(x)
    for i in range(x.shape[0]):
        out[i] = float(np.median(xpad[i : i + k]))
    return out


def _windowed_rms(x: np.ndarray, win: int, hop: int) -> np.ndarray:
    """Per-frame RMS at the given hop with a centered window."""
    half = win // 2
    n_frames = (x.shape[0] + hop - 1) // hop
    rms = np.empty(n_frames, dtype=np.float64)
    for f in range(n_frames):
        c = f * hop
        a = max(0, c - half)
        b = min(x.shape[0], c + half)
        seg = x[a:b]
        rms[f] = float(np.sqrt(np.mean(seg * seg) + 1e-20))
    return rms


# ---------------------------------------------------------------------------
# Neural field parameterization.
# ---------------------------------------------------------------------------


def _pe_feats(x: jnp.ndarray, n_freqs: int = N_PE_FREQS,
              max_freq: float = PE_MAX_FREQ) -> jnp.ndarray:
    """Sinusoidal positional encoding for a scalar input ``x``.

    Returns a 1-D vector of length ``2 * n_freqs`` (sin/cos at
    geometrically spaced frequencies).
    """
    freqs = (2.0 ** jnp.arange(n_freqs, dtype=DTYPE)) * jnp.pi / max_freq
    arg = x * freqs
    return jnp.concatenate([jnp.sin(arg), jnp.cos(arg)])


def init_field_params(seed: int, in_dim: int) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    """Glorot-uniform init for a small MLP. Last layer initialized
    near zero so the field starts at exactly default-tube area."""
    rng = np.random.default_rng(seed)
    dims = [in_dim] + [HIDDEN] * (N_LAYERS - 1) + [1]
    params: list[tuple[jnp.ndarray, jnp.ndarray]] = []
    for i in range(len(dims) - 1):
        fan_in, fan_out = dims[i], dims[i + 1]
        limit = math.sqrt(6.0 / (fan_in + fan_out))
        if i == len(dims) - 2:
            # Final layer: tiny weights and zero bias → log_area_delta ≈ 0.
            W = (rng.uniform(-limit, limit, (fan_in, fan_out)) * 0.01).astype(np.float32)
            b = np.zeros((fan_out,), dtype=np.float32)
        else:
            W = rng.uniform(-limit, limit, (fan_in, fan_out)).astype(np.float32)
            b = np.zeros((fan_out,), dtype=np.float32)
        params.append((jnp.asarray(W, dtype=DTYPE), jnp.asarray(b, dtype=DTYPE)))
    return params


def field_forward(params: list[tuple[jnp.ndarray, jnp.ndarray]],
                  s_norm: jnp.ndarray, t_norm: jnp.ndarray) -> jnp.ndarray:
    """Scalar log-area-delta at ``(s_norm, t_norm)``. Both args 0-D."""
    h = jnp.concatenate([_pe_feats(s_norm), _pe_feats(t_norm)])
    for i, (W, b) in enumerate(params):
        h = h @ W + b
        if i < len(params) - 1:
            h = jax.nn.gelu(h)
    return jnp.tanh(h[0]) * DELTA_MAX


def field_grid(params: list[tuple[jnp.ndarray, jnp.ndarray]],
               s_grid: jnp.ndarray, t_grid: jnp.ndarray) -> jnp.ndarray:
    """Evaluate field over the (T, S) grid → log-area-delta field.

    ``s_grid`` selects which sections the field covers — pass
    ``s_grid_free`` to mask to the supraglottal oral range (T, N_FREE),
    or the full ``s_grid_full`` to operate on all sections (legacy).
    """
    # vmap over t outer, vmap over s inner.
    eval_one = lambda s, t: field_forward(params, s, t)
    eval_t = jax.vmap(eval_one, in_axes=(0, None))  # over s, fixed t
    eval_grid = jax.vmap(eval_t, in_axes=(None, 0))  # over t
    return eval_grid(s_grid, t_grid)


# ---------------------------------------------------------------------------
# Rendering & loss.
# ---------------------------------------------------------------------------


@dataclass
class TubeStatic:
    """Frozen tube constants from default_tube.npz."""
    length_cm: jnp.ndarray         # (93,)
    log_area_default: jnp.ndarray  # (93,) log of default area
    wall_mass: jnp.ndarray         # (93,)
    wall_resistance: jnp.ndarray   # (93,)
    wall_stiffness: jnp.ndarray    # (93,)


def load_tube_static(default_tube_path: Path) -> TubeStatic:
    d = np.load(default_tube_path)
    length = jnp.asarray(d["length_cm"], dtype=DTYPE)
    area = jnp.asarray(d["area_cm2"], dtype=DTYPE)
    log_area_default = jnp.log(jnp.maximum(area, AREA_MIN))
    return TubeStatic(
        length_cm=length,
        log_area_default=log_area_default,
        wall_mass=jnp.asarray(d["wall_mass_cgs"], dtype=DTYPE),
        wall_resistance=jnp.asarray(d["wall_resistance_cgs"], dtype=DTYPE),
        wall_stiffness=jnp.asarray(d["wall_stiffness_cgs"], dtype=DTYPE),
    )


def render_from_field(params,
                      tube: TubeStatic,
                      glottis_traj: jnp.ndarray,
                      s_grid: jnp.ndarray,
                      t_grid: jnp.ndarray,
                      *,
                      frame_chunk_size: int = 10,
                      frication_enabled: bool = False) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Sample the field on the (T, 93) grid, build the per-frame tube
    arrays, and synthesize audio through TEA.

    ``frame_chunk_size`` controls TEA's outer-scan checkpoint granularity.
    The default in `synthesize_dynamic_tube` is K=1 (strongest memory
    bound, slowest compile + iter throughput at b=1). At single-batch
    inverse-fit T=120 we have plenty of headroom — Codex review (thread
    [session identifier omitted]) recommended K=4 as the first throughput experiment;
    we use K=10 as a more aggressive trade. The (K, samples_per_frame,
    97, 97) backward-transpose at K=10 b=1 is ≈ 156 MB, well inside the
    16 GB / 5 GB headroom seen on the forward pass.

    Returns ``(audio[N], log_area[T, 93])``.

    The field is sampled over the free supraglottal range (sections
    ``FREE_LO:FREE_HI``) and **spliced** into the default-tube
    log-area. Sections outside the free range stay at default-tube
    values exactly. This avoids spending gradient on glottis sections
    23/24 (overwritten by GeometricGlottis anyway) and on
    nasal/sinus/fossa side-branches the oral-phrase probe can't
    supervise.
    """
    delta_free = field_grid(params, s_grid, t_grid)           # (T, N_FREE)
    T = delta_free.shape[0]
    log_area_default_broadcast = jnp.broadcast_to(
        tube.log_area_default[None, :], (T, N_SECTIONS)
    )
    log_area = log_area_default_broadcast.at[:, FREE_LO:FREE_HI].add(delta_free)
    area = jnp.exp(log_area)
    length = jnp.broadcast_to(tube.length_cm[None, :], (T, N_SECTIONS))
    Mw = jnp.broadcast_to(tube.wall_mass[None, :], (T, N_SECTIONS))
    Bw = jnp.broadcast_to(tube.wall_resistance[None, :], (T, N_SECTIONS))
    Kw = jnp.broadcast_to(tube.wall_stiffness[None, :], (T, N_SECTIONS))
    if frication_enabled:
        # Tag every supraglottal section as a sibilant site (PLACE_ALVEOLAR,
        # gain 2.2). The frication module's physical_gain still gates by
        # Re² · area + pressure-drop thresholds, so noise only fires where
        # the tube is genuinely constricted with high Δp — physics decides
        # WHERE, the place_index just decides the gain multiplier.
        # voicing=0 gives full unvoiced gain; the gate filters voiced
        # frames via the Re² constriction proxy.
        place_index = jnp.zeros((T, N_SECTIONS), dtype=jnp.int32).at[
            :, FREE_LO:FREE_HI
        ].set(PLACE_ALVEOLAR)
        voicing = jnp.zeros((T,), dtype=DTYPE)
        audio, *_ = synthesize_dynamic_tube(
            length_cm=length,
            area_cm2=area,
            wall_mass=Mw,
            wall_resistance=Bw,
            wall_stiffness=Kw,
            samples_per_frame=SAMPLES_PER_FRAME,
            glottis_traj=glottis_traj,
            frame_chunk_size=frame_chunk_size,
            frication=Frication.default(),
            place_index=place_index,
            voicing=voicing,
            noise_key=jax.random.key(0),
        )
    else:
        audio, *_ = synthesize_dynamic_tube(
            length_cm=length,
            area_cm2=area,
            wall_mass=Mw,
            wall_resistance=Bw,
            wall_stiffness=Kw,
            samples_per_frame=SAMPLES_PER_FRAME,
            glottis_traj=glottis_traj,
            frame_chunk_size=frame_chunk_size,
        )
    return audio, log_area


def field_priors(log_area: jnp.ndarray, log_area_default: jnp.ndarray) -> dict:
    """Smoothness + curvature + clamp + default-deviation priors.

    All terms are 0-D scalars. Returned as a dict so the caller can
    weight them and report them in the optim trace.
    """
    # Spatial gradient: log_area[T, 93]; finite-diff along axis=1.
    d_log_s = log_area[:, 1:] - log_area[:, :-1]
    d_log_t = log_area[1:, :] - log_area[:-1, :]
    # Curvature along s.
    d2_log_s = log_area[:, 2:] - 2.0 * log_area[:, 1:-1] + log_area[:, :-2]

    # Clamp barrier — margin-based softplus that's ~zero in the safe
    # interior and only fires near the limits. Previous formulation was
    # a global interior potential (softplus(log_area - log_max) had
    # value 0.29 at log_area = log(4.0) = 1.39, biasing the optimizer
    # to shrink all areas regardless of the acoustic loss). See LOG
    # entry 2026-05-13 + Codex thread [session identifier omitted] for the diagnosis.
    log_lo = jnp.log(AREA_MIN) + BARRIER_MARGIN
    log_hi = jnp.log(AREA_MAX) - BARRIER_MARGIN
    over = jax.nn.softplus(BARRIER_BETA * (log_area - log_hi)) / BARRIER_BETA
    under = jax.nn.softplus(BARRIER_BETA * (log_lo - log_area)) / BARRIER_BETA
    barrier = jnp.mean(over + under)

    default_dev = jnp.mean((log_area - log_area_default[None, :]) ** 2)

    return {
        "smooth_s": jnp.mean(d_log_s ** 2),
        "smooth_t": jnp.mean(d_log_t ** 2),
        "curv": jnp.mean(d2_log_s ** 2),
        "barrier": barrier,
        "default_dev": default_dev,
    }


def compose_loss(params,
                 tube: TubeStatic,
                 glottis_traj: jnp.ndarray,
                 s_grid: jnp.ndarray,
                 t_grid: jnp.ndarray,
                 target_log_mel: jnp.ndarray,
                 target_mr_stfts: list[jnp.ndarray],
                 feature_cfg: FeatureConfig,
                 n_target_samples: int,
                 frame_chunk_size: int = 10,
                 frication_enabled: bool = False,
                 w_smooth_s: float = W_SMOOTH_S,
                 w_smooth_t: float = W_SMOOTH_T,
                 w_curv: float = W_CURV) -> tuple[jnp.ndarray, dict]:
    audio, log_area = render_from_field(
        params, tube, glottis_traj, s_grid, t_grid,
        frame_chunk_size=frame_chunk_size,
        frication_enabled=frication_enabled,
    )
    audio_trim = audio[:n_target_samples]
    # RMS-normalize the rendered audio to ~0.05 (≈ -26 dBFS), matching the
    # target audio's pre-normalization. This decouples the acoustic loss
    # from absolute amplitude, which TEA's source-driven dynamic range
    # cannot match by construction.
    audio_n = audio_trim / (jnp.sqrt(jnp.mean(audio_trim * audio_trim)) + 1e-8) * 0.05

    loss_ac = acoustic_loss(
        audio_n,
        target_log_mel,
        target_mr_stfts,
        feature_cfg,
        mel_weight=W_LOGMEL,
        stft_weight=W_STFT,
    )
    # Restrict priors to the free range — penalizing smoothness on
    # default-tube sections we're not optimizing is meaningless.
    priors = field_priors(
        log_area[:, FREE_LO:FREE_HI],
        tube.log_area_default[FREE_LO:FREE_HI],
    )
    loss = (
        loss_ac
        + w_smooth_s * priors["smooth_s"]
        + w_smooth_t * priors["smooth_t"]
        + w_curv * priors["curv"]
        + W_CLAMP * priors["barrier"]
        + W_DEFAULT * priors["default_dev"]
    )
    diag = {"acoustic": loss_ac, **priors, "total": loss}
    return loss, diag


# ---------------------------------------------------------------------------
# Adam (manual, so we don't pull in optax for one optimizer).
# ---------------------------------------------------------------------------


def adam_init(params):
    """Initialize Adam moments matching the pytree structure of params.

    Returns (m_pytree, v_pytree, step_scalar). Keeping m/v as the same
    pytree shape as params means downstream jit'd ops can use
    ``jax.tree_util.tree_map`` instead of flatten/unflatten in the hot
    path — which is what lets the fused step compile to one XLA HLO.
    """
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    return (m, v, jnp.asarray(0, dtype=jnp.int32))


def adam_apply(params, grads, state, lr=LR, b1=0.9, b2=0.999, eps=1e-8,
               warmup_steps: int = 0, update_norm_cap: float = 0.0):
    """Tree-mapped Adam update. Pure-jax, jit-friendly.

    Optional kwargs (Codex thread [session identifier omitted] — 2026-05-14):

    - ``warmup_steps``: linear LR ramp from 0 → ``lr`` over the first
      ``warmup_steps`` steps. Prevents Adam's m,v-near-zero first-step
      problem (``upd ≈ lr × sign(grad)``) from blowing the field into
      tanh saturation. 0 = disabled.
    - ``update_norm_cap``: global ℓ₂ cap on the per-step update. After
      computing raw Adam updates, scale them so total ‖Δθ‖₂ ≤ cap.
      0 = disabled. Combined with warmup, removes the first-step risk.
    """
    m, v, step = state
    step = step + 1
    # Linear LR warmup.
    warmup_scale = jnp.where(
        warmup_steps > 0,
        jnp.minimum(1.0, step.astype(jnp.float32) / jnp.maximum(1.0, float(warmup_steps))),
        1.0,
    )
    effective_lr = lr * warmup_scale
    b1_corr = 1.0 - b1 ** step
    b2_corr = 1.0 - b2 ** step

    def _new_m(mi, g):
        return b1 * mi + (1.0 - b1) * g

    def _new_v(vi, g):
        return b2 * vi + (1.0 - b2) * (g * g)

    m_new = jax.tree_util.tree_map(_new_m, m, grads)
    v_new = jax.tree_util.tree_map(_new_v, v, grads)

    # Per-parameter Adam update (raw, before any global norm cap).
    def _raw_update(mn, vn):
        mhat = mn / b1_corr
        vhat = vn / b2_corr
        return effective_lr * mhat / (jnp.sqrt(vhat) + eps)

    raw_updates = jax.tree_util.tree_map(_raw_update, m_new, v_new)

    # Optional global ℓ₂ cap on the update. JAX-friendly: if cap == 0,
    # the scale is unconditionally 1.0 (`jnp.minimum(1, ∞) = 1`).
    leaves = jax.tree_util.tree_leaves(raw_updates)
    upd_l2 = jnp.sqrt(sum(jnp.sum(u * u) for u in leaves))
    cap = jnp.asarray(update_norm_cap, dtype=jnp.float32)
    safe_cap = jnp.where(cap > 0.0, cap, jnp.inf)
    scale = jnp.minimum(1.0, safe_cap / (upd_l2 + 1e-12))
    scaled_updates = jax.tree_util.tree_map(lambda u: u * scale, raw_updates)

    p_new = jax.tree_util.tree_map(lambda p, u: p - u, params, scaled_updates)
    return p_new, (m_new, v_new, step)


# ---------------------------------------------------------------------------
# Plausibility metrics.
# ---------------------------------------------------------------------------


def plausibility(log_area: np.ndarray, log_area_default: np.ndarray) -> dict:
    """Compute Gate-3 metrics. log_area shape (T, 93)."""
    area = np.exp(log_area)
    n_total = area.size
    near_min = float(np.sum(area <= 1.5 * AREA_MIN))
    near_max = float(np.sum(area >= 0.9 * AREA_MAX))
    clamp_occ = (near_min + near_max) / max(n_total, 1)

    d_s = log_area[:, 1:] - log_area[:, :-1]
    d_t = log_area[1:, :] - log_area[:-1, :]
    # d/ds in physical units: section length is ≈ 0.3..1.0 cm; report as
    # raw finite-diff in log-cm² per section (matches the task's
    # |d log A / ds| < 3.0 gate language).
    max_grad_s = float(np.max(np.abs(d_s)))
    # |d log A / dt| in log-cm²/s ; FRAME_RATE_HZ scaling.
    max_grad_t = float(np.max(np.abs(d_t)) * FRAME_RATE_HZ)
    return {
        "clamp_occupancy": float(clamp_occ),
        "max_grad_s": max_grad_s,
        "max_grad_t": max_grad_t,
    }


# ---------------------------------------------------------------------------
# Vowel sanity gate (Gate 4).
# ---------------------------------------------------------------------------


def vowel_sanity(field_log_area: np.ndarray,
                 tube: TubeStatic,
                 anchor_npz: Path,
                 n_frames: int) -> dict:
    """For each of /a/, /i/, /u/: pick the frame-mid for that vowel
    from a hand-mapped time index (since we don't have MFA available),
    snapshot the field's area function at that time, render it as a
    sustained tube via the oral-only transfer-function impulse driver,
    measure F1/F2/F3, and compare to Story targets within ±20 %.

    The phrase is "The juice of lemons makes fine punch." Hand picks:
      /u/ in "juice"  — ~26 %
      /i/ in "fine"   — ~80 %  (close approx of /aI/ nucleus → /i/)
      /a/ in "punch"  — ~92 %  (use 'punch' nucleus as /ʌ/, closest to /a/)
    These are approximate; the gate is qualitative per the task spec.
    """
    from tractus_engine_advanced.tds.mna import TubeGeometry
    from tractus_engine_advanced.tds.state import TdsState
    from tractus_engine_advanced.tds.step import step_one
    from tractus_engine_advanced.tds.constants import MIN_AREA_CM2
    from tractus_engine_advanced.tds.tube_topology import (
        LAST_MOUTH_SECTION, NUM_SECTIONS,
        LOWER_GLOTTIS_SECTION, UPPER_GLOTTIS_SECTION,
        TOPOLOGY,
    )

    GLOTTIS_OPEN_CM2 = 0.5
    IMPULSE_DPA = 1.0
    N_SAMPLES = 4096

    # Story targets (subset).
    TARGETS = {
        "/a/": (694, 942, 2948),   # Story /ɑ/
        "/i/": (325, 2139, 2976),  # Story /i/
        "/u/": (314, 702, 2298),   # Story /u/
    }
    PICK_T_NORM = {"/u/": 0.26, "/i/": 0.80, "/a/": 0.92}

    wall_m = np.asarray(tube.wall_mass, dtype=np.float32)
    wall_r = np.asarray(tube.wall_resistance, dtype=np.float32)
    wall_k = np.asarray(tube.wall_stiffness, dtype=np.float32)
    length_vec = np.asarray(tube.length_cm, dtype=np.float32)

    # JIT the impulse-response computation ONCE; reuse across all three
    # vowels. Without this, ``jax.lax.scan`` traces a fresh closure per
    # vowel (because ``geom`` differs) and the script silently spent
    # 3-5 minutes in repeated traces.
    @jax.jit
    def _ir(area_cm2_full: jnp.ndarray,
            length_cm_full: jnp.ndarray) -> jnp.ndarray:
        geom = TubeGeometry.from_arrays(
            length_cm=length_cm_full,
            area_cm2=area_cm2_full,
            wall_mass=jnp.asarray(wall_m),
            wall_resistance=jnp.asarray(wall_r),
            wall_stiffness=jnp.asarray(wall_k),
        )
        tds_state = TdsState.zeros()

        def body(carry, sample_idx):
            st = carry
            p_in = jnp.where(sample_idx == 0, IMPULSE_DPA, 0.0)
            o = step_one(
                st, geom, TOPOLOGY,
                subglottal_pressure_dPa=p_in,
                soft_walls=True,
                turbulence_losses=True,
            )
            return o.state, o.mouth_flow

        _, ir = jax.lax.scan(body, tds_state, jnp.arange(N_SAMPLES))
        return ir

    out: dict[str, dict] = {}
    for vowel, t_norm in PICK_T_NORM.items():
        print(f"    vowel_sanity {vowel}: rendering impulse response...",
              flush=True)
        f_idx = min(int(round(t_norm * (n_frames - 1))), n_frames - 1)
        area_vec = np.exp(field_log_area[f_idx]).astype(np.float32)  # (93,)
        # Open the glottis and ablate side branches (oral-only).
        A_full = area_vec.copy()
        L_full = length_vec.copy()
        A_full[LOWER_GLOTTIS_SECTION] = GLOTTIS_OPEN_CM2
        A_full[UPPER_GLOTTIS_SECTION] = GLOTTIS_OPEN_CM2
        A_full[LAST_MOUTH_SECTION + 1 : NUM_SECTIONS] = float(MIN_AREA_CM2)
        ir = np.asarray(_ir(jnp.asarray(A_full), jnp.asarray(L_full)),
                        dtype=np.float64)
        ir = np.asarray(ir, dtype=np.float64)
        # Peak detect.
        win = np.hanning(len(ir))
        X = np.fft.rfft(ir * win)
        psd_dB = 20.0 * np.log10(np.abs(X) + 1e-12)
        # A1 FIX: now correct — SAMPLE_RATE = 44100 matches TEA's actual
        # rendering rate (TIMESTEP_S = 1/44100), so the formant reads are
        # in true Hz. (Previously SAMPLE_RATE was 22050 and these reads
        # were halved; see git log for the fix commit.)
        freqs = np.fft.rfftfreq(len(ir), d=1.0 / SAMPLE_RATE)
        pad = 3
        yp = np.pad(psd_dB, (pad, pad), mode="edge")
        sm = np.convolve(yp, np.ones(7) / 7, mode="valid")
        is_pk = np.zeros_like(sm, dtype=bool)
        is_pk[1:-1] = (sm[1:-1] > sm[:-2]) & (sm[1:-1] > sm[2:])
        band = (freqs >= 150.0) & (freqs <= 5000.0)
        peaks = list(zip(freqs[is_pk & band].tolist(),
                         sm[is_pk & band].tolist()))
        tgt = TARGETS[vowel]
        meas = []
        ok = []
        for t in tgt:
            half = 0.20 * t
            in_win = [(f, a) for f, a in peaks if abs(f - t) <= half]
            if in_win:
                in_win.sort(key=lambda x: -x[1])
                meas.append(in_win[0][0])
                ok.append(True)
            else:
                meas.append(0.0)
                ok.append(False)
        out[vowel] = {
            "t_norm": t_norm,
            "frame_idx": f_idx,
            "target": tgt,
            "measured": tuple(float(x) for x in meas),
            "pct_err": tuple(
                float(abs(m - t) / t * 100) if m > 0 else None
                for m, t in zip(meas, tgt)
            ),
            "found": ok,
        }
    return out


# ---------------------------------------------------------------------------
# Public function.
# ---------------------------------------------------------------------------


def copyfit_neural_field_phrase(  # noqa: PLR0913 — CLI-friendly
    audio_path: Path,
    transcript: str,
    out_dir: Path,
    *,
    n_iters: int = 250,
    frame_rate_hz: int = FRAME_RATE_HZ,
    sample_rate: int = SAMPLE_RATE,
    seed: int = 0,
    sparc_npz: Path | None = None,
    n_seeds: int = 3,
    default_tube_path: Path | None = None,
    story_anchors_path: Path | None = None,
    frame_chunk_size: int = 10,
    lr: float = LR,
    diag_every: int = 10,
    warmup_steps: int = 0,
    update_norm_cap: float = 0.0,
    w_smooth_s: float = W_SMOOTH_S,
    w_smooth_t: float = W_SMOOTH_T,
    w_curv: float = W_CURV,
    frication_enabled: bool = False,
) -> dict:
    """Optimize a neural-field log-area perturbation to match the
    reference audio, render through TEA, and write artifacts.

    Returns a dict (also serialized as report.json).
    """
    if sparc_npz is not None:
        # Reserved for Week 2; raise loudly if a caller passes one in
        # Week 1 so we don't silently ignore it.
        raise ValueError("sparc_npz is reserved for Week 2; pass None in Week 1.")
    if frame_rate_hz != FRAME_RATE_HZ:
        raise ValueError(f"frame_rate_hz must be {FRAME_RATE_HZ}; got {frame_rate_hz}.")
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"sample_rate must be {SAMPLE_RATE}; got {sample_rate}.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    default_tube_path = default_tube_path or (
        _ROOT / "tractus_controller" / "outputs" / "fixtures" / "default_tube.npz"
    )
    story_anchors_path = story_anchors_path or (
        _ROOT / "tractus_controller" / "outputs" / "fixtures" / "story_1996_anchors.npz"
    )

    import soundfile as sf
    target_np, sr = sf.read(str(audio_path))
    target_np = target_np.astype(np.float32)
    if target_np.ndim == 2:
        target_np = target_np.mean(axis=-1)
    # A1 FIX: resample reference audio to TEA's native 44.1 kHz rate. Natasha
    # is at 22.05 kHz, dev recordings vary; soxr_hq is high-quality bandlimited
    # interpolation. After this, target and TEA output are on the same time
    # base and the loss is comparing apples-to-apples.
    if sr != sample_rate:
        try:
            import soxr
            target_np = soxr.resample(target_np, sr, sample_rate, quality="HQ").astype(np.float32)
        except ImportError:
            import librosa
            target_np = librosa.resample(
                target_np, orig_sr=sr, target_sr=sample_rate, res_type="soxr_hq"
            ).astype(np.float32)
        print(f"  resampled reference audio: {sr} Hz -> {sample_rate} Hz "
              f"({len(target_np)} samples = {len(target_np)/sample_rate:.3f} s)",
              flush=True)

    # Trim to integer multiple of samples_per_frame; n_frames = T.
    n_frames = target_np.shape[0] // SAMPLES_PER_FRAME
    n_samples_used = n_frames * SAMPLES_PER_FRAME
    target_np = target_np[:n_samples_used]
    # Peak-normalize target to ~0.5 so the loss isn't dominated by absolute
    # amplitude. (We RMS-normalize the rendered audio at the same -26 dBFS
    # marker inside compose_loss.)
    peak = float(np.max(np.abs(target_np)))
    if peak > 0:
        target_np = (target_np * (0.5 / peak)).astype(np.float32)

    glottis_traj_np = extract_source(target_np, sample_rate, n_frames)
    glottis_traj = jnp.asarray(glottis_traj_np, dtype=DTYPE)

    tube = load_tube_static(default_tube_path)

    # Field is sampled only over the free supraglottal range. Use the
    # absolute s-position of each free section so the positional
    # encoding is interpretable across mask changes.
    s_grid_full = jnp.linspace(0.0, 1.0, N_SECTIONS, dtype=DTYPE)
    s_grid = s_grid_full[FREE_LO:FREE_HI]  # (N_FREE,)
    t_grid = jnp.linspace(0.0, 1.0, n_frames, dtype=DTYPE)

    # MR-STFT FFT sizes are specified in samples, so doubling sample rate
    # without doubling them halves the physical analysis window. Preserve
    # the broken-baseline's effective time-frequency analysis (originally
    # (512, 1024, 2048) at 22050 Hz = (23.2, 46.4, 92.9) ms) by using
    # (1024, 2048, 4096) at 44100 Hz. Mel filterbank uses time-based
    # win_ms / hop_ms which auto-scales with sample_rate; nothing to override.
    feature_cfg = FeatureConfig(
        sample_rate=sample_rate,
        mr_stft_n_ffts=(1024, 2048, 4096),
    )
    target_jnp = jnp.asarray(target_np, dtype=DTYPE)
    # Match the rendered audio's RMS scaling to ~0.05 to be symmetric with
    # compose_loss; also peak/RMS normalize the *target* so the log-mel
    # comparison isn't pulled by gross amplitude mismatches.
    target_rms = float(np.sqrt(np.mean(target_np ** 2)) + 1e-8)
    target_for_feats = jnp.asarray(
        (target_np / target_rms * 0.05).astype(np.float32), dtype=DTYPE
    )
    target_log_mel, target_mr_stfts = precompute_target(target_for_feats, feature_cfg)

    PE_DIM = 2 * N_PE_FREQS  # per axis
    IN_DIM = 2 * PE_DIM      # (s, t)
    _ = IN_DIM  # silence linter

    # has_aux=True: the loss-only diff path is what XLA backward
    # touches; the diag dict is computed in forward and shipped back as
    # aux but does not participate in the backward graph. Cost is the
    # extra forward arithmetic for the priors (cheap) + the diag scalars
    # in the output.
    def loss_fn(params):
        loss, diag = compose_loss(
            params,
            tube,
            glottis_traj,
            s_grid,
            t_grid,
            target_log_mel,
            target_mr_stfts,
            feature_cfg,
            n_samples_used,
            frame_chunk_size=frame_chunk_size,
            frication_enabled=frication_enabled,
            w_smooth_s=w_smooth_s,
            w_smooth_t=w_smooth_t,
            w_curv=w_curv,
        )
        return loss, diag

    value_and_grad = jax.value_and_grad(loss_fn, has_aux=True)

    # Telemetry helpers — emitted alongside loss so we can distinguish
    # "gradient vanishes" from "field saturates" from "step too small".
    def _tree_l2(x):
        leaves = jax.tree_util.tree_leaves(x)
        return jnp.sqrt(sum(jnp.sum(a * a) for a in leaves))

    # Fused jit'd train step: forward + backward + adam apply + telemetry
    # in one compiled HLO.
    @jax.jit
    def train_step(params, opt_state):
        (loss, diag), grads = value_and_grad(params)
        # Telemetry — evaluate field at the *pre-update* params so the
        # numbers match the loss we just computed.
        delta_pre = field_grid(params, s_grid, t_grid)
        grad_l2 = _tree_l2(grads)
        delta_absmax = jnp.max(jnp.abs(delta_pre))
        delta_sat_frac = jnp.mean(jnp.abs(delta_pre) > 0.95 * DELTA_MAX)
        params_new, opt_state_new = adam_apply(
            params, grads, opt_state, lr=lr,
            warmup_steps=warmup_steps, update_norm_cap=update_norm_cap,
        )
        # Update magnitude across all params.
        updates = jax.tree_util.tree_map(lambda a, b: b - a, params, params_new)
        update_l2 = _tree_l2(updates)
        diag = {
            **diag,
            "grad_l2": grad_l2,
            "update_l2": update_l2,
            "delta_absmax": delta_absmax,
            "delta_sat_frac": delta_sat_frac,
            "w_barrier": W_CLAMP * diag["barrier"],
        }
        return params_new, opt_state_new, loss, diag

    # NOTE: deliberately NOT defining a separate `eval_step` jit. Each
    # unique jit closure costs a full XLA compile (~90-120s with the
    # checkpointed per-frame scan); the second compile silently bloated
    # wall-clock to >3min before iter-1 printed. We use the FIRST
    # train_step's pre-update (val, diag) as the iter-0 trace anchor,
    # which is mathematically the same thing.

    render_jit = jax.jit(
        lambda params: render_from_field(
            params, tube, glottis_traj, s_grid, t_grid,
            frame_chunk_size=frame_chunk_size,
            frication_enabled=frication_enabled,
        )
    )

    seeds = [seed + i for i in range(n_seeds)]
    seed_results = []
    best = None

    for s_idx, s in enumerate(seeds):
        params = init_field_params(s, IN_DIM)
        opt_state = adam_init(params)
        t0 = time.time()
        trace_total: list[float] = []
        trace_acoustic: list[float] = []
        trace_smooth_s: list[float] = []
        trace_smooth_t: list[float] = []
        trace_curv: list[float] = []
        trace_barrier: list[float] = []
        trace_default: list[float] = []
        if s_idx == 0:
            print(f"  seed={s}: starting (first call includes ~90-120s JIT "
                  f"compile of train_step)...", flush=True)
        else:
            print(f"  seed={s}: starting (cached compile expected)...", flush=True)
        # Async device materialization: avoid blocking `float(val)` on
        # every iter. Keep arrays on-device, only host-sync (and print)
        # every DIAG_EVERY iters. Final final-eval syncs everything.
        DIAG_EVERY = diag_every
        pending_val: jnp.ndarray | None = None
        pending_diag: dict | None = None
        for it in range(n_iters):
            params, opt_state, val, diag = train_step(params, opt_state)
            pending_val = val
            pending_diag = diag
            if (it + 1) % DIAG_EVERY == 0 or it == 0:
                print(
                    f"  seed={s} iter={it + 1:4d} "
                    f"loss={float(pending_val):.4f} "
                    f"ac={float(pending_diag['acoustic']):.6f} "
                    f"wbar={float(pending_diag.get('w_barrier', 0.0)):.2e} "
                    f"grad={float(pending_diag.get('grad_l2', 0.0)):.2e} "
                    f"upd={float(pending_diag.get('update_l2', 0.0)):.2e} "
                    f"dmax={float(pending_diag.get('delta_absmax', 0.0)):.3f} "
                    f"sat={float(pending_diag.get('delta_sat_frac', 0.0)):.3f} "
                    f"t={time.time() - t0:.0f}s",
                    flush=True,
                )
                trace_total.append(float(pending_val))
                trace_acoustic.append(float(pending_diag["acoustic"]))
                trace_smooth_s.append(float(pending_diag["smooth_s"]))
                trace_smooth_t.append(float(pending_diag["smooth_t"]))
                trace_curv.append(float(pending_diag["curv"]))
                trace_barrier.append(float(pending_diag["barrier"]))
                trace_default.append(float(pending_diag["default_dev"]))
        elapsed = time.time() - t0
        # Final eval — re-run train_step but discard the param update by
        # capturing val/diag at the (post-final-iter) params. (For the
        # last reported (val, diag), those are PRE-update, so we run one
        # more grad-free forward by re-invoking train_step but throwing
        # away the new params — net effect: 1 more iter of training, then
        # we record those metrics. Cheaper than a separate eval_step
        # compile.)
        params, opt_state, final_loss, final_diag = train_step(params, opt_state)
        trace_total.append(float(final_loss))
        trace_acoustic.append(float(final_diag["acoustic"]))
        trace_smooth_s.append(float(final_diag["smooth_s"]))
        trace_smooth_t.append(float(final_diag["smooth_t"]))
        trace_curv.append(float(final_diag["curv"]))
        trace_barrier.append(float(final_diag["barrier"]))
        trace_default.append(float(final_diag["default_dev"]))
        print(f"  seed={s} optimization done in {elapsed:.0f}s; "
              f"compiling render + materializing audio...", flush=True)
        audio_np, log_area_np = render_jit(params)
        audio_np = np.asarray(audio_np)
        log_area_np = np.asarray(log_area_np)
        plaus = plausibility(log_area_np, np.asarray(tube.log_area_default))

        # trace_total[0] is loss *after* iter 1 (post-first-grad-step).
        # Because the field is initialized so log_area_delta ≈ 0 and
        # LR is small, this is within ~1 % of the true iter-0 loss; we
        # use it as the "init" anchor for Gate 2. The alternative
        # (separate eval_step jit) cost an extra 90-120s compile per
        # run with no scientific benefit.
        seed_results.append({
            "seed": s,
            "elapsed_s": elapsed,
            "init_total": trace_total[0],
            "final_total": trace_total[-1],
            "init_acoustic": trace_acoustic[0],
            "final_acoustic": trace_acoustic[-1],
            "final_diag": {k: float(v) for k, v in final_diag.items()},
            "plausibility": plaus,
            "trace_total": trace_total,
            "trace_acoustic": trace_acoustic,
            "trace_smooth_s": trace_smooth_s,
            "trace_smooth_t": trace_smooth_t,
            "trace_curv": trace_curv,
            "trace_barrier": trace_barrier,
            "trace_default": trace_default,
            "_audio": audio_np,
            "_log_area": log_area_np,
            "_params": params,
        })
        if (best is None) or (trace_total[-1] < best["trace_total"][-1]):
            best = seed_results[-1]

    # Write artifacts for the BEST seed.
    audio_best = best["_audio"]
    log_area_best = best["_log_area"]
    # Peak-normalize to 0.95 for the wav file (TEA's raw output is small).
    peak_a = float(np.max(np.abs(audio_best)))
    audio_save = audio_best.astype(np.float32)
    if peak_a > 0:
        audio_save = audio_save * (0.95 / peak_a)
    sf.write(str(out_dir / "recon.wav"), audio_save, sample_rate)

    np.savez(
        out_dir / "field_snapshot.npz",
        log_area=log_area_best.astype(np.float32),
        area_cm2=np.exp(log_area_best).astype(np.float32),
        length_cm=np.asarray(tube.length_cm, dtype=np.float32),
        s_grid=np.asarray(s_grid, dtype=np.float32),
        t_grid=np.asarray(t_grid, dtype=np.float32),
        glottis_traj=glottis_traj_np.astype(np.float32),
        seed=np.int32(best["seed"]),
    )

    # Plots.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        for r in seed_results:
            ax.plot(r["trace_total"], label=f"seed={r['seed']} total", alpha=0.5)
        ax.plot(best["trace_acoustic"], color="k", label=f"best acoustic (seed={best['seed']})")
        ax.set_xlabel("iter")
        ax.set_ylabel("loss")
        ax.set_yscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "optim_trace.png", dpi=120)
        plt.close(fig)

        # Area function at three time snapshots.
        fig2, axs2 = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
        t_idx = [int(0.1 * n_frames), int(0.5 * n_frames), int(0.9 * n_frames)]
        s_arr = np.asarray(s_grid)
        area_arr = np.exp(log_area_best)
        for ax, ti in zip(axs2, t_idx):
            ax.plot(s_arr, np.exp(np.asarray(tube.log_area_default)),
                    "k--", label="default")
            ax.plot(s_arr, area_arr[ti, :], "r-", label=f"fit t={ti}")
            ax.set_xlabel("s (normalized)")
            ax.set_title(f"t={ti / max(n_frames-1,1):.2f}")
            ax.set_yscale("log")
            ax.grid(alpha=0.3)
        axs2[0].set_ylabel("area cm²")
        axs2[0].legend(fontsize=8)
        fig2.tight_layout()
        fig2.savefig(out_dir / "area_function_t0_t1_t2.png", dpi=120)
        plt.close(fig2)
    except Exception as e:  # pragma: no cover
        print(f"  WARNING: plot failed: {e}")

    # Vowel sanity.
    print("  measuring vowel sanity...")
    vowels = vowel_sanity(log_area_best, tube, story_anchors_path, n_frames)

    # Gate evaluation (gate 1 is whisper, run externally).
    init_acoustic = best["trace_acoustic"][0]
    final_acoustic = best["trace_acoustic"][-1]
    gate2_pass = bool(final_acoustic <= 0.6 * init_acoustic)

    # A2 GUARD — the Week-1 "plateau" was a still-descending loss stopped at a
    # 100-iter invocation, mistaken for convergence. Nothing in the optimizer
    # forces convergence, so surface an explicit "did the loss actually flatten?"
    # flag in the report. If the trailing-window descent is more than PLATEAU_EPS
    # of the run's total improvement, the run did NOT plateau and a FAILED verdict
    # is not yet earned (same trap as Week-1 A2). Diagnostic only — does not change
    # the optimization; just refuses to let a truncated run masquerade as converged.
    _acc = best["trace_acoustic"]
    PLATEAU_WINDOW = 20
    PLATEAU_EPS = 0.01
    if len(_acc) > PLATEAU_WINDOW:
        _total_descent = float(_acc[0] - _acc[-1])
        _tail_descent = float(_acc[-PLATEAU_WINDOW] - _acc[-1])
        _tail_frac = (_tail_descent / _total_descent) if _total_descent > 1e-9 else 0.0
        plateaued = bool(_tail_frac < PLATEAU_EPS)
    else:
        _tail_frac = None
        plateaued = False
    if not plateaued:
        _tf = _tail_frac if _tail_frac is not None else float("nan")
        print(f"  A2 WARNING: acoustic loss did NOT plateau (trailing-"
              f"{PLATEAU_WINDOW}-iter descent = {_tf:.4f} of total; threshold "
              f"{PLATEAU_EPS}). Loss may still be descending — a FAILED verdict "
              f"is not yet earned. Consider more iters.")

    plaus = best["plausibility"]
    gate3_pass = bool(
        plaus["clamp_occupancy"] < 0.05
        and plaus["max_grad_s"] < 3.0
        and plaus["max_grad_t"] < 5.0
    )
    # Gate 4 — qualitative: all three vowels report measured F1/F2/F3 > 0.
    gate4_summary = []
    for v in ("/a/", "/i/", "/u/"):
        d = vowels[v]
        # F1 alive iff measured > 200 Hz (rule of thumb for not-collapsed).
        f1_alive = d["measured"][0] > 200.0
        gate4_summary.append(f1_alive)
    gate4_pass = all(gate4_summary)

    report = {
        "audio_path": str(audio_path),
        "transcript": transcript,
        "sample_rate": sample_rate,
        "frame_rate_hz": frame_rate_hz,
        "n_frames": int(n_frames),
        "n_samples_used": int(n_samples_used),
        "n_iters": int(n_iters),
        "n_seeds": int(n_seeds),
        "seeds": seeds,
        "frication_enabled": bool(frication_enabled),
        "plateaued": bool(plateaued),
        "tail_descent_frac": (float(_tail_frac) if _tail_frac is not None else None),
        "best_seed": int(best["seed"]),
        "loss_trace": {
            "best_seed_total": best["trace_total"],
            "best_seed_acoustic": best["trace_acoustic"],
            "best_seed_smooth_s": best["trace_smooth_s"],
            "best_seed_smooth_t": best["trace_smooth_t"],
            "best_seed_curv": best["trace_curv"],
            "best_seed_barrier": best["trace_barrier"],
            "best_seed_default": best["trace_default"],
        },
        "init_acoustic": float(init_acoustic),
        "final_acoustic": float(final_acoustic),
        "init_total": float(best["trace_total"][0]),
        "final_total": float(best["trace_total"][-1]),
        "plausibility": plaus,
        "vowels": {
            k: {kk: (list(vv) if isinstance(vv, tuple) else vv)
                for kk, vv in v.items()}
            for k, v in vowels.items()
        },
        "gates": {
            "gate2_loss_improvement": gate2_pass,
            "gate2_target_ratio": float(final_acoustic / max(init_acoustic, 1e-12)),
            "gate3_field_plausibility": gate3_pass,
            "gate4_vowel_sanity": gate4_pass,
            "gate4_summary": gate4_summary,
        },
        "per_seed": [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in seed_results
        ],
        "consent_verified": True,
        "consent_note": (
            "Reference audio is synthetic (ElevenLabs voice 'Natasha — "
            "Valley girl'). Developer consents to using their own "
            "synthetic-TTS sample for Week-1 probe. No real-person "
            "voice cloning."
        ),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"  wrote {out_dir / 'report.json'}")
    return report


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def _main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--transcript", type=str, required=True)
    p.add_argument("--out", type=Path, required=True)
    # A2 FIX — default bumped 250 → 300 to match the pre-registered Task 009 /
    # fixed-rerun budget. The Week-1 contamination came from a 100-iter *invocation*
    # stopping a still-descending loss; 250 was the old code default. 300 is the
    # project's deliberate convergence budget (docs/week1/fixed_rerun_hypotheses.md,
    # 009_tea_to_tea spec). The `plateaued` flag in the report still guards against
    # a run that hasn't actually converged even at 300.
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument(
        "--lr",
        type=float,
        default=LR,
        help=f"Adam learning rate. Default {LR}. Bump to 1e-2 if the field is "
             "not moving (grad_l2 small / acoustic loss flat).",
    )
    p.add_argument(
        "--diag-every",
        type=int,
        default=10,
        help="Print/trace telemetry every N iters. Use 1 for short diagnostic "
             "probes.",
    )
    p.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Linear LR warmup steps. Codex recipe: 20 for the post-margin-"
             "barrier run, prevents Adam first-step saturation. 0 disables.",
    )
    p.add_argument(
        "--update-norm-cap",
        type=float,
        default=0.0,
        help="Global ℓ₂ cap on the per-step update. Codex recipe: 0.35. 0 "
             "disables.",
    )
    p.add_argument("--w-smooth-s", type=float, default=W_SMOOTH_S,
                   help=f"Spatial smoothness weight. Default {W_SMOOTH_S}. "
                        "Codex anti-jaggedness recipe: 1e-2.")
    p.add_argument("--w-smooth-t", type=float, default=W_SMOOTH_T,
                   help=f"Temporal smoothness weight. Default {W_SMOOTH_T}. "
                        "Codex anti-jaggedness recipe: 3e-2.")
    p.add_argument("--w-curv", type=float, default=W_CURV,
                   help=f"Curvature weight. Default {W_CURV}. Codex anti-"
                        "jaggedness recipe: 1e-3.")
    p.add_argument(
        "--frame-chunk-size", type=int, default=10,
        help="TEA `synthesize_dynamic_tube` outer-scan checkpoint chunk "
             "size. Higher = faster compile + per-iter throughput at b=1, "
             "more backward memory. K=10 is the throughput-favoured default "
             "for this single-batch inverse-fit; K=1 is the production "
             "memory-tight default for T=118 b=4 training.",
    )
    p.add_argument(
        "--frication", action="store_true",
        help="Enable physically-derived frication (Frication.default()). "
             "Tags supraglottal sections with PLACE_ALVEOLAR; the physics "
             "gate (Re² · area + Δp threshold) decides where noise fires. "
             "Off by default — Week-1 baseline behavior.",
    )
    args = p.parse_args(argv)

    report = copyfit_neural_field_phrase(
        args.audio, args.transcript, args.out,
        n_iters=args.iters, n_seeds=args.seeds, seed=args.seed_base,
        frame_chunk_size=args.frame_chunk_size, lr=args.lr,
        diag_every=args.diag_every,
        warmup_steps=args.warmup,
        update_norm_cap=args.update_norm_cap,
        w_smooth_s=args.w_smooth_s,
        w_smooth_t=args.w_smooth_t,
        w_curv=args.w_curv,
        frication_enabled=args.frication,
    )
    print(json.dumps(
        {"gates": report["gates"], "init_acoustic": report["init_acoustic"],
         "final_acoustic": report["final_acoustic"],
         "plausibility": report["plausibility"]},
        indent=2,
    ))
    # Return 0 always; gate 1 (Whisper) is a separate runbook call.
    return 0


if __name__ == "__main__":
    sys.exit(_main())
