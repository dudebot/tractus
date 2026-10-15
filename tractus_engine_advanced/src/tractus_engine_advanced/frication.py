"""Physically-derived per-section frication noise.

ARCHITECTURE.md §2.3 / §496-504 / §516-524 / §1684-1685; reference papers
Narayanan & Alwan 2000 (sustained fricatives), Stevens 1971 (turbulence
onset). This module owns the *physical* derivation of frication gain from
constriction geometry + pressure + place + voicing + obstacle proxy:

    frication_gain[t, s] = f_phys(constriction_area_cm2[t, s],
                                  pressure_drop_dPa[t, s],
                                  place_index[t, s],
                                  voicing[t],
                                  obstacle_proxy[t, s])

It does NOT introduce a free `frication_gates` channel; the cheat-channel
discipline in §3 forbids that. A small *learned* color residual (per
section, energy-capped) is allowed for fine spectral calibration but
**multiplies** the physical gain — so if physics says zero, color cannot
synthesize energy.

Stochastic noise is reparameterized: `noise(t, s) = tanh(z) · gain` of a
unit-Gaussian draw `z ~ N(0, 1)` keyed by ``noise_key``. Gradients flow
through ``gain`` deterministically; the noise pattern is fixed given the
key. Re-using `tanh` (instead of raw normals) bounds the per-sample
amplitude — frication is dominated by gain not tail draws — and removes
the rejection-sampling step the deleted upstream used.

Articulator-class behaviour (sibilant /s/ vs labio-dental /f/ vs dental
/θ/ vs glottal /h/) emerges from the per-section *place_index* the body
model emits, NOT from a learned switch. The gain function reads
``PLACE_GAIN[place_index]`` to scale the broadband turbulence amplitude
appropriately for each articulator class.

Public API
----------

* :class:`Frication` — dataclass config (enabled, place gains, noise
  amplitude scale, color cap).
* :func:`physical_gain` — pure-numpy/JAX `f_phys(...) -> [93]` returning
  the monopole noise amplitude per section.
* :func:`gain_with_color` — applies the calibration residual (multiplicative,
  capped). Cannot create gain where physical conditions don't.
* :func:`noise_sample` — reparameterized `tanh(N(0,1)) · gain`, deterministic
  given ``noise_key``.
* :func:`derive_frication_from_state` — convenience: pull constriction
  geometry, pressure drop, voicing from the per-sample TDS state.
* :data:`PLACE_*` — integer place codes used in tests + body-model
  emission tables.

Constants
---------

The thresholds and exponents are calibrated to the Narayanan & Alwan 2000
sustained-fricative dataset (mid-band intensity vs constriction area /
pressure-drop sweep). They are physical defaults, not learned. Re-tuning
requires re-running the calibration sweep against measured fricatives;
do NOT use them as free knobs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int

from tractus_engine_advanced.precision import DTYPE
from tractus_engine_advanced.smoothing import Smoothing, soft_step

# ---------------------------------------------------------------------------
# Place-of-articulation integer codes. The body model emits
# ``place_index[T, 93]`` per frame; values that don't match any place
# fall back to ``PLACE_NONE`` (no frication injection at that section).
# ---------------------------------------------------------------------------

PLACE_NONE: int = 0           # no frication site
PLACE_LABIO_DENTAL: int = 1   # /f/ /v/ — lower-lip + upper-teeth
PLACE_DENTAL: int = 2         # /θ/ /ð/ — tongue-tip + upper-teeth
PLACE_ALVEOLAR: int = 3       # /s/ /z/ — tongue-tip + alveolar ridge (sibilant)
PLACE_POST_ALVEOLAR: int = 4  # /ʃ/ /ʒ/ — tongue-blade + post-alveolar (sibilant)
PLACE_VELAR: int = 5          # /x/ /ɣ/ — tongue-body + velum
PLACE_GLOTTAL: int = 6        # /h/ — aspiration at glottis
PLACE_BURST: int = 7          # stop-release transient (k-burst, p-burst, t-burst)

NUM_PLACES: int = 8

# Per-place broadband gain (dimensionless). Sibilants (/s/, /ʃ/) need an
# obstacle (teeth) downstream of the constriction to develop the radiated
# high-frequency energy that distinguishes them from /θ/ /f/. The relative
# weights here are calibrated so a fixed (Δp, A_c) operating point yields
# Narayanan-Alwan-class spectral magnitudes (sibilants ~6 dB above
# non-sibilant fricatives in the 4-8 kHz band). These are static physical
# constants, NOT learned per-frame.
PLACE_GAIN_DEFAULT: tuple[float, ...] = (
    0.0,    # PLACE_NONE
    1.0,    # PLACE_LABIO_DENTAL  /f/ /v/
    0.9,    # PLACE_DENTAL        /θ/ /ð/
    2.2,    # PLACE_ALVEOLAR      /s/ /z/  — sibilant boost
    2.0,    # PLACE_POST_ALVEOLAR /ʃ/ /ʒ/  — sibilant boost
    0.8,    # PLACE_VELAR         /x/
    1.4,    # PLACE_GLOTTAL       /h/, aspiration
    1.6,    # PLACE_BURST         stop-release
)


# Onset thresholds (per Stevens 1971, Catford 1977). Below these the flow
# is laminar and produces no broadband noise. Re-using mks-CGS units of
# the existing constants module: pressure in dPa (= µbar; 1 cmH2O ≈ 980 dPa),
# area in cm². The thresholds here are deliberately conservative — they
# bracket the operating region where measured fricatives are voiced;
# sub-threshold conditions yield zero gain (smooth ramp via `soft_step`).
PRESSURE_DROP_THRESHOLD_DPA: float = 200.0    # ≈ 0.2 cmH2O
AREA_TURBULENT_CEILING_CM2: float = 0.30      # above this, no constriction
AREA_FRICATIVE_FLOOR_CM2: float = 1e-3        # below: total closure, no flow

# Smoothness for the threshold gates (in the same units as the inputs).
DEFAULT_TAU_PRESSURE_DPA: float = 100.0
DEFAULT_TAU_AREA_CM2: float = 0.05

# Color residual cap (multiplicative, ±cap around 1.0). 0.30 means the
# learned residual can only multiply the physical gain by [0.7, 1.3] —
# 30% wiggle room for spectral calibration. Larger caps approach
# cheat-channel territory (the §3 ledger says <20% loss change on
# ablation; this cap is the energy lever that sets that).
COLOR_CAP_DEFAULT: float = 0.30


@dataclass(frozen=True)
class Frication:
    """Frication-module config. All fields have defaults.

    Attributes
    ----------
    enabled
        Master switch. When False, the module is a no-op (zero gain) and
        the rest of the pipeline behaves identically to pre-frication.
    place_gain
        Per-place broadband-gain multiplier. Length-``NUM_PLACES`` tuple.
        Defaults to :data:`PLACE_GAIN_DEFAULT`.
    color_cap
        Symmetric cap on the multiplicative color residual. The
        residual passed to :func:`gain_with_color` is clipped to
        ``[-color_cap, color_cap]`` before multiplying ``(1 + clipped)``
        into the physical gain.
    noise_amp_scale
        Global amplitude scale (cm³/s of equivalent monopole flow per
        unit ``Re² · area`` operating point). Calibrated so default
        sibilant fricative on a 0.05 cm² constriction at 800 dPa
        Δp gives ~50 cm³/s monopole — within the operating range of
        TEA's MNA injection. NOT a learned knob; physics-bound.
    pressure_drop_threshold_dPa, area_turbulent_ceiling_cm2,
    area_fricative_floor_cm2, tau_pressure_dPa, tau_area_cm2
        See module-level constants. Exposed for ablation only.
    """

    enabled: bool = False
    place_gain: tuple[float, ...] = PLACE_GAIN_DEFAULT
    color_cap: float = COLOR_CAP_DEFAULT
    # correctness/004b: with the post-solve injection point in
    # ``update_variables`` (matrix RHS no longer sees source_amp), the
    # MNA-stability ceiling that pinned 004a at ~0.5 cm³/s is gone.
    # Stevens 1971 puts physiological turbulent monopole flow at
    # 5-15 cm³/s peak for sibilants. Default at 8 cm³/s — middle of the
    # measured range, gives ~6-10 dB sibilant lift in the 4-8 kHz band
    # at matched A_c, Δp on the smoke fricative-rich corpus, and stays
    # well clear of the (now-irrelevant) numerical-instability ceiling.
    # The unit-test gates remain under this larger amplitude because
    # they test ratios + signs, not absolute magnitudes.
    noise_amp_scale: float = 8.0
    pressure_drop_threshold_dPa: float = PRESSURE_DROP_THRESHOLD_DPA
    area_turbulent_ceiling_cm2: float = AREA_TURBULENT_CEILING_CM2
    area_fricative_floor_cm2: float = AREA_FRICATIVE_FLOOR_CM2
    tau_pressure_dPa: float = DEFAULT_TAU_PRESSURE_DPA
    tau_area_cm2: float = DEFAULT_TAU_AREA_CM2

    @classmethod
    def default(cls) -> "Frication":
        """Default-on physical frication: enabled=True, all defaults."""
        return cls(enabled=True)

    @classmethod
    def disabled(cls) -> "Frication":
        return cls(enabled=False)


# ---------------------------------------------------------------------------
# Physical gain. Pure function over JAX-or-numpy arrays + a Frication config.
# ---------------------------------------------------------------------------


def _place_gain_table(cfg: Frication) -> Float[Array, "P"]:
    return jnp.asarray(cfg.place_gain, dtype=DTYPE)


def physical_gain(
    constriction_area_cm2,           # (..., 93) — per-section, includes silent
    pressure_drop_dPa,               # (..., 93) — Δp across each section
    place_index,                     # (..., 93) — int32, PLACE_* code
    voicing,                         # (...) or (..., 93) — voicing scalar in [0,1]
    obstacle_proxy,                  # (..., 93) — dental/teeth presence in [0,1]
    *,
    cfg: Frication | None = None,
):
    """Per-section turbulence-noise amplitude (cm³/s monopole equivalent).

    Standard turbulent-noise scaling (Narayanan & Alwan 2000 eq. 4–5,
    Stevens 1971) gives ``q_turb ∝ Re²·A_c`` with ``Re ∝ √(2·Δp/ρ)·A_c/ν``.
    Folding the constants and dropping ν (treated as constant) yields:

        gain = K · place_gain · (Δp/Δp_crit - 1)+ · A_c · σ_turb(A_c)
               · (1 - 0.7·voicing)   ← partial voicing damping
               · (1 + 0.5·obstacle_proxy)  ← downstream-obstacle boost

    where ``(·)+`` is a smooth ReLU, ``σ_turb(A_c)`` is a smooth indicator
    that the area is in the turbulent-flow regime
    ``[A_floor, A_ceiling]``, and ``K`` is :attr:`Frication.noise_amp_scale`.

    Hard invariants (mathematical, NOT learned):
      * ``constriction_area_cm2 == 0`` → gain == 0 (multiplicative).
      * ``pressure_drop_dPa <= 0`` → gain → 0 (smooth ramp; exactly 0 at
        the floor through the soft-ReLU surrogate when the smoothing
        config sets τ=0).
      * ``place_index == PLACE_NONE`` → gain == 0.

    The voicing damping factor matches the empirical observation that
    fully-voiced fricatives ([z], [v]) carry roughly 30% of the broadband
    energy of their voiceless counterparts at matched (A_c, Δp) — the
    glottal source masks part of the supraglottal turbulence.

    Returns
    -------
    Float[Array, "... 93"]
        Per-section gain in cm³/s monopole-flow units.
    """
    cfg = cfg if cfg is not None else Frication.default()
    if not cfg.enabled:
        return jnp.zeros_like(jnp.asarray(constriction_area_cm2, dtype=DTYPE))

    A = jnp.asarray(constriction_area_cm2, dtype=DTYPE)
    dP = jnp.asarray(pressure_drop_dPa, dtype=DTYPE)
    pi = jnp.asarray(place_index, dtype=jnp.int32)
    v = jnp.asarray(voicing, dtype=DTYPE)
    obs = jnp.asarray(obstacle_proxy, dtype=DTYPE)

    # Place-class gain lookup. Out-of-range place_index → PLACE_NONE = 0.0.
    p_gain_table = _place_gain_table(cfg)
    pi_clipped = jnp.clip(pi, 0, NUM_PLACES - 1)
    g_place = p_gain_table[pi_clipped]
    in_range = (pi >= 1) & (pi < NUM_PLACES)
    g_place = jnp.where(in_range, g_place, jnp.zeros_like(g_place))

    # Pressure-drop threshold via smooth ReLU (Δp/Δp_crit - 1)+.
    # cfg fields are python floats (frozen dataclass), so soft_step's
    # ``tau`` arg accepts them directly without going through jnp.asarray.
    # correctness/004b: tanh-saturate the factor at the upper end. Catford
    # 1977 reports turbulent sibilant intensity saturates at ~8-12 dB
    # above the onset operating point — i.e. the linear (Δp/Δp_crit - 1)
    # ramp must asymptote. Without the cap the
    # noise → net_flow → pressure → noise feedback loop on the smoke
    # corpus burst phones (/p/ /k/ /t/) drives source_amp into
    # nonlinear-saturation territory of the MNA pressure update and
    # produces audio clipping (output_step audio peak = 1.0 RMS = 0.47
    # observed pre-cap on burst_kindly). A tanh cap at 8× the linear
    # ramp gives ~18 dB headroom from threshold to saturation, well
    # past the empirical fricative range, while bounding the feedback
    # loop's worst-case amplitude.
    dP_crit_py = float(cfg.pressure_drop_threshold_dPa)
    tau_p_py = float(cfg.tau_pressure_dPa)
    excess_p = dP - dP_crit_py
    pressure_ramp = soft_step(excess_p, tau_p_py) * jnp.maximum(
        excess_p / dP_crit_py, 0.0
    )
    # Saturate at PRESSURE_FACTOR_SAT (asymptote) via tanh; small-x
    # behaviour is preserved (tanh(x/SAT)*SAT ≈ x for x << SAT). The
    # asymptote at 30 covers the typical sibilant operating range
    # (Δp/Δp_crit ~ 5-15) without compression while still bounding
    # the worst-case pressure-noise feedback amplitude (where Δp can
    # transiently grow to ≫ 30·Δp_crit during burst-frication onsets).
    PRESSURE_FACTOR_SAT = jnp.asarray(30.0, dtype=DTYPE)
    pressure_factor = jnp.tanh(pressure_ramp / PRESSURE_FACTOR_SAT) * PRESSURE_FACTOR_SAT

    # Area in the turbulent regime: A_floor < A < A_ceiling. Smooth product
    # of two soft_steps gives a window function in A.
    A_floor_py = float(cfg.area_fricative_floor_cm2)
    A_ceil_py = float(cfg.area_turbulent_ceiling_cm2)
    tau_A_py = float(cfg.tau_area_cm2)
    area_window = (
        soft_step(A - A_floor_py, tau_A_py)
        * soft_step(A_ceil_py - A, tau_A_py)
    )

    # Voicing damping: voiced fricatives carry ~30% of voiceless energy.
    if v.ndim < A.ndim:
        # Broadcast trailing-axis voicing scalar over sections.
        v_broadcast = jnp.broadcast_to(v[..., None], A.shape)
    else:
        v_broadcast = v
    voicing_factor = 1.0 - 0.7 * jnp.clip(v_broadcast, 0.0, 1.0)

    # Obstacle/teeth boost: scales sibilant energy. Bounded [1, 1.5].
    obstacle_factor = 1.0 + 0.5 * jnp.clip(obs, 0.0, 1.0)

    K = jnp.asarray(cfg.noise_amp_scale, dtype=DTYPE)
    gain = (
        K
        * g_place
        * pressure_factor
        * area_window
        * A
        * voicing_factor
        * obstacle_factor
    )
    return gain


def gain_with_color(
    constriction_area_cm2,
    pressure_drop_dPa,
    place_index,
    voicing,
    obstacle_proxy,
    color,                            # (..., 93) — learned residual ∈ [-cap, +cap]
    *,
    cfg: Frication | None = None,
):
    """Apply the ``frication_color`` calibration residual to the physical gain.

    The color residual is **multiplicative** with cap ``cfg.color_cap``:
    ``gain_calibrated = gain_phys · (1 + clip(color, -cap, +cap))``.

    Mathematical guarantee: when ``physical_gain == 0`` (no physical
    support) the calibrated gain is identically zero regardless of the
    color residual. This is the §1684-1685 invariant: color CANNOT
    synthesize frication where physics doesn't support it.
    """
    cfg = cfg if cfg is not None else Frication.default()
    base = physical_gain(
        constriction_area_cm2=constriction_area_cm2,
        pressure_drop_dPa=pressure_drop_dPa,
        place_index=place_index,
        voicing=voicing,
        obstacle_proxy=obstacle_proxy,
        cfg=cfg,
    )
    c = jnp.asarray(color, dtype=DTYPE)
    cap = jnp.asarray(cfg.color_cap, dtype=DTYPE)
    c_clip = jnp.clip(c, -cap, cap)
    return base * (1.0 + c_clip)


# ---------------------------------------------------------------------------
# Reparameterized noise sampling. Replaces the deleted upstream's
# rejection-sampling routine (algorithm reference only).
# ---------------------------------------------------------------------------


def noise_sample(
    noise_key,                      # jax.random.key
    gain,                           # (..., 93) — per-section gain (cm^3/s)
    *,
    bound: float = 1.0,
):
    """Sample one frame of monopole-flow noise, deterministic in ``noise_key``.

    Returns ``tanh(z / bound) · gain`` where ``z ~ N(0, bound²)``. The
    ``tanh`` reparameterization both bounds the sample magnitude (so a
    rare large normal draw can't dominate the spectrum) and gives a
    smooth deterministic gradient through the sampling — the gradient
    flows through ``gain``, the ``z`` is non-differentiable but fixed
    given the key.

    Shape conventions: ``gain.shape`` may be ``(..., 93)``; the noise
    sample shares the shape of ``gain``.
    """
    g = jnp.asarray(gain, dtype=DTYPE)
    z = jax.random.normal(noise_key, shape=g.shape, dtype=DTYPE) * bound
    return jnp.tanh(z / bound) * g


# ---------------------------------------------------------------------------
# Convenience: derive (constriction_area, pressure_drop, voicing) from
# the per-sample TDS state for callers that don't want to hand-build
# every input. The body model is expected to emit these directly in
# production; this helper is a calibration / smoke convenience.
# ---------------------------------------------------------------------------


def derive_pressure_drop_dPa(section_pressure_dPa: Float[Array, "93"]) -> Float[Array, "93"]:
    """Per-section pressure drop = pressure - downstream pressure.

    For the last section, treats the downstream-of-mouth pressure as zero
    (radiation termination, atmospheric reference). Approximate but
    sufficient for the frication operating window — the actual radiated
    pressure is small compared to the drop across a constriction.
    """
    p = jnp.asarray(section_pressure_dPa, dtype=DTYPE)
    # Forward difference, padding with zero at the right.
    p_next = jnp.concatenate(
        [p[1:], jnp.zeros((1,), dtype=DTYPE)], axis=0
    )
    return p - p_next


def constriction_area_from_geom(area_cm2: Float[Array, "93"]) -> Float[Array, "93"]:
    """Identity helper. ``constriction_area`` for the frication module is
    just the per-section tube area at the noise-injection site.

    The body model is expected to emit a *crisp* per-section area; this
    helper is here as a documentation anchor — frication does NOT
    compute a soft-argmin to "find" the constriction (forbidden by the
    SCOPE.md §3.A anti-deliverable). If a section has high area, its
    physical gain is automatically zero through the area_window factor
    in :func:`physical_gain`.
    """
    return jnp.asarray(area_cm2, dtype=DTYPE)


def derive_frication_from_state(
    section_pressure_dPa: Float[Array, "93"],
    area_cm2: Float[Array, "93"],
    place_index: Int[Array, "93"],
    voicing_scalar: Float[Array, ""],
    obstacle_proxy: Float[Array, "93"],
    *,
    cfg: Frication | None = None,
) -> Float[Array, "93"]:
    """Single-frame physical-gain derivation from MNA-state quantities.

    Per-sample helper used by the pipeline integration to plumb the
    physical inputs out of the TDS state.
    """
    cfg = cfg if cfg is not None else Frication.default()
    A = constriction_area_from_geom(area_cm2)
    dP = derive_pressure_drop_dPa(section_pressure_dPa)
    return physical_gain(
        constriction_area_cm2=A,
        pressure_drop_dPa=dP,
        place_index=place_index,
        voicing=voicing_scalar,
        obstacle_proxy=obstacle_proxy,
        cfg=cfg,
    )


__all__ = [
    "Frication",
    "PLACE_NONE",
    "PLACE_LABIO_DENTAL",
    "PLACE_DENTAL",
    "PLACE_ALVEOLAR",
    "PLACE_POST_ALVEOLAR",
    "PLACE_VELAR",
    "PLACE_GLOTTAL",
    "PLACE_BURST",
    "NUM_PLACES",
    "PLACE_GAIN_DEFAULT",
    "PRESSURE_DROP_THRESHOLD_DPA",
    "AREA_TURBULENT_CEILING_CM2",
    "AREA_FRICATIVE_FLOOR_CM2",
    "COLOR_CAP_DEFAULT",
    "physical_gain",
    "gain_with_color",
    "noise_sample",
    "derive_pressure_drop_dPa",
    "constriction_area_from_geom",
    "derive_frication_from_state",
    "place_psd_sos",
    "place_psd_sos_table",
    "psd_filter_step",
    "PsdFilterState",
]


# ---------------------------------------------------------------------------
# Place-conditional PSD shaping (correctness/004b).
# ---------------------------------------------------------------------------
#
# The bare ``noise_sample`` is broadband-flat (``tanh(z) · gain``); audible
# sibilants need spectral peaks at 6-8 kHz (/s/ /z/) and 3 kHz (/ʃ/ /ʒ/).
# Non-sibilant fricatives /f/ /v/ /θ/ /ð/ are ~flat through ~5 kHz, /h/ is
# low-pass through tract resonances, and bursts are broadband with a slight
# HPF.
#
# Implementation: per-place cascaded SOS biquad. Coefficients are derived
# from canonical analog peaking-EQ / lowpass / highpass prototypes and
# bilinear-transformed at TEA's native 44.1 kHz. 4-pole (M=2 biquads)
# is enough for the spectral-class distinction the gate measures; higher
# orders gain nothing here.
#
# Each biquad section is stored as ``(M, 6)`` rows ``[b0, b1, b2, 1, a1, a2]``
# (scipy's standard SOS layout). The full table is ``(NUM_PLACES, M, 6)``.
# Filter state per place class is ``(M, 2)`` DF2T delays — same convention
# as ``glottis/geometric.py:_sos_biquad_step`` (correctness/006).


def _design_peak_bandpass(
    f_peak_hz: float,
    bw_hz: float,
    sample_rate_hz: float,
    *,
    gain_db: float = 12.0,
) -> "np.ndarray":  # (1, 6)
    """Single-section peaking EQ at ``f_peak_hz`` with bandwidth ``bw_hz``.

    Audio EQ "peak" filter cookbook (Robert Bristow-Johnson). Returns one
    SOS row ``[b0, b1, b2, 1, a1, a2]``.
    """
    import numpy as np

    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f_peak_hz / sample_rate_hz
    cw = np.cos(w0)
    sw = np.sin(w0)
    Q = max(f_peak_hz / max(bw_hz, 1e-3), 0.5)
    alpha = sw / (2.0 * Q)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cw
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cw
    a2 = 1.0 - alpha / A
    return np.asarray(
        [[b0 / a0, b1 / a0, b2 / a0, 1.0, a1 / a0, a2 / a0]], dtype=np.float64
    )


def _design_butter_lp(
    cutoff_hz: float, sample_rate_hz: float, order: int = 4
) -> "np.ndarray":  # (M, 6)
    import numpy as np
    import scipy.signal as sps

    sos = sps.butter(order, cutoff_hz, btype="low", fs=sample_rate_hz, output="sos")
    return np.asarray(sos, dtype=np.float64)


def _design_butter_hp(
    cutoff_hz: float, sample_rate_hz: float, order: int = 2
) -> "np.ndarray":
    import numpy as np
    import scipy.signal as sps

    sos = sps.butter(order, cutoff_hz, btype="high", fs=sample_rate_hz, output="sos")
    return np.asarray(sos, dtype=np.float64)


def _design_butter_bp(
    lo_hz: float,
    hi_hz: float,
    sample_rate_hz: float,
    order: int = 2,
) -> "np.ndarray":
    """Bandpass with N-th-order Butterworth (returns 2N biquads)."""
    import numpy as np
    import scipy.signal as sps

    sos = sps.butter(
        order, [lo_hz, hi_hz], btype="band", fs=sample_rate_hz, output="sos"
    )
    return np.asarray(sos, dtype=np.float64)


def _pad_or_truncate_sos(sos, target_M: int):
    """Pad with identity sections (b0=1, b1=b2=a1=a2=0) or truncate to ``target_M``."""
    import numpy as np

    cur_M = sos.shape[0]
    if cur_M == target_M:
        return sos
    if cur_M > target_M:
        return sos[:target_M]
    pad = np.tile(np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=sos.dtype), (target_M - cur_M, 1))
    return np.concatenate([sos, pad], axis=0)


# Number of SOS biquads per place. 2 biquads = 4 poles, enough to give the
# /s/ vs /ʃ/ vs /f/ class distinction at the 4-8 kHz / 6-8 kHz gate
# measurement bands without spending state.
PSD_NUM_SOS: int = 2


def place_psd_sos(place: int, sample_rate_hz: float = 44100.0) -> "np.ndarray":
    """SOS biquad coefficients for one place class.

    Designed against a 44.1 kHz native rate (TEA's TIMESTEP_S). Each
    place gets a `(PSD_NUM_SOS, 6)` table (scipy SOS layout). PLACE_NONE
    returns identity (all-pass) so a misrouted noise stream is
    unmodified — but in practice PLACE_NONE noise is zero by physics,
    so the filter never fires.

    The canonical envelopes encode the spectral character of each
    fricative class measured against the Narayanan-Alwan 2000 sibilant
    centre-frequencies and Stevens 1971 turbulence spectra:

    * PLACE_ALVEOLAR (/s/ /z/): peak ~7 kHz (sibilant), 4-9 kHz BP.
    * PLACE_POST_ALVEOLAR (/ʃ/ /ʒ/): peak ~3 kHz, 2-5 kHz BP.
    * PLACE_LABIO_DENTAL (/f/ /v/): broadband, low-shelf rolloff ~7 kHz.
    * PLACE_DENTAL (/θ/ /ð/): broadband, slightly bright (~5 kHz peak).
    * PLACE_VELAR (/x/): mid 1.5-3 kHz emphasis.
    * PLACE_GLOTTAL (/h/): low-passed below 4 kHz (radiates through
      tract resonances rather than direct turbulence).
    * PLACE_BURST: broadband, HPF at 500 Hz to suppress sub-band rumble.
    """
    import numpy as np

    sr = float(sample_rate_hz)
    nyq = sr * 0.5

    if place == PLACE_NONE:
        sos = np.tile(
            np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
            (PSD_NUM_SOS, 1),
        )
    elif place == PLACE_ALVEOLAR:
        # Sibilant /s/ /z/: bandpass 4-9 kHz centred ~7 kHz.
        sos = _design_butter_bp(4000.0, min(9000.0, nyq * 0.95), sr, order=1)
    elif place == PLACE_POST_ALVEOLAR:
        # /ʃ/ /ʒ/: bandpass 2-5 kHz centred ~3 kHz.
        sos = _design_butter_bp(2000.0, 5000.0, sr, order=1)
    elif place == PLACE_LABIO_DENTAL:
        # /f/ /v/: broadband to ~6 kHz, slightly down above.
        sos = _design_butter_lp(6000.0, sr, order=2)
    elif place == PLACE_DENTAL:
        # /θ/ /ð/: broadband to ~7 kHz, mild peak at 5 kHz.
        lp = _design_butter_lp(7000.0, sr, order=2)
        peak = _design_peak_bandpass(5000.0, 3000.0, sr, gain_db=4.0)
        sos = np.concatenate([lp[:1], peak], axis=0)
    elif place == PLACE_VELAR:
        # /x/: 1.5-3 kHz emphasis.
        sos = _design_butter_bp(1500.0, 3000.0, sr, order=1)
    elif place == PLACE_GLOTTAL:
        # /h/ aspiration: low-pass through tract resonances ~4 kHz.
        sos = _design_butter_lp(4000.0, sr, order=2)
    elif place == PLACE_BURST:
        # Stop-release: HPF 500 Hz to suppress sub-band rumble; flat above.
        sos = _design_butter_hp(500.0, sr, order=2)
    else:
        # Unknown place — identity.
        sos = np.tile(
            np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float64),
            (PSD_NUM_SOS, 1),
        )

    sos = _pad_or_truncate_sos(sos, PSD_NUM_SOS)
    return sos.astype(np.float64)


def place_psd_sos_table(sample_rate_hz: float = 44100.0) -> Float[Array, "P M 6"]:
    """Stack of per-place PSD SOS coefficients. Indexed by ``PLACE_*``."""
    import numpy as np

    rows = [place_psd_sos(p, sample_rate_hz) for p in range(NUM_PLACES)]
    table = np.stack(rows, axis=0)
    return jnp.asarray(table, dtype=DTYPE)


# Filter state per place class: ``(NUM_PLACES, PSD_NUM_SOS, 2)`` DF2T delays.
PsdFilterState = Float[Array, "P M 2"]


def init_psd_filter_state() -> PsdFilterState:
    """Zero-initialised per-place PSD filter state."""
    return jnp.zeros((NUM_PLACES, PSD_NUM_SOS, 2), dtype=DTYPE)


def _sos_biquad_step_scalar(
    x: Float[Array, ""],
    state: Float[Array, "M 2"],
    sos: Float[Array, "M 6"],
) -> tuple[Float[Array, ""], Float[Array, "M 2"]]:
    """Per-place scalar SOS-DF2T biquad step. Mirrors the convention of
    ``glottis/geometric.py:_sos_biquad_step`` (correctness/006)."""
    M = sos.shape[0]
    new_state = jnp.zeros_like(state)
    cur = x
    for i in range(M):
        s = state[i]
        b0_i, b1_i, b2_i = sos[i, 0], sos[i, 1], sos[i, 2]
        a1_i, a2_i = sos[i, 4], sos[i, 5]
        y_i = b0_i * cur + s[0]
        new_s0 = b1_i * cur - a1_i * y_i + s[1]
        new_s1 = b2_i * cur - a2_i * y_i
        new_state = new_state.at[i].set(jnp.stack([new_s0, new_s1]))
        cur = y_i
    return cur, new_state


def psd_filter_step(
    per_place_input: Float[Array, "P"],
    state: PsdFilterState,
    sos_table: Float[Array, "P M 6"],
) -> tuple[Float[Array, ""], PsdFilterState]:
    """One sample of place-conditional PSD shaping.

    ``per_place_input[p]`` is the (already-physically-gained) noise
    contribution attributable to place ``p``, summed across all sections
    that emit at place ``p``. Each place's stream is filtered through
    its own SOS biquad cascade; the shaped streams are summed to give
    the scalar ``shaped_noise`` injected at the mouth-radiate output.
    """
    out = jnp.asarray(0.0, dtype=DTYPE)
    new_state_per_place = []
    for p in range(NUM_PLACES):
        y_p, st_p = _sos_biquad_step_scalar(
            per_place_input[p], state[p], sos_table[p]
        )
        out = out + y_p
        new_state_per_place.append(st_p)
    new_state = jnp.stack(new_state_per_place, axis=0)
    return out, new_state
