"""GeometricGlottis: smooth JAX port of upstream `GeometricGlottis.cpp`.

The model is an analytical phase-warped sinusoidal glottal-area model,
a standard analytical glottal-area model.
hard mode (parity target ≤ 1e-4 RMSE on glottal areas/lengths over a parameter
sweep) and add smooth surrogates for the kinks in soft mode.

The module exposes pure functions over (state, params, anatomy) tuples — no
Python object state. The per-sample carry is small (~12 floats) so reverse-mode
backprop through `lax.scan` is cheap.

Upstream references (line numbers in `upstream/backend/`):
"""

from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import scipy.signal as sps

from tractus_engine_advanced.precision import DTYPE
from tractus_engine_advanced.smoothing import Smoothing, soft_relu

# Parameter index conventions match the upstream `enum`s (see
# Standard geometric-glottis source model.
# Control params (length 11):
F0, PR, XB, XT, CA, PL, RA, DP, PS, FL, AS = range(11)
# Static params (length 4):
RT, RL, RF0, CL = range(4)

NUM_CONTROL_PARAMS = 11
NUM_STATIC_PARAMS = 4
NUM_GLOTTIS_SECTIONS = 2

# Hard parameter clamps from upstream (see GeometricGlottis::GeometricGlottis()):
CONTROL_MIN = jnp.array([40.0, 0.0, -0.05, -0.05, -0.25, 0.0, -1.0, 0.0, -0.5, 0.0, -40.0])
CONTROL_MAX = jnp.array([600.0, 20000.0, 0.3, 0.3, 0.25, 3.1415, 1.0, 1.0, 0.5, 100.0, 0.0])
STATIC_MIN = jnp.array([0.3, 0.5, 50.0, 0.1])
STATIC_MAX = jnp.array([1.0, 2.0, 400.0, 0.5])

# Default static params.
STATIC_DEFAULT = jnp.array([0.45, 1.6, 120.0, 0.4])  # RT, RL, RF0, CL


# ---------------------------------------------------------------------------
# Chebyshev IIR — direct port of upstream's hand-rolled createChebyshev.
# Run at init to compute (a, b) for the 4th-order, 25 Hz lowpass on
# supraglottal pressure. Per-sample stateful update is in `_iir_step`.
# ---------------------------------------------------------------------------


def _create_chebyshev(cutoff_ratio: float, is_highpass: bool, num_poles: int):
    """Compute IIR coefficients matching upstream `IirFilter::createChebyshev`.

    Returns (a, b) numpy arrays of length `num_poles + 1`. Convention follows
    upstream: a[i] are FIR-like (input) taps, b[i] are recursive (output) taps;
    output sample is `y = a[0]*x + sum(a[i]*x[-i] + b[i]*y[-i] for i in 1..N)`.
    """
    PERCENT_RIPPLE = 0.5
    MAX_ORDER = 32  # upstream's MAX_IIR_ORDER

    if num_poles % 2 != 0:
        num_poles += 1
    num_poles = min(num_poles, MAX_ORDER)

    a = np.zeros(MAX_ORDER + 3)
    b = np.zeros(MAX_ORDER + 3)
    a[2] = 1.0
    b[2] = 1.0

    for pol in range(1, num_poles // 2 + 1):
        # Pole on the unit circle.
        pol_real = -np.cos(np.pi / (2.0 * num_poles) + (np.pi * (pol - 1)) / num_poles)
        pol_imag = np.sin(np.pi / (2.0 * num_poles) + (np.pi * (pol - 1)) / num_poles)

        # Warp circle → ellipse for the ripple.
        temp = 100.0 / (100.0 - PERCENT_RIPPLE)
        es = np.sqrt(temp * temp - 1.0)
        vx = (1.0 / num_poles) * np.log((1.0 / es) + np.sqrt((1.0 / (es * es)) + 1))
        kx = (1.0 / num_poles) * np.log((1.0 / es) + np.sqrt((1.0 / (es * es)) - 1))
        kx = 0.5 * (np.exp(kx) + np.exp(-kx))
        pol_real = pol_real * (0.5 * (np.exp(vx) - np.exp(-vx))) / kx
        pol_imag = pol_imag * (0.5 * (np.exp(vx) + np.exp(-vx))) / kx

        # s → z (upstream's idiosyncratic pre-warp uses `tan(0.5)`, not π·fc/fs).
        t = 2.0 * np.tan(0.5)
        w = 2.0 * np.pi * cutoff_ratio
        m = pol_real * pol_real + pol_imag * pol_imag
        d = 4.0 - 4.0 * pol_real * t + m * t * t

        x0 = (t * t) / d
        x1 = (2.0 * t * t) / d
        x2 = (t * t) / d
        y1 = (8.0 - 2.0 * m * t * t) / d
        y2 = (-4.0 - 4.0 * pol_real * t - m * t * t) / d

        if is_highpass:
            k = -np.cos(0.5 * w + 0.5) / np.cos(0.5 * w - 0.5)
        else:
            k = np.sin(0.5 - 0.5 * w) / np.sin(0.5 + 0.5 * w)

        d = 1.0 + y1 * k - y2 * k * k
        a0 = (x0 - x1 * k + x2 * k * k) / d
        a1 = (-2.0 * x0 * k + x1 + x1 * k * k - 2.0 * x2 * k) / d
        a2 = (x0 * k * k - x1 * k + x2) / d
        b1 = (2.0 * k + y1 + y1 * k * k - 2.0 * y2 * k) / d
        b2 = (-(k * k) - y1 * k + y2) / d
        if is_highpass:
            a1 = -a1
            b1 = -b1

        ta = a.copy()
        tb = b.copy()
        for i in range(2, MAX_ORDER + 1):
            a[i] = a0 * ta[i] + a1 * ta[i - 1] + a2 * ta[i - 2]
            b[i] = tb[i] - b1 * tb[i - 1] - b2 * tb[i - 2]

    b[2] = 0.0
    a_final = np.zeros(MAX_ORDER + 1)
    b_final = np.zeros(MAX_ORDER + 1)
    for i in range(MAX_ORDER - 1):
        a_final[i] = a[i + 2]
        b_final[i] = -b[i + 2]

    # Normalize DC gain (or Nyquist for highpass).
    sum_a = 0.0
    sum_b = 0.0
    for i in range(MAX_ORDER - 1):
        if not is_highpass or (i & 1) == 0:
            sum_a += a_final[i]
            sum_b += b_final[i]
        else:
            sum_a -= a_final[i]
            sum_b -= b_final[i]
    gain = sum_a / (1.0 - sum_b)
    a_final /= gain

    # Truncate to the actual order. Upstream stores order+1 coefficients
    # (indices 0..order) but only uses indices 1..order on the recursive side.
    return a_final[: num_poles + 1].copy(), b_final[: num_poles + 1].copy()


def _iir_step(x: jax.Array, in_buf: jax.Array, out_buf: jax.Array, a: jax.Array, b: jax.Array):
    """One sample of upstream's IirFilter::getOutputSample.

    Upstream uses a circular buffer; we use a shift register since order is
    small (4) and shift is 4 ops per sample — cheaper than modular indexing
    in JAX. State convention: in_buf[0] is the most recent prior input, etc.

    Returns: (y, new_in_buf, new_out_buf).
    """
    # y = a[0]*x + sum_{i=1..N} (a[i]*x_prev[i-1] + b[i]*y_prev[i-1])
    y = a[0] * x + jnp.dot(a[1:], in_buf) + jnp.dot(b[1:], out_buf)
    new_in_buf = jnp.concatenate([x[None], in_buf[:-1]])
    new_out_buf = jnp.concatenate([y[None], out_buf[:-1]])
    return y, new_in_buf, new_out_buf


def _sos_biquad_step(
    x: jax.Array,
    state: jax.Array,    # (M, 2) DF2T delays per section
    sos: jax.Array,       # (M, 6) [b0, b1, b2, a0(=1), a1, a2] per section
) -> tuple[jax.Array, jax.Array]:
    """Cascaded SOS (transposed direct-form II) biquad filter — one sample.

    correctness/006: replaces direct-form-I evaluation of the 25 Hz Chebyshev
    on supraglottal pressure (geometric.py). DF1 is fp32-unstable for filters
    with poles within ~0.005 of the unit circle (which this 25 Hz/44.1 kHz
    filter has): the b·y_prev recursion accumulates catastrophic cancellation
    across thousands of samples, eventually overflowing fp32. SOS-DF2T keeps
    each biquad's state bounded and is bit-exact-equivalent to the original
    transfer function.

    State convention (transposed DF II, length 2 per section):
      y[n] = b0·x[n] + s1[n]
      s1[n+1] = b1·x[n] - a1·y[n] + s2[n]
      s2[n+1] = b2·x[n] - a2·y[n]
    """
    M = sos.shape[0]
    b0, b1, b2 = sos[:, 0], sos[:, 1], sos[:, 2]
    # sos[:, 3] is a0 = 1.0 by construction (scipy normalizes).
    a1, a2 = sos[:, 4], sos[:, 5]

    # Sequential application across sections. M is small (2 for 4-pole filter)
    # so the Python loop unrolls cleanly under JIT.
    new_state = jnp.zeros_like(state)
    cur = x
    for i in range(M):
        s = state[i]
        y_i = b0[i] * cur + s[0]
        new_s0 = b1[i] * cur - a1[i] * y_i + s[1]
        new_s1 = b2[i] * cur - a2[i] * y_i
        new_state = new_state.at[i].set(jnp.stack([new_s0, new_s1]))
        cur = y_i
    return cur, new_state


# ---------------------------------------------------------------------------
# Anatomy + State + main step
# ---------------------------------------------------------------------------


class GeometricGlottisAnatomy(eqx.Module):
    """Static (per-speaker) GeometricGlottis configuration."""

    static_params: jax.Array  # (4,) RT, RL, RF0, CL  — see STATIC_DEFAULT
    sampling_rate_hz: float = eqx.field(static=True)
    # IIR coefficients for the 4th-order Chebyshev-1 25 Hz lowpass.
    # Kept for back-compat / bit-equivalence test fixtures (DF1 form).
    iir_a: jax.Array  # (5,)
    iir_b: jax.Array  # (5,)
    # SOS biquad form of the same filter (correctness/006). DF1 is fp32-unstable
    # at this Q; SOS is the standard remedy and is mathematically identical.
    iir_sos: jax.Array  # (M, 6)


def make_anatomy(
    static_params: jnp.ndarray = STATIC_DEFAULT,
    sampling_rate_hz: float = 44100.0,
    supra_filter_cutoff_hz: float = 25.0,
) -> GeometricGlottisAnatomy:
    a, b = _create_chebyshev(
        supra_filter_cutoff_hz / sampling_rate_hz, is_highpass=False, num_poles=4
    )
    # SOS form (correctness/006): scipy expects (num, den) with den[0]=1 and
    # den[k] = -our_b[k]; numerator = our `a`. Convert to SOS.
    a_scipy_den = np.concatenate([[1.0], -np.asarray(b)[1:]])
    sos = sps.tf2sos(np.asarray(a), a_scipy_den)
    return GeometricGlottisAnatomy(
        static_params=jnp.asarray(static_params, dtype=DTYPE),
        sampling_rate_hz=sampling_rate_hz,
        iir_a=jnp.asarray(a, dtype=DTYPE),
        iir_b=jnp.asarray(b, dtype=DTYPE),
        iir_sos=jnp.asarray(sos, dtype=DTYPE),
    )


class GeometricGlottisState(eqx.Module):
    """Per-sample carry for GeometricGlottis."""

    phase: jax.Array  # () radians, accumulator (no modulo)
    time_s: jax.Array  # () seconds, used for Klatt jitter
    supra_filtered_dPa: jax.Array  # () supraglottal pressure after IIR
    # DF2T per-biquad state for the 25 Hz Chebyshev (correctness/006). Two
    # biquads × 2 delays = 4 floats; replaces the prior DF1 (in_buf, out_buf).
    iir_sos_state: jax.Array  # (M, 2)


def init_state(num_sos: int = 2) -> GeometricGlottisState:
    return GeometricGlottisState(
        phase=jnp.asarray(0.0, dtype=DTYPE),
        time_s=jnp.asarray(0.0, dtype=DTYPE),
        supra_filtered_dPa=jnp.asarray(0.0, dtype=DTYPE),
        iir_sos_state=jnp.zeros((num_sos, 2), dtype=DTYPE),
    )


# ---------------------------------------------------------------------------
# calc_geometry: stateless, per-sample geometry from (params, phase, supra_p).
# ---------------------------------------------------------------------------


def _restrict(params: jax.Array, lo: jax.Array, hi: jax.Array) -> jax.Array:
    return jnp.clip(params, lo, hi)


def _piecewise_linear_warp(
    orig_phase: jax.Array,
    alpha: jax.Array,
    beta: jax.Array,
    tau_segment: float = 0.0,
) -> jax.Array:
    """Piecewise-linear interpolation of `orig_phase` on the table (alpha → beta).

    `alpha` and `beta` are length-6 arrays; `orig_phase ∈ [0, 4π]`. Mirrors
    upstream's loop in `GeometricGlottis.cpp:227-242` exactly: pick the segment
    `i` such that alpha[i] ≤ phase ≤ alpha[i+1], then linearly interpolate.

    Vectorized via masks since there are only 5 segments. Differentiable
    everywhere except at the breakpoint themselves (sub-grad fine).

    correctness/003 (M6 soft): with ``tau_segment > 0`` the segment-pick
    becomes a softmax over the in-segment indicator (one-hot in hard
    mode → exact equivalence at τ=0). The interp values themselves are
    already smooth; only the discrete pick was a kink.
    """
    EPSILON = 1e-9
    # 5 segments
    seg_lo = alpha[:-1]  # (5,)
    seg_hi = alpha[1:]
    beta_lo = beta[:-1]
    beta_hi = beta[1:]

    denom = jnp.maximum(seg_hi - seg_lo, EPSILON)
    interp = beta_lo + (beta_hi - beta_lo) * (orig_phase - seg_lo) / denom
    # Mask: 1 if orig_phase falls into [seg_lo, seg_hi], else 0.
    # Upstream's loop takes the *first* matching segment; segments are
    # contiguous and non-overlapping (sorted alpha) so taking the segment
    # whose midpoint is closest to orig_phase is equivalent and SIMD-friendly.
    in_segment = (orig_phase >= seg_lo) & (orig_phase <= seg_hi)
    in_segment_f = in_segment.astype(orig_phase.dtype)

    if tau_segment <= 0.0:
        # In case of float ties at boundaries, prefer the lowest-index segment
        # (matches upstream's break-on-first-match behavior).
        first_match = jnp.argmax(in_segment.astype(jnp.int32))
        return interp[first_match]

    # Soft pick: softmax over the in-segment indicator with temperature τ.
    # On a one-hot indicator this reduces to weighted-average around the
    # active segment; near a boundary it smoothly blends two segments.
    soft_w = jax.nn.softmax(in_segment_f / tau_segment)
    return jnp.sum(soft_w * interp)


def calc_geometry(
    control_params: jax.Array,  # (11,) F0, PR, XB, XT, CA, PL, RA, DP, PS, FL, AS
    state: GeometricGlottisState,
    anatomy: GeometricGlottisAnatomy,
    smoothing: Smoothing | None = None,
) -> tuple[jax.Array, jax.Array, dict]:
    """Compute (length_cm[2], area_cm2[2], aux) for the current sample.

    Pure function. No state mutation. Mirrors `GeometricGlottis::calcGeometry`.

    correctness/003 (M6 soft): when ``smoothing`` is non-None and
    ``smoothing.enabled``, hard kinks (``jnp.maximum`` ReLUs, segment
    pick, in-second-pulse blend) are replaced with τ-controlled smooth
    surrogates. ``smoothing=None`` or ``Smoothing.hard()`` runs the
    original hard path (bit-equal forward).
    """
    sm = smoothing if smoothing is not None else Smoothing.hard()
    use_soft = sm.enabled
    tau_relu = float(sm.tau_relu) if use_soft else 0.0
    tau_seg = float(sm.tau_segment) if use_soft else 0.0
    tau_blend = float(sm.tau_blend) if use_soft else 0.0
    cp = _restrict(control_params, CONTROL_MIN, CONTROL_MAX)
    sp = _restrict(anatomy.static_params, STATIC_MIN, STATIC_MAX)

    # Titze-1989 length / thickness from F0.
    length = sp[RL] * jnp.sqrt(cp[F0] / sp[RF0])
    thickness = (sp[RT] * sp[RL]) / length

    # Two phases: lower edge raw, upper edge lagged by PHASE_LAG.
    skewness = cp[PS]
    pi = jnp.pi
    raw_lower = state.phase
    raw_upper = state.phase + 4.0 * pi - cp[PL]

    # Modulo 4π — upstream computes N = floor(p / 4π), p -= N·4π. JAX fmod
    # gives the same result for non-negative inputs (raw_upper has +4π so
    # it's always positive even with PL up to π).
    orig_lower = jnp.fmod(raw_lower, 4.0 * pi)
    orig_lower = jnp.where(orig_lower < 0.0, orig_lower + 4.0 * pi, orig_lower)
    orig_upper = jnp.fmod(raw_upper, 4.0 * pi)
    orig_upper = jnp.where(orig_upper < 0.0, orig_upper + 4.0 * pi, orig_upper)

    # Piecewise-linear warp (6 breakpoints).
    alpha = jnp.array(
        [
            0.0,
            (1.0 + skewness) * pi / 2.0,
            pi + (1.0 - skewness) * pi / 2.0,
            2.0 * pi + (1.0 + skewness) * pi / 2.0,
            3.0 * pi + (1.0 - skewness) * pi / 2.0,
            4.0 * pi,
        ]
    )
    beta = jnp.array([0.0, 0.5 * pi, 1.5 * pi, 2.5 * pi, 3.5 * pi, 4.0 * pi])

    warped_lower = _piecewise_linear_warp(orig_lower, alpha, beta, tau_seg)
    warped_upper = _piecewise_linear_warp(orig_upper, alpha, beta, tau_seg)

    # Amplitude from transglottal pressure (Titze 1989).
    # correctness/006: ``jnp.sqrt`` has gradient ``1/(2·sqrt(x))`` which is
    # NaN at x=0. ``trans_pressure`` is exactly zero whenever
    # ``cp[PR] <= supra_filtered`` (notably the first frames where
    # ``cp[PR] = 0``), so ``jax.grad`` of the chained loss returns NaN unless
    # we lift the argument off zero. ``+1e-12 dPa`` is well below dynamic
    # range (typical pressures are 1e3..1e4 dPa) so forward output is
    # bit-identical within fp32 precision.
    trans_pressure = soft_relu(cp[PR] - state.supra_filtered_dPa, tau_relu)
    amplitude = 0.0011 * jnp.sqrt(trans_pressure + 1e-12)
    amplitude = amplitude * (sp[RL] / length)
    rel_amp = soft_relu(cp[RA], tau_relu)
    amplitude = amplitude * rel_amp

    # Per-edge displacement with double-pulsing dip in [1.5π, 3.5π].
    double_pulsing = cp[DP]

    def _edge_displacement(warped_phase: jax.Array) -> jax.Array:
        normal = amplitude * jnp.sin(warped_phase)
        # Inside the second-pulse region.
        reduced = -amplitude + amplitude * (jnp.sin(warped_phase) + 1.0) * (1.0 - 0.5 * double_pulsing)
        if tau_blend <= 0.0:
            in_second_pulse = (warped_phase > 1.5 * pi) & (warped_phase < 3.5 * pi)
            return jnp.where(in_second_pulse, reduced, normal)
        # Soft blend: product of two sigmoids forms a smooth boundary
        # bump on (1.5π, 3.5π). At τ → 0+ this collapses to the hard
        # jnp.where above (within sigmoid-tail precision).
        gate = (
            jax.nn.sigmoid((warped_phase - 1.5 * pi) / tau_blend)
            * jax.nn.sigmoid((3.5 * pi - warped_phase) / tau_blend)
        )
        return gate * reduced + (1.0 - gate) * normal

    disp_lower = _edge_displacement(warped_lower)
    disp_upper = _edge_displacement(warped_upper)

    # Numerical glottal area integration: 32 lateral samples per edge.
    # x(t) = endX·t + displacement·sin(πt), with t∈(0, 1] and ReLU clipping.
    # Then area = chinkArea + Σ DELTA_LENGTH·(prev_x + x). Note upstream uses
    # NUM_SAMPLES=32 and t = k/N for k=1..32, prev_x = x(k=0)=0.
    chink_area = soft_relu(cp[CA], tau_relu)
    NUM_SAMPLES = 32
    DELTA_LENGTH = length / (NUM_SAMPLES - 1)
    t = jnp.arange(1, NUM_SAMPLES + 1, dtype=DTYPE) / NUM_SAMPLES  # shape (32,)
    sin_pi_t = jnp.sin(t * pi)

    def _edge_area(end_x: jax.Array, disp: jax.Array) -> jax.Array:
        x = soft_relu(end_x * t + disp * sin_pi_t, tau_relu)  # (32,)
        prev_x = jnp.concatenate([jnp.zeros(1, dtype=x.dtype), x[:-1]])
        return chink_area + DELTA_LENGTH * jnp.sum(prev_x + x)

    area_lower = _edge_area(cp[XB], disp_lower)
    area_upper = _edge_area(cp[XT], disp_upper)

    # Tube-section lengths: each glottis section is half the thickness.
    length_cm = jnp.stack([thickness * 0.5, thickness * 0.5])
    area_cm2 = jnp.stack([area_lower, area_upper])

    aux = {
        "length": length,
        "thickness": thickness,
        "amplitude": amplitude,
        "lower_displacement": disp_lower,
        "upper_displacement": disp_upper,
        "warped_lower": warped_lower,
        "warped_upper": warped_upper,
        "lower_area": area_lower,
        "upper_area": area_upper,
        "chink_width": chink_area / jnp.maximum(sp[CL], 1e-6),
    }
    return length_cm, area_cm2, aux


# ---------------------------------------------------------------------------
# inc_time: per-sample state advance.
# ---------------------------------------------------------------------------


def inc_time(
    state: GeometricGlottisState,
    control_params: jax.Array,  # (11,)
    pressure_dPa: jax.Array,  # (4,) [subglottal, lower, upper, supra]
    dt_s: float,
    anatomy: GeometricGlottisAnatomy,
) -> GeometricGlottisState:
    """Advance state by one sample. Mirrors `GeometricGlottis::incTime`."""
    cp = _restrict(control_params, CONTROL_MIN, CONTROL_MAX)
    supra_raw = pressure_dPa[3]

    # Run supra through the SOS-form Chebyshev IIR (correctness/006).
    supra_filtered, new_sos_state = _sos_biquad_step(
        supra_raw, state.iir_sos_state, anatomy.iir_sos
    )
    # Hard clamp to ±40000 dPa.
    supra_filtered = jnp.clip(supra_filtered, -40000.0, 40000.0)

    # Klatt-Klatt-style flutter — frequency components 12.7, 7.1, 4.7 Hz.
    flutter_pct = cp[FL]
    f0_base = cp[F0]
    flutter = (flutter_pct / 50.0) * (f0_base / 100.0) * (
        jnp.sin(2.0 * jnp.pi * 12.7 * state.time_s)
        + jnp.sin(2.0 * jnp.pi * 7.1 * state.time_s)
        + jnp.sin(2.0 * jnp.pi * 4.7 * state.time_s)
    )
    f0 = f0_base + flutter

    new_phase = state.phase + f0 * dt_s * 2.0 * jnp.pi
    new_time = state.time_s + dt_s

    return GeometricGlottisState(
        phase=new_phase,
        time_s=new_time,
        supra_filtered_dPa=supra_filtered,
        iir_sos_state=new_sos_state,
    )


# ---------------------------------------------------------------------------
# step: combined inc_time + calc_geometry.
# ---------------------------------------------------------------------------


def step(
    state: GeometricGlottisState,
    control_params: jax.Array,
    pressure_dPa: jax.Array,
    dt_s: float,
    anatomy: GeometricGlottisAnatomy,
    smoothing: Smoothing | None = None,
) -> tuple[GeometricGlottisState, jax.Array, jax.Array]:
    """One full sample: advance state, then compute geometry from new state.

    Upstream's call order in `Synthesizer::add` (line 200-244) is:
      1. interpolate tube + glottis params
      2. glottis->calcGeometry()        (uses the *current* state)
      3. glottis->getTubeData(...)
      4. tdsModel->setTube(...)
      5. (read pressures from TDS)
      6. glottis->incTime(...)           (advances state for *next* sample)

    So upstream's geometry at sample N uses state advanced through N-1 samples
    (i.e., state *after* incTime was called N-1 times). Our `step` mirrors
    this: we advance state first using the *previous* sample's pressures, then
    compute geometry. Callers must pass pressures from sample N-1 to step(N).
    """
    new_state = inc_time(state, control_params, pressure_dPa, dt_s, anatomy)
    length_cm, area_cm2, _aux = calc_geometry(
        control_params, new_state, anatomy, smoothing=smoothing,
    )
    return new_state, length_cm, area_cm2
