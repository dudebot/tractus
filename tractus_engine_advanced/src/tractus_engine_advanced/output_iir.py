"""M4 — output-stage Chebyshev IIR + skin radiation filter.

Direct port of upstream's `IirFilter` (`IirFilter.cpp:31-466`) for the two
filters that touch the audio path:

  - `outputPressureFilter`: 8-pole low-pass, cutoff = 12 000 Hz / 44 100 Hz,
    applied to the pressure derivative (`Synthesizer.cpp:40,251`).
  - `glottalToneFilter`: 4-pole, fixed coefficients shared with
    `transvelarCouplingFilter1` (`TdsModel.cpp:567`); converts the
    `FIRST_PHARYNX` section pressure to a skin-radiation volume velocity
    (`TdsModel.cpp:899`).

Both filters share the same direct-form-I recursion (the same as the existing
glottis IIR step in `glottis/geometric.py:_iir_step`), but with the higher
order kept generic. We re-export the same `_create_chebyshev` helper so the
output filter does not depend on the glottis module.

Reference for stage 4 of `notes/pipeline_map.md`: SCOPE.md §2.2 step 9.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from tractus_engine_advanced.glottis_geometric import _create_chebyshev
from tractus_engine_advanced.precision import DTYPE

# Upstream constants.
SAMPLING_RATE_HZ = 44100
SYNTHETIC_SPEECH_BANDWIDTH_HZ = 12000.0  # Constants.h:44
OUTPUT_FILTER_ORDER = 8                  # Synthesizer.cpp:40

# Skin-radiation filter coefficients (TdsModel.cpp:520-554, reused at :567).
# These are hand-tuned, NOT generated from createChebyshev — they're the
# transvelarCouplingFilter1 coefficients, also assigned to glottalToneFilter
# via setCoefficients. Order 4. Verbatim from upstream.
_GLOTTAL_TONE_A = np.array(
    [
        5.027640021717718e-007,
        -7.995535578908732e-007,
        2.967895557191014e-007,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)
_GLOTTAL_TONE_B = np.array(
    [
        0.0,
        3.986308869708467,
        -5.959669638387298,
        3.960408461107104,
        -0.987047716233603,
    ],
    dtype=np.float64,
)
GLOTTAL_TONE_ORDER = 4

# 1e-7 final scaling (Synthesizer.cpp:251).
AUDIO_SCALE = 1e-7


# ---------------------------------------------------------------------------
# Filter coefficient builders


def output_chebyshev_coeffs() -> tuple[np.ndarray, np.ndarray]:
    """Coefficients for the 8th-order Chebyshev-I lowpass (Synthesizer.cpp:40).

    Cutoff ratio = 12000/44100. Lowpass. Returns (a, b) of length order+1=9.
    """
    cutoff_ratio = SYNTHETIC_SPEECH_BANDWIDTH_HZ / SAMPLING_RATE_HZ
    a, b = _create_chebyshev(cutoff_ratio, is_highpass=False, num_poles=OUTPUT_FILTER_ORDER)
    return a, b


def glottal_tone_coeffs() -> tuple[np.ndarray, np.ndarray]:
    """Skin-radiation filter (TdsModel.cpp:567). Hand-coded — not Chebyshev."""
    return _GLOTTAL_TONE_A.copy(), _GLOTTAL_TONE_B.copy()


# ---------------------------------------------------------------------------
# Stateful per-sample step (matches upstream IirFilter::getOutputSample)


class IirState(eqx.Module):
    """Direct-form-I shift-register state for an IIR filter of order N.

    `in_buf[0]` is the most-recent prior input x[n-1]; `out_buf[0]` is y[n-1].
    """

    in_buf: Float[Array, "N"]
    out_buf: Float[Array, "N"]

    @classmethod
    def zeros(cls, order: int) -> "IirState":
        return cls(
            in_buf=jnp.zeros(order, dtype=DTYPE),
            out_buf=jnp.zeros(order, dtype=DTYPE),
        )


def iir_step(
    state: IirState,
    x: Float[Array, ""],
    a: Float[Array, "N1"],  # length order+1
    b: Float[Array, "N1"],
) -> tuple[IirState, Float[Array, ""]]:
    """One direct-form-I step of `IirFilter::getOutputSample` (IirFilter.cpp:61).

    `y[n] = a[0]*x[n] + sum_{i=1..N} a[i]*x[n-i] + sum_{i=1..N} b[i]*y[n-i]`.
    """
    y = a[0] * x + jnp.dot(a[1:], state.in_buf) + jnp.dot(b[1:], state.out_buf)
    new_in = jnp.concatenate([x[None], state.in_buf[:-1]])
    new_out = jnp.concatenate([y[None], state.out_buf[:-1]])
    return IirState(in_buf=new_in, out_buf=new_out), y


# ---------------------------------------------------------------------------
# Combined output state + driver


class OutputState(eqx.Module):
    """Per-sample carry for the M4 output stage."""

    output_filter: IirState           # 8-pole Chebyshev on pressure
    skin_filter: IirState             # 4-pole on FIRST_PHARYNX pressure
    prev_total_flow_cm3_s: Float[Array, ""]  # for the pressure derivative

    @classmethod
    def zeros(cls) -> "OutputState":
        return cls(
            output_filter=IirState.zeros(OUTPUT_FILTER_ORDER),
            skin_filter=IirState.zeros(GLOTTAL_TONE_ORDER),
            prev_total_flow_cm3_s=jnp.asarray(0.0, dtype=DTYPE),
        )


class OutputCoeffs(eqx.Module):
    """Static (per-init) IIR coefficients for the output stage."""

    out_a: Float[Array, "9"]
    out_b: Float[Array, "9"]
    skin_a: Float[Array, "5"]
    skin_b: Float[Array, "5"]
    dt_s: float = eqx.field(static=True)

    @classmethod
    def default(cls) -> "OutputCoeffs":
        oa, ob = output_chebyshev_coeffs()
        sa, sb = glottal_tone_coeffs()
        return cls(
            out_a=jnp.asarray(oa, dtype=DTYPE),
            out_b=jnp.asarray(ob, dtype=DTYPE),
            skin_a=jnp.asarray(sa, dtype=DTYPE),
            skin_b=jnp.asarray(sb, dtype=DTYPE),
            dt_s=1.0 / SAMPLING_RATE_HZ,
        )


def output_step(
    state: OutputState,
    coeffs: OutputCoeffs,
    mouth_flow_cm3_s: Float[Array, ""],
    nostril_flow_cm3_s: Float[Array, ""],
    first_pharynx_pressure_dPa: Float[Array, ""],
) -> tuple[OutputState, Float[Array, ""]]:
    """One sample of the upstream output stage (Synthesizer.cpp:230-251).

    Computes:
      skin_flow      = glottalToneFilter(P[FIRST_PHARYNX])
      total_flow     = mouth_flow + nostril_flow + skin_flow
      pressure       = (total_flow - prev_total_flow) / dt
      audio          = outputPressureFilter(pressure) * 1e-7

    The `(k - 1) & TDS_BUFFER_MASK` indexing in upstream is a circular
    1-sample delay. State on entry holds `prev_total_flow` from the previous
    sample (zero on first call, matching `outputFlow[]` zero-init).
    """
    skin_filter, skin_flow = iir_step(
        state.skin_filter, first_pharynx_pressure_dPa, coeffs.skin_a, coeffs.skin_b
    )
    total_flow = mouth_flow_cm3_s + nostril_flow_cm3_s + skin_flow
    pressure = (total_flow - state.prev_total_flow_cm3_s) / coeffs.dt_s
    output_filter, filtered = iir_step(
        state.output_filter, pressure, coeffs.out_a, coeffs.out_b
    )
    audio = filtered * AUDIO_SCALE
    new_state = OutputState(
        output_filter=output_filter,
        skin_filter=skin_filter,
        prev_total_flow_cm3_s=total_flow,
    )
    return new_state, audio
