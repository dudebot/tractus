"""End-to-end synthesis entry point.

For M4 (output stage parity) we expose `synthesize_static_tube`: drive a fixed
93-section tube with a fixed subglottal pressure for N samples and return
audio samples in `[-1, 1]`. This matches the M2 parity-test driver shape
(`tests/test_tds_parity.py:_drive`) and is the harness the M4 acceptance
criterion is written against.

`synthesize_dynamic_tube` is the per-frame M5/G3a driver: tube + glottis
trajectories at frame rate, audio at sample rate. The glottis is the JAX
`GeometricGlottis` (`tea/glottis_geometric.py`), advanced per sample inside
the inner scan and used to override sections 23/24 of the tube each sample.
Per-sample call order mirrors `Synthesizer::add` (`Synthesizer.cpp:200-242`):
  1. linear-interp glottis params (prev_frame → cur_frame) at sample fraction
  2. calc_geometry on current glottis state → glottis section length/area
  3. override tube sections 23/24 with those values
  4. read 4 pressures from current TDS state (last sample's post-step result)
  5. inc_time glottis with those pressures (advance phase + IIR)
  6. step_one TDS with subglottal_pressure_dPa = cp[PR]

For frames at sample n in `[f*samples_per_frame, (f+1)*samples_per_frame)`
the glottis-param interp uses `prev_cp = glottis_traj[f-1]` (or
`glottis_traj[0]` for frame 0) and `cur_cp = glottis_traj[f]`. This matches
upstream's `prevGlottisParams/newGlottisParams` semantics.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from tractus_engine_advanced import frication as _frication_mod
from tractus_engine_advanced.frication import Frication
from tractus_engine_advanced import glottis_geometric as gg
from tractus_engine_advanced.output_iir import (
    OutputCoeffs,
    OutputState,
    output_step,
)
from tractus_engine_advanced.precision import DTYPE
from tractus_engine_advanced.smoothing import Smoothing
from tractus_engine_advanced.tds.constants import TIMESTEP_S
from tractus_engine_advanced.tds.mna import TubeGeometry
from tractus_engine_advanced.tds.state import PRESSURE, TdsState
from tractus_engine_advanced.tds.step import step_one
from tractus_engine_advanced.tds.tube_topology import (
    FIRST_MOUTH_SECTION,
    FIRST_PHARYNX_SECTION,
    LAST_MOUTH_SECTION,
    LAST_TRACHEA_SECTION,
    LOWER_GLOTTIS_SECTION,
    TOPOLOGY,
    UPPER_GLOTTIS_SECTION,
    TubeTopology,
)

# Neutral 11-vector glottis-param trajectory for static-glottis callers
# (kept as a `jnp` array at the right dtype). Matches the test fixture in
# tests/test_geometric_glottis.py NEUTRAL_PARAMS verbatim.
NEUTRAL_GLOTTIS_PARAMS = jnp.array(
    [120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0],
    dtype=DTYPE,
)


def synthesize_static_tube(
    geom: TubeGeometry,
    n_samples: int,
    *,
    topology: TubeTopology = TOPOLOGY,
    subglottal_pressure_dPa: float = 8000.0,
    soft_walls: bool = True,
    turbulence_losses: bool = True,
    initial_tds_state: TdsState | None = None,
    initial_output_state: OutputState | None = None,
    output_coeffs: OutputCoeffs | None = None,
    dt: float = TIMESTEP_S,
    smoothing: Smoothing | None = None,
) -> tuple[Float[Array, "N"], TdsState, OutputState]:
    """Run `n_samples` of TDS + output stage on a fixed tube.

    Returns `(audio, final_tds_state, final_output_state)`. Audio is in `[-1, 1]`
    (after the upstream `1e-7` scaling).

    The pressure source is held constant at `subglottal_pressure_dPa` for the
    full window. To get an impulse response, set it nonzero for one sample and
    zero otherwise — but the simpler "step response" (constant pressure from
    rest) is the canonical parity fixture and matches `test_tds_parity._drive`.
    """
    tds_state = initial_tds_state if initial_tds_state is not None else TdsState.zeros()
    out_state = initial_output_state if initial_output_state is not None else OutputState.zeros()
    coeffs = output_coeffs if output_coeffs is not None else OutputCoeffs.default()

    sm_static = smoothing if smoothing is not None else Smoothing.hard()

    def body(carry, _):
        tds_st, out_st = carry
        step = step_one(
            tds_st, geom, topology,
            subglottal_pressure_dPa=subglottal_pressure_dPa,
            soft_walls=soft_walls,
            turbulence_losses=turbulence_losses,
            dt=dt,
            smoothing=sm_static,
        )
        # M4 owns skin radiation: read FIRST_PHARYNX pressure from the *new*
        # state (TdsModel.cpp:899 reads ts->pressure after updateVariables ran).
        first_pharynx_p = step.state.section[FIRST_PHARYNX_SECTION, PRESSURE]
        new_out_st, audio_sample = output_step(
            out_st, coeffs,
            mouth_flow_cm3_s=step.mouth_flow,
            nostril_flow_cm3_s=step.nostril_flow,
            first_pharynx_pressure_dPa=first_pharynx_p,
        )
        return (step.state, new_out_st), audio_sample

    (final_tds, final_out), audio = jax.lax.scan(
        body, (tds_state, out_state), jnp.arange(n_samples)
    )
    return audio, final_tds, final_out


def synthesize_static_tube_streaming(
    geom: TubeGeometry,
    n_samples: int,
    *,
    topology: TubeTopology = TOPOLOGY,
    subglottal_pressure_dPa: float = 8000.0,
    soft_walls: bool = True,
    turbulence_losses: bool = True,
) -> tuple[
    Float[Array, "N"],   # audio
    Float[Array, "N"],   # mouth_flow
    Float[Array, "N"],   # nostril_flow
    Float[Array, "N"],   # skin_flow
]:
    """Same as `synthesize_static_tube` but also returns per-sample flows."""
    tds_state = TdsState.zeros()
    out_state = OutputState.zeros()
    coeffs = OutputCoeffs.default()

    def body(carry, _):
        tds_st, out_st = carry
        step = step_one(
            tds_st, geom, topology,
            subglottal_pressure_dPa=subglottal_pressure_dPa,
            soft_walls=soft_walls,
            turbulence_losses=turbulence_losses,
        )
        first_pharynx_p = step.state.section[FIRST_PHARYNX_SECTION, PRESSURE]
        # Pre-compute skin flow exactly as the output_step would, so we can
        # report it independently. We reuse the iir_step internals via a
        # second filter pass on a scratch state — but that double-passes; the
        # cleaner route is to mirror output_step inline.
        new_out_st, audio_sample = output_step(
            out_st, coeffs,
            mouth_flow_cm3_s=step.mouth_flow,
            nostril_flow_cm3_s=step.nostril_flow,
            first_pharynx_pressure_dPa=first_pharynx_p,
        )
        # Recompute the skin sample = new_out_st.skin_filter.out_buf[0] (= y[n]).
        skin_sample = new_out_st.skin_filter.out_buf[0]
        return (step.state, new_out_st), (
            audio_sample, step.mouth_flow, step.nostril_flow, skin_sample
        )

    _, (audio, mouth, nostril, skin) = jax.lax.scan(
        body, (tds_state, out_state), jnp.arange(n_samples)
    )
    return audio, mouth, nostril, skin


def _override_glottis_geom(
    geom: TubeGeometry,
    glottis_length_cm: Float[Array, "2"],
    glottis_area_cm2: Float[Array, "2"],
) -> TubeGeometry:
    """Replace tube sections 23/24 (glottis) with per-sample length+area from
    `GeometricGlottis.calc_geometry`. Walls are disabled on glottis sections in
    `_wall_alpha_beta` (alpha=beta=0), so wall_mass/resistance/stiffness on those
    sections don't enter the MNA assembly — we leave them at the frame's values.
    """
    idx = jnp.array([LOWER_GLOTTIS_SECTION, UPPER_GLOTTIS_SECTION])
    new_L = geom.length_cm.at[idx].set(glottis_length_cm)
    # Mirror the MIN_AREA clamp applied in `TubeGeometry.from_arrays`.
    from tractus_engine_advanced.tds.constants import MIN_AREA_CM2
    A_clamped = jnp.maximum(glottis_area_cm2, MIN_AREA_CM2)
    new_A = geom.area_cm2.at[idx].set(A_clamped)
    new_V = geom.volume_cm3.at[idx].set(new_L[idx] * new_A[idx])
    return eqx.tree_at(
        lambda g: (g.length_cm, g.area_cm2, g.volume_cm3),
        geom,
        (new_L, new_A, new_V),
    )


def synthesize_dynamic_tube(
    length_cm: Float[Array, "F 93"],
    area_cm2: Float[Array, "F 93"],
    wall_mass: Float[Array, "F 93"],
    wall_resistance: Float[Array, "F 93"],
    wall_stiffness: Float[Array, "F 93"],
    *,
    samples_per_frame: int,
    glottis_traj: Float[Array, "F 11"] | None = None,
    glottis_anatomy: Float[Array, "4"] | None = None,
    topology: TubeTopology = TOPOLOGY,
    soft_walls: bool = True,
    turbulence_losses: bool = True,
    initial_tds_state: TdsState | None = None,
    initial_output_state: OutputState | None = None,
    initial_glottis_state: gg.GeometricGlottisState | None = None,
    output_coeffs: OutputCoeffs | None = None,
    dt: float = TIMESTEP_S,
    interpolate_glottis_within_frame: bool = True,
    subglottal_pressure_dPa_legacy: float = 8000.0,
    frame_chunk_size: int = 1,
    smoothing: Smoothing | None = None,
    frication: Frication | None = None,
    place_index: jnp.ndarray | None = None,
    obstacle_proxy: jnp.ndarray | None = None,
    voicing: jnp.ndarray | None = None,
    frication_color: jnp.ndarray | None = None,
    noise_key: "jax.random.KeyArray | None" = None,
) -> tuple[
    Float[Array, "N"], TdsState, OutputState, gg.GeometricGlottisState
]:
    """Per-frame dynamic-tube synthesis with integrated GeometricGlottis.

    For F frames each holding the *tube* fixed for ``samples_per_frame`` audio
    samples, run TDS + output stage. The *glottis* state is advanced per
    sample using `GeometricGlottis`, which produces the per-sample length+area
    for tube sections 23 and 24 (overriding the frame's tube there). The
    subglottal-pressure source for the TDS step is `glottis_traj[:, PR]`
    (i.e., per-frame, optionally interpolated to per-sample).

    Glottis-param interpolation
    ---------------------------
    With ``interpolate_glottis_within_frame=True`` (default; matches upstream
    `Synthesizer::add`), within frame f sample i uses
    ``cp(i) = (1 - i/N)·prev_cp + (i/N)·glottis_traj[f]``, where ``prev_cp``
    is `glottis_traj[f-1]` (or `glottis_traj[0]` for frame 0). With
    ``False``, the param vector is held constant within a frame at
    ``glottis_traj[f]`` (matches the per-frame harness in `test_m5_parity`).

    Backward-compat: if ``glottis_traj`` is None, the GeometricGlottis path
    is bypassed entirely — sections 23/24 stay at the values in
    ``area_cm2[f, :]`` / ``length_cm[f, :]`` and the subglottal pressure
    source is held at the legacy default (`subglottal_pressure_dPa_legacy`).
    This preserves the pre-G3a M5 corpus semantics (frozen glottis area,
    constant `setPressureSource`).

    Returns ``(audio, final_tds_state, final_output_state, final_glottis_state)``
    with audio shape ``(F * samples_per_frame,)``.

    Backward-pass memory
    --------------------
    ``frame_chunk_size`` controls the rematerialization granularity of the
    *outer* per-frame scan for backward memory. The outer scan is split into
    ``ceil(F / K)`` chunks of K frames; each chunk's body (which itself runs
    an inner per-sample scan of length ``samples_per_frame``) is wrapped in
    ``jax.checkpoint(policy=nothing_saveable)`` so backward rebuilds it from
    the chunk-boundary carry rather than holding the full inner-scan tape
    in memory. The forward pass is bit-equal regardless of K.

    K=1 (default) gives the strongest memory bound — the only inner-scan
    activations live during backward are for one frame at a time, which is
    what brings the production ``T=118`` backward under the 2 GB / batch=4
    G2 budget. Larger K trades memory for fewer recompute steps; for the
    production short-window driver the default is correct.

    F need not be divisible by K — the input is right-padded by repeating
    the last frame (and last glottis-traj entry) up to ``ceil(F/K)·K``
    frames; trailing audio samples beyond ``F·samples_per_frame`` are
    trimmed before return.
    """
    if frame_chunk_size < 1:
        raise ValueError(f"frame_chunk_size must be >= 1; got {frame_chunk_size}")
    sm = smoothing if smoothing is not None else Smoothing.hard()
    # Track whether the caller passed explicit init state. When all three
    # state kwargs are None, we're starting a fresh render — apply the
    # frame-0 pressure ramp-in (see `quiescent_cp` below). When any explicit
    # state is supplied, we're continuing a streaming render — DON'T ramp,
    # the prior state already carries the steady-state context.
    starting_fresh = (
        initial_tds_state is None
        and initial_output_state is None
        and initial_glottis_state is None
    )
    tds_state = initial_tds_state if initial_tds_state is not None else TdsState.zeros()
    out_state = initial_output_state if initial_output_state is not None else OutputState.zeros()
    coeffs = output_coeffs if output_coeffs is not None else OutputCoeffs.default()

    # correctness/004: physically-derived per-section frication. ``frication``
    # config defaults to disabled (zero gain → zero monopole noise; the MNA
    # ``source_amp`` plumbing is bit-equal to pre-fric). When enabled the
    # body model is expected to emit ``place_index``, ``obstacle_proxy``,
    # ``voicing``, and (optionally) ``frication_color``; ``noise_key``
    # derives the deterministic-given-key noise pattern.
    fr_cfg = frication if frication is not None else Frication.disabled()
    fr_enabled = bool(fr_cfg.enabled)

    L = jnp.asarray(length_cm, dtype=DTYPE)
    A = jnp.asarray(area_cm2, dtype=DTYPE)
    Mw = jnp.asarray(wall_mass, dtype=DTYPE)
    Bw = jnp.asarray(wall_resistance, dtype=DTYPE)
    Kw = jnp.asarray(wall_stiffness, dtype=DTYPE)
    n_frames = L.shape[0]

    # Build per-frame frication-input arrays. When the corresponding kwarg
    # is None we synthesise a zero default; place_index defaults to
    # PLACE_NONE so any sections without explicit articulator-class
    # assignment emit no frication.
    if fr_enabled:
        if place_index is None:
            pi_arr = jnp.zeros((n_frames, 93), dtype=jnp.int32)
        else:
            pi_arr = jnp.asarray(place_index, dtype=jnp.int32)
            if pi_arr.shape != (n_frames, 93):
                raise ValueError(
                    f"place_index must be ({n_frames}, 93); got {pi_arr.shape}"
                )
        if obstacle_proxy is None:
            obs_arr = jnp.zeros((n_frames, 93), dtype=DTYPE)
        else:
            obs_arr = jnp.asarray(obstacle_proxy, dtype=DTYPE)
            if obs_arr.shape != (n_frames, 93):
                raise ValueError(
                    f"obstacle_proxy must be ({n_frames}, 93); got {obs_arr.shape}"
                )
        if voicing is None:
            v_arr = jnp.zeros((n_frames,), dtype=DTYPE)
        else:
            v_arr = jnp.asarray(voicing, dtype=DTYPE)
            if v_arr.shape != (n_frames,):
                raise ValueError(
                    f"voicing must be ({n_frames},); got {v_arr.shape}"
                )
        if frication_color is None:
            color_arr = jnp.zeros((n_frames, 93), dtype=DTYPE)
        else:
            color_arr = jnp.asarray(frication_color, dtype=DTYPE)
            if color_arr.shape != (n_frames, 93):
                raise ValueError(
                    f"frication_color must be ({n_frames}, 93); got {color_arr.shape}"
                )
        if noise_key is None:
            n_key = jax.random.key(0)
        else:
            n_key = noise_key
    else:
        pi_arr = jnp.zeros((n_frames, 93), dtype=jnp.int32)
        obs_arr = jnp.zeros((n_frames, 93), dtype=DTYPE)
        v_arr = jnp.zeros((n_frames,), dtype=DTYPE)
        color_arr = jnp.zeros((n_frames, 93), dtype=DTYPE)
        n_key = jax.random.key(0)

    use_glottis = glottis_traj is not None
    g_state = (
        initial_glottis_state if initial_glottis_state is not None else gg.init_state()
    )

    # correctness/004b: per-place PSD-shaping SOS table + filter state.
    # The PSD filter shapes the broadband-flat ``noise_sample`` output
    # into place-conditional spectra (sibilant 6-8 kHz peak for /s/,
    # 3 kHz peak for /ʃ/, low-pass for /h/, etc.). When frication is
    # disabled the table and state are unused, but we always build them
    # so downstream tracing is shape-stable.
    psd_sos_table = _frication_mod.place_psd_sos_table(
        sample_rate_hz=1.0 / float(dt)
    )
    psd_state_init = _frication_mod.init_psd_filter_state()

    # correctness/005 + 007: rematerialize the per-sample scan body AND the
    # outer per-frame chunk via `jax.checkpoint`. Without the outer wrap, XLA
    # materializes a per-sample-output Jacobian transpose with leading shape
    # ``(F, samples_per_frame, 97, 97)`` during the per-frame scan's backward,
    # which is ~3.9 GB at F=118 — exceeding the 16 GB GPU's static budget.
    # Chunking the outer scan into ``frame_chunk_size``-frame chunks bounds
    # that transpose to ``(K, samples_per_frame, 97, 97)``: at K=1 the leading
    # dim collapses entirely, bringing T=118 b=4 under the 2 GB G2 budget.
    _ckpt_policy = jax.checkpoint_policies.nothing_saveable

    K = int(frame_chunk_size)
    # Right-pad to a multiple of K by repeating the last frame; trim audio
    # at the end. ``n_pad`` may be 0.
    n_pad = (-n_frames) % K
    n_padded = n_frames + n_pad
    n_chunks = n_padded // K

    def _pad_repeat(x: jnp.ndarray) -> jnp.ndarray:
        if n_pad == 0:
            return x
        last = x[-1:]  # shape (1, ...)
        tail = jnp.broadcast_to(last, (n_pad,) + x.shape[1:])
        return jnp.concatenate([x, tail], axis=0)

    L = _pad_repeat(L)
    A = _pad_repeat(A)
    Mw = _pad_repeat(Mw)
    Bw = _pad_repeat(Bw)
    Kw = _pad_repeat(Kw)
    pi_padded = _pad_repeat(pi_arr)
    obs_padded = _pad_repeat(obs_arr)
    v_padded = _pad_repeat(v_arr)
    color_padded = _pad_repeat(color_arr)

    if use_glottis:
        gtraj = jnp.asarray(glottis_traj, dtype=DTYPE)
        if gtraj.shape != (n_frames, gg.NUM_CONTROL_PARAMS):
            raise ValueError(
                f"glottis_traj must be ({n_frames}, {gg.NUM_CONTROL_PARAMS}); got {gtraj.shape}"
            )
        gtraj = _pad_repeat(gtraj)
        static_params = jnp.asarray(
            glottis_anatomy if glottis_anatomy is not None else gg.STATIC_DEFAULT,
            dtype=DTYPE,
        )
        anatomy = gg.make_anatomy(
            static_params=static_params,
            sampling_rate_hz=1.0 / float(dt),
        )
        interp_within = bool(interpolate_glottis_within_frame)
        inv_N = 1.0 / float(samples_per_frame)

        def per_frame_body(carry, frame_inputs):
            tds_st, out_st, g_st, psd_st, prev_cp = carry
            (
                l_f, a_f, m_f, b_f, k_f, cur_cp,
                pi_f, obs_f, v_f, color_f, frame_key,
            ) = frame_inputs
            geom = TubeGeometry.from_arrays(
                length_cm=l_f, area_cm2=a_f,
                wall_mass=m_f, wall_resistance=b_f, wall_stiffness=k_f,
                smoothing=sm,
            )

            def per_sample_body(inner_carry, i):
                t_st, o_st, gs, psd_st = inner_carry
                if interp_within:
                    ratio = jnp.asarray(i, dtype=DTYPE) * inv_N
                    cp = (1.0 - ratio) * prev_cp + ratio * cur_cp
                else:
                    cp = cur_cp

                # (1) calc_geometry on CURRENT glottis state — mirrors
                # Synthesizer.cpp:222 (calcGeometry then getTubeData).
                length_g, area_g, _aux = gg.calc_geometry(cp, gs, anatomy, smoothing=sm)

                # (2) Override tube sections 23/24.
                geom_s = _override_glottis_geom(geom, length_g, area_g)

                # (3) Read the four pressures from the CURRENT TDS state.
                # Synthesizer.cpp:236-239 does this BEFORE advancing TDS this
                # sample, so it sees the last sample's post-step pressures —
                # the one-sample algebraic-loop break (SCOPE.md §3.C row N).
                sec_p = t_st.section[:, PRESSURE]
                pressures = jnp.stack([
                    sec_p[LAST_TRACHEA_SECTION],
                    sec_p[LOWER_GLOTTIS_SECTION],
                    sec_p[UPPER_GLOTTIS_SECTION],
                    sec_p[FIRST_PHARYNX_SECTION],
                ])

                # (4) Advance glottis state (Synthesizer.cpp:241).
                new_gs = gg.inc_time(gs, cp, pressures, dt, anatomy)

                # (5a) Frication monopole noise. Pre-step pressures + current
                # geometry feed the physical-gain derivation, then
                # ``noise_sample`` produces the deterministic per-sample
                # turbulent flow source. When ``fr_enabled`` is False this
                # is a zero array and bit-equal to the pre-fric forward.
                if fr_enabled:
                    sample_key = jax.random.fold_in(frame_key, i)
                    fr_gain = _frication_mod.gain_with_color(
                        constriction_area_cm2=geom_s.area_cm2,
                        pressure_drop_dPa=_frication_mod.derive_pressure_drop_dPa(sec_p),
                        place_index=pi_f,
                        voicing=v_f,
                        obstacle_proxy=obs_f,
                        color=color_f,
                        cfg=fr_cfg,
                    )
                    monopole_amp = _frication_mod.noise_sample(sample_key, fr_gain)
                    # correctness/004b: place-conditional PSD shaping on the
                    # mouth-radiate path. The broadband ``monopole_amp`` is
                    # grouped by place class (/s/-alveolar, /ʃ/-post-alveolar,
                    # etc.), each group sum is run through its own SOS biquad
                    # cascade, and the filtered streams are summed to one
                    # scalar ``extra_mouth`` value. This produces the 6-8 kHz
                    # sibilant peaks for /s/ /z/, the 3 kHz peak for /ʃ/ /ʒ/,
                    # and the low-pass envelope for /h/ aspiration that the
                    # 4-8 kHz frication-band gate measures. The MNA monopole
                    # path (post-solve net-flow injection in
                    # ``update_variables``) still carries the unfiltered
                    # noise into the tract — that's what gives the per-frame
                    # phone-aligned RMS bumps; the place-conditional shaping
                    # gives the spectral-class identity.
                    #
                    # Glottal aspiration sections (23-24) and trachea/sinus
                    # are excluded — direct mouth radiation only models the
                    # supraglottal contribution. /h/ does emit at glottal
                    # sections but the place_index there is PLACE_GLOTTAL
                    # whose biquad is configured for the appropriate
                    # low-pass envelope; we keep glottis-site noise out of
                    # this radiate path so it doesn't double-count via the
                    # glottal source's own radiation route.
                    sup_lo = FIRST_PHARYNX_SECTION
                    sup_hi = LAST_MOUTH_SECTION + 1
                    sup_amp = monopole_amp[sup_lo:sup_hi]
                    sup_pi = pi_f[sup_lo:sup_hi]
                    # Per-place sum — segment_sum across NUM_PLACES classes.
                    per_place_input = jax.ops.segment_sum(
                        sup_amp,
                        jnp.clip(sup_pi, 0, _frication_mod.NUM_PLACES - 1),
                        num_segments=_frication_mod.NUM_PLACES,
                    )
                    extra_mouth, new_psd_st = _frication_mod.psd_filter_step(
                        per_place_input, psd_st, psd_sos_table,
                    )
                else:
                    monopole_amp = None
                    extra_mouth = jnp.asarray(0.0, dtype=DTYPE)
                    new_psd_st = psd_st

                # (5) TDS step. Subglottal pressure source = cp[PR].
                step = step_one(
                    t_st, geom_s, topology,
                    subglottal_pressure_dPa=cp[gg.PR],
                    soft_walls=soft_walls,
                    turbulence_losses=turbulence_losses,
                    dt=dt,
                    smoothing=sm,
                    monopole_source_amp=monopole_amp,
                )
                first_pharynx_p = step.state.section[FIRST_PHARYNX_SECTION, PRESSURE]
                new_o_st, audio_sample = output_step(
                    o_st, coeffs,
                    mouth_flow_cm3_s=step.mouth_flow + extra_mouth,
                    nostril_flow_cm3_s=step.nostril_flow,
                    first_pharynx_pressure_dPa=first_pharynx_p,
                )
                return (step.state, new_o_st, new_gs, new_psd_st), audio_sample

            per_sample_body_ck = jax.checkpoint(
                per_sample_body, policy=_ckpt_policy
            )
            (new_tds, new_out, new_gs, new_psd_st), frame_audio = jax.lax.scan(
                per_sample_body_ck,
                (tds_st, out_st, g_st, psd_st),
                jnp.arange(samples_per_frame),
            )
            return (new_tds, new_out, new_gs, new_psd_st, cur_cp), frame_audio

        # Onset-transient fix (correctness/004c): when starting from a fresh
        # zero-initialized state, seed `prev_cp` with a "quiescent" copy of
        # `gtraj[0]` that has lung pressure (`PR`) zeroed. The per-sample
        # interpolation `(1-ratio)*prev_cp + ratio*cur_cp` inside frame 0
        # then ramps PR linearly from 0 to `gtraj[0, PR]` over the first
        # `samples_per_frame` samples (~10 ms at 44.1 kHz / spf=441), which
        # eliminates the audible click that an instantaneous 8 kPa step
        # produced. Streaming-continuation callers (those passing explicit
        # initial_*_state) keep the historical behavior since their prior
        # state already carries the steady-state context.
        if starting_fresh:
            quiescent_cp = gtraj[0].at[gg.PR].set(jnp.asarray(0.0, dtype=DTYPE))
        else:
            quiescent_cp = gtraj[0]
        init_carry = (tds_state, out_state, g_state, psd_state_init, quiescent_cp)

        # Reshape per-frame inputs to (n_chunks, K, ...) and run an outer
        # scan over chunks. The chunk body runs the original per-frame scan
        # over K frames; wrapping it in `jax.checkpoint` is what bounds the
        # backward-pass tape footprint to a single chunk.
        def _chunk(x):
            return x.reshape((n_chunks, K) + x.shape[1:])

        L_c = _chunk(L)
        A_c = _chunk(A)
        Mw_c = _chunk(Mw)
        Bw_c = _chunk(Bw)
        Kw_c = _chunk(Kw)
        gtraj_c = _chunk(gtraj)
        pi_c = _chunk(pi_padded)
        obs_c = _chunk(obs_padded)
        v_c = _chunk(v_padded)
        color_c = _chunk(color_padded)
        # Per-frame keys derived from the master noise_key by folding in
        # the (padded) frame index. Deterministic in (noise_key, frame_idx);
        # the per-sample fold-in inside per_sample_body adds the sample
        # index, giving a stable (key, frame, sample) triple.
        frame_indices = jnp.arange(n_padded)
        per_frame_keys = jax.vmap(lambda i: jax.random.fold_in(n_key, i))(frame_indices)
        keys_c = per_frame_keys.reshape((n_chunks, K) + per_frame_keys.shape[1:])

        def per_chunk_body(carry, chunk_inputs):
            (new_tds, new_out, new_gs, new_psd_st, last_cp), chunk_audio = jax.lax.scan(
                per_frame_body, carry, chunk_inputs
            )
            return (new_tds, new_out, new_gs, new_psd_st, last_cp), chunk_audio

        per_chunk_body_ck = jax.checkpoint(per_chunk_body, policy=_ckpt_policy)

        (final_tds, final_out, final_gs, _, _), audio_blocks = jax.lax.scan(
            per_chunk_body_ck,
            init_carry,
            (L_c, A_c, Mw_c, Bw_c, Kw_c, gtraj_c, pi_c, obs_c, v_c, color_c, keys_c),
        )
        # audio_blocks shape: (n_chunks, K, samples_per_frame).
        audio = audio_blocks.reshape(-1)
        if n_pad > 0:
            audio = audio[: n_frames * samples_per_frame]
        return audio, final_tds, final_out, final_gs

    # ---------- legacy / pre-G3a path: no GeometricGlottis ----------
    psub = float(subglottal_pressure_dPa_legacy)

    def per_frame_body_legacy(carry, frame_inputs):
        tds_st, out_st = carry
        l_f, a_f, m_f, b_f, k_f = frame_inputs
        geom = TubeGeometry.from_arrays(
            length_cm=l_f, area_cm2=a_f,
            wall_mass=m_f, wall_resistance=b_f, wall_stiffness=k_f,
            smoothing=sm,
        )

        def per_sample_body(inner_carry, _):
            t_st, o_st = inner_carry
            step = step_one(
                t_st, geom, topology,
                subglottal_pressure_dPa=psub,
                soft_walls=soft_walls,
                turbulence_losses=turbulence_losses,
                dt=dt,
                smoothing=sm,
            )
            first_pharynx_p = step.state.section[FIRST_PHARYNX_SECTION, PRESSURE]
            new_o_st, audio_sample = output_step(
                o_st, coeffs,
                mouth_flow_cm3_s=step.mouth_flow,
                nostril_flow_cm3_s=step.nostril_flow,
                first_pharynx_pressure_dPa=first_pharynx_p,
            )
            return (step.state, new_o_st), audio_sample

        per_sample_body_ck = jax.checkpoint(
            per_sample_body, policy=_ckpt_policy
        )
        (new_tds, new_out), frame_audio = jax.lax.scan(
            per_sample_body_ck, (tds_st, out_st), jnp.arange(samples_per_frame)
        )
        return (new_tds, new_out), frame_audio

    def _chunk(x):
        return x.reshape((n_chunks, K) + x.shape[1:])

    L_c = _chunk(L)
    A_c = _chunk(A)
    Mw_c = _chunk(Mw)
    Bw_c = _chunk(Bw)
    Kw_c = _chunk(Kw)

    def per_chunk_body_legacy(carry, chunk_inputs):
        (new_tds, new_out), chunk_audio = jax.lax.scan(
            per_frame_body_legacy, carry, chunk_inputs
        )
        return (new_tds, new_out), chunk_audio

    per_chunk_body_ck = jax.checkpoint(per_chunk_body_legacy, policy=_ckpt_policy)

    (final_tds, final_out), audio_blocks = jax.lax.scan(
        per_chunk_body_ck, (tds_state, out_state), (L_c, A_c, Mw_c, Bw_c, Kw_c)
    )
    audio = audio_blocks.reshape(-1)
    if n_pad > 0:
        audio = audio[: n_frames * samples_per_frame]
    return audio, final_tds, final_out, g_state
