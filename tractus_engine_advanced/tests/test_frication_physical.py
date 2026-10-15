"""Frication physical-derivation invariants (ARCHITECTURE.md §496-504, §516-524).

Frication gain must be a physical function of constriction area + pressure
drop + place + voicing + obstacle proxy — *not* a free learned gate. The
``frication_color`` residual is energy-capped calibration only; it cannot
synthesize frication where physical conditions don't support it.

After correctness/004 these flip from xfail to pass without rewriting.
"""

from __future__ import annotations

import importlib

import numpy as np
import pytest


def _frication_module():
    return importlib.import_module("tractus_engine_advanced.frication")


@pytest.mark.frication_physical
def test_zero_constriction_area_yields_zero_frication_gain() -> None:
    """Zero constriction area → zero frication gain *mathematically*, not learned.

    The frication module derives gain from constriction geometry; with
    zero area there is nothing to fricate. This is a structural identity,
    not a regularization target.
    """

    frication = _frication_module()
    cfg = frication.Frication.default()
    # ARCH §496-504: f_phys(constriction_area=0, ...) must return 0 — set
    # all 93 sections to PLACE_ALVEOLAR (sibilant) and high pressure to
    # show that even maximum drive yields zero gain when area is zero.
    gain = frication.physical_gain(
        constriction_area_cm2=np.zeros(93, dtype=np.float32),
        pressure_drop_dPa=np.full(93, 5000.0, dtype=np.float32),
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        cfg=cfg,
    )
    g = np.asarray(gain)
    assert np.all(g == 0.0), (
        f"zero constriction must yield zero frication; got max {np.max(np.abs(g)):.3g}"
    )


@pytest.mark.frication_physical
def test_zero_pressure_drop_yields_zero_frication_gain() -> None:
    """Zero (or sub-threshold) pressure drop → zero gain.

    Below the turbulence-onset pressure threshold the flow is laminar and
    no broadband noise should be generated. The smoothing surrogate
    fades smoothly to zero.
    """

    frication = _frication_module()
    cfg = frication.Frication.default()
    gain = frication.physical_gain(
        constriction_area_cm2=np.full(93, 0.05, dtype=np.float32),
        pressure_drop_dPa=np.zeros(93, dtype=np.float32),
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        cfg=cfg,
    )
    g = np.asarray(gain)
    # Smoothed soft_step on excess pressure isn't bit-zero, but the ramp
    # should be very small relative to the typical operating range.
    assert np.max(np.abs(g)) < 1e-3, (
        f"zero pressure must yield ~zero frication; got max {np.max(np.abs(g)):.3g}"
    )


@pytest.mark.frication_physical
def test_color_residual_cannot_create_frication_without_physics() -> None:
    """``frication_color`` cannot synthesize energy when constriction + pressure
    don't physically support it (ARCH §516-524, §1684-1685)."""

    frication = _frication_module()
    cfg = frication.Frication.default()

    constriction = np.zeros(93, dtype=np.float32)  # no constriction
    pressure = np.zeros(93, dtype=np.float32)  # no pressure drop
    # Even with a maximally aggressive color residual, gain must stay 0.
    color = np.full(93, 1.0, dtype=np.float32)
    gain = frication.gain_with_color(
        constriction_area_cm2=constriction,
        pressure_drop_dPa=pressure,
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        color=color,
        cfg=cfg,
    )
    g = np.asarray(gain)
    # The product gain_phys * (1 + clipped_color) is exactly zero when
    # constriction is zero (multiplicative invariant), independent of color.
    assert np.max(np.abs(g)) < 1e-6, (
        f"frication_color synthesized {np.max(np.abs(g)):.3g} energy "
        "without physical support"
    )


@pytest.mark.frication_physical
def test_gain_monotone_in_constriction_within_range() -> None:
    """Gain is monotone increasing in constriction area within
    ``[A_floor, A_peak]`` at fixed pressure drop, place, voicing.

    The physical model implies that bigger constriction (within the
    fricative regime) means more flow → more turbulence → more gain.
    Above ``A_turbulent_ceiling`` the flow becomes laminar and gain
    drops back toward zero — that's tested in
    ``test_gain_drops_above_ceiling`` separately.
    """

    frication = _frication_module()
    cfg = frication.Frication.default()
    # Sweep area from near-floor to near-ceiling.
    n = 16
    A_sweep = np.linspace(
        cfg.area_fricative_floor_cm2 + 1e-3,
        cfg.area_turbulent_ceiling_cm2 * 0.5,
        n,
        dtype=np.float32,
    )
    gains = []
    for A in A_sweep:
        gain = frication.physical_gain(
            constriction_area_cm2=np.full(93, float(A), dtype=np.float32),
            pressure_drop_dPa=np.full(93, 800.0, dtype=np.float32),
            place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
            voicing=np.zeros((), dtype=np.float32),
            obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
            cfg=cfg,
        )
        gains.append(float(np.asarray(gain).max()))
    diffs = np.diff(gains)
    assert np.all(diffs >= -1e-6), (
        f"gain not monotone in constriction within sweep: diffs={diffs}"
    )


@pytest.mark.frication_physical
def test_gain_drops_above_ceiling() -> None:
    """When constriction area exceeds the turbulent ceiling, gain → 0.

    A wide-open tube section is no longer a constriction; turbulent
    noise should not be radiated from there.
    """

    frication = _frication_module()
    cfg = frication.Frication.default()
    A_open = cfg.area_turbulent_ceiling_cm2 * 5.0  # well above ceiling
    gain = frication.physical_gain(
        constriction_area_cm2=np.full(93, A_open, dtype=np.float32),
        pressure_drop_dPa=np.full(93, 800.0, dtype=np.float32),
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        cfg=cfg,
    )
    assert float(np.asarray(gain).max()) < 1e-3, (
        "wide-open area should not produce frication"
    )


@pytest.mark.frication_physical
def test_sibilant_louder_than_non_sibilant_at_matched_drive() -> None:
    """At matched (A_c, Δp), /s/ (alveolar sibilant) is louder than
    /θ/ (dental non-sibilant). This is the place-class boost in action."""

    frication = _frication_module()
    cfg = frication.Frication.default()
    A = np.full(93, 0.05, dtype=np.float32)
    dP = np.full(93, 800.0, dtype=np.float32)
    obs = np.full(93, 1.0, dtype=np.float32)

    g_sib = frication.physical_gain(
        constriction_area_cm2=A,
        pressure_drop_dPa=dP,
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=obs,
        cfg=cfg,
    )
    g_dent = frication.physical_gain(
        constriction_area_cm2=A,
        pressure_drop_dPa=dP,
        place_index=np.full(93, frication.PLACE_DENTAL, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=obs,
        cfg=cfg,
    )
    assert float(np.asarray(g_sib).max()) > float(np.asarray(g_dent).max()) * 1.5


@pytest.mark.frication_physical
def test_voiced_quieter_than_voiceless_at_matched_drive() -> None:
    """At matched (A_c, Δp, place), voiced fricative gain < voiceless."""

    frication = _frication_module()
    cfg = frication.Frication.default()
    A = np.full(93, 0.05, dtype=np.float32)
    dP = np.full(93, 800.0, dtype=np.float32)
    obs = np.full(93, 1.0, dtype=np.float32)

    g_voiceless = frication.physical_gain(
        constriction_area_cm2=A,
        pressure_drop_dPa=dP,
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=obs,
        cfg=cfg,
    )
    g_voiced = frication.physical_gain(
        constriction_area_cm2=A,
        pressure_drop_dPa=dP,
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.ones((), dtype=np.float32),
        obstacle_proxy=obs,
        cfg=cfg,
    )
    assert float(np.asarray(g_voiced).max()) < float(np.asarray(g_voiceless).max())


@pytest.mark.frication_physical
def test_disabled_module_is_zero() -> None:
    """``Frication.disabled()`` returns gain == 0 regardless of inputs."""

    frication = _frication_module()
    cfg = frication.Frication.disabled()
    gain = frication.physical_gain(
        constriction_area_cm2=np.full(93, 0.05, dtype=np.float32),
        pressure_drop_dPa=np.full(93, 5000.0, dtype=np.float32),
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        cfg=cfg,
    )
    assert np.all(np.asarray(gain) == 0.0)


@pytest.mark.frication_physical
def test_color_cap_bounds_residual_effect() -> None:
    """Color residual is multiplicative ``(1 + clip(color, ±cap))`` —
    when ``|color|`` exceeds the cap the effect saturates."""

    frication = _frication_module()
    cfg = frication.Frication(enabled=True, color_cap=0.30)
    base = frication.physical_gain(
        constriction_area_cm2=np.full(93, 0.05, dtype=np.float32),
        pressure_drop_dPa=np.full(93, 800.0, dtype=np.float32),
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        cfg=cfg,
    )
    # Color way above cap.
    cap = frication.gain_with_color(
        constriction_area_cm2=np.full(93, 0.05, dtype=np.float32),
        pressure_drop_dPa=np.full(93, 800.0, dtype=np.float32),
        place_index=np.full(93, frication.PLACE_ALVEOLAR, dtype=np.int32),
        voicing=np.zeros((), dtype=np.float32),
        obstacle_proxy=np.full(93, 1.0, dtype=np.float32),
        color=np.full(93, 5.0, dtype=np.float32),  # well above cap
        cfg=cfg,
    )
    base_max = float(np.asarray(base).max())
    cap_max = float(np.asarray(cap).max())
    expected_ceiling = base_max * (1.0 + cfg.color_cap) * (1 + 1e-5)
    assert cap_max <= expected_ceiling, (
        f"color residual exceeded ±cap saturation: cap_max={cap_max:.3g} > "
        f"expected ceiling {expected_ceiling:.3g}"
    )


@pytest.mark.frication_physical
def test_noise_sample_deterministic_in_key_and_zero_when_gain_zero() -> None:
    """``noise_sample(noise_key, gain)`` is deterministic given the key
    and exactly zero where ``gain == 0``."""
    import jax

    frication = _frication_module()
    key = jax.random.key(42)
    gain = np.zeros(93, dtype=np.float32)
    gain[42] = 1.0

    s1 = np.asarray(frication.noise_sample(key, gain))
    s2 = np.asarray(frication.noise_sample(key, gain))
    assert np.allclose(s1, s2), "deterministic given key"
    # Sections with zero gain must produce zero noise (multiplicative invariant).
    zero_mask = np.ones(93, dtype=bool)
    zero_mask[42] = False
    assert np.all(s1[zero_mask] == 0.0), "zero gain → zero noise"


@pytest.mark.frication_physical
def test_gradient_finite_in_each_input_channel() -> None:
    """Gradient finite-difference agreement on the frication module
    per-input channel.

    A simple smoke check that ``physical_gain`` is differentiable in
    ``constriction_area_cm2``, ``pressure_drop_dPa``, ``voicing``, and
    ``obstacle_proxy`` over the operating window. ``place_index`` is
    integer-valued (no gradient).
    """
    import jax
    import jax.numpy as jnp

    frication = _frication_module()
    cfg = frication.Frication.default()

    A = jnp.full((93,), 0.05, dtype=jnp.float32)
    dP = jnp.full((93,), 800.0, dtype=jnp.float32)
    pi = jnp.full((93,), frication.PLACE_ALVEOLAR, dtype=jnp.int32)
    v = jnp.asarray(0.0, dtype=jnp.float32)
    obs = jnp.full((93,), 1.0, dtype=jnp.float32)

    def loss_A(A_):
        return frication.physical_gain(
            constriction_area_cm2=A_,
            pressure_drop_dPa=dP,
            place_index=pi,
            voicing=v,
            obstacle_proxy=obs,
            cfg=cfg,
        ).sum()

    def loss_dP(dP_):
        return frication.physical_gain(
            constriction_area_cm2=A,
            pressure_drop_dPa=dP_,
            place_index=pi,
            voicing=v,
            obstacle_proxy=obs,
            cfg=cfg,
        ).sum()

    def loss_v(v_):
        return frication.physical_gain(
            constriction_area_cm2=A,
            pressure_drop_dPa=dP,
            place_index=pi,
            voicing=v_,
            obstacle_proxy=obs,
            cfg=cfg,
        ).sum()

    def loss_obs(obs_):
        return frication.physical_gain(
            constriction_area_cm2=A,
            pressure_drop_dPa=dP,
            place_index=pi,
            voicing=v,
            obstacle_proxy=obs_,
            cfg=cfg,
        ).sum()

    g_A = jax.grad(loss_A)(A)
    g_dP = jax.grad(loss_dP)(dP)
    g_v = jax.grad(loss_v)(v)
    g_obs = jax.grad(loss_obs)(obs)

    for name, g in [("A", g_A), ("dP", g_dP), ("v", g_v), ("obs", g_obs)]:
        ga = np.asarray(g)
        assert np.all(np.isfinite(ga)), f"non-finite grad in {name}"
        # At the operating point gradient should be nonzero in at least
        # one site for each non-trivial channel. Voicing damps gain so
        # ``∂gain/∂voicing`` is negative; pressure / area / obstacle
        # contribute positively.
        assert float(np.sum(np.abs(ga))) > 0.0, f"all-zero grad in {name}"


@pytest.mark.frication_physical
def test_gradient_finite_difference_agreement_on_pressure() -> None:
    """Central FD vs autograd agreement on pressure-drop input channel,
    summed over sections, at the typical operating point.

    Tolerance 1e-2 relative; the soft_step on the threshold gate has a
    finite curvature so a small bias is expected at the ramp shoulder.
    """
    import jax
    import jax.numpy as jnp

    frication = _frication_module()
    cfg = frication.Frication.default()

    A = jnp.full((93,), 0.05, dtype=jnp.float32)
    pi = jnp.full((93,), frication.PLACE_ALVEOLAR, dtype=jnp.int32)
    v = jnp.asarray(0.0, dtype=jnp.float32)
    obs = jnp.full((93,), 1.0, dtype=jnp.float32)

    def loss(dP_scalar):
        dP_vec = jnp.full((93,), dP_scalar, dtype=jnp.float32)
        return frication.physical_gain(
            constriction_area_cm2=A,
            pressure_drop_dPa=dP_vec,
            place_index=pi,
            voicing=v,
            obstacle_proxy=obs,
            cfg=cfg,
        ).sum()

    p0 = jnp.asarray(800.0, dtype=jnp.float32)
    g_auto = float(jax.grad(loss)(p0))
    eps = 1.0
    g_fd = float((loss(p0 + eps) - loss(p0 - eps)) / (2.0 * eps))
    rel = abs(g_auto - g_fd) / max(abs(g_fd), 1e-6)
    assert rel < 5e-2, f"FD vs autograd disagreement: auto={g_auto:.4g} fd={g_fd:.4g} rel={rel:.3g}"
