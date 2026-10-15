"""correctness/003 — soft-mode test slice.

Exercises the Smoothing config + soft-surrogate sites on a small
representative seed set so the gate runs in single-digit seconds. The
full 10K-seed sweep + per-module FD report lives in
``scripts/m6_soft_sweep.py``; this file verifies the same primitives
on a tiny budget so it can stay in default ``pytest`` regression CI
once correctness/003 closes.

Acceptance gates mirrored from the task file:

* NaN-free / Inf-free forward across a small batch under ``default_soft``.
* Per-module FD agreement (smaller N here; 10 samples each) within
  fp32 tolerance.
* τ → 0 forward within 1% RMS-relative of hard.
"""
from __future__ import annotations

import os

# Pin fp32 BEFORE any TEA / JAX import — TEA's precision policy reads it
# at module import.
os.environ.setdefault("TEA_PRECISION", "fp32")

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from tractus_engine_advanced import Smoothing
from tractus_engine_advanced import glottis_geometric as gg
from tractus_engine_advanced import output_iir as iir
from tractus_engine_advanced.tds.constants import MIN_AREA_CM2
from tractus_engine_advanced.tds.mna import TubeGeometry
from tractus_engine_advanced.tds.state import TdsState
from tractus_engine_advanced.tds.step import step_one
from tractus_engine_advanced.tds.tube_topology import TOPOLOGY


pytestmark = [pytest.mark.m6_soft]


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(abs(a), abs(b), 1e-9)


# ---------------------------------------------------------------------------
# Surrogate primitives
# ---------------------------------------------------------------------------


def test_smoothing_hard_disables_soft():
    s = Smoothing.hard()
    assert s.enabled is False


def test_smoothing_default_soft_has_positive_taus():
    s = Smoothing.default_soft()
    assert s.enabled is True
    assert s.tau_relu > 0.0
    assert s.tau_segment > 0.0
    assert s.tau_blend > 0.0
    assert s.tau_area_clamp > 0.0
    assert s.tau_bernoulli > 0.0


# ---------------------------------------------------------------------------
# Glottis — calc_geometry
# ---------------------------------------------------------------------------


def test_glottis_soft_forward_finite():
    """16 random in-envelope cp samples produce finite area & length under
    ``Smoothing.default_soft()``."""
    anatomy = gg.make_anatomy()
    state = gg.init_state()
    smoothing = Smoothing.default_soft()
    rng = np.random.default_rng(42)
    glottis_min = np.asarray(gg.CONTROL_MIN, dtype=np.float32)
    glottis_max = np.asarray(gg.CONTROL_MAX, dtype=np.float32)
    for _ in range(16):
        cp = (glottis_min + (glottis_max - glottis_min) * rng.random(11).astype(np.float32))
        cp[gg.F0] = float(rng.uniform(60.0, 400.0))
        cp[gg.PR] = float(rng.uniform(2000.0, 14000.0))
        length, area, _ = gg.calc_geometry(jnp.asarray(cp), state, anatomy, smoothing=smoothing)
        assert bool(jnp.all(jnp.isfinite(length)))
        assert bool(jnp.all(jnp.isfinite(area)))


def test_glottis_fd_agreement_default_soft():
    """jax.grad of Σ area_cm2 w.r.t. cp matches scaled-axis FD within fp32 tol.
    10 random in-envelope inputs, accept if ≥ 9/10 below 1e-2 rel err.
    """
    anatomy = gg.make_anatomy()
    state = gg.init_state()
    smoothing = Smoothing.default_soft()
    glottis_min = np.asarray(gg.CONTROL_MIN, dtype=np.float32)
    glottis_max = np.asarray(gg.CONTROL_MAX, dtype=np.float32)
    cp_scale = (glottis_max - glottis_min) * 0.5

    def scalar(cp):
        _, area, _ = gg.calc_geometry(cp, state, anatomy, smoothing=smoothing)
        return jnp.sum(area)

    rng = np.random.default_rng(43)
    eps = 1e-2
    n_pass = 0
    n_total = 10
    for _ in range(n_total):
        cp = (glottis_min + (glottis_max - glottis_min) * rng.random(11).astype(np.float32))
        cp[gg.F0] = float(rng.uniform(60.0, 400.0))
        cp[gg.PR] = float(rng.uniform(2000.0, 14000.0))
        g = np.asarray(jax.grad(scalar)(jnp.asarray(cp)))
        v = rng.standard_normal(11).astype(np.float32)
        v /= np.linalg.norm(v)
        step = (cp_scale * v) * eps
        analytic = float(np.sum(g * (cp_scale * v)))
        f_p = float(scalar(jnp.asarray(cp + step)))
        f_m = float(scalar(jnp.asarray(cp - step)))
        fd = (f_p - f_m) / (2.0 * eps)
        if _rel_err(analytic, fd) < 1e-2:
            n_pass += 1
    assert n_pass >= 9, f"glottis FD: only {n_pass}/{n_total} below 1e-2 rel err"


# ---------------------------------------------------------------------------
# TDS step
# ---------------------------------------------------------------------------


def test_tds_step_soft_forward_finite():
    """One TDS step with default-soft on a perturbed default tube produces
    finite section_pressure / branch_current."""
    pytest.importorskip("scipy")
    smoothing = Smoothing.default_soft()
    rng = np.random.default_rng(44)
    # 93-section default-ish tube. We don't import the controller here; build
    # a reasonable uniform tube as a stand-in (this is the engine-side test).
    L = np.full(93, 0.4, dtype=np.float32)
    A = np.full(93, 1.0, dtype=np.float32)
    Mw = np.full(93, 1.5, dtype=np.float32)
    Bw = np.full(93, 8000.0, dtype=np.float32)
    Kw = np.full(93, 1e6, dtype=np.float32)
    state = TdsState.zeros()
    for _ in range(8):
        A_p = np.maximum(A * (1.0 + rng.uniform(-0.3, 0.3, 93)), 3.0 * MIN_AREA_CM2).astype(np.float32)
        geom = TubeGeometry.from_arrays(L, A_p, Mw, Bw, Kw, smoothing=smoothing)
        out = step_one(
            state, geom, TOPOLOGY,
            subglottal_pressure_dPa=8000.0,
            smoothing=smoothing,
        )
        assert bool(jnp.all(jnp.isfinite(out.section_pressure_dPa)))
        assert bool(jnp.all(jnp.isfinite(out.branch_current_cm3_s)))


# ---------------------------------------------------------------------------
# Output step (linear; no soft surrogate)
# ---------------------------------------------------------------------------


def test_output_step_fd_agreement():
    coeffs = iir.OutputCoeffs.default()
    state = iir.OutputState.zeros()
    mouth = jnp.asarray(0.0, dtype=jnp.float32)
    nostril = jnp.asarray(0.0, dtype=jnp.float32)

    def scalar(p):
        _, audio = iir.output_step(state, coeffs, mouth, nostril, p)
        return audio.reshape(())

    rng = np.random.default_rng(45)
    eps = 1.0
    n_pass = 0
    for _ in range(10):
        p = float(rng.uniform(-1000.0, 1000.0))
        p_j = jnp.asarray(p, dtype=jnp.float32)
        g = float(jax.grad(scalar)(p_j))
        f_p = float(scalar(p_j + eps))
        f_m = float(scalar(p_j - eps))
        fd = (f_p - f_m) / (2.0 * eps)
        if _rel_err(g, fd) < 1e-2:
            n_pass += 1
    assert n_pass >= 9, f"output_step FD: {n_pass}/10"


# ---------------------------------------------------------------------------
# τ → 0 identity property — soft surrogates reduce to hard
# ---------------------------------------------------------------------------


def test_glottis_soft_reduces_to_hard_at_small_tau():
    """At τ = 1e-6, calc_geometry's soft path matches the hard path on
    a typical operating point within fp32 noise (~1% relative)."""
    anatomy = gg.make_anatomy()
    state = gg.init_state()
    cp = jnp.asarray(
        [120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0],
        dtype=jnp.float32,
    )
    _, area_hard, _ = gg.calc_geometry(cp, state, anatomy, smoothing=Smoothing.hard())
    near_zero = Smoothing(
        enabled=True,
        tau_relu=1e-6, tau_segment=1e-6, tau_blend=1e-6,
        tau_area_clamp=1e-6, tau_bernoulli=1e-6,
    )
    _, area_soft, _ = gg.calc_geometry(cp, state, anatomy, smoothing=near_zero)
    diff = float(jnp.max(jnp.abs(area_hard - area_soft)))
    rel = diff / max(float(jnp.max(jnp.abs(area_hard))), 1e-9)
    # 1% relative is the documented identity bar (see m6_soft_sweep
    # docstring); fp32 cancellation prevents tighter than ~1e-3 across
    # multi-frame chains, so on a single-step probe ≪ 1e-2 is expected.
    assert rel < 1e-2, f"glottis τ→0 identity: rel={rel}"


def test_tube_geometry_soft_area_clamp_reduces_to_hard():
    A = np.array([1.0, 1e-3, 5e-5, 0.5, 1e-4, 2.0], dtype=np.float32)
    L = np.full_like(A, 0.4)
    Mw = np.full_like(A, 1.5)
    Bw = np.full_like(A, 8000.0)
    Kw = np.full_like(A, 1e6)
    # hard
    g_hard = TubeGeometry.from_arrays(L, A, Mw, Bw, Kw, smoothing=Smoothing.hard())
    # τ very small: the softplus floor approaches max(A, MIN_AREA_CM2).
    near_zero = Smoothing(
        enabled=True, tau_area_clamp=1e-9,
        tau_relu=0., tau_segment=0., tau_blend=0., tau_bernoulli=0.,
    )
    g_soft = TubeGeometry.from_arrays(L, A, Mw, Bw, Kw, smoothing=near_zero)
    diff = float(jnp.max(jnp.abs(g_hard.area_cm2 - g_soft.area_cm2)))
    assert diff < 1e-5, f"area clamp τ→0 diff={diff}"


# ---------------------------------------------------------------------------
# Master rollup gate (the m6_soft slice deliberately reuses the unit
# checks above so a `pytest -m m6_soft` invocation reports each gate
# individually).
# ---------------------------------------------------------------------------
