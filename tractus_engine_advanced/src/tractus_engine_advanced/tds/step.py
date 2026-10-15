"""Glue: one full TDS step (prepare → assemble → solve → update → outputs).

Mirrors `TdsModel::proceedTimeStep` (`TdsModel.cpp:849`). Skin radiation flow
defers to M4 (Chebyshev IIR port); for M2 we report 0.0 there.
"""

from __future__ import annotations

import os
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from tractus_engine_advanced.precision import DTYPE
from tractus_engine_advanced.smoothing import Smoothing
from tractus_engine_advanced.tds.constants import TIMESTEP_S
from tractus_engine_advanced.tds.mna import (
    PrepValues,
    TubeGeometry,
    assemble_matrix,
    prepare_time_step,
    update_variables,
)
from tractus_engine_advanced.tds.state import TdsState
from tractus_engine_advanced.tds.tube_topology import (
    LAST_MOUTH_SECTION,
    LAST_NOSE_SECTION,
    NUM_BRANCH_CURRENTS,
    NUM_SECTIONS,
    TubeTopology,
)


# correctness/007 iteration 2: wrap the per-sample MNA Cholesky solve in
# `jax.lax.custom_linear_solve`. This bypasses default-autodiff materialization
# of a per-sample Jacobian transpose ``f32[F, samples_per_frame, 97, 97]``
# (~3.9 GB for the production T=118 grid) by using implicit-function-theorem
# differentiation: backward solves ``M^T y = cotangent`` and folds that into
# the cotangents of ``M`` and ``b`` matrix-free, instead of building the dense
# JVP/VJP of the solver. Forward semantics are bit-equal: same Cholesky on
# the same ``(-M, -b)`` — only the autodiff rule changes.
#
# Set ``TEA_DISABLE_CUSTOM_SOLVE=1`` to fall back to the plain ``cho_solve``.
# Used by the verifier to measure VRAM/time delta vs. autodiff baseline.
_USE_CUSTOM_LINEAR_SOLVE = os.environ.get(
    "TEA_DISABLE_CUSTOM_SOLVE", ""
).lower() not in ("1", "true", "yes")


def _cholesky_solve_spd(
    M_pd: Float[Array, "97 97"], b_pd: Float[Array, "97"]
) -> Float[Array, "97"]:
    """Plain SPD solve via Cholesky. Used both as the forward solver under
    ``custom_linear_solve`` and as the legacy direct path."""
    chol = jax.scipy.linalg.cho_factor(M_pd, lower=True)
    return jax.scipy.linalg.cho_solve(chol, b_pd)


def _mna_solve(
    M_pd: Float[Array, "97 97"], b_pd: Float[Array, "97"]
) -> Float[Array, "97"]:
    """Solve ``M_pd @ x = b_pd`` where ``M_pd`` is treated as SPD.

    With ``custom_linear_solve`` (default) the VJP is computed implicitly:
    if ``f(M, b) = solve(M, b)`` then ``∂L/∂b = solve(M^T, ∂L/∂x)`` and the
    contribution to ``∂L/∂M`` is ``-∂L/∂b ⊗ x``. JAX synthesises both via
    a transpose of the matvec closure — we just need to provide a
    ``transpose_solve`` (or set ``symmetric=True`` to reuse ``solve``).

    The pre-negated MNA matrix is symmetric (post-Bernoulli the upstream
    solver Cholesky-factorises an SPD form), but to stay defensive against
    asymmetry creeping in we provide an explicit transpose solver and leave
    ``symmetric=False``. The cost is one extra Cholesky factorisation in the
    backward pass per sample — negligible vs. the dense Jacobian materialisation
    it replaces.
    """
    if not _USE_CUSTOM_LINEAR_SOLVE:
        return _cholesky_solve_spd(M_pd, b_pd)

    def matvec(x):
        return M_pd @ x

    def solve(_matvec, rhs):
        return _cholesky_solve_spd(M_pd, rhs)

    def transpose_solve(_vecmat, rhs):
        return _cholesky_solve_spd(M_pd.T, rhs)

    return jax.lax.custom_linear_solve(
        matvec, b_pd, solve, transpose_solve, symmetric=False
    )


class StepOutput(eqx.Module):
    state: TdsState
    section_pressure_dPa: Float[Array, "93"]
    branch_current_cm3_s: Float[Array, "97"]
    mouth_flow: Float[Array, ""]
    nostril_flow: Float[Array, ""]
    skin_flow: Float[Array, ""]


def _step_one_impl(
    state: TdsState,
    geom: TubeGeometry,
    subglottal_pressure_dPa: Float[Array, ""],
    monopole_source_amp: Float[Array, "93"],
    topology: TubeTopology,
    soft_walls: bool,
    turbulence_losses: bool,
    dt: float,
    smoothing: Smoothing,
) -> StepOutput:
    prep = prepare_time_step(
        state, geom, topology,
        soft_walls=soft_walls,
        turbulence_losses=turbulence_losses,
        dt=dt,
        smoothing=smoothing,
        monopole_source_amp=monopole_source_amp,
    )
    M, b = assemble_matrix(
        state, geom, prep, topology,
        subglottal_pressure_dPa=subglottal_pressure_dPa,
        dt=dt,
    )
    # Upstream's `solveEquationsCholesky` negates M and b at entry so the SPD
    # form is positive-definite. JAX's `cho_solve` expects an SPD matrix; we
    # match that by negating both sides — the solution itself is unchanged
    # since (-M) x = (-b) ⟺ M x = b. The solve is routed through
    # ``_mna_solve`` so backward uses the implicit-function VJP (see module
    # docstring for ``_USE_CUSTOM_LINEAR_SOLVE``).
    M_pd = -M
    b_pd = -b
    flow_vector = _mna_solve(M_pd, b_pd)

    # correctness/004b: monopole noise enters as a post-solve net-flow
    # term; ``prepare_time_step`` no longer couples it into the MNA RHS.
    new_state = update_variables(
        state, flow_vector, prep, topology, dt=dt,
        monopole_source_amp=monopole_source_amp,
    )

    cur_out = jnp.asarray(topology.current_out)
    mouth_out = cur_out[LAST_MOUTH_SECTION]
    nostril_out = cur_out[LAST_NOSE_SECTION]
    mouth_flow = (
        jnp.where(mouth_out[0] >= 0, flow_vector[jnp.where(mouth_out[0] >= 0, mouth_out[0], 0)], 0.0)
        + jnp.where(mouth_out[1] >= 0, flow_vector[jnp.where(mouth_out[1] >= 0, mouth_out[1], 0)], 0.0)
    )
    nostril_flow = (
        jnp.where(nostril_out[0] >= 0, flow_vector[jnp.where(nostril_out[0] >= 0, nostril_out[0], 0)], 0.0)
        + jnp.where(nostril_out[1] >= 0, flow_vector[jnp.where(nostril_out[1] >= 0, nostril_out[1], 0)], 0.0)
    )
    # Skin radiation: requires the Chebyshev IIR `glottalToneFilter`. M4 owns
    # that port; for M2 we return 0 (excluded from acceptance).
    skin_flow = jnp.asarray(0.0, dtype=DTYPE)

    return StepOutput(
        state=new_state,
        section_pressure_dPa=new_state.section[:, 0],  # PRESSURE
        branch_current_cm3_s=flow_vector,
        mouth_flow=mouth_flow,
        nostril_flow=nostril_flow,
        skin_flow=skin_flow,
    )


# correctness/008: ``jax.custom_vjp`` at the **step granularity** so the
# per-sample dense ``M`` cotangent (97×97 fp32) never crosses the scan-body
# boundary. With autodiff straight through ``step_one``, JAX scan stacks
# any tensor that the bwd-of-scan-body needs in its co-output, including the
# ``dM = -y · x^T`` tensor produced by the implicit-function VJP of the linear
# solve. At ``T=118 b=4 samples_per_frame=882`` that becomes
# ``f32[118, 882, 4, 97, 97] ≈ 14.6 GB`` — the residual blocker post-iter-2.
#
# The fix: hand JAX a custom VJP rule whose **fwd** returns only the small
# residuals (state, geom, subglottal_pressure_dPa) and whose **bwd** uses
# ``jax.vjp`` to recompute the input cotangents *inside* the bwd region. The
# bwd produces only ``(state_cotangent, geom_cotangent, p_cotangent)``, none
# of which are 97×97. The 97×97 ``dM`` tensor is local to bwd and never
# appears at the scan-stacked boundary.
#
# Forward semantics are bit-identical: ``_step_one_impl`` is unchanged and is
# what the fwd rule executes. The custom_linear_solve from iter 2 stays
# wrapped around the Cholesky in ``_mna_solve``; it is a no-op inside the
# fwd path of this custom_vjp but harmless. Set ``TEA_DISABLE_CUSTOM_VJP=1``
# to fall back to plain autodiff (used by the verifier to A/B the gate).
_USE_CUSTOM_VJP = os.environ.get(
    "TEA_DISABLE_CUSTOM_VJP", ""
).lower() not in ("1", "true", "yes")


# ``nondiff_argnums`` indexes the positional args of the decorated function.
# The signature below intentionally pushes static-ish args (topology, flags,
# dt, smoothing) to the tail so we can mark them all non-differentiable in
# one block. ``smoothing`` is a frozen dataclass of floats (hashable) so it
# can ride as a non-diff arg without breaking the custom_vjp signature.
@partial(jax.custom_vjp, nondiff_argnums=(4, 5, 6, 7, 8))
def _step_one_cvjp(
    state: TdsState,
    geom: TubeGeometry,
    subglottal_pressure_dPa: Float[Array, ""],
    monopole_source_amp: Float[Array, "93"],
    topology: TubeTopology,
    soft_walls: bool,
    turbulence_losses: bool,
    dt: float,
    smoothing: Smoothing,
) -> StepOutput:
    return _step_one_impl(
        state, geom, subglottal_pressure_dPa, monopole_source_amp,
        topology, soft_walls, turbulence_losses, dt, smoothing,
    )


def _step_one_cvjp_fwd(
    state: TdsState,
    geom: TubeGeometry,
    subglottal_pressure_dPa: Float[Array, ""],
    monopole_source_amp: Float[Array, "93"],
    topology: TubeTopology,
    soft_walls: bool,
    turbulence_losses: bool,
    dt: float,
    smoothing: Smoothing,
):
    out = _step_one_impl(
        state, geom, subglottal_pressure_dPa, monopole_source_amp,
        topology, soft_walls, turbulence_losses, dt, smoothing,
    )
    # Stash only the differentiable inputs as residuals. The bwd will replay
    # forward via ``jax.vjp`` to materialise intermediates locally — those
    # never cross the scan boundary because they're computed inside the bwd
    # call, not stored as fwd residuals.
    residuals = (state, geom, subglottal_pressure_dPa, monopole_source_amp)
    return out, residuals


def _step_one_cvjp_bwd(
    topology: TubeTopology,
    soft_walls: bool,
    turbulence_losses: bool,
    dt: float,
    smoothing: Smoothing,
    residuals,
    cotangent_out: StepOutput,
):
    state, geom, subglottal_pressure_dPa, monopole_source_amp = residuals

    def f(s, g, p, m):
        return _step_one_impl(
            s, g, p, m, topology, soft_walls, turbulence_losses, dt, smoothing,
        )

    _, vjp_fn = jax.vjp(f, state, geom, subglottal_pressure_dPa, monopole_source_amp)
    cot_state, cot_geom, cot_p, cot_m = vjp_fn(cotangent_out)
    return cot_state, cot_geom, cot_p, cot_m


_step_one_cvjp.defvjp(_step_one_cvjp_fwd, _step_one_cvjp_bwd)


def step_one(
    state: TdsState,
    geom: TubeGeometry,
    topology: TubeTopology,
    *,
    subglottal_pressure_dPa: float = 0.0,
    soft_walls: bool = True,
    turbulence_losses: bool = True,
    dt: float = TIMESTEP_S,
    smoothing: Smoothing | None = None,
    monopole_source_amp: Float[Array, "93"] | None = None,
) -> StepOutput:
    """Advance one TDS sample.

    correctness/004: ``monopole_source_amp`` plumbs the per-section
    physically-derived frication noise (``frication.noise_sample(...)``)
    into the MNA RHS as a flow source. ``None`` keeps the legacy
    pre-frication behaviour (zero source amp). The arg is differentiable
    in the gradient path so the body model's frication output is
    end-to-end trainable.
    """
    p = jnp.asarray(subglottal_pressure_dPa, dtype=DTYPE)
    sm = smoothing if smoothing is not None else Smoothing.hard()
    if monopole_source_amp is None:
        m = jnp.zeros((NUM_SECTIONS,), dtype=DTYPE)
    else:
        m = jnp.asarray(monopole_source_amp, dtype=DTYPE)
    if _USE_CUSTOM_VJP:
        return _step_one_cvjp(
            state, geom, p, m, topology, soft_walls, turbulence_losses, dt, sm,
        )
    return _step_one_impl(
        state, geom, p, m, topology, soft_walls, turbulence_losses, dt, sm,
    )
