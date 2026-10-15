"""Per-sample MNA assembly for the TDS step — pure JAX.

Standard 1-D transmission-line vocal-tract acoustic model:
  - `prepare_time_step` — predict pressures/flows at next sample
  - `assemble_matrix`   — build the symmetric system for the solve

Sign convention: the matrix and RHS produced here use the pre-negation
form; the Cholesky solver in `step.py` applies the negation.

For M2 the noise sources are off (monopole_sample = dipole_sample = 0),
piriform fossa coupling is on, transvelar coupling is off.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Bool, Float, Int

from tractus_engine_advanced.precision import DTYPE
from tractus_engine_advanced.smoothing import Smoothing, soft_max_floor, soft_step

from tractus_engine_advanced.tds.constants import (
    AIR_VISCOSITY_CGS,
    AMBIENT_DENSITY_CGS,
    CRITICAL_AREA_CM2,
    MAX_STIFFNESS_BOOST,
    MIN_AREA_CM2,
    MIN_RADIUS_CM,
    NOISE_CUTOFF_FREQ_HZ,
    SOUND_VELOCITY_CGS,
    THETA,
    THETA1,
    TIMESTEP_S,
)
from tractus_engine_advanced.tds.state import (
    MAGNITUDE,
    MAGNITUDE_RATE,
    NUM_BRANCH_FIELDS,
    NUM_SECTION_FIELDS,
    PRESSURE,
    PRESSURE_RATE,
    TdsState,
    WALL_CURRENT,
    WALL_CURRENT_RATE,
    WALL_CURRENT_RATE2,
)
from tractus_engine_advanced.tds.tube_topology import (
    FIRST_FOSSA_SECTION,
    FIRST_MOUTH_SECTION,
    FIRST_NOSE_SECTION,
    FIRST_PHARYNX_SECTION,
    FIRST_SINUS_SECTION,
    FIRST_TRACHEA_SECTION,
    LAST_MOUTH_SECTION,
    LAST_NOSE_SECTION,
    LAST_PHARYNX_SECTION,
    LAST_SINUS_SECTION,
    LAST_TRACHEA_SECTION,
    LOWER_GLOTTIS_SECTION,
    NUM_BRANCH_CURRENTS,
    NUM_SECTIONS,
    UPPER_GLOTTIS_SECTION,
    TubeTopology,
)


# ---------------------------------------------------------------------------
# Tube geometry — static-ish per-frame inputs (length, area, walls).
# ---------------------------------------------------------------------------


class TubeGeometry(eqx.Module):
    """93-section tube as TdsModel sees it post-`setTube` (CRITICAL_AREA boost applied).

    Articulator is int32 — cast to float32 if needed for any soft-mixture work
    later; for M2 it's only consumed by noise placement (off).
    """
    length_cm: Float[Array, "93"]
    area_cm2: Float[Array, "93"]
    volume_cm3: Float[Array, "93"]
    wall_mass: Float[Array, "93"]           # Mw, post-stiffness-boost
    wall_resistance: Float[Array, "93"]     # Bw
    wall_stiffness: Float[Array, "93"]      # Kw, post-CRITICAL_AREA boost

    @classmethod
    def from_arrays(
        cls,
        length_cm: np.ndarray,         # (93,)
        area_cm2: np.ndarray,          # (93,) — raw; will be MIN_AREA_CM2-clamped
        wall_mass: np.ndarray,         # (93,) raw Mw from speaker defaults
        wall_resistance: np.ndarray,   # (93,) raw Bw
        wall_stiffness: np.ndarray,    # (93,) raw Kw (pre-boost)
        smoothing: Smoothing | None = None,
    ) -> "TubeGeometry":
        """Apply the same area clamp + CRITICAL_AREA stiffness boost upstream's
        setTube does. Volume = length * (clamped area).

        correctness/003 (M6 soft): when ``smoothing`` is enabled and
        ``tau_area_clamp > 0``, the hard ``jnp.maximum(A, MIN_AREA_CM2)``
        floor is replaced by a softplus-shifted floor. At τ → 0 the soft
        floor reduces to the hard one (within softplus tail precision).
        """
        L = jnp.asarray(length_cm, dtype=DTYPE)
        A_raw = jnp.asarray(area_cm2, dtype=DTYPE)
        sm = smoothing if smoothing is not None else Smoothing.hard()
        if sm.enabled and sm.tau_area_clamp > 0.0:
            A = soft_max_floor(A_raw, MIN_AREA_CM2, float(sm.tau_area_clamp))
        else:
            A = jnp.maximum(A_raw, MIN_AREA_CM2)
        V = L * A
        Mw = jnp.asarray(wall_mass, dtype=DTYPE)
        Bw = jnp.asarray(wall_resistance, dtype=DTYPE)
        Kw_raw = jnp.asarray(wall_stiffness, dtype=DTYPE)
        # CRITICAL_AREA boost: K /= max(area / CRITICAL_AREA_CM2, 1/MAX_BOOST).
        ratio = jnp.maximum(A / CRITICAL_AREA_CM2, 1.0 / MAX_STIFFNESS_BOOST)
        boost = jnp.where(A < CRITICAL_AREA_CM2, 1.0 / ratio, 1.0)
        Kw = Kw_raw * boost
        return cls(
            length_cm=L,
            area_cm2=A,
            volume_cm3=V,
            wall_mass=Mw,
            wall_resistance=Bw,
            wall_stiffness=Kw,
        )


# ---------------------------------------------------------------------------
# Pre-step quantities — output of `prepare_time_step`.
# ---------------------------------------------------------------------------


class PrepValues(eqx.Module):
    R0: Float[Array, "93"]      # ohm-resistance, left side of section
    R1: Float[Array, "93"]      # ohm-resistance, right side of section (post-Bernoulli)
    L: Float[Array, "93"]       # inductivity per section
    C: Float[Array, "93"]       # capacitance per section
    alpha: Float[Array, "93"]   # wall-vibration THETA coef
    beta: Float[Array, "93"]    # wall-vibration THETA forcing
    D: Float[Array, "93"]       # pre-step pressure constant (TdsModel.cpp:1169)
    E: Float[Array, "93"]       # pre-step pressure-from-flow gain  (= dt*THETA/(C+alpha))


# ---------------------------------------------------------------------------
# Per-section R, L, C — `prepareTimeStep:944-978`.
# ---------------------------------------------------------------------------


def _per_section_rlc(geom: TubeGeometry, section_idx: Int[Array, "93"]) -> tuple[
    Float[Array, "93"], Float[Array, "93"], Float[Array, "93"], Float[Array, "93"]
]:
    """Returns (R0, R1, L, C). R0 and R1 are equal here (Bernoulli applied later).

    Two formulae:
      - sinus (FIRST_SINUS..LAST_SINUS): Helmholtz-resonator branch.
      - all other sections: circular-or-elliptical-cross-section branch.
    """
    A = geom.area_cm2          # already clamped
    L_len = geom.length_cm
    V = geom.volume_cm3

    # --- Helmholtz (sinus) branch ---
    L_sinus = AMBIENT_DENSITY_CGS * (L_len / A)
    C_sinus = V / (AMBIENT_DENSITY_CGS * SOUND_VELOCITY_CGS * SOUND_VELOCITY_CGS)
    R_sinus = (8.0 * AIR_VISCOSITY_CGS * jnp.pi * L_len) / (A * A)

    # --- Normal (cylindrical / elliptical) branch ---
    a_circ = jnp.sqrt(A / jnp.pi)               # would-be circular half-axis
    use_ellipse = a_circ < MIN_RADIUS_CM
    a = jnp.where(use_ellipse, MIN_RADIUS_CM, a_circ)
    b = jnp.where(use_ellipse, A / (jnp.pi * a), a_circ)

    L_normal = (AMBIENT_DENSITY_CGS * 0.5 * L_len) / A
    C_normal = V / (AMBIENT_DENSITY_CGS * SOUND_VELOCITY_CGS * SOUND_VELOCITY_CGS)
    R_normal = ((2.0 * AIR_VISCOSITY_CGS * L_len) * (a * a + b * b)) / (
        jnp.pi * a * a * a * b * b * b
    )

    is_sinus = (section_idx >= FIRST_SINUS_SECTION) & (section_idx <= LAST_SINUS_SECTION)
    L = jnp.where(is_sinus, L_sinus, L_normal)
    C = jnp.where(is_sinus, C_sinus, C_normal)
    R = jnp.where(is_sinus, R_sinus, R_normal)
    return R, R, L, C


# ---------------------------------------------------------------------------
# Wall-vibration alpha/beta — `prepareTimeStep:980-1015`.
# ---------------------------------------------------------------------------


def _wall_alpha_beta(
    geom: TubeGeometry,
    state: TdsState,
    section_idx: Int[Array, "93"],
    soft_walls: bool,
    dt: float,
) -> tuple[Float[Array, "93"], Float[Array, "93"]]:
    """alpha, beta = 0 for the two glottis sections even with soft_walls=True
    (`prepareTimeStep:988`). Sinus sections use a sphere-surface model; all
    others use cylinder-barrel surface.
    """
    A = geom.area_cm2
    L_len = geom.length_cm
    V = geom.volume_cm3
    Mw = geom.wall_mass
    Bw = geom.wall_resistance
    Kw = geom.wall_stiffness

    circ = 2.0 * jnp.sqrt(A * jnp.pi)
    surface_cyl = circ * L_len
    surface_sphere = 4.0 * jnp.pi * jnp.power((3.0 * V) / (4.0 * jnp.pi), 2.0 / 3.0)
    is_sinus = (section_idx >= FIRST_SINUS_SECTION) & (section_idx <= LAST_SINUS_SECTION)
    surface = jnp.where(is_sinus, surface_sphere, surface_cyl)
    surface = jnp.maximum(surface, MIN_AREA_CM2)

    Rw = Bw / surface
    Lw = Mw / surface
    Cw = surface / Kw

    alpha = 1.0 / (
        Lw / (dt * dt * THETA * THETA)
        + Rw / (dt * THETA)
        + 1.0 / Cw
    )

    wc = state.section[:, WALL_CURRENT]
    wcr = state.section[:, WALL_CURRENT_RATE]
    wcr2 = state.section[:, WALL_CURRENT_RATE2]
    beta_unscaled = (
        wc * (Lw / (dt * dt * THETA * THETA) + Rw / (dt * THETA))
        + wcr * (Lw * (THETA1 / THETA + 1.0) / (dt * THETA) + Rw * (THETA1 / THETA))
        + wcr2 * Lw * (THETA1 / THETA)
    )
    beta = alpha * beta_unscaled

    is_glottis = (section_idx == LOWER_GLOTTIS_SECTION) | (section_idx == UPPER_GLOTTIS_SECTION)
    enabled = jnp.asarray(soft_walls and True, dtype=bool) & ~is_glottis
    alpha = jnp.where(enabled, alpha, 0.0)
    beta = jnp.where(enabled, beta, 0.0)
    return alpha, beta


# ---------------------------------------------------------------------------
# Bernoulli losses — `prepareTimeStep:1033-1117`.
# ---------------------------------------------------------------------------


def _bernoulli_chain_loss(
    R0: Float[Array, "93"],
    R1: Float[Array, "93"],
    geom: TubeGeometry,
    state: TdsState,
    topology: TubeTopology,
    tau_bernoulli: float = 0.0,
) -> tuple[Float[Array, "93"], Float[Array, "93"]]:
    """Wide→narrow / narrow→wide chain Bernoulli updates for pharynx-mouth and
    nose, where the source has exactly one outflow (`prepareTimeStep:1033-1064`).

    Skipped sections: glottis, sinus, fossa, trachea (chain coupling already
    reflects boundary conditions there). Specifically, the upstream loop is
        for i in [FIRST_PHARYNX+1..LAST_MOUTH] ∪ [FIRST_NOSE+1..LAST_NOSE]:
            if section[i-1].currentOut[1] == -1: apply
    """
    n = NUM_SECTIONS
    A = geom.area_cm2
    cur_out = jnp.asarray(topology.current_out)

    # Per-section "is in the loop": True for i-1 chain-feeds i, i in pharynx-mouth or nose.
    in_pharynx_mouth = jnp.array(
        [(FIRST_PHARYNX_SECTION + 1) <= i <= LAST_MOUTH_SECTION for i in range(n)],
        dtype=bool,
    )
    in_nose = jnp.array(
        [(FIRST_NOSE_SECTION + 1) <= i <= LAST_NOSE_SECTION for i in range(n)],
        dtype=bool,
    )
    in_loop = in_pharynx_mouth | in_nose

    # u = currentOut[0] of section i-1 — read from prior-sample state.branch.magnitude.
    # Need to clip i-1 to [0, NUM_SECTIONS) and the currentOut to [-1 → safe].
    prev_idx = jnp.clip(jnp.arange(n) - 1, 0, n - 1)
    prev_out0 = cur_out[prev_idx, 0]
    prev_out1 = cur_out[prev_idx, 1]
    out0_safe = jnp.where(prev_out0 == -1, 0, prev_out0)
    u_prev = jnp.where(prev_out0 == -1, 0.0, state.branch[out0_safe, MAGNITUDE])

    A_prev = A[prev_idx]
    A_self = A
    is_simple_chain = prev_out1 == -1
    # Bernoulli applies when (A_self < A_prev AND u > 0) OR (A_self > A_prev AND u < 0).
    cond = (
        ((A_self < A_prev) & (u_prev > 0.0))
        | ((A_self > A_prev) & (u_prev < 0.0))
    )
    apply = in_loop & is_simple_chain & cond

    delta_prev_R1 = -u_prev * 0.5 * AMBIENT_DENSITY_CGS / (A_prev * A_prev)
    delta_self_R0 = u_prev * 0.5 * AMBIENT_DENSITY_CGS / (A_self * A_self)

    if tau_bernoulli > 0.0:
        # Soft direction-gate: sigmoid((A_prev - A_self)/τ_area)·sigmoid(u/τ_u)
        # gives the (A_self < A_prev) ∧ (u > 0) leaf, mirrored for the
        # (A_self > A_prev) ∧ (u < 0) leaf, summed. As τ → 0 reduces to the
        # hard ``cond`` above. The ``in_loop`` and ``is_simple_chain``
        # gates are static-topology-driven booleans (no autograd path),
        # left hard.
        gate_pos = jax.nn.sigmoid((A_prev - A_self) / tau_bernoulli) \
                 * jax.nn.sigmoid(u_prev / tau_bernoulli)
        gate_neg = jax.nn.sigmoid((A_self - A_prev) / tau_bernoulli) \
                 * jax.nn.sigmoid(-u_prev / tau_bernoulli)
        soft_cond = gate_pos + gate_neg
        soft_apply = in_loop.astype(soft_cond.dtype) \
                   * is_simple_chain.astype(soft_cond.dtype) \
                   * soft_cond
        delta_prev_R1_masked = soft_apply * delta_prev_R1
        delta_self_R0_masked = soft_apply * delta_self_R0
    else:
        # Scatter-add into R1[i-1] and R0[i] via segment_sum.
        delta_prev_R1_masked = jnp.where(apply, delta_prev_R1, 0.0)
        delta_self_R0_masked = jnp.where(apply, delta_self_R0, 0.0)

    # Add to R1 at index (i-1): use scatter via segment_sum-like at[].add.
    R1_out = R1.at[prev_idx].add(delta_prev_R1_masked)
    R0_out = R0.at[jnp.arange(n)].add(delta_self_R0_masked)
    return R0_out, R1_out


def _glottal_entrance_loss(
    R0: Float[Array, "93"],
    geom: TubeGeometry,
    state: TdsState,
    topology: TubeTopology,
    tau_bernoulli: float = 0.0,
) -> Float[Array, "93"]:
    """Glottal entrance Bernoulli (k_ent = 1.0): `prepareTimeStep:1083-1100`.
    Reads u = getCurrentIn(LOWER_GLOTTIS) from prior-sample state.
    """
    sourceArea = geom.area_cm2[LAST_TRACHEA_SECTION]
    targetArea = geom.area_cm2[LOWER_GLOTTIS_SECTION]
    ci = int(topology.current_in[LOWER_GLOTTIS_SECTION])
    u = state.branch[ci, MAGNITUDE] if ci != -1 else jnp.asarray(0.0, dtype=DTYPE)
    body = (
        1.0 * 0.5 * AMBIENT_DENSITY_CGS * jnp.abs(u)
        * (1.0 / (targetArea * targetArea) - 1.0 / (sourceArea * sourceArea))
    )
    if tau_bernoulli > 0.0:
        gate = jax.nn.sigmoid(u / tau_bernoulli)
        delta = gate * body
    else:
        delta = jnp.where(u > 0.0, body, 0.0)
    return R0.at[LOWER_GLOTTIS_SECTION].add(delta)


def _glottal_transition_loss(
    R1: Float[Array, "93"],
    geom: TubeGeometry,
    state: TdsState,
    topology: TubeTopology,
    tau_bernoulli: float = 0.0,
) -> Float[Array, "93"]:
    """Glottis transition Bernoulli (`prepareTimeStep:1102-1114`).
    Reads u = getCurrentOut(LOWER_GLOTTIS) from prior-sample state.
    """
    sourceArea = geom.area_cm2[LOWER_GLOTTIS_SECTION]
    targetArea = geom.area_cm2[UPPER_GLOTTIS_SECTION]
    co0 = int(topology.current_out[LOWER_GLOTTIS_SECTION, 0])
    u = state.branch[co0, MAGNITUDE] if co0 != -1 else jnp.asarray(0.0, dtype=DTYPE)
    body = (
        jnp.abs(u) * 0.5 * AMBIENT_DENSITY_CGS
        * (1.0 / (targetArea * targetArea) - 1.0 / (sourceArea * sourceArea))
    )
    if tau_bernoulli > 0.0:
        gate = (
            jax.nn.sigmoid(u / tau_bernoulli)
            * jax.nn.sigmoid((sourceArea - targetArea) / tau_bernoulli)
        )
        delta = gate * body
    else:
        delta = jnp.where(
            (u > 0.0) & (targetArea < sourceArea), body, 0.0,
        )
    return R1.at[LOWER_GLOTTIS_SECTION].add(delta)


# ---------------------------------------------------------------------------
# `prepare_time_step` — top-level prep for one sample.
# ---------------------------------------------------------------------------


def prepare_time_step(
    state: TdsState,
    geom: TubeGeometry,
    topology: TubeTopology,
    *,
    soft_walls: bool = True,
    turbulence_losses: bool = True,
    dt: float = TIMESTEP_S,
    smoothing: Smoothing | None = None,
    monopole_source_amp: Float[Array, "93"] | None = None,
) -> PrepValues:
    """Prepare per-sample MNA quantities.

    Parameters
    ----------
    monopole_source_amp
        Per-section monopole flow source (cm³/s). Set by the frication
        module (correctness/004): the physically-derived turbulent-noise
        amplitude per section, modulated by a deterministic
        ``tanh(z) · gain`` reparameterized noise sample. **Note** —
        correctness/004b moved the actual injection out of ``D`` into
        the post-solve ``update_variables`` net-flow term. The matrix
        solve in this function therefore no longer sees ``source_amp``
        at all — the kwarg is kept for API compatibility but is now
        ignored at the prep stage. See module docstring of
        ``update_variables`` for the new injection point and rationale.
        Bit-equal to legacy at ``source_amp = 0``; stable at
        physiological turbulence amplitudes (5-15 cm³/s) because the
        MNA RHS magnitude is decoupled from the noise drive.
    """
    sm = smoothing if smoothing is not None else Smoothing.hard()
    tau_b = float(sm.tau_bernoulli) if sm.enabled else 0.0
    section_idx = jnp.arange(NUM_SECTIONS)

    R0, R1, L, C = _per_section_rlc(geom, section_idx)

    alpha, beta = _wall_alpha_beta(geom, state, section_idx, soft_walls, dt)

    if turbulence_losses:
        R0, R1 = _bernoulli_chain_loss(R0, R1, geom, state, topology, tau_b)

    R0 = _glottal_entrance_loss(R0, geom, state, topology, tau_b)
    R1 = _glottal_transition_loss(R1, geom, state, topology, tau_b)

    # D, E (TdsModel.cpp:1167-1170). correctness/004b: the
    # ``monopole_source_amp`` injection used to enter via
    # ``D += d · source_amp`` here, but that coupled the noise drive
    # into the matrix RHS through ``b_i = H + tgt_D[i] - src_D[i]``,
    # which made the per-sample solve unstable above ~0.5 cm³/s
    # (integrator divergence: large RHS → large branch flows → next
    # sample blows up). The fix is to inject the flow source post-solve
    # in ``update_variables`` as an additive net-flow term, leaving the
    # MNA matrix and RHS bit-identical to the no-noise legacy path.
    # See ``update_variables`` for details.
    d = dt * THETA / (C + alpha)
    E = d
    D = (
        state.section[:, PRESSURE]
        + dt * THETA1 * state.section[:, PRESSURE_RATE]
        - d * beta
    )

    return PrepValues(R0=R0, R1=R1, L=L, C=C, alpha=alpha, beta=beta, D=D, E=E)


# ---------------------------------------------------------------------------
# `assemble_matrix` — `calcMatrix:1961-2215`. Returns LHS (97×97) and RHS (97).
# Pre-negation sign convention; the Cholesky solver flips signs at entry.
# ---------------------------------------------------------------------------


def _safe_idx(idx: Int[Array, "..."]) -> tuple[Int[Array, "..."], Bool[Array, "..."]]:
    """Map -1 → (0, False); otherwise → (idx, True)."""
    valid = idx >= 0
    safe = jnp.where(valid, idx, 0)
    return safe, valid


def assemble_matrix(
    state: TdsState,
    geom: TubeGeometry,
    prep: PrepValues,
    topology: TubeTopology,
    *,
    subglottal_pressure_dPa: float = 0.0,
    dt: float = TIMESTEP_S,
) -> tuple[Float[Array, "97 97"], Float[Array, "97"]]:
    n_br = NUM_BRANCH_CURRENTS
    # Lift topology arrays to jnp so traced indexing works under vmap.
    src = jnp.asarray(topology.source_section)
    tgt = jnp.asarray(topology.target_section)
    cur_in = jnp.asarray(topology.current_in)
    cur_out = jnp.asarray(topology.current_out)
    is_rad = jnp.asarray(topology.is_radiation)
    is_br = jnp.asarray(topology.is_branching)
    is_simple = jnp.asarray(topology.is_simple)
    branching_off = jnp.asarray(topology.branching_off)

    # --- per-branch lookups, with -1 sentinels handled ---
    src_safe, src_valid = _safe_idx(src)
    tgt_safe, tgt_valid = _safe_idx(tgt)

    src_L = jnp.where(src_valid, prep.L[src_safe], 0.0)
    src_R1 = jnp.where(src_valid, prep.R1[src_safe], 0.0)
    src_E = jnp.where(src_valid, prep.E[src_safe], 0.0)
    src_D = jnp.where(src_valid, prep.D[src_safe], 0.0)
    src_area = jnp.where(src_valid, geom.area_cm2[src_safe], 0.0)
    src_ci = jnp.where(src_valid, cur_in[src_safe], -1)
    src_co0 = jnp.where(src_valid, cur_out[src_safe, 0], -1)
    src_co1 = jnp.where(src_valid, cur_out[src_safe, 1], -1)

    tgt_L = jnp.where(tgt_valid, prep.L[tgt_safe], 0.0)
    tgt_R0 = jnp.where(tgt_valid, prep.R0[tgt_safe], 0.0)
    tgt_E = jnp.where(tgt_valid, prep.E[tgt_safe], 0.0)
    tgt_D = jnp.where(tgt_valid, prep.D[tgt_safe], 0.0)
    tgt_co0 = jnp.where(tgt_valid, cur_out[tgt_safe, 0], -1)
    tgt_co1 = jnp.where(tgt_valid, cur_out[tgt_safe, 1], -1)

    L_AB = src_L + tgt_L
    R_AB = src_R1 + tgt_R0

    u = state.branch[:, MAGNITUDE]
    u_rate = state.branch[:, MAGNITUDE_RATE]

    bo_safe, bo_valid = _safe_idx(branching_off)
    uD = jnp.where(bo_valid, u[bo_safe], 0.0)
    uD_rate = jnp.where(bo_valid, u_rate[bo_safe], 0.0)

    # Pressure-source injection: branch with target == FIRST_TRACHEA.
    S = jnp.where(
        tgt_valid & (tgt == FIRST_TRACHEA_SECTION),
        -subglottal_pressure_dPa,
        0.0,
    )

    dt_theta = dt * THETA
    th1_th = THETA1 / THETA

    # --- per-row functions ---

    def row_simple(i):
        G = L_AB[i] / dt_theta + R_AB[i]
        H = -u_rate[i] * L_AB[i] * th1_th - L_AB[i] * u[i] / dt_theta + S[i]

        row = jnp.zeros(n_br, dtype=DTYPE)

        ci_safe_i, ci_valid_i = _safe_idx(src_ci[i])
        row = row.at[ci_safe_i].add(jnp.where(ci_valid_i, src_E[i], 0.0))

        diag = -tgt_E[i] - G - jnp.where(src_valid[i], src_E[i], 0.0)
        row = row.at[i].add(diag)

        co0_safe_i, co0_valid_i = _safe_idx(tgt_co0[i])
        row = row.at[co0_safe_i].add(jnp.where(co0_valid_i, tgt_E[i], 0.0))
        co1_safe_i, co1_valid_i = _safe_idx(tgt_co1[i])
        row = row.at[co1_safe_i].add(jnp.where(co1_valid_i, tgt_E[i], 0.0))

        b_i = H + tgt_D[i] - jnp.where(src_valid[i], src_D[i], 0.0)
        return row, b_i

    def row_branching(i):
        F = L_AB[i] / dt_theta + R_AB[i]
        G = src_L[i] / dt_theta + src_R1[i]
        H = (
            -(L_AB[i] * u[i] + src_L[i] * uD[i]) / dt_theta
            - th1_th * (L_AB[i] * u_rate[i] + src_L[i] * uD_rate[i])
            + S[i]
        )

        row = jnp.zeros(n_br, dtype=DTYPE)

        # branching-off partner.
        row = row.at[bo_safe[i]].add(jnp.where(bo_valid[i], -src_E[i] - G, 0.0))
        ci_safe_i, ci_valid_i = _safe_idx(src_ci[i])
        row = row.at[ci_safe_i].add(jnp.where(ci_valid_i, src_E[i], 0.0))
        # diagonal.
        row = row.at[i].add(-tgt_E[i] - src_E[i] - F)

        co0_safe_i, co0_valid_i = _safe_idx(tgt_co0[i])
        row = row.at[co0_safe_i].add(jnp.where(co0_valid_i, tgt_E[i], 0.0))
        co1_safe_i, co1_valid_i = _safe_idx(tgt_co1[i])
        row = row.at[co1_safe_i].add(jnp.where(co1_valid_i, tgt_E[i], 0.0))

        b_i = H + tgt_D[i] - src_D[i]
        return row, b_i

    def row_radiation(i):
        # Radiation rows always have source valid + 2 outflows, but be defensive
        # so non-radiation lanes don't OOB-scatter when this fn is computed-then-discarded.
        rc = src_co0[i]
        ic = src_co1[i]
        rc_safe, rc_valid = _safe_idx(rc)
        ic_safe, ic_valid = _safe_idx(ic)

        uR = jnp.where(rc_valid, u[rc_safe], 0.0)
        uL = jnp.where(ic_valid, u[ic_safe], 0.0)
        uR_rate = jnp.where(rc_valid, u_rate[rc_safe], 0.0)
        uL_rate = jnp.where(ic_valid, u_rate[ic_safe], 0.0)
        L_A = src_L[i]
        R_A = src_R1[i]
        S_rad = 0.0  # M2: lipsDipoleSource.sample = 0
        rad_area = src_area[i]

        # The R-current path (i is currentOut[0]).
        R_rad = (128.0 * AMBIENT_DENSITY_CGS * SOUND_VELOCITY_CGS) / (
            9.0 * jnp.pi * jnp.pi * rad_area
        )
        F_R = L_A / dt_theta + R_A + R_rad
        G_R = L_A / dt_theta + R_A
        H_R = (
            -L_A * (uR + uL) / dt_theta
            - L_A * th1_th * (uR_rate + uL_rate)
            + S_rad
        )

        # The L-current path (i is currentOut[1]).
        L_rad = (8.0 * AMBIENT_DENSITY_CGS) / (3.0 * jnp.pi * jnp.sqrt(rad_area * jnp.pi))
        L_AB_rad = L_A + L_rad
        F_L = L_A / dt_theta + R_A
        G_L = L_AB_rad / dt_theta + R_A
        H_L = (
            -(L_A * uR + L_AB_rad * uL) / dt_theta
            - th1_th * (L_A * uR_rate + L_AB_rad * uL_rate)
            + S_rad
        )

        is_R_row = i == rc
        F = jnp.where(is_R_row, F_R, F_L)
        G = jnp.where(is_R_row, G_R, G_L)
        H = jnp.where(is_R_row, H_R, H_L)

        row = jnp.zeros(n_br, dtype=DTYPE)
        ci_safe_i, ci_valid_i = _safe_idx(src_ci[i])
        row = row.at[ci_safe_i].add(jnp.where(ci_valid_i, src_E[i], 0.0))
        # The two parallel currents — set the resistance and inductivity entries.
        row = row.at[rc_safe].add(jnp.where(rc_valid, -src_E[i] - F, 0.0))
        row = row.at[ic_safe].add(jnp.where(ic_valid, -src_E[i] - G, 0.0))

        b_i = H - src_D[i]
        return row, b_i

    branch_idx = jnp.arange(n_br)
    rows_s, b_s = jax.vmap(row_simple)(branch_idx)
    rows_b, b_br = jax.vmap(row_branching)(branch_idx)
    rows_r, b_r = jax.vmap(row_radiation)(branch_idx)

    M = jnp.where(
        is_simple[:, None],
        rows_s,
        jnp.where(is_br[:, None], rows_b, rows_r),
    )
    b = jnp.where(is_simple, b_s, jnp.where(is_br, b_br, b_r))
    return M, b


# ---------------------------------------------------------------------------
# `update_variables` — `TdsModel.cpp:2222-2274`. Advances state after the solve.
# ---------------------------------------------------------------------------


def update_variables(
    state: TdsState,
    flow_vector: Float[Array, "97"],
    prep: PrepValues,
    topology: TubeTopology,
    *,
    dt: float = TIMESTEP_S,
    monopole_source_amp: Float[Array, "93"] | None = None,
) -> TdsState:
    """Advance state after the MNA solve.

    Parameters
    ----------
    monopole_source_amp
        Per-section turbulent flow source (cm³/s; correctness/004b).
        Added to the post-solve per-section net flow before the pressure
        update so the noise drives ``new_pressure`` without entering the
        linear-solve RHS. Matrix-solve stability is therefore decoupled
        from the noise amplitude — the same Cholesky that handles the
        no-noise case handles arbitrary physiological turbulence
        amplitudes (5-15 cm³/s tested). When ``None`` the term is zero
        and the path is bit-equal to the pre-frication legacy.
    """
    dt_theta = dt * THETA
    th1_th = THETA1 / THETA
    noise_alpha = jnp.exp(-2.0 * jnp.pi * NOISE_CUTOFF_FREQ_HZ * dt)

    # ---- branch currents ----
    old_mag = state.branch[:, MAGNITUDE]
    old_rate = state.branch[:, MAGNITUDE_RATE]
    old_noise = state.branch[:, 2]  # NOISE_MAGNITUDE
    new_mag = flow_vector
    new_rate = (new_mag - old_mag) / dt_theta - th1_th * old_rate
    new_noise = (1.0 - noise_alpha) * new_mag + noise_alpha * old_noise
    new_branch = jnp.stack([new_mag, new_rate, new_noise], axis=-1)

    # ---- per-section pressures (use NEW branch magnitudes for net flow) ----
    cur_in = jnp.asarray(topology.current_in)
    cur_out = jnp.asarray(topology.current_out)

    ci_safe, ci_valid = _safe_idx(cur_in)
    co0_safe, co0_valid = _safe_idx(cur_out[:, 0])
    co1_safe, co1_valid = _safe_idx(cur_out[:, 1])

    flow_in = jnp.where(ci_valid, new_mag[ci_safe], 0.0)
    flow_out = (
        jnp.where(co0_valid, new_mag[co0_safe], 0.0)
        + jnp.where(co1_valid, new_mag[co1_safe], 0.0)
    )
    net_flow = flow_in - flow_out
    # correctness/004b: post-solve flow injection (see docstring).
    if monopole_source_amp is not None:
        net_flow = net_flow + jnp.asarray(monopole_source_amp, dtype=DTYPE)

    old_pressure = state.section[:, PRESSURE]
    old_pres_rate = state.section[:, PRESSURE_RATE]
    old_wc = state.section[:, WALL_CURRENT]
    old_wcr = state.section[:, WALL_CURRENT_RATE]
    old_wcr2 = state.section[:, WALL_CURRENT_RATE2]

    new_pressure = prep.D + prep.E * net_flow
    new_pres_rate = (new_pressure - old_pressure) / dt_theta - th1_th * old_pres_rate
    new_wc = new_pres_rate * prep.alpha + prep.beta
    new_wcr = (new_wc - old_wc) / dt_theta - th1_th * old_wcr
    new_wcr2 = (new_wcr - old_wcr) / dt_theta - th1_th * old_wcr2

    new_section = jnp.stack(
        [new_pressure, new_pres_rate, new_wc, new_wcr, new_wcr2],
        axis=-1,
    )
    return TdsState(section=new_section, branch=new_branch)
