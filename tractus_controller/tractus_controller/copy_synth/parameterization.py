"""Reduced-articulator knot parameterization for renderer copy-synth.

We optimize a small additive **residual** on a small number of "free"
channels of the 30-d (T, 30) tract+glottis trajectory at a low knot rate
(e.g. 50 Hz). The residual is interpolated from K knots and ADDED on top
of the full-rate template; non-free channels are held at template values
exactly. The other 22 channels stay at their template values.

**Identity property:** when all knot residuals are zero, ``build_trajectory``
returns the template byte-for-byte. This is the oracle-zero-loss
reproduction property D8 (Phase A) is meant to deliver — without it,
"init from template" can't reproduce the template's audio.

History: an earlier parameterization stored per-channel **values** at K
knots and *overwrote* the free channels with linear interpolation between
them. With K=4, that smashed every variation in the template down to a
piecewise-linear approximation; even when init_knots were sampled from the
template, the round trip through linear interp produced audible 1 Hz
triangle-wave articulator artifacts ("yyyyeuh ne neushujushjuush") and
made oracle-zero loss unreachable. Codex finding #5 / D8 (2026-05-02).

Trajectory layout (per ``tractus_shared.tract_sequence``):
    [:, 0:11]   glottis params (Geometric glottis: F0, PR, XB, XT, CA, PL,
                                RA, DP, PS, FL, AS)
    [:, 11:30]  tract params (HX, HY, JX, JA, LP, LD, VS, VO, TCX, TCY,
                              TTX, TTY, TBX, TBY, TRX, TRY, TS1, TS2, TS3)

Default 8 free channels (chosen post-Codex review of plan/30 §Phase 1):
    F0  (glottis 0)   30-d idx 0    pitch
    PR  (glottis 1)   30-d idx 1    lung pressure (loudness, voicing onset)
    JA  (tract 3)     30-d idx 14   jaw angle
    LP  (tract 4)     30-d idx 15   lip protrusion
    LD  (tract 5)     30-d idx 16   lip distance (aperture)
    VO  (tract 7)     30-d idx 18   velar opening (nasals)
    TCX (tract 8)     30-d idx 19   tongue body horz
    TCY (tract 9)     30-d idx 20   tongue body vert

Note: ``LP`` is overloaded (lip protrusion vs. lung pressure). We use the
30-d index throughout to disambiguate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FreeChannel:
    """One free articulator channel + the global ``min_max_bounds`` from the speaker file."""
    name: str
    idx: int           # column in the 30-d trajectory
    global_min: float
    global_max: float


# Static bounds for the 8 default free channels.
DEFAULT_FREE_CHANNELS: tuple[FreeChannel, ...] = (
    FreeChannel("F0",  0,  40.0,    600.0),
    FreeChannel("PR",  1,  0.0,     20000.0),
    FreeChannel("JA",  14, -7.0,    0.0),
    FreeChannel("LP",  15, -1.0,    1.0),
    FreeChannel("LD",  16, -2.0,    4.0),
    FreeChannel("VO",  18, -0.1,    1.0),
    FreeChannel("TCX", 19, -3.0,    4.0),
    FreeChannel("TCY", 20, -3.0,    1.0),
)

# Like DEFAULT but without VO (velum opening). Velum drift was the smoking
# gun for the lip-flap / babbling failure mode on real audio (drift 0.60
# of bound width on natasha_1_s00 seed 0). Locking VO at template is more
# robust on non-nasal-heavy speech and removes the optimizer's largest
# physically-implausible escape hatch.
NO_VELUM_FREE_CHANNELS: tuple[FreeChannel, ...] = tuple(
    ch for ch in DEFAULT_FREE_CHANNELS if ch.name != "VO"
)


# Interpolation kind for the residual:
#   "linear" — np.interp; matches Phase-A identity; cheapest.
#   "cubic"  — Fritsch-Carlson PCHIP (shape-preserving cubic Hermite) through
#              the K residual knots. Identity preserved (zero knots → zero
#              residual). Unlike a natural cubic spline, PCHIP cannot
#              overshoot adjacent knots — interpolated residuals stay between
#              neighboring knot extrema by construction. This closes the
#              R-V4-1 bug where natural-cubic overshoot pushed residuals past
#              the search bounds. See plan/70-handoff-2026-05-03.md.
INTERP_KINDS: tuple[str, ...] = ("linear", "cubic")


def derive_bounds(
    template: np.ndarray,
    channels: tuple[FreeChannel, ...] = DEFAULT_FREE_CHANNELS,
    delta_frac: float = 0.20,
) -> np.ndarray:
    """Per-channel symmetric **residual** bounds (lo, hi) centered at 0.

    Under residual-on-template, the optimizer searches for an additive
    delta on each free channel. The natural bound is ``±delta_frac × W``
    where ``W`` is the template's own dynamic range on that channel
    (or ``delta_frac × global_range × 0.10`` if the template is constant).

    Hard cap: the bound is clipped so that ``template + max_residual``
    never exceeds the global per-channel bounds anywhere along the
    trajectory. Concretely::

        lo = max(-delta, ch.global_min - col.min())
        hi = min( delta, ch.global_max - col.max())

    Returns array of shape ``(len(channels), 2)`` — columns ``(lo, hi)``.

    The returned interval always contains 0 (so the residual=0 init is a
    legal trial and ``build_trajectory`` reproduces the template). If the
    template is at a global bound, the interval is one-sided.
    """
    if template.ndim != 2 or template.shape[1] != 30:
        raise ValueError(f"template must be (T, 30); got {template.shape}")

    bounds = np.empty((len(channels), 2), dtype=np.float64)
    for i, ch in enumerate(channels):
        col = template[:, ch.idx]
        col_min = float(col.min())
        col_max = float(col.max())
        template_range = col_max - col_min
        global_range = ch.global_max - ch.global_min
        if template_range > 0:
            delta = delta_frac * template_range
        else:
            # Constant template on this channel — open a small window so the
            # optimizer has SOME slack, but tighter than the global range.
            delta = max(delta_frac * global_range * 0.10, 1e-3 * global_range)

        # Symmetric residual interval around 0, then clamp so adding the
        # residual never pushes any column value past the global bounds.
        lo = max(-delta, ch.global_min - col_min)
        hi = min(+delta, ch.global_max - col_max)
        # Force the interval to contain 0 even at extreme edges (so init=0
        # is always a legal trial).
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        if hi <= lo:
            # Degenerate (shouldn't happen now that 0 is always inside);
            # guarantee a tiny window so optimizer doesn't crash.
            lo, hi = -1e-6, 1e-6
        bounds[i] = (lo, hi)
    return bounds


def template_initial_knots(
    template: np.ndarray,
    knot_times_s: np.ndarray,
    knot_rate_hz: float,
    channels: tuple[FreeChannel, ...] = DEFAULT_FREE_CHANNELS,
) -> np.ndarray:
    """Initial **residual** knot values: zeros.

    Residual-on-template parameterization: the optimizer's "do nothing"
    initial guess is residual=0, which makes ``build_trajectory`` return
    the template exactly. ``knot_times_s`` and ``template`` are taken for
    API parity (and so callers can swap in alternate inits later) but only
    the shape ``(K, C)`` is used.

    Returns shape ``(K, len(channels))``.
    """
    if template.ndim != 2 or template.shape[1] != 30:
        raise ValueError(f"template must be (T, 30); got {template.shape}")
    return np.zeros((len(knot_times_s), len(channels)), dtype=np.float64)


def _interp_residual(
    knot_times_s: np.ndarray,
    knot_values: np.ndarray,
    times_s: np.ndarray,
    kind: str,
) -> np.ndarray:
    """Interpolate ``knot_values`` (K,) at ``knot_times_s`` to ``times_s``.

    ``kind`` is one of ``INTERP_KINDS``. All kinds preserve the identity
    ``zero knots → zero residual``.
    """
    if kind == "linear":
        return np.interp(times_s, knot_times_s, knot_values)
    if kind == "cubic":
        # Fritsch-Carlson PCHIP: shape-preserving piecewise cubic Hermite.
        # Interpolated values stay between adjacent knot extrema by
        # construction — fixes R-V4-1 (natural-cubic overshoot beyond
        # search bounds). Identity preserved: y == 0 ⇒ all slopes zero ⇒
        # output identically zero.
        if len(knot_times_s) < 2:
            return np.interp(times_s, knot_times_s, knot_values)
        return _pchip_interpolate(knot_times_s, knot_values, times_s)
    raise ValueError(f"unknown interp kind {kind!r}; want one of {INTERP_KINDS}")


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson PCHIP knot slopes.

    Returns per-knot slope ``m[i]`` such that the piecewise cubic Hermite
    through ``(x[i], y[i], m[i])`` is monotone on each segment whenever
    the data is locally monotone. y == 0 implies all secants and slopes
    are 0, so the interpolant is identically zero.
    """
    n = x.shape[0]
    h = np.diff(x)                    # (n-1,)
    delta = np.diff(y) / h             # secant slopes (n-1,)
    m = np.zeros(n, dtype=np.float64)

    if n == 2:
        # Linear segment: slope is the single secant.
        m[:] = delta[0]
        return m

    # Interior knots: weighted harmonic mean when secants share sign, else 0.
    same_sign = np.sign(delta[:-1]) * np.sign(delta[1:]) > 0
    w1 = 2.0 * h[1:] + h[:-1]
    w2 = h[1:] + 2.0 * h[:-1]
    # Avoid division by zero where same_sign is False (the result is masked
    # out anyway). Use safe sentinel `1.0` for any zero secant.
    safe_d_left = np.where(delta[:-1] != 0, delta[:-1], 1.0)
    safe_d_right = np.where(delta[1:] != 0, delta[1:], 1.0)
    denom = w1 / safe_d_left + w2 / safe_d_right
    safe_denom = np.where(denom != 0, denom, 1.0)
    interior = np.where(
        same_sign & (delta[:-1] != 0) & (delta[1:] != 0),
        (w1 + w2) / safe_denom,
        0.0,
    )
    m[1:-1] = interior

    # Endpoint slopes per Fritsch-Carlson (non-overshooting one-sided
    # three-point estimate).
    def _endpoint_slope(h0, h1, d0, d1):
        s = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
        if np.sign(s) != np.sign(d0):
            return 0.0
        if np.sign(d0) != np.sign(d1) and abs(s) > abs(3.0 * d0):
            return 3.0 * d0
        return s

    m[0] = _endpoint_slope(h[0], h[1], delta[0], delta[1])
    m[-1] = _endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    return m


def _pchip_interpolate(x: np.ndarray, y: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson PCHIP evaluation. ``x`` strictly increasing.

    Identity property: y == 0 ⇒ result == 0 (all slopes vanish, all
    Hermite basis terms are scaled by zero).

    Shape-preserving property: ``min(y[i], y[i+1]) <= result <= max(y[i], y[i+1])``
    for any ``x[i] <= x_eval <= x[i+1]``. This is what closes R-V4-1.
    """
    n = x.shape[0]
    if n < 2:
        return np.full_like(x_eval, fill_value=float(y[0]) if n else 0.0, dtype=np.float64)
    m = _pchip_slopes(x, y)
    h = np.diff(x)

    # Locate segment for each evaluation point.
    idx = np.searchsorted(x, x_eval, side="right") - 1
    idx = np.clip(idx, 0, n - 2)

    h_i = h[idx]
    t = (x_eval - x[idx]) / h_i  # in [0, 1] within each segment
    t2 = t * t
    t3 = t2 * t
    # Hermite basis.
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    return (
        h00 * y[idx]
        + h10 * h_i * m[idx]
        + h01 * y[idx + 1]
        + h11 * h_i * m[idx + 1]
    )


def build_trajectory(
    template: np.ndarray,
    knot_values: np.ndarray,
    knot_times_s: np.ndarray,
    knot_rate_hz: float,
    channels: tuple[FreeChannel, ...] = DEFAULT_FREE_CHANNELS,
    interp_kind: str = "linear",
) -> np.ndarray:
    """Build a (T, 30) trajectory by ADDING an interpolated residual to the
    template on each free channel.

    **Identity:** ``build_trajectory(template, zeros, ...) == template``
    bit-for-bit (within ``1e-12``; the only operation on free channels is
    `+ 0.0`, which is exact in IEEE 754).

    Args:
        template: (T, 30) template trajectory at ``knot_rate_hz``.
        knot_values: (K, len(channels)) **residual** knot values for each
            free channel (0 means "stay on template").
        knot_times_s: (K,) time positions of each knot, in seconds.
        knot_rate_hz: rate of the output trajectory (matches ``template``).
        channels: free channels.
        interp_kind: one of ``INTERP_KINDS``. ``"linear"`` is the Phase A
            default and matches ``np.interp``. ``"cubic"`` uses a natural
            cubic spline and is the Phase B option for less rough between-
            knot motion.

    Returns:
        (T, 30) trajectory ready for ``the legacy C++ synth path``. The non-free 22
        channels are byte-identical to the template; free channels equal
        ``template[:, idx] + interp(residual_knots)``. Final values are
        clipped to per-channel global bounds so renderer never sees an
        out-of-range value (the residual bounds make this rare, but still).
    """
    if template.ndim != 2 or template.shape[1] != 30:
        raise ValueError(f"template must be (T, 30); got {template.shape}")
    if knot_values.shape != (len(knot_times_s), len(channels)):
        raise ValueError(
            f"knot_values shape {knot_values.shape} != "
            f"({len(knot_times_s)}, {len(channels)})"
        )
    if interp_kind not in INTERP_KINDS:
        raise ValueError(
            f"interp_kind {interp_kind!r} not in {INTERP_KINDS}"
        )

    n_frames = template.shape[0]
    times_s = np.arange(n_frames, dtype=np.float64) / knot_rate_hz
    out = template.astype(np.float64, copy=True)
    for i, ch in enumerate(channels):
        residual = _interp_residual(knot_times_s, knot_values[:, i], times_s, interp_kind)
        out[:, ch.idx] = out[:, ch.idx] + residual
        # Clip to global bounds (defensive — derive_bounds already shapes
        # the search space so this normally never fires).
        np.clip(out[:, ch.idx], ch.global_min, ch.global_max, out=out[:, ch.idx])
    return out


def make_uniform_knot_times(duration_s: float, n_knots: int, edge_pad_s: float = 0.0) -> np.ndarray:
    """K uniformly-spaced knots covering [edge_pad_s, duration_s - edge_pad_s].

    Knots include the endpoints so interpolation never extrapolates.
    """
    if n_knots < 2:
        raise ValueError(f"need at least 2 knots; got {n_knots}")
    a = edge_pad_s
    b = max(a + 1e-3, duration_s - edge_pad_s)
    return np.linspace(a, b, n_knots, dtype=np.float64)


def fit_metadata(
    *,
    interp_kind: str,
    knot_rate_hz: float,
    channels: tuple[FreeChannel, ...],
    backend: str,
    init_loss: float,
    fit_loss: float,
) -> dict:
    """Stamp the load-bearing metadata that should land on every fit NPZ.

    Per plan/70-handoff-2026-05-03.md §"Cross-cutting (lands with T1)":
    every fit's NPZ output stamps interp_kind, knot_rate_hz, channels,
    backend, init_loss, fit_loss, jax/jaxlib versions, and jax devices.
    Keeping this in one helper means the cross-runner stamping stays
    consistent.

    Returns a flat dict — savez will store each value as an array.
    """
    try:
        import jax  # type: ignore[import-not-found]
        import jaxlib  # type: ignore[import-not-found]
        jax_version = str(getattr(jax, "__version__", "unknown"))
        jaxlib_version = str(getattr(jaxlib, "__version__", "unknown"))
        jax_devices = ",".join(str(d) for d in jax.devices())
    except Exception:  # noqa: BLE001
        jax_version = "unavailable"
        jaxlib_version = "unavailable"
        jax_devices = ""

    channel_names = ",".join(ch.name for ch in channels)
    return {
        "interp_kind": np.asarray(interp_kind),
        "knot_rate_hz": np.asarray(float(knot_rate_hz)),
        "channels": np.asarray(channel_names),
        "backend": np.asarray(backend),
        "init_loss": np.asarray(float(init_loss)),
        "fit_loss": np.asarray(float(fit_loss)),
        "jax_version": np.asarray(jax_version),
        "jaxlib_version": np.asarray(jaxlib_version),
        "jax_devices": np.asarray(jax_devices),
    }


def normalized_knot_error(
    fitted: np.ndarray,
    target: np.ndarray,
    bounds: np.ndarray,
) -> float:
    """Mean per-knot absolute error normalized to per-channel bound width.

    fitted, target shape: (K, C). bounds shape: (C, 2). Returns a scalar in [0, ~1+].
    """
    if fitted.shape != target.shape:
        raise ValueError(f"shape mismatch {fitted.shape} vs {target.shape}")
    width = bounds[:, 1] - bounds[:, 0]  # (C,)
    width = np.where(width <= 0, 1.0, width)
    err = np.abs(fitted - target) / width[None, :]
    return float(err.mean())
