"""50 Hz frame interpolator — composes asynchronous overlapping
GestureEvent lists into a `(T, D_gesture)` numpy array.

Contract (load-bearing per ARCH §2.2)
-------------------------------------

For each gesture channel (column of the output array):

1. Start at the channel's neutral baseline value (see
   :data:`channels.GESTURE_NEUTRAL`).
2. For each ``GestureEvent`` on that channel, emit a per-event ramp
   shape:

   - ``hold_release`` (default): linear ramp from baseline to
     ``target_value`` over ``min(0.04 s, duration_s)``, hold for
     remainder of ``duration_s``, then linear release back to baseline
     over ``release_s`` (default 0.0 = instantaneous).
   - ``pchip``: monotone cubic Hermite ramp; same hold/release.
   - ``critically_damped``: 2nd-order step response ``1 - (1 + s) e^{-s}``
     where ``s = (t - onset) / tau`` and ``tau = duration_s / 4``.
   - ``ballistic``: cosine raised pulse; release back to neutral over
     ``release_s``.

3. Multiple overlapping events on the same channel **compose by
   strength-weighted average** of their (active) per-event contributions.
   When the sum of strengths drops to zero, the baseline is returned.

The interpolator is implemented in pure numpy (CPU); it's used at
corpus generation time and at inference dispatch, not inside the JAX
backward pass. The neural-field body model consumes the resulting
`(T, 20)` array as a per-frame conditioning input — encoded to a
small `z_t` and fed to `delta_θ(s, z_t, speaker_vec)`.

Frame rate: **50 Hz** (T_frame = 20 ms). This is the reference rate
the body model trains at; per ARCH §2.3 the renderer interface is
expanded at this rate. The choice of 50 Hz is fixed here; if a future
session wants 100 Hz it changes :data:`FRAME_RATE_HZ` and re-renders
the corpus.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..gestures import GestureEvent
from .channels import (
    GESTURE_CHANNELS,
    GESTURE_NEUTRAL,
    N_GESTURE_CHANNELS,
    gesture_neutral_vec,
)


FRAME_RATE_HZ: float = 50.0
FRAME_DT_S: float = 1.0 / FRAME_RATE_HZ


def _ramp_attack(progress: np.ndarray, shape: str) -> np.ndarray:
    """progress is in [0, 1]; returns the attack/hold/release blending
    profile for that progress fraction. Used for the per-event window."""
    if shape == "hold_release":
        # Linear up to 1.0 fast, then hold.
        return np.minimum(progress * 4.0, 1.0)
    if shape == "pchip":
        # Smoothstep (3p^2 - 2p^3) is the simplest monotone cubic that
        # matches PCHIP-on-2-knots identically. Cheaper than the full
        # PCHIP routine and has bounded derivative at endpoints.
        p = np.clip(progress, 0.0, 1.0)
        return p * p * (3.0 - 2.0 * p)
    if shape == "critically_damped":
        # 1 - (1 + s) e^{-s}. tau is implicitly progress * 4 (so the
        # response is ~98% complete by progress=1).
        s = progress * 4.0
        return 1.0 - (1.0 + s) * np.exp(-s)
    if shape == "ballistic":
        # Raised cosine over [0, 1].
        p = np.clip(progress, 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * p)
    raise ValueError(f"unknown ramp_shape: {shape!r}")


def _event_window(
    event: GestureEvent,
    t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (active_mask, contribution) for a single event on the
    timeline ``t`` (shape (T,) in seconds).

    ``contribution`` is a fraction in [0, 1] representing how much the
    event has reached its target_value. The caller multiplies it by
    ``(target_value - baseline) * strength`` and adds to the baseline.
    """
    onset = event.onset_s
    dur = event.duration_s
    rel = event.release_s if event.release_s is not None else 0.0
    end = onset + dur + rel

    active = (t >= onset) & (t <= end)
    contrib = np.zeros_like(t)

    # Attack / hold phase
    if dur > 0:
        in_hold = active & (t < onset + dur)
        progress = np.clip((t - onset) / dur, 0.0, 1.0)
        contrib = np.where(
            in_hold,
            _ramp_attack(progress, event.ramp_shape),
            contrib,
        )
        # End of hold: full target value
        contrib = np.where(
            active & (t >= onset + dur),
            1.0,
            contrib,
        )
    # Release phase: linear back to 0 over `rel` seconds
    if rel > 0:
        in_release = active & (t >= onset + dur)
        rel_progress = np.clip((t - (onset + dur)) / rel, 0.0, 1.0)
        contrib = np.where(in_release, 1.0 - rel_progress, contrib)

    return active, contrib


def compose_gesture_trajectory(
    events: Iterable[GestureEvent],
    duration_s: float,
    frame_rate_hz: float = FRAME_RATE_HZ,
) -> np.ndarray:
    """Compose a list of overlapping GestureEvents into a (T, D_gesture)
    trajectory at ``frame_rate_hz`` Hz.

    Parameters
    ----------
    events
        Iterable of :class:`GestureEvent`. Channel IDs that are not in
        :data:`GESTURE_CHANNELS` are silently dropped (with no error)
        — this keeps the interpolator robust against schema drift in
        upstream callers. A future strict-mode kwarg can flip this.
    duration_s
        Total trajectory duration. Will be quantized up to the next
        whole frame.
    frame_rate_hz
        Frame rate. Default 50 Hz (the body-model contract).

    Returns
    -------
    np.ndarray of shape ``(T, D_gesture)``, dtype float32. Frame ``i`` is
    the channel state at time ``(i + 0.5) / frame_rate_hz`` (frame
    centers).
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be > 0, got {duration_s}")
    dt = 1.0 / frame_rate_hz
    T = int(np.ceil(duration_s * frame_rate_hz))
    # Frame centers
    t = (np.arange(T) + 0.5) * dt

    # Per-channel accumulator (target_delta * strength * contribution),
    # weighted by sum of strengths. Initialize from neutral.
    out = np.broadcast_to(
        gesture_neutral_vec(), (T, N_GESTURE_CHANNELS)
    ).copy()

    # Group events by channel index (skip unknown channels)
    by_channel: dict[int, list[GestureEvent]] = {}
    for ev in events:
        if ev.channel_id not in GESTURE_CHANNELS:
            continue
        idx = GESTURE_CHANNELS.index(ev.channel_id)
        by_channel.setdefault(idx, []).append(ev)

    for col, evs in by_channel.items():
        baseline = float(GESTURE_NEUTRAL[GESTURE_CHANNELS[col]])
        # weighted combination of (target * strength * contrib) numerators
        # divided by sum of (strength * contrib) denominators. Where
        # denominator is 0, fall back to baseline.
        num = np.zeros(T, dtype=np.float64)
        den = np.zeros(T, dtype=np.float64)
        for ev in evs:
            active, contrib = _event_window(ev, t)
            w = ev.strength * contrib
            num += w * ev.target_value
            den += w
        active_mask = den > 1e-9
        composed = np.where(active_mask, num / np.maximum(den, 1e-12), baseline)
        out[:, col] = composed.astype(np.float32)

    return out.astype(np.float32, copy=False)
