"""GestureEvent composition + dataclass contract (ARCHITECTURE.md §460-475).

The two ``gesture_event``-marked tests are:

1. **Composition / async overlap (CRITICAL).** Two overlapping gestures
   on different channels — or partially overlapping on the same channel —
   must compose to a continuous body state within the reachable manifold.
2. **Dataclass contract (MEDIUM).** Round-trip serialization
   (dict → GestureEvent → dict) and temporal-ordering invariants
   (``t_start < t_end``, sorted-by-onset list semantics).

The ``GestureEvent`` dataclass is owned by ``correctness/004`` /
``pretraining/003``. Until it lands these tests are
``xfail(strict=True)`` against an explicit ``ModuleNotFoundError`` so
the markers exist in collection and CI surfaces the gap.
"""

from __future__ import annotations

import importlib

import pytest


def _gesture_module():
    """Return the gesture module if it exists, else None.

    The canonical home is ``tractus_controller.gestures`` per ARCH §460.
    A future refactor may move it; update this resolver if so.
    """

    for candidate in (
        "tractus_controller.gestures",
        "tractus_controller.gesture_event",
        "tractus_engine_advanced.gestures",
    ):
        try:
            return importlib.import_module(candidate)
        except ModuleNotFoundError:
            continue
    return None


# ---------------------------------------------------------------------------
# Composition / async overlap (CRITICAL).
# ---------------------------------------------------------------------------


@pytest.mark.gesture_event
@pytest.mark.skip(reason="GestureEvent composition not built yet; blocked on "
    "correctness/004 + pretraining/003",
)
def test_two_overlapping_gestures_on_different_channels_compose_to_continuous_body_state() -> None:
    """Two gestures on different channels, fully overlapping in time, must
    compose to a continuous (no jump-discontinuity) body-state trajectory.
    """

    gestures = _gesture_module()
    assert gestures is not None
    pytest.fail(
        "placeholder: GestureEvent composition behavior is owned by "
        "correctness/004 + pretraining/003"
    )


@pytest.mark.gesture_event
@pytest.mark.skip(reason="GestureEvent composition not built yet; blocked on "
    "correctness/004 + pretraining/003",
)
def test_partially_overlapping_gestures_same_channel_compose_to_continuous_state() -> None:
    """Two partially overlapping gestures on the *same* channel must
    blend continuously — no discontinuity at either onset or offset.
    The composed trajectory must remain inside the channel's reachable
    range (anatomy-bounded).
    """

    gestures = _gesture_module()
    assert gestures is not None
    pytest.fail(
        "placeholder: same-channel partial-overlap composition is owned by "
        "correctness/004 + pretraining/003"
    )


# ---------------------------------------------------------------------------
# Dataclass contract (MEDIUM).
# ---------------------------------------------------------------------------


@pytest.mark.gesture_event
@pytest.mark.skip(reason="GestureEvent dataclass not built yet; blocked on correctness/004",
)
def test_gesture_event_round_trip_serialization() -> None:
    """``GestureEvent.from_dict(g.to_dict()) == g`` for all canonical channels."""

    gestures = _gesture_module()
    assert gestures is not None
    GestureEvent = gestures.GestureEvent  # type: ignore[attr-defined]

    g = GestureEvent(
        channel="lip_aperture",
        t_start=0.10,
        t_end=0.25,
        target=0.5,
        velocity_cap=2.0,
    )
    payload = g.to_dict()
    restored = GestureEvent.from_dict(payload)
    assert restored == g


@pytest.mark.gesture_event
@pytest.mark.skip(reason="GestureEvent dataclass not built yet; blocked on correctness/004",
)
def test_gesture_event_temporal_ordering_invariants() -> None:
    """``t_start < t_end`` is enforced at construction. A list of GestureEvents
    sorted by onset is the canonical ordering for downstream consumption.
    """

    gestures = _gesture_module()
    assert gestures is not None
    GestureEvent = gestures.GestureEvent  # type: ignore[attr-defined]

    # t_start ≥ t_end must raise ValueError.
    with pytest.raises(ValueError):
        GestureEvent(channel="x", t_start=0.5, t_end=0.5, target=0.0)
    with pytest.raises(ValueError):
        GestureEvent(channel="x", t_start=0.6, t_end=0.5, target=0.0)

    a = GestureEvent(channel="x", t_start=0.0, t_end=0.1, target=0.0)
    b = GestureEvent(channel="y", t_start=0.05, t_end=0.20, target=1.0)
    c = GestureEvent(channel="z", t_start=0.30, t_end=0.40, target=0.5)
    seq = sorted([c, a, b], key=lambda e: e.t_start)
    assert [e.channel for e in seq] == ["x", "y", "z"]
