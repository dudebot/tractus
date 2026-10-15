"""GestureEvent dataclass — ARCH §315.

Asynchronous, overlapping articulatory/source events. Per
`ARCHITECTURE.md` §315 the canonical schema is:

    @dataclass(frozen=True)
    class GestureEvent:
        channel_id:   str
        onset_s:      float
        target_value: float
        duration_s:   float
        release_s:    float | None
        strength:     float
        ramp_shape:   Literal["pchip", "critically_damped",
                              "ballistic", "hold_release"]
        prosody_tags: tuple[str, ...]
        affect_tags:  tuple[str, ...]

The smoke/001 path uses GestureEvent only as a typed bag for sidecar
provenance and a forward-compatible interface; gesture *composition*
into a continuous body-state trajectory is owned by
correctness/004 + pretraining/003. Smoke renders the events directly
through a per-channel hold/release ramp on the 30-d (T, 30) trajectory
template — no pretraining/003 controller in the loop.

Existing placeholder tests in ``tests/test_gesture_event.py`` were
written against an older proposed schema (``channel`` / ``t_start`` /
``t_end`` / ``target``). Those tests are ``pytest.skip``ed and slated
for rewrite by correctness/004; this module follows ARCH §315.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

RampShape = Literal["pchip", "critically_damped", "ballistic", "hold_release"]

VALID_RAMP_SHAPES: frozenset[str] = frozenset(
    {"pchip", "critically_damped", "ballistic", "hold_release"}
)


@dataclass(frozen=True)
class GestureEvent:
    """One asynchronous gesture intent on a single channel.

    Per ARCH §315. ``target_value`` is in physical units (cm, Hz, dPa,
    etc.) interpreted by the receiving channel; consumers are expected
    to know each channel's units.
    """

    channel_id: str
    onset_s: float
    target_value: float
    duration_s: float
    release_s: float | None = None
    strength: float = 1.0
    ramp_shape: RampShape = "hold_release"
    prosody_tags: tuple[str, ...] = field(default_factory=tuple)
    affect_tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.duration_s <= 0:
            raise ValueError(
                f"duration_s must be > 0, got {self.duration_s!r}"
            )
        if self.release_s is not None and self.release_s < 0:
            raise ValueError(
                f"release_s must be >= 0 if set, got {self.release_s!r}"
            )
        if not (0.0 <= self.strength <= 1.0):
            raise ValueError(
                f"strength must be in [0, 1], got {self.strength!r}"
            )
        if self.ramp_shape not in VALID_RAMP_SHAPES:
            raise ValueError(
                f"ramp_shape must be one of {sorted(VALID_RAMP_SHAPES)}, "
                f"got {self.ramp_shape!r}"
            )

    @property
    def offset_s(self) -> float:
        """Time of the start of release (end of hold). For composition
        the gesture's audible influence may extend through release_s."""
        return self.onset_s + self.duration_s

    def end_s(self) -> float:
        """Last time the gesture exerts influence (offset + release)."""
        return self.offset_s + (self.release_s or 0.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GestureEvent:
        prosody = tuple(d.get("prosody_tags") or ())
        affect = tuple(d.get("affect_tags") or ())
        return cls(
            channel_id=d["channel_id"],
            onset_s=float(d["onset_s"]),
            target_value=float(d["target_value"]),
            duration_s=float(d["duration_s"]),
            release_s=(
                None if d.get("release_s") is None else float(d["release_s"])
            ),
            strength=float(d.get("strength", 1.0)),
            ramp_shape=d.get("ramp_shape", "hold_release"),
            prosody_tags=prosody,
            affect_tags=affect,
        )


def sort_by_onset(events: list[GestureEvent]) -> list[GestureEvent]:
    """Stable sort a list of GestureEvents by onset time."""
    return sorted(events, key=lambda e: (e.onset_s, e.channel_id))
