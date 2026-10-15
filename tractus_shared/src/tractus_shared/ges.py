"""GestureScore: Python data model for the `.ges` XML format used by
the TEA fixture parity tests (ARCH §10).

Read, write, and programmatically build a `GestureScore`. Phoneme-level
semantics (which tier values produce a given phone) are not represented
here — this is the file-format layer only.

Format::

    <gestural_score>
      <gesture_sequence type="vowel-gestures" unit="">
        <gesture value="a" slope="0.000000" duration_s="0.600000"
                 time_constant_s="0.012000" neutral="0" />
        ...
      </gesture_sequence>
      ... 8 sequences total, fixed names ...
    </gestural_score>

For the numeric tiers (``f0-gestures`` in semitones, ``lung-pressure-gestures``
in dPa) the gesture's ``value`` attribute is a number; otherwise it's a
named target shape resolved by the speaker file (e.g. ``"a"``, ``"modal"``,
``"tt-alveolar-closure"``).
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Iterator

# Tier order is part of the file format — preserve it on round-trip.
TIER_TYPES: tuple[str, ...] = (
    "vowel-gestures",
    "lip-gestures",
    "tongue-tip-gestures",
    "tongue-body-gestures",
    "velic-gestures",
    "glottal-shape-gestures",
    "f0-gestures",
    "lung-pressure-gestures",
)

# Tiers whose ``value`` attribute is numeric (everything else is a shape name).
NUMERIC_TIERS: frozenset[str] = frozenset({"f0-gestures", "lung-pressure-gestures"})

# Default unit on the f0 tier observed upstream.
_DEFAULT_UNITS: dict[str, str] = {
    "f0-gestures": "st",
    "lung-pressure-gestures": "dPa",
}


def _format_float(x: float) -> str:
    """Match upstream's "%.6f" formatting so round-trip diffs are clean."""
    return f"{x:.6f}"


@dataclass
class Gesture:
    """One gesture event on one tier.

    `value` is stored as a string so we can losslessly round-trip both
    named target shapes ("a", "modal") and the numeric tiers.
    """

    value: str = ""
    slope: float = 0.0
    duration_s: float = 0.0
    time_constant_s: float = 0.012
    neutral: bool = False

    def to_element(self) -> ET.Element:
        attrib = {
            "value": self.value,
            "slope": _format_float(self.slope),
            "duration_s": _format_float(self.duration_s),
            "time_constant_s": _format_float(self.time_constant_s),
            "neutral": "1" if self.neutral else "0",
        }
        return ET.Element("gesture", attrib=attrib)

    @classmethod
    def from_element(cls, el: ET.Element) -> "Gesture":
        return cls(
            value=el.get("value", ""),
            slope=float(el.get("slope", "0")),
            duration_s=float(el.get("duration_s", "0")),
            time_constant_s=float(el.get("time_constant_s", "0")),
            neutral=el.get("neutral", "0") == "1",
        )


@dataclass
class GestureTier:
    """One <gesture_sequence> in the .ges XML."""

    type: str
    unit: str = ""
    gestures: list[Gesture] = field(default_factory=list)

    KNOWN: ClassVar[tuple[str, ...]] = TIER_TYPES

    def add(self, *gestures: Gesture) -> "GestureTier":
        self.gestures.extend(gestures)
        return self

    def total_duration_s(self) -> float:
        return sum(g.duration_s for g in self.gestures)

    def is_numeric(self) -> bool:
        return self.type in NUMERIC_TIERS

    def to_element(self) -> ET.Element:
        seq = ET.Element("gesture_sequence", attrib={"type": self.type, "unit": self.unit})
        for g in self.gestures:
            seq.append(g.to_element())
        return seq

    @classmethod
    def from_element(cls, el: ET.Element) -> "GestureTier":
        tier = cls(type=el.get("type", ""), unit=el.get("unit", ""))
        for child in el.findall("gesture"):
            tier.gestures.append(Gesture.from_element(child))
        return tier


@dataclass
class GestureScore:
    """Top-level <gestural_score>. Tiers are kept in their declared order."""

    tiers: list[GestureTier] = field(default_factory=list)

    # ----- factories -----

    @classmethod
    def empty(cls) -> "GestureScore":
        """Score with all 8 tiers present and empty — the canonical baseline."""
        return cls(
            tiers=[
                GestureTier(type=t, unit=_DEFAULT_UNITS.get(t, ""))
                for t in TIER_TYPES
            ]
        )

    @classmethod
    def read(cls, path: str | Path) -> "GestureScore":
        tree = ET.parse(str(path))
        root = tree.getroot()
        if root.tag != "gestural_score":
            raise ValueError(
                f"expected <gestural_score> root, got <{root.tag}> at {path}"
            )
        tiers = [GestureTier.from_element(seq) for seq in root.findall("gesture_sequence")]
        return cls(tiers=tiers)

    # ----- mutation -----

    def tier(self, type_: str) -> GestureTier:
        for t in self.tiers:
            if t.type == type_:
                return t
        if type_ not in TIER_TYPES:
            raise KeyError(f"unknown tier type {type_!r}; must be one of {TIER_TYPES}")
        new = GestureTier(type=type_, unit=_DEFAULT_UNITS.get(type_, ""))
        self.tiers.append(new)
        return new

    def __iter__(self) -> Iterator[GestureTier]:
        return iter(self.tiers)

    # ----- output -----

    def to_xml(self) -> str:
        root = ET.Element("gestural_score")
        for tier in self.tiers:
            root.append(tier.to_element())
        ET.indent(root, space="  ")
        body = ET.tostring(root, encoding="unicode")
        # ET self-closes empty elements; upstream keeps an explicit closing
        # tag for empty <gesture_sequence>. Patch only that case so leaf
        # <gesture/> elements stay self-closed (matching upstream).
        body = re.sub(
            r'<gesture_sequence ([^>/]*?) />',
            r'<gesture_sequence \1>\n  </gesture_sequence>',
            body,
        )
        return body + "\n"

    def write(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_xml())
        return out


# ----------------------------------------------------------------------------
# Equality helpers used by the round-trip test.
# Float comparisons go through an absolute tolerance because upstream rounds
# everything to 6 decimals.
# ----------------------------------------------------------------------------


def _gestures_equal(a: Gesture, b: Gesture, *, atol: float = 1e-6) -> bool:
    return (
        a.value == b.value
        and abs(a.slope - b.slope) <= atol
        and abs(a.duration_s - b.duration_s) <= atol
        and abs(a.time_constant_s - b.time_constant_s) <= atol
        and a.neutral == b.neutral
    )


def scores_structurally_equal(
    a: GestureScore, b: GestureScore, *, atol: float = 1e-6
) -> bool:
    if [t.type for t in a.tiers] != [t.type for t in b.tiers]:
        return False
    for ta, tb in zip(a.tiers, b.tiers):
        if ta.type != tb.type or ta.unit != tb.unit:
            return False
        if len(ta.gestures) != len(tb.gestures):
            return False
        for ga, gb in zip(ta.gestures, tb.gestures):
            if not _gestures_equal(ga, gb, atol=atol):
                return False
    return True
