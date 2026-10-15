"""Per-phone default durations + jitter."""

from __future__ import annotations

import pytest

from tractus_controller.phonemes import durations as d
from tractus_controller.phonemes.arpabet import UnsupportedPhone


def test_unstressed_consonants_use_class_defaults() -> None:
    assert d.duration_of("D") == d.BASE_DURATION_S["stop"]
    assert d.duration_of("S") == d.BASE_DURATION_S["fricative"]
    assert d.duration_of("M") == d.BASE_DURATION_S["nasal"]


def test_primary_stress_lengthens_vowels() -> None:
    base = d.base_duration_of("EY1")
    assert d.duration_of("EY1") == pytest.approx(base * d.STRESS_MULTIPLIER["1"])
    assert d.duration_of("EY1") > d.duration_of("EY0")


def test_zero_stress_shortens() -> None:
    assert d.duration_of("IY0") < d.base_duration_of("IY0")


def test_oov_phone_raises() -> None:
    # OW is in the expanded English inventory; use a bogus phone for OOV.
    with pytest.raises(UnsupportedPhone):
        d.duration_of("XX1")


def test_default_durations_produces_one_per_token() -> None:
    seq = ["D", "EY1", "Z", "IY0"]
    out = d.default_durations(seq)
    assert len(out) == len(seq)
    assert all(x > 0 for x in out)


def test_jitter_is_reproducible_under_same_seed() -> None:
    seq = ["D", "EY1", "Z", "IY0"]
    a = d.jittered_durations(seq, d.JitterConfig(jitter=0.2, seed=42))
    b = d.jittered_durations(seq, d.JitterConfig(jitter=0.2, seed=42))
    assert a == b


def test_jitter_differs_under_different_seed() -> None:
    seq = ["D", "EY1", "Z", "IY0"]
    a = d.jittered_durations(seq, d.JitterConfig(jitter=0.2, seed=1))
    b = d.jittered_durations(seq, d.JitterConfig(jitter=0.2, seed=2))
    assert a != b


def test_jitter_stays_within_bounds() -> None:
    seq = ["D", "EY1", "Z", "IY0", "G", "IH1", "V", "M"]
    base = d.default_durations(seq)
    jittered = d.jittered_durations(seq, d.JitterConfig(jitter=0.15, seed=99))
    for b, j in zip(base, jittered):
        # Compare against the unstressed base since duration_of applies
        # stress AND default_durations does too — the jitter ratio is on
        # the stressed default. Bounds: ±15%.
        ratio = j / b
        assert 0.85 - 1e-9 <= ratio <= 1.15 + 1e-9, (ratio, b, j)


def test_silence_padding_constants_are_safe_for_pressure_ramp() -> None:
    """Per memory historical pressure-ramp note, leading silence must
    be >= 0.25s or word-initial consonants vanish under the ramp."""
    assert d.LEADING_SILENCE_S >= 0.25
    assert d.TRAILING_SILENCE_S > 0
