"""Per-phone default durations.

Used by ``synth-template`` and ``generate-training-data`` when the caller
doesn't supply explicit per-phone durations. Numbers are coarse averages
in the 60–250 ms range (vowels 80–120 ms, stops 60 ms, fricatives 80 ms);
refine by listening when a corpus is in hand.

These are *defaults*. Stress/word-boundary lengthening is a separate
pass; see ``apply_stress_lengthening``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .arpabet import (
    DAISY_SUBSET,
    UnsupportedPhone,
    phone_class_of,
    split_stress,
)


# Base durations per phone class, in seconds. Vowels are mid-range
# unstressed; stress lengthening is applied separately.
BASE_DURATION_S: dict[str, float] = {
    "vowel":       0.10,
    "stop":        0.06,
    "fricative":   0.08,
    "nasal":       0.07,
    "approximant": 0.07,
}

# Per-phone overrides (in seconds) when the class default isn't right.
# Diphthongs and long vowels need longer durations than monophthongs.
PHONE_OVERRIDE_S: dict[str, float] = {
    # Diphthongs / long vowels
    "EY": 0.15,
    "AY": 0.15,
    "AW": 0.15,
    "OY": 0.15,
    "OW": 0.15,
    "AA": 0.13,
    "AO": 0.13,
    "UW": 0.13,
    "AE": 0.12,
    "ER": 0.12,
    # Other vowels (short class default 0.10 fits, listed for explicit calibration)
    "IY": 0.10,
    "EH": 0.10,
    "AH": 0.09,
    "UH": 0.10,
    "IH": 0.09,
}

# Multipliers for ARPAbet stress digits.
STRESS_MULTIPLIER: dict[str, float] = {
    "0": 0.85,  # reduced
    "":  1.00,  # no stress digit (consonants, or unspecified)
    "2": 1.10,  # secondary
    "1": 1.30,  # primary
}

# Always-on padding for the leading and trailing silence segments.
LEADING_SILENCE_S: float = 0.30
TRAILING_SILENCE_S: float = 0.10


def base_duration_of(arpabet_token: str) -> float:
    """Default unstressed duration for a phone, in seconds."""
    phone, _ = split_stress(arpabet_token)
    if phone not in DAISY_SUBSET:
        raise UnsupportedPhone(phone)
    if phone in PHONE_OVERRIDE_S:
        return PHONE_OVERRIDE_S[phone]
    return BASE_DURATION_S[phone_class_of(arpabet_token)]


def duration_of(arpabet_token: str) -> float:
    """Stress-aware default duration for a phone, in seconds."""
    _, stress = split_stress(arpabet_token)
    return base_duration_of(arpabet_token) * STRESS_MULTIPLIER.get(stress, 1.0)


def default_durations(arpabet_tokens: list[str]) -> list[float]:
    return [duration_of(t) for t in arpabet_tokens]


@dataclass
class JitterConfig:
    """Multiplicative jitter for training-data variety.

    Each phone duration is multiplied by ``1 + uniform(-jitter, +jitter)``.
    """

    jitter: float = 0.15
    seed: int | None = None


def jittered_durations(
    arpabet_tokens: list[str], config: JitterConfig
) -> list[float]:
    rng = random.Random(config.seed)
    out = []
    for tok in arpabet_tokens:
        base = duration_of(tok)
        out.append(base * (1.0 + rng.uniform(-config.jitter, config.jitter)))
    return out
