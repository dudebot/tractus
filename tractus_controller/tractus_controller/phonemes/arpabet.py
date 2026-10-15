"""ARPAbet phone inventory and small utilities for the G2P / phoneme path.

The 39-phoneme CMU ARPAbet inventory used by the planner (ARCH §2.1)
and the phonemizer pipeline. ``DAISY_SUBSET`` is the canonical mapping
from each supported phone to its phone class and voicing; the canonical
"Daisy, Daisy" smoke fixture lives at
``tests/fixtures/daisy_phonemes.json``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PhoneClass = Literal["vowel", "stop", "fricative", "nasal", "approximant"]


@dataclass(frozen=True)
class PhoneMapping:
    arpabet: str
    phone_class: PhoneClass
    voiced: bool


DAISY_SUBSET: dict[str, PhoneMapping] = {
    # ---- Vowels (15) ------------------------------------------------
    "AA": PhoneMapping("AA", "vowel", True),
    "AE": PhoneMapping("AE", "vowel", True),
    "AH": PhoneMapping("AH", "vowel", True),
    "AO": PhoneMapping("AO", "vowel", True),
    "AW": PhoneMapping("AW", "vowel", True),
    "AY": PhoneMapping("AY", "vowel", True),
    "EH": PhoneMapping("EH", "vowel", True),
    "ER": PhoneMapping("ER", "vowel", True),
    "EY": PhoneMapping("EY", "vowel", True),
    "IH": PhoneMapping("IH", "vowel", True),
    "IY": PhoneMapping("IY", "vowel", True),
    "OW": PhoneMapping("OW", "vowel", True),
    "OY": PhoneMapping("OY", "vowel", True),
    "UH": PhoneMapping("UH", "vowel", True),
    "UW": PhoneMapping("UW", "vowel", True),
    # ---- Stops (6) --------------------------------------------------
    "B":  PhoneMapping("B",  "stop", True),
    "D":  PhoneMapping("D",  "stop", True),
    "G":  PhoneMapping("G",  "stop", True),
    "K":  PhoneMapping("K",  "stop", False),
    "P":  PhoneMapping("P",  "stop", False),
    "T":  PhoneMapping("T",  "stop", False),
    # ---- Affricates (2) — short-stop class for duration purposes ----
    "CH": PhoneMapping("CH", "stop", False),
    "JH": PhoneMapping("JH", "stop", True),
    # ---- Fricatives (9) ---------------------------------------------
    "DH": PhoneMapping("DH", "fricative", True),
    "F":  PhoneMapping("F",  "fricative", False),
    "HH": PhoneMapping("HH", "fricative", False),
    "S":  PhoneMapping("S",  "fricative", False),
    "SH": PhoneMapping("SH", "fricative", False),
    "TH": PhoneMapping("TH", "fricative", False),
    "V":  PhoneMapping("V",  "fricative", True),
    "Z":  PhoneMapping("Z",  "fricative", True),
    "ZH": PhoneMapping("ZH", "fricative", True),
    # ---- Nasals (3) -------------------------------------------------
    "M":  PhoneMapping("M",  "nasal", True),
    "N":  PhoneMapping("N",  "nasal", True),
    "NG": PhoneMapping("NG", "nasal", True),
    # ---- Approximants & lateral (4) ---------------------------------
    "L":  PhoneMapping("L",  "approximant", True),
    "R":  PhoneMapping("R",  "approximant", True),
    "W":  PhoneMapping("W",  "approximant", True),
    "Y":  PhoneMapping("Y",  "approximant", True),
}

DAISY_PHONES: frozenset[str] = frozenset(DAISY_SUBSET)


class UnsupportedPhone(KeyError):
    """ARPAbet phone is not in the currently supported subset."""


def split_stress(arpabet_token: str) -> tuple[str, str]:
    """Split "EY1" → ("EY", "1"), "D" → ("D", "")."""
    if arpabet_token and arpabet_token[-1].isdigit():
        return arpabet_token[:-1], arpabet_token[-1]
    return arpabet_token, ""


def is_supported(arpabet_token: str) -> bool:
    """True if the (un-stressed) phone is in the supported inventory."""
    phone, _ = split_stress(arpabet_token)
    return phone in DAISY_SUBSET


def parse_arpabet_string(s: str) -> list[str]:
    """Split a whitespace-separated ARPAbet string into tokens.

    Tolerates commas as visual separators (writing "D EY1 , Z IY0" is
    equivalent to "D EY1 Z IY0") so users can paste fixture-style input.
    Empty tokens are dropped.
    """
    return [tok for tok in s.replace(",", " ").split() if tok]


def phone_class_of(arpabet_token: str) -> PhoneClass:
    phone, _ = split_stress(arpabet_token)
    if phone not in DAISY_SUBSET:
        raise UnsupportedPhone(phone)
    return DAISY_SUBSET[phone].phone_class
