"""eSpeak-NG IPA -> ARPAbet bridge for the full 39-phoneme English inventory.

Wires the ``synth --text`` CLI path: ``phonemize`` (espeak-ng) emits IPA
with stress markers (`ˈ` primary, `ˌ` secondary) and length markers
(`ː`); this module folds those into ARPAbet tokens with a trailing
stress digit (``EY1``, ``IY0``, ``IY2``).

Scope: the full CMU ARPAbet 39-phoneme inventory in
``arpabet.DAISY_SUBSET``. Anything outside raises ``UnmappableIpaSegment``
— the synth CLI's OOV guard refuses, with no silent substitution.

Stress rules (per espeak-ng convention):
    `ˈ` before a syllable -> primary stress (1) on that syllable's vowel
    `ˌ` before a syllable -> secondary stress (2) on that syllable's vowel
    no marker before a syllable -> unstressed (0) on that syllable's vowel

Length-marker `ː` is folded into the preceding vowel symbol (e.g. ``iː``
and ``i`` both map to ``IY``; ARPAbet doesn't distinguish length).

Word boundaries follow the espeak whitespace convention: each whitespace
gap in the IPA string ends a word. ``word_boundaries_after`` lists the
indices into the returned ARPAbet sequence AFTER which a word ends —
matching the convention in ``tests/fixtures/daisy_phonemes.json``.
"""

from __future__ import annotations

from .arpabet import DAISY_PHONES

# Stress markers used by espeak-ng with `with_stress=True`.
STRESS_PRIMARY = "ˈ"  # ˈ
STRESS_SECONDARY = "ˌ"  # ˌ
STRESS_MARKERS = frozenset({STRESS_PRIMARY, STRESS_SECONDARY})
LENGTH_MARK = "ː"  # ː

# IPA -> ARPAbet phone mapping, restricted to the CMU ARPAbet inventory.
#
# Vowels are listed twice for length variants where espeak-ng emits both
# (English /i/ and /u/ surface as either short ``i``/``u`` or long
# ``iː``/``uː`` depending on stress and position; ARPAbet collapses both
# into IY/UW). Diphthongs are matched as multi-codepoint keys via the
# longest-match scan in ``_consume_segment``.
#
# Vowels marked as such here drive stress placement: the most recent
# `ˈ` / `ˌ` before the vowel attaches to it, and any consonant before
# the marker is unstressed.
IPA_TO_ARPABET: dict[str, tuple[str, bool]] = {
    # ---- Consonants (phone, is_vowel=False) ------------------------
    "p": ("P", False),
    "b": ("B", False),
    "t": ("T", False),
    "d": ("D", False),
    "k": ("K", False),
    # espeak-ng uses U+0261 (LATIN SMALL LETTER SCRIPT G) for /g/, not the
    # ASCII letter g. Both are accepted defensively across phonemizer versions.
    "ɡ": ("G", False),
    "g": ("G", False),
    # Affricates (multi-codepoint — longest-match scan picks these first)
    "tʃ": ("CH", False),
    "dʒ": ("JH", False),
    # Fricatives
    "f": ("F", False),
    "v": ("V", False),
    "θ": ("TH", False),
    "ð": ("DH", False),
    "s": ("S", False),
    "z": ("Z", False),
    "ʃ": ("SH", False),
    "ʒ": ("ZH", False),
    "h": ("HH", False),
    # Nasals
    "m": ("M", False),
    "n": ("N", False),
    "ŋ": ("NG", False),
    # Approximants and lateral
    "l": ("L", False),
    # espeak-ng emits the IPA alveolar approximant U+0279 (ɹ) for English /r/.
    # Accept ASCII ``r`` too in case a different backend supplies it.
    "ɹ": ("R", False),
    "r": ("R", False),
    "j": ("Y", False),
    "w": ("W", False),

    # ---- Vowels (phone, is_vowel=True) -----------------------------
    # Diphthongs (multi-codepoint — match before component vowels)
    "eɪ": ("EY", True),
    "aɪ": ("AY", True),
    "aʊ": ("AW", True),
    "ɔɪ": ("OY", True),
    "oʊ": ("OW", True),
    # Monophthongs and length variants. ARPAbet collapses long/short
    # surface variants of /i/, /u/, /ɔ/, /ɑ/ that espeak emits depending
    # on stress and position.
    "i": ("IY", True),
    "iː": ("IY", True),
    "ɪ": ("IH", True),
    "e": ("EH", True),     # espeak occasionally emits /e/ for non-diphthong /ɛ/
    "ɛ": ("EH", True),
    "æ": ("AE", True),
    "ɑ": ("AA", True),
    "ɑː": ("AA", True),
    "ɒ": ("AA", True),     # British /ɒ/ — collapse onto AA in en-us style
    "ʌ": ("AH", True),
    "ə": ("AH", True),     # CMU ARPAbet merges /ʌ/ and /ə/ onto AH
    "ɔ": ("AO", True),
    "ɔː": ("AO", True),
    "o": ("OW", True),     # bare /o/ in en-us ≈ /oʊ/
    "oː": ("OW", True),
    "ʊ": ("UH", True),
    "u": ("UW", True),
    "uː": ("UW", True),
    # Rhotacized schwa: ARPAbet collapses /ɚ/ (schwa-r) and /ɝ/ (stressed
    # vocalic R) both onto ER.
    "ɚ": ("ER", True),
    "ɝ": ("ER", True),
}

# Validate at import time that every emitted ARPAbet phone is in the
# Daisy subset — defensive against future edits introducing a stray phone.
for _ipa, (_phone, _) in IPA_TO_ARPABET.items():
    assert _phone in DAISY_PHONES, (
        f"ipa_arpabet.IPA_TO_ARPABET maps {_ipa!r} -> {_phone!r}, "
        f"which is outside DAISY_PHONES={sorted(DAISY_PHONES)}"
    )

# Longest-first sort so multi-codepoint keys (eɪ, iː, uː, ɔː) match before
# single-codepoint keys.
_SORTED_KEYS: list[str] = sorted(IPA_TO_ARPABET.keys(), key=len, reverse=True)


class UnmappableIpaSegment(ValueError):
    """An IPA segment is outside the CMU ARPAbet inventory.

    Raised by :func:`ipa_to_arpabet`. The synth CLI's OOV guard converts
    this to a user-facing refusal — never silently substitute.
    """


def _consume_segment(word: str, pos: int) -> tuple[str, bool, int]:
    """Match the next IPA segment at ``word[pos:]``.

    Returns ``(arpabet_phone, is_vowel, new_pos)``. Raises
    :class:`UnmappableIpaSegment` if no key matches.
    """
    for key in _SORTED_KEYS:
        if word.startswith(key, pos):
            phone, is_vowel = IPA_TO_ARPABET[key]
            return phone, is_vowel, pos + len(key)
    # Build a small context window for the error message.
    bad = word[pos]
    raise UnmappableIpaSegment(
        f"IPA segment {bad!r} (U+{ord(bad):04X}) at position {pos} of "
        f"{word!r} is outside the CMU ARPAbet inventory. Supported phones: "
        f"{sorted(DAISY_PHONES)}. Re-phonemize with a different word, or "
        f"extend the IPA->ARPAbet table."
    )


def _convert_word(word: str) -> list[str]:
    """Convert a single whitespace-delimited IPA word to ARPAbet tokens.

    Stress logic: scan left-to-right. A stress marker (`ˈ` or `ˌ`)
    arms a "pending stress" state. The next vowel encountered consumes
    that pending stress as its trailing digit; consonants before that
    vowel are emitted without stress digits (ARPAbet convention).
    Vowels with no preceding marker get stress 0.
    """
    tokens: list[str] = []
    pending_stress: str = "0"  # "0" | "1" | "2"
    pos = 0
    while pos < len(word):
        ch = word[pos]
        if ch == STRESS_PRIMARY:
            pending_stress = "1"
            pos += 1
            continue
        if ch == STRESS_SECONDARY:
            pending_stress = "2"
            pos += 1
            continue
        if ch.isspace():
            # Words are split by the caller; spaces inside a "word" string
            # shouldn't happen, but tolerate by skipping.
            pos += 1
            continue
        phone, is_vowel, new_pos = _consume_segment(word, pos)
        if is_vowel:
            tokens.append(f"{phone}{pending_stress}")
            pending_stress = "0"
        else:
            tokens.append(phone)
        pos = new_pos
    return tokens


def ipa_to_arpabet(ipa: str) -> tuple[list[str], list[int]]:
    """Convert an espeak-ng IPA string to ARPAbet tokens + word boundaries.

    Returns ``(arpabet_tokens, word_boundaries_after)`` where:

    * ``arpabet_tokens`` is a flat list like ``['D', 'EY1', 'Z', 'IY0',
      'D', 'EY1', 'Z', 'IY0', ...]`` — vowels carry a trailing stress
      digit, consonants do not.
    * ``word_boundaries_after`` is a list of indices into
      ``arpabet_tokens`` AFTER which a word ends — matching the
      convention in ``tests/fixtures/daisy_phonemes.json``. The implicit
      final-word boundary is NOT included (the fixture's comment: "7
      words = 6 internal boundaries, plus the implicit final boundary").

    Raises :class:`UnmappableIpaSegment` on any IPA segment outside the
    CMU ARPAbet inventory.
    """
    if not ipa.strip():
        return [], []

    tokens: list[str] = []
    boundaries: list[int] = []
    words = [_convert_word(w) for w in ipa.split()]
    # Drop empty IPA "words" (only stress markers, etc.) — they shouldn't
    # contribute a phantom boundary.
    words = [w for w in words if w]
    for i, word_tokens in enumerate(words):
        tokens.extend(word_tokens)
        # Internal boundary only — caller treats the final-word boundary
        # as implicit (matches daisy_phonemes.json convention).
        if i < len(words) - 1:
            boundaries.append(len(tokens) - 1)
    return tokens, boundaries
