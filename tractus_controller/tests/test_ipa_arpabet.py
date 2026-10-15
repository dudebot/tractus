"""IPA -> ARPAbet bridge tests (full 39-phoneme CMU ARPAbet inventory; the
Daisy fixture remains a smoke test inside the larger inventory).

Wires `phonemize` (espeak-ng IPA) to the existing ARPAbet path used by
`tractus-controller synth`. The mapping covers the 39-phoneme CMU set;
any IPA segment outside that (e.g. exotic non-English phones) must
raise, never silently substitute.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tractus_controller.phonemes.ipa_arpabet import (
    UnmappableIpaSegment,
    ipa_to_arpabet,
)
from tractus_controller.phonemes.phonemize import phonemize

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "daisy_phonemes.json"


def _load_fixture() -> dict:
    return json.loads(FIXTURE_PATH.read_text())


# ---------------------------------------------------------------------------
# Bridge mechanics: hand-fed IPA -> fixture-exact ARPAbet
# ---------------------------------------------------------------------------


def test_idealized_daisy_ipa_round_trips_to_fixture() -> None:
    """The bridge produces the canonical fixture token sequence + boundaries
    when fed an idealized IPA transcription of the Daisy phrase.

    "Idealized" because en-us espeak-ng emits a few divergent surface forms
    (secondary stress on connected ``me``, /ʊ/ rather than /ɔː/ for ``your``)
    that don't match the hand-curated fixture. This test isolates the
    bridge logic from those espeak quirks; see the espeak-marked test
    below for the actual phonemizer round-trip.
    """
    fixture = _load_fixture()
    # Stress markers placed before each stressed syllable's onset, length
    # marker on long vowels — exactly what a faithful IPA transcription
    # of the fixture's ARPAbet+stress would look like.
    ideal_ipa = "dˈeɪzi dˈeɪzi ɡˈɪv mi jˈɔːɹ ˈænsɚ dˈuː"
    tokens, bounds = ipa_to_arpabet(ideal_ipa)
    assert tokens == fixture["arpabet_with_stress"]
    assert bounds == fixture["word_boundaries_after"]


def test_stress_marker_attaches_to_following_vowel_not_consonant() -> None:
    """`ˈ` between a consonant and vowel marks the vowel, not the consonant."""
    tokens, _ = ipa_to_arpabet("dˈeɪ")
    # D has no stress digit, EY1 carries the primary stress.
    assert tokens == ["D", "EY1"]


def test_secondary_stress_marker_maps_to_digit_2() -> None:
    tokens, _ = ipa_to_arpabet("mˌiː")
    assert tokens == ["M", "IY2"]


def test_no_stress_marker_means_stress_digit_0() -> None:
    tokens, _ = ipa_to_arpabet("zi")
    assert tokens == ["Z", "IY0"]


def test_length_marker_folds_into_preceding_vowel() -> None:
    """`iː` and `i` both collapse to IY; `uː` and `u` both collapse to UW."""
    long_tokens, _ = ipa_to_arpabet("iː uː")
    short_tokens, _ = ipa_to_arpabet("i u")
    assert long_tokens == ["IY0", "UW0"]
    assert short_tokens == ["IY0", "UW0"]


def test_word_boundaries_match_fixture_convention() -> None:
    """Fixture convention: indices AFTER which a word ends, EXCLUDING the
    implicit final-word boundary (per ``daisy_phonemes.json:_comment``)."""
    # Two words: "daisy" (4 phones) + "give" (3 phones) = 7 total. One
    # internal boundary at index 3.
    tokens, bounds = ipa_to_arpabet("dˈeɪzi ɡˈɪv")
    assert len(tokens) == 7
    assert bounds == [3]


def test_empty_input_returns_empty() -> None:
    assert ipa_to_arpabet("") == ([], [])
    assert ipa_to_arpabet("   ") == ([], [])


def test_g_accepts_both_ascii_and_ipa_script_g() -> None:
    """espeak emits U+0261 (ɡ) for /g/. Accept ASCII g defensively too."""
    a, _ = ipa_to_arpabet("ɡ")
    b, _ = ipa_to_arpabet("g")
    assert a == b == ["G"]


def test_r_accepts_both_ascii_and_ipa_alveolar_approximant() -> None:
    """espeak emits U+0279 (ɹ) for English /r/."""
    a, _ = ipa_to_arpabet("ɹ")
    b, _ = ipa_to_arpabet("r")
    assert a == b == ["R"]


# ---------------------------------------------------------------------------
# OOV handling: out-of-subset IPA segments must raise, never substitute
# ---------------------------------------------------------------------------


def test_out_of_subset_ipa_raises_with_offending_token_in_message() -> None:
    """`ɸ` (voiceless bilabial fricative, e.g. Japanese /f/) isn't in CMU ARPAbet."""
    with pytest.raises(UnmappableIpaSegment) as exc_info:
        ipa_to_arpabet("ɸɪn")
    msg = str(exc_info.value)
    assert "ɸ" in msg
    assert "subset" in msg or "inventory" in msg or "ARPAbet" in msg


def test_out_of_subset_ipa_includes_codepoint_in_message() -> None:
    """The offending codepoint should be printable in the error message."""
    with pytest.raises(UnmappableIpaSegment) as exc_info:
        ipa_to_arpabet("ɣ")  # voiced velar fricative — not in English ARPAbet
    msg = str(exc_info.value)
    assert "ɣ" in msg
    assert "U+0263" in msg


def test_out_of_subset_consonant_raises() -> None:
    """`ʔ` (glottal stop, common in some languages) isn't a CMU ARPAbet phone."""
    with pytest.raises(UnmappableIpaSegment) as exc_info:
        ipa_to_arpabet("ʔa")
    assert "ʔ" in str(exc_info.value)


# ---------------------------------------------------------------------------
# End-to-end: phonemize -> ipa_to_arpabet (espeak-marked, auto-skipped if
# espeak-ng is missing).
# ---------------------------------------------------------------------------


pytestmark_espeak = pytest.mark.espeak


@pytest.mark.espeak
def test_phonemize_round_trip_on_in_subset_phrase_succeeds() -> None:
    """Use a sub-phrase whose espeak-ng IPA stays inside the Daisy subset.

    Avoids "your" (espeak emits /ʊ/) and "answer" (en-us is fine; just
    keeping the phrase short and stress-stable).
    """
    ipa = phonemize("Daisy Daisy give")
    tokens, bounds = ipa_to_arpabet(ipa)
    # Two "Daisy"s + "give" = 4 + 4 + 3 = 11 phones, two words ending
    # before "give".
    assert tokens == [
        "D", "EY1", "Z", "IY0",
        "D", "EY1", "Z", "IY0",
        "G", "IH1", "V",
    ]
    assert bounds == [3, 7]


@pytest.mark.espeak
def test_phonemize_full_daisy_phrase_round_trips_after_inventory_expansion() -> None:
    """Post-2026-05-01 inventory expansion: the full canonical Daisy phrase
    is now mappable end-to-end via --text. en-us espeak emits "your" as
    /jʊɹ/ where /ʊ/ used to be OOV; with UH in the 39-phone inventory it
    now maps to UH cleanly.
    """
    ipa = phonemize("Daisy, Daisy, give me your answer do")
    tokens, bounds = ipa_to_arpabet(ipa)
    # Sanity: 7 words → 6 internal boundaries.
    assert len(bounds) == 6
    # Every token is a valid ARPAbet phone (basic shape check).
    assert all(t and t[0].isalpha() for t in tokens)
    # The "your" word produces UH (the previously-OOV vowel).
    assert "UH0" in tokens or "UH1" in tokens or "UH2" in tokens
