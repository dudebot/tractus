"""Phase-5 phonemize: eSpeak NG wrapper. espeak-marked, auto-skipped if missing."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from tractus_controller.cli import app
from tractus_controller.phonemes import phonemize as phon

pytestmark = pytest.mark.espeak

runner = CliRunner()


def test_hello_world_matches_pinned_reference() -> None:
    """Asserts the exact pinned IPA output. Bumping phonemizer or
    espeak-ng versions and getting a different string should fail this
    test loudly — update EXPECTED_HELLO_WORLD intentionally, not silently.
    """
    out = phon.phonemize("hello world")
    assert out == phon.EXPECTED_HELLO_WORLD, (out, phon.EXPECTED_HELLO_WORLD)


def test_empty_input_returns_empty() -> None:
    assert phon.phonemize("") == ""
    assert phon.phonemize("   ") == ""


def test_each_token_uses_only_known_unicode_letters_or_marks() -> None:
    """Fallback assertion: even if pinning is impractical later, at
    minimum every token should consist of valid IPA-ish characters
    (letters, combining marks, length marks).
    """
    out = phon.phonemize("the quick brown fox")
    assert out, out
    for word in out.split():
        for ch in word:
            assert (
                ch.isalpha()
                or ch in "ːˈˌ"  # IPA length, primary, secondary stress
                or 0x0300 <= ord(ch) <= 0x036F  # combining diacritics
            ), f"unexpected char {ch!r} in {word!r}"


def test_cli_phonemize_emits_pinned_reference() -> None:
    result = runner.invoke(app, ["phonemize", "hello world"])
    assert result.exit_code == 0, result.output
    assert phon.EXPECTED_HELLO_WORLD in result.output


def test_espeak_version_string_is_reported() -> None:
    v = phon.espeak_version()
    assert v is not None
    assert "espeak-ng" in v.lower() or "espeak" in v.lower()
