"""eSpeak-NG-backed phonemization, used by `tractus-controller phonemize`.

The plan pins phonemizer (3.3.0 in pyproject.toml) and relies on the
system espeak-ng version. Output is asserted exactly for "hello world"
by the tests. If upstream behavior drifts, update the EXPECTED_HELLO_WORLD
constant rather than relaxing the test — that keeps regressions visible.
"""

from __future__ import annotations

import shutil
import subprocess
from functools import lru_cache

DEFAULT_LANGUAGE = "en-us"
DEFAULT_BACKEND = "espeak"

# Pinned reference output for the smoke phonemize test. Locked against
# phonemizer==3.3.0 + espeak-ng 1.51.x. If you bump either dependency and
# the assertion fails, regenerate this string and bump the test rev.
EXPECTED_HELLO_WORLD = "həlˈoʊ wˈɜːld"


class PhonemizerError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _import_backend():
    try:
        from phonemizer.backend import EspeakBackend  # type: ignore
    except ImportError as e:
        raise PhonemizerError(
            f"phonemizer not installed: {e}. "
            f"`pip install -e .` should pull it in."
        ) from e
    return EspeakBackend


def _ensure_espeak_available() -> None:
    if shutil.which("espeak-ng") is None and shutil.which("espeak") is None:
        raise PhonemizerError(
            "espeak-ng not found on PATH. "
            "Install with `apt install espeak-ng espeak-ng-data`."
        )


@lru_cache(maxsize=4)
def _backend(language: str) -> object:
    _ensure_espeak_available()
    EspeakBackend = _import_backend()
    return EspeakBackend(
        language=language,
        preserve_punctuation=False,
        with_stress=True,
        language_switch="remove-flags",
    )


def phonemize(text: str, language: str = DEFAULT_LANGUAGE) -> str:
    """Return IPA phonemes for `text`. Whitespace separates words.

    Empty input returns an empty string.
    """
    if not text.strip():
        return ""
    backend = _backend(language)
    result = backend.phonemize([text], strip=True)
    if not result:
        return ""
    return result[0].strip()


def espeak_version() -> str | None:
    """Return the system espeak-ng version, or None if not installed."""
    exe = shutil.which("espeak-ng") or shutil.which("espeak")
    if exe is None:
        return None
    try:
        out = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, check=True, timeout=5
        )
        first_line = (out.stdout or out.stderr).splitlines()[0]
        return first_line
    except (subprocess.SubprocessError, OSError, IndexError):
        return None
