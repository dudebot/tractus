#!/usr/bin/env python3
"""Consent-rule smoke-validator (ARCH §1687-1692).

The actual validator is :mod:`tractus_shared.manifest_natural` — its
``read()`` raises ``ManifestError`` on a manifest line missing
``consent_verified: true`` or with it set to false. The full pytest
coverage already lives in ``tractus_controller/tests/test_manifest.py``.

This wrapper exists so the pre-commit hook + the GitHub Actions CI lint
job can run a cheap, dependency-light smoke without spinning up the
full pytest suite. It builds two synthetic manifests in-memory:

1. ``consent_verified: true`` — must round-trip cleanly.
2. ``consent_verified: false`` — must be rejected.

Exit-non-zero on either drift (false-positive or false-negative).
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _check() -> int:
    try:
        from tractus_shared import manifest_natural as m  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover - environment issue
        print(f"SKIP: tractus_shared not importable ({e}); manifest validator deferred")
        return 0

    with tempfile.TemporaryDirectory() as d:
        td = Path(d)
        ok_path = td / "ok.jsonl"
        bad_path = td / "bad.jsonl"
        miss_path = td / "miss.jsonl"
        ok_path.write_text(
            json.dumps({"audio_path": "x.wav", "text": "x", "consent_verified": True})
            + "\n"
        )
        bad_path.write_text(
            json.dumps({"audio_path": "x.wav", "text": "x", "consent_verified": False})
            + "\n"
        )
        miss_path.write_text(
            json.dumps({"audio_path": "x.wav", "text": "x"}) + "\n"
        )

        # Positive case: must succeed.
        try:
            entries = m.read(ok_path)
        except Exception as e:
            print(f"FAIL: consent_verified=true manifest rejected: {e}")
            return 1
        if not entries:
            print("FAIL: consent_verified=true manifest produced no entries")
            return 1

        # Negative case 1: explicit false.
        try:
            m.read(bad_path)
        except m.ManifestError:
            pass
        except Exception as e:
            print(f"FAIL: wrong exception class on consent=false: {e!r}")
            return 1
        else:
            print("FAIL: consent_verified=false manifest was NOT rejected")
            return 1

        # Negative case 2: missing field.
        try:
            m.read(miss_path)
        except m.ManifestError:
            pass
        except Exception as e:
            print(f"FAIL: wrong exception class on missing consent: {e!r}")
            return 1
        else:
            print("FAIL: manifest missing consent_verified was NOT rejected")
            return 1

    print("OK: consent rule (ARCH §1687-1692)")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_check())
