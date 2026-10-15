"""JSONL manifest tooling for training/evaluation data.

Schema for a manifest entry (extensible — keep additions opt-in):

    {
        "audio_path": "data/clips/0001.wav",        # required
        "text": "hello world",                       # required
        "consent_verified": true,                    # required, must be true
        "phonemes": ["HH","AH0","L","OW1", ...],     # optional
        "speaker_id": "spk-001",                     # optional
        "duration_sec": 1.234,                       # optional
        "language": "en-us",                         # optional, default en-us
        "license": "CC0-1.0",                        # optional
        "extra": {...}                               # opaque, preserved
    }

`consent_verified: true` is enforced everywhere — see project README §1
and plan/00-overview.md. Entries without it raise `ManifestError` at read
time, by design. No silent skipping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator


REQUIRED_FIELDS: tuple[str, ...] = ("audio_path", "text", "consent_verified")


class ManifestError(ValueError):
    """Raised on schema violations. Includes the offending line number."""


@dataclass
class ManifestEntry:
    audio_path: str
    text: str
    consent_verified: bool
    phonemes: list[str] | None = None
    speaker_id: str | None = None
    duration_sec: float | None = None
    language: str | None = None
    license: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "audio_path": self.audio_path,
            "text": self.text,
            "consent_verified": self.consent_verified,
        }
        for key in ("phonemes", "speaker_id", "duration_sec", "language", "license"):
            val = getattr(self, key)
            if val is not None:
                d[key] = val
        if self.extra:
            d["extra"] = self.extra
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], *, line_no: int | None = None) -> "ManifestEntry":
        missing = [f for f in REQUIRED_FIELDS if f not in d]
        if missing:
            where = f" at line {line_no}" if line_no else ""
            raise ManifestError(f"missing required field(s) {missing}{where}: {d}")
        if d["consent_verified"] is not True:
            where = f" at line {line_no}" if line_no else ""
            raise ManifestError(
                f"consent_verified must be true{where} (got {d['consent_verified']!r})"
            )
        recognized = {*REQUIRED_FIELDS, "phonemes", "speaker_id", "duration_sec",
                      "language", "license", "extra"}
        extra_payload = dict(d.get("extra") or {})
        for key in d:
            if key not in recognized:
                extra_payload[key] = d[key]
        return cls(
            audio_path=d["audio_path"],
            text=d["text"],
            consent_verified=True,
            phonemes=d.get("phonemes"),
            speaker_id=d.get("speaker_id"),
            duration_sec=d.get("duration_sec"),
            language=d.get("language"),
            license=d.get("license"),
            extra=extra_payload,
        )


def read(path: str | Path) -> list[ManifestEntry]:
    entries: list[ManifestEntry] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as e:
                raise ManifestError(f"line {i}: invalid JSON ({e})") from e
            entries.append(ManifestEntry.from_dict(payload, line_no=i))
    return entries


def iter_read(path: str | Path) -> Iterator[ManifestEntry]:
    """Streaming variant — yield entries one at a time without loading the whole file."""
    with Path(path).open("r", encoding="utf-8") as f:
        for i, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            payload = json.loads(line)
            yield ManifestEntry.from_dict(payload, line_no=i)


def write(entries: Iterable[ManifestEntry], path: str | Path) -> int:
    """Write entries to JSONL. Returns the number of entries written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def round_trip(in_path: str | Path, out_path: str | Path) -> int:
    """Read, validate, write — used by `tractus-controller prepare-manifest`."""
    entries = read(in_path)
    return write(entries, out_path)
