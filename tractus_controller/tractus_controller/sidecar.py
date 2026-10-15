"""Sidecar JSON metadata for any rendered WAV.

Project constraint (README §3, plan/00-overview.md):
every output WAV gets a sidecar JSON with git hash, config, and the
fixture/input path it came from. This module is the single place that
defines that schema.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import __version__
from .paths import workspace_root


def _git_describe() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(workspace_root()), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(workspace_root()), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=True,
            timeout=2,
        ).stdout.strip()
        return f"{sha}{'-dirty' if dirty else ''}"
    except (subprocess.SubprocessError, OSError):
        return None


@dataclass
class RenderSidecar:
    """Metadata persisted alongside every rendered WAV.

    Keep this schema additive — downstream tooling will key off it.
    """

    package_version: str
    git_hash: str | None
    timestamp_unix: float
    fixture_path: str
    speaker_path: str | None
    sampling_rate: int
    num_samples: int
    duration_sec: float
    rms: float
    peak: float
    platform: str
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, wav_path: str | os.PathLike) -> Path:
        """Write the sidecar next to the WAV at `wav_path` with a `.json` suffix."""
        sidecar_path = Path(wav_path).with_suffix(".json")
        sidecar_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))
        return sidecar_path


def make_sidecar(
    *,
    fixture_path: str | os.PathLike,
    speaker_path: str | os.PathLike | None,
    sampling_rate: int,
    audio,  # numpy.ndarray, untyped to keep numpy out of the import path
    extra: dict[str, Any] | None = None,
) -> RenderSidecar:
    import numpy as np  # local import

    arr = np.asarray(audio, dtype=np.float64)
    rms = float(np.sqrt(np.mean(arr * arr))) if arr.size else 0.0
    peak = float(np.max(np.abs(arr))) if arr.size else 0.0
    return RenderSidecar(
        package_version=__version__,
        git_hash=_git_describe(),
        timestamp_unix=time.time(),
        fixture_path=str(Path(fixture_path).resolve()),
        speaker_path=str(Path(speaker_path).resolve()) if speaker_path else None,
        sampling_rate=sampling_rate,
        num_samples=int(arr.size),
        duration_sec=float(arr.size / sampling_rate) if sampling_rate else 0.0,
        rms=rms,
        peak=peak,
        platform=f"{platform.system()} {platform.release()} {platform.machine()}",
        extra=extra or {},
    )
