"""Environment diagnostics. Each check returns (ok, value) so --strict can exit non-zero."""

from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

@dataclass
class CheckResult:
    name: str
    ok: bool
    value: str
    required: bool

    @property
    def status_glyph(self) -> str:
        if self.ok:
            return "OK"
        return "FAIL" if self.required else "WARN"


Check = Callable[[], tuple[bool, str]]


def _check_os() -> tuple[bool, str]:
    return True, f"{platform.system()} {platform.release()} ({platform.machine()})"


def _check_python() -> tuple[bool, str]:
    v = sys.version_info
    ok = (v.major, v.minor) in {(3, 11), (3, 12)}
    return ok, f"{v.major}.{v.minor}.{v.micro} ({sys.executable})"


def _check_torch() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
    except ImportError as e:
        return False, f"not installed ({e})"
    return True, f"{torch.__version__} (built with CUDA={torch.version.cuda})"


def _check_cuda() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "CUDA not available (CPU fallback only)"
    return True, f"available ({torch.cuda.device_count()} device(s))"


def _check_gpu() -> tuple[bool, str]:
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False, "torch not installed"
    if not torch.cuda.is_available():
        return False, "no CUDA device"
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    cap = f"sm_{props.major}{props.minor}"
    return True, f"{name} ({vram_gb:.1f} GB, {cap})"


def _check_espeak() -> tuple[bool, str]:
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not espeak:
        return False, "espeak-ng not in PATH"
    try:
        out = subprocess.run(
            [espeak, "--version"], capture_output=True, text=True, check=True, timeout=5
        )
        version = (out.stdout or out.stderr).splitlines()[0]
    except (subprocess.SubprocessError, OSError, IndexError) as e:
        return False, f"espeak-ng --version failed: {e}"
    return True, f"{version} ({espeak})"


def _check_phonemizer() -> tuple[bool, str]:
    try:
        import phonemizer  # noqa: PLC0415
    except ImportError as e:
        return False, f"not installed ({e})"
    return True, getattr(phonemizer, "__version__", "unknown")


def _check_libsndfile() -> tuple[bool, str]:
    try:
        import soundfile  # noqa: PLC0415
    except ImportError as e:
        return False, f"soundfile not installed ({e})"
    libv = getattr(soundfile, "__libsndfile_version__", "unknown")
    return True, f"soundfile {soundfile.__version__} (libsndfile {libv})"


# (name, fn, required-for-strict).
# Required = must be green for `doctor --strict` to exit 0.
# Non-required = informational only (e.g., GPU is "nice to have").
_CHECKS: list[tuple[str, Check, bool]] = [
    ("os", _check_os, True),
    ("python", _check_python, True),
    ("libsndfile", _check_libsndfile, True),
    ("espeak-ng", _check_espeak, True),
    ("phonemizer", _check_phonemizer, True),
    ("torch", _check_torch, False),
    ("cuda", _check_cuda, False),
    ("gpu", _check_gpu, False),
]


def run_checks() -> list[CheckResult]:
    results: list[CheckResult] = []
    for name, fn, required in _CHECKS:
        try:
            ok, value = fn()
        except Exception as e:  # never let one check crash the diagnostic
            ok, value = False, f"check raised: {e}"
        results.append(CheckResult(name=name, ok=ok, value=value, required=required))
    return results


def report(results: list[CheckResult], use_color: bool = True) -> str:
    width = max(len(r.name) for r in results)
    lines = []
    for r in results:
        glyph = r.status_glyph
        if use_color:
            color = {"OK": "\033[32m", "FAIL": "\033[31m", "WARN": "\033[33m"}[glyph]
            glyph = f"{color}{glyph}\033[0m"
        marker = "" if r.required else " (informational)"
        lines.append(f"  [{glyph}] {r.name.ljust(width)}  {r.value}{marker}")
    return "\n".join(lines)


def has_failures(results: list[CheckResult], strict: bool) -> bool:
    """Return True if --strict should exit non-zero.

    strict=True: any required check failure is fatal.
    strict=False: only programmatic errors are fatal (none right now).
    """
    if strict:
        return any(not r.ok and r.required for r in results)
    return False
