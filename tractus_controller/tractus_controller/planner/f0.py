"""PyWorld-DIO F0 extraction wrapper.

DIO + StoneMask is the task-spec choice (see task 04_pretraining/001).
Output is in log-Hz at a 100 Hz frame rate (10 ms hop), with a voiced
mask. Unvoiced frames carry log_hz = 0 and mask = 0.

Loudness envelope is a simple RMS energy in dB at the same frame rate;
this is "good enough for v0" per the task spec ("(Optional) loudness
envelopes"). Replace with a perceptual-loudness measure post-Phase-4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_FRAME_PERIOD_MS: float = 10.0  # 100 Hz frame rate
DEFAULT_F0_FLOOR: float = 50.0
DEFAULT_F0_CEIL: float = 600.0


@dataclass(frozen=True)
class F0Result:
    log_hz: np.ndarray  # shape [T_f0], float32
    voiced: np.ndarray  # shape [T_f0], bool
    loud_db: np.ndarray  # shape [T_f0], float32
    frame_period_ms: float


def extract_f0(
    wav: np.ndarray,
    sr: int,
    frame_period_ms: float = DEFAULT_FRAME_PERIOD_MS,
    f0_floor: float = DEFAULT_F0_FLOOR,
    f0_ceil: float = DEFAULT_F0_CEIL,
) -> F0Result:
    """Extract DIO+StoneMask F0 + simple RMS loudness for ``wav``.

    ``wav`` must be float64 mono in [-1, 1]. ``sr`` in Hz.
    """
    try:
        import pyworld  # type: ignore
    except ImportError as e:  # pragma: no cover — exercised at install
        raise RuntimeError(
            "pyworld required for F0 extraction; "
            "install with `pip install pyworld`."
        ) from e
    x = np.asarray(wav, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"extract_f0 wants mono audio, got shape {x.shape}")
    f0_raw, t = pyworld.dio(
        x, sr, frame_period=frame_period_ms, f0_floor=f0_floor, f0_ceil=f0_ceil
    )
    f0 = pyworld.stonemask(x, f0_raw, t, sr)
    voiced = f0 > 0.0
    log_hz = np.zeros_like(f0, dtype=np.float32)
    log_hz[voiced] = np.log(f0[voiced]).astype(np.float32)
    loud_db = _frame_rms_db(x, sr, frame_period_ms, n_frames=len(f0))
    return F0Result(
        log_hz=log_hz.astype(np.float32),
        voiced=voiced,
        loud_db=loud_db.astype(np.float32),
        frame_period_ms=frame_period_ms,
    )


def _frame_rms_db(
    x: np.ndarray, sr: int, frame_period_ms: float, n_frames: int
) -> np.ndarray:
    """Per-frame RMS in dBFS, length-aligned to PyWorld's frame grid."""
    hop = int(round(sr * frame_period_ms / 1000.0))
    win = max(hop * 2, 1)
    out = np.zeros(n_frames, dtype=np.float32)
    eps = 1e-8
    for i in range(n_frames):
        center = i * hop
        a = max(center - win // 2, 0)
        b = min(center + win // 2, len(x))
        if b <= a:
            out[i] = -120.0
            continue
        rms = math.sqrt(float(np.mean(x[a:b] ** 2)) + eps)
        out[i] = 20.0 * math.log10(rms + eps)
    return out


def cents_rmse(
    pred_log_hz: np.ndarray, target_log_hz: np.ndarray, voiced: np.ndarray
) -> float:
    """Cents-RMSE between predicted and target log-Hz on voiced frames.

    1 octave = 1200 cents = log(2)·1200/log(2) ≈ ln(2)·1731.234 (no — see
    below). Cents are a *log2* scale: 1200 cents per octave. So if both
    inputs are in *natural* log-Hz, then a 1-cent gap is
    ``ln(2) / 1200 ≈ 5.776e-4`` log-Hz. Convert by dividing the log-Hz
    diff by ``ln(2) / 1200``.
    """
    voiced = voiced.astype(bool)
    if not voiced.any():
        return float("nan")
    diff = (pred_log_hz - target_log_hz)[voiced]
    cents = diff * (1200.0 / math.log(2.0))
    return float(np.sqrt(np.mean(cents**2)))
