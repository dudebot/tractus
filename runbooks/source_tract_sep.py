#!/usr/bin/env python3
"""source_tract_sep.py — per-axis source/tract separation metrics.

Compares a synthesized audio file against a reference and emits
per-axis distance metrics:

- F0 RMSE (cents)
- spectral tilt (dB/octave) difference
- frication band energy (dB) difference, mean over 4-6 / 6-8 / 8-11 kHz
- nasalization metric difference (low-band / total energy proxy)
- formants F1-F3 percent difference (LPC-based)

Pass criterion (defaults): F0 RMSE < 50 cents, |tilt| < 2 dB/oct,
|frication band| < 3 dB, |formants| < 5%, |nasal proxy| < 0.10.

CLI:
    python source_tract_sep.py \\
        --audio /path/to/synth.wav \\
        --reference /path/to/ref.wav \\
        [--ref-anatomy /path/to/anatomy.npz] \\
        [--config /path/to/thresholds.json]

Output: JSON on stdout, one-line summary on stderr.
Exit 0 if all axes pass; non-zero otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lib.json_io import emit  # noqa: E402


DEFAULT_THRESHOLDS = {
    "f0_rmse_cents": 50.0,
    "tilt_db_per_oct_abs": 2.0,
    "frication_band_db_abs": 3.0,
    "nasal_metric_abs": 0.10,
    "formant_pct_abs": 0.05,
}

SCHEMA_PATH = (
    Path(__file__).resolve().parent / "schemas" / "source_tract_sep.schema.json"
)


def _load_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _f0_pyin(y: np.ndarray, sr: int) -> np.ndarray:
    import librosa

    f0, _voiced, _vp = librosa.pyin(
        y,
        fmin=float(librosa.note_to_hz("C2")),
        fmax=float(librosa.note_to_hz("C7")),
        sr=sr,
    )
    return f0


def _f0_rmse_cents(f_a: np.ndarray, f_b: np.ndarray) -> float:
    n = min(len(f_a), len(f_b))
    a = f_a[:n]
    b = f_b[:n]
    mask = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not np.any(mask):
        # No mutual voiced frames — treat as failure-magnitude.
        return 1e6
    cents = 1200.0 * np.log2(a[mask] / b[mask])
    return float(np.sqrt(np.mean(cents**2)))


def _spectral_tilt_db_per_oct(y: np.ndarray, sr: int) -> float:
    """Linear regression on log10|S(f)| vs log10(f) over 100Hz-4kHz.
    Returns dB/octave."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    spec = np.mean(S, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    band = (freqs >= 100.0) & (freqs <= 4000.0) & (spec > 1e-9)
    if not np.any(band):
        return 0.0
    log_f = np.log10(freqs[band])
    log_mag_db = 20.0 * np.log10(spec[band])
    slope_db_per_decade, _ = np.polyfit(log_f, log_mag_db, 1)
    return float(slope_db_per_decade * np.log10(2.0))


def _frication_band_energy_db(y: np.ndarray, sr: int) -> dict[str, float]:
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    bands = {"4-6kHz": (4000, 6000), "6-8kHz": (6000, 8000), "8-11kHz": (8000, 11000)}
    out: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            out[name] = -120.0
            continue
        e = np.mean(S[mask, :] ** 2)
        out[name] = 10.0 * np.log10(e + 1e-12)
    return out


def _nasal_proxy(y: np.ndarray, sr: int) -> float:
    """Crude nasalance proxy: low-band (200-1000 Hz) energy share /
    total in-voice band (200-4000 Hz). Uniform 1D scalar."""
    import librosa

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    low = (freqs >= 200) & (freqs < 1000)
    full = (freqs >= 200) & (freqs < 4000)
    e_low = np.mean(S[low, :])
    e_full = np.mean(S[full, :]) + 1e-12
    return float(e_low / e_full)


def _formants_lpc(y: np.ndarray, sr: int, n_formants: int = 3) -> list[float]:
    """LPC-based formants. Returns [F1, F2, F3] in Hz (or NaN). Uses
    the average frame of voiced energy as a single estimate — adequate
    for the runbook contract; deeper formant tracking belongs in the
    inverse-fit task."""
    import librosa

    n_lpc = 2 + sr // 1000
    try:
        a = librosa.lpc(y, order=n_lpc)
    except Exception:
        return [float("nan")] * n_formants
    roots = np.roots(a)
    roots = roots[np.imag(roots) >= 0.0]
    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = sorted(angles * (sr / (2.0 * np.pi)))
    formants = [f for f in freqs if 90.0 < f < 5500.0]
    formants = formants[:n_formants]
    while len(formants) < n_formants:
        formants.append(float("nan"))
    return [float(f) for f in formants]


def main() -> int:
    p = argparse.ArgumentParser(prog="source_tract_sep")
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--ref-anatomy", type=Path, default=None)
    p.add_argument("--config", type=Path, default=None)
    args = p.parse_args()

    if not args.audio.exists() or not args.reference.exists():
        report = {
            "axes": {},
            "pass": False,
            "error": "audio or reference missing",
        }
        emit(report, "FAIL source_tract_sep: missing input", SCHEMA_PATH)
        return 2

    thresholds = dict(DEFAULT_THRESHOLDS)
    if args.config and args.config.exists():
        thresholds.update(json.loads(args.config.read_text()))

    y_a, sr_a = _load_audio(args.audio)
    y_b, sr_b = _load_audio(args.reference)
    if sr_a != sr_b:
        # Resample reference to match.
        import librosa

        y_b = librosa.resample(y_b, orig_sr=sr_b, target_sr=sr_a)
        sr_b = sr_a
    sr = sr_a

    f0_a = _f0_pyin(y_a, sr)
    f0_b = _f0_pyin(y_b, sr)
    f0_rmse_c = _f0_rmse_cents(f0_a, f0_b)

    tilt_a = _spectral_tilt_db_per_oct(y_a, sr)
    tilt_b = _spectral_tilt_db_per_oct(y_b, sr)
    tilt_diff = tilt_a - tilt_b

    fric_a = _frication_band_energy_db(y_a, sr)
    fric_b = _frication_band_energy_db(y_b, sr)
    fric_diff = {k: fric_a[k] - fric_b[k] for k in fric_a}
    fric_max_abs = max(abs(v) for v in fric_diff.values())

    nasal_a = _nasal_proxy(y_a, sr)
    nasal_b = _nasal_proxy(y_b, sr)
    nasal_diff = nasal_a - nasal_b

    form_a = _formants_lpc(y_a, sr)
    form_b = _formants_lpc(y_b, sr)
    form_pct = []
    for fa, fb in zip(form_a, form_b):
        if not np.isfinite(fa) or not np.isfinite(fb) or fb <= 0:
            form_pct.append(float("nan"))
        else:
            form_pct.append(float(abs(fa - fb) / fb))
    form_max_pct = max((p for p in form_pct if np.isfinite(p)), default=0.0)

    axis_pass = {
        "f0": f0_rmse_c < thresholds["f0_rmse_cents"],
        "tilt": abs(tilt_diff) < thresholds["tilt_db_per_oct_abs"],
        "frication": fric_max_abs < thresholds["frication_band_db_abs"],
        "nasal": abs(nasal_diff) < thresholds["nasal_metric_abs"],
        "formants": form_max_pct < thresholds["formant_pct_abs"],
    }
    overall = all(axis_pass.values())

    report = {
        "axes": {
            "f0_rmse_cents": float(f0_rmse_c),
            "tilt_db_per_oct_diff": float(tilt_diff),
            "frication_band_db_diff": {k: float(v) for k, v in fric_diff.items()},
            "nasal_metric_diff": float(nasal_diff),
            "formant_pct_diff": [float(p) for p in form_pct],
        },
        "axis_pass": {k: bool(v) for k, v in axis_pass.items()},
        "thresholds": {k: float(v) for k, v in thresholds.items()},
        "pass": bool(overall),
    }
    if overall:
        emit(report, f"PASS source_tract_sep: f0_rmse={f0_rmse_c:.1f}c tilt_diff={tilt_diff:.2f}", SCHEMA_PATH)
        return 0
    failed = [k for k, v in axis_pass.items() if not v]
    emit(report, "FAIL source_tract_sep: axes failed: " + ",".join(failed), SCHEMA_PATH)
    return 1


if __name__ == "__main__":
    sys.exit(main())
