"""Independent FFT-based formant probe for Story anchor renders.

The 002a worker's gate uses LPC root analysis on a downsampled-to-10kHz
audio. LPC has known biases on extreme vowels (close-spaced poles for
back vowels; low-F1 vowels confused by source pre-emphasis). This
probe is a parallel measurement using direct FFT peak-picking on a
windowed-then-averaged steady-state PSD — different methodology, same
input — to verify whether TEA's render is genuinely off or whether the
LPC gate is reading badly.

Run::
    python scripts/probe_story_spectrum.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("TEA_PRECISION", "fp32")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tractus_controller"))
sys.path.insert(0, str(ROOT / "tractus_engine_advanced" / "src"))

import jax.numpy as jnp  # noqa: E402

from tractus_controller.body_model.anchors import iter_story_anchors  # noqa: E402
from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: E402
from tractus_engine_advanced.tds.tube_topology import (  # noqa: E402
    FIRST_PHARYNX_SECTION, LAST_MOUTH_SECTION,
)


SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 441
F = 50  # 500 ms render — give the steady state plenty of room
FREE_LO = FIRST_PHARYNX_SECTION
FREE_HI = LAST_MOUTH_SECTION + 1


def load_default_speaker() -> dict:
    npz = np.load(ROOT / "tractus_controller/outputs/fixtures/default_tube.npz")
    return {k: np.asarray(npz[k], dtype=np.float32) for k in
            ("length_cm", "area_cm2", "wall_mass_cgs",
             "wall_resistance_cgs", "wall_stiffness_cgs")}


def voiced_glottis(f0: float = 120.0):
    g = np.array(
        [f0, 8000.0, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
        dtype=np.float32,
    )
    return np.tile(g[None, :], (F, 1))


def render_story_shape(default: dict, area_tea: np.ndarray,
                       length_tea: np.ndarray | None = None,
                       f0: float = 120.0) -> np.ndarray:
    """Render Story area function through TEA. Splice into the default
    speaker's full 93-section tube at the supraglottal slice.

    If length_tea is provided, also override the length of those sections
    (Story shapes have variable per-section lengths, not the default
    fixture's lengths)."""
    A = jnp.asarray(default["area_cm2"]).at[FREE_LO:FREE_HI].set(jnp.asarray(area_tea))
    if length_tea is not None:
        L = jnp.asarray(default["length_cm"]).at[FREE_LO:FREE_HI].set(jnp.asarray(length_tea))
    else:
        L = jnp.asarray(default["length_cm"])
    Mw = jnp.asarray(default["wall_mass_cgs"])
    Bw = jnp.asarray(default["wall_resistance_cgs"])
    Kw = jnp.asarray(default["wall_stiffness_cgs"])
    L_t = jnp.tile(L[None, :], (F, 1))
    A_t = jnp.tile(A[None, :], (F, 1))
    Mw_t = jnp.tile(Mw[None, :], (F, 1))
    Bw_t = jnp.tile(Bw[None, :], (F, 1))
    Kw_t = jnp.tile(Kw[None, :], (F, 1))
    g = jnp.asarray(voiced_glottis(f0))
    audio, *_ = synthesize_dynamic_tube(
        L_t, A_t, Mw_t, Bw_t, Kw_t,
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=g,
    )
    return np.asarray(audio)


def fft_psd(audio: np.ndarray, sr: int, win_ms: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Mean PSD across overlapping Hann windows on the steady-state region.

    Drops the first 100 ms of voicing transient. Returns (freqs_Hz, psd_dB)."""
    drop = int(0.1 * sr)
    x = audio[drop:]
    win = int(win_ms * sr / 1000)
    hop = win // 2
    n_frames = max(1, (len(x) - win) // hop)
    window = np.hanning(win)
    psd = np.zeros(win // 2 + 1, dtype=np.float64)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + win]
        if len(seg) < win:
            break
        S = np.fft.rfft(seg.astype(np.float64) * window)
        psd += np.abs(S) ** 2
    psd /= max(n_frames, 1)
    freqs = np.fft.rfftfreq(win, d=1.0 / sr)
    psd_dB = 10.0 * np.log10(psd + 1e-12)
    return freqs, psd_dB


def smooth(y: np.ndarray, w: int = 7) -> np.ndarray:
    """Boxcar smoothing for peak-pick stability."""
    if w < 2:
        return y.copy()
    pad = w // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(w) / w
    return np.convolve(yp, kernel, mode="valid")


def find_peaks_simple(freqs: np.ndarray, psd_dB: np.ndarray,
                      f_min: float = 150.0, f_max: float = 4500.0,
                      min_separation_Hz: float = 250.0,
                      n_peaks: int = 3) -> list[float]:
    """Find local maxima above f_min, enforce minimum separation. No LPC."""
    smoothed = smooth(psd_dB, w=9)
    mask = (freqs >= f_min) & (freqs <= f_max)
    fs = freqs[mask]
    ps = smoothed[mask]
    # Local maxima
    is_peak = np.zeros_like(ps, dtype=bool)
    is_peak[1:-1] = (ps[1:-1] > ps[:-2]) & (ps[1:-1] > ps[2:])
    peaks = [(fs[i], ps[i]) for i in np.where(is_peak)[0]]
    # Sort by amplitude desc; greedily take with separation
    peaks.sort(key=lambda x: -x[1])
    chosen: list[float] = []
    for f, _ in peaks:
        if all(abs(f - c) >= min_separation_Hz for c in chosen):
            chosen.append(f)
        if len(chosen) >= n_peaks:
            break
    return sorted(chosen)


# Story 2008 published computed formants from the worker's table, plus
# Peterson & Barney measured human formants. Two reference sets help
# triangulate.
TARGETS_STORY = {
    "/i/": (325, 2139, 2976), "/ɪ/": (413, 2103, 2626), "/e/": (484, 1887, 2404),
    "/ɛ/": (565, 1595, 2200), "/æ/": (810, 1655, 2310), "/ʌ/": (593, 836, 3099),
    "/ɑ/": (694, 942, 2948), "/ɔ/": (557, 709, 3176), "/o/": (499, 768, 2421),
    "/ʊ/": (477, 789, 2499), "/u/": (314, 702, 2298),
}
TARGETS_PB = {
    "/i/": (270, 2290, 3010), "/ɪ/": (390, 1990, 2550), "/e/": (530, 1840, 2480),
    "/ɛ/": (580, 1840, 2480), "/æ/": (660, 1720, 2410), "/ʌ/": (640, 1190, 2390),
    "/ɑ/": (730, 1090, 2440), "/ɔ/": (570, 840, 2410), "/o/": (570, 840, 2410),
    "/ʊ/": (450, 1030, 2380), "/u/": (300, 870, 2240),
}


def main() -> int:
    default = load_default_speaker()

    shapes = list(iter_story_anchors())
    if not shapes:
        print("ERROR: no shapes loaded", file=sys.stderr)
        return 1

    # JIT warmup
    name0, area0, length0 = shapes[0]
    _ = render_story_shape(default, area0, length0, f0=120.0)

    for f0 in (120.0, 250.0, 400.0):
        print(f"\n=== F0 = {f0:.0f} Hz "
              f"(harmonic spacing {f0:.0f} Hz; "
              f"{'dense — formants smeared' if f0 < 200 else 'sparse — formants visible' if f0 > 350 else 'moderate'}) ===")
        print(f"{'shape':<6} {'F1 fft':>8} {'F1 St':>7} {'F1 P&B':>7}  "
              f"{'F2 fft':>8} {'F2 St':>7} {'F2 P&B':>7}  "
              f"{'F3 fft':>8} {'F3 St':>7} {'F3 P&B':>7}")
        print("-" * 96)
        # Subset to /i/, /a/, /u/ for cross-F0 sweep — most diagnostic
        for name, area, length in shapes:
            if name not in ("/i/", "/ɑ/", "/u/"):
                continue
            audio = render_story_shape(default, area, length, f0=f0)
            freqs, psd_dB = fft_psd(audio, SAMPLE_RATE)
            peaks = find_peaks_simple(freqs, psd_dB, n_peaks=3,
                                       min_separation_Hz=max(150.0, 0.6 * f0))
            f1, f2, f3 = (peaks + [0.0, 0.0, 0.0])[:3]
            st = TARGETS_STORY.get(name, (0, 0, 0))
            pb = TARGETS_PB.get(name, (0, 0, 0))
            print(f"{name:<6} {f1:>8.0f} {st[0]:>7.0f} {pb[0]:>7.0f}  "
                  f"{f2:>8.0f} {st[1]:>7.0f} {pb[1]:>7.0f}  "
                  f"{f3:>8.0f} {st[2]:>7.0f} {pb[2]:>7.0f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
