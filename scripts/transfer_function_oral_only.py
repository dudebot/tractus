"""Oral-only transfer-function gate for Story anchor shapes.

Iteration 2, per Codex 2026-05-09 (thread [session identifier omitted]): the previous
attempt left side branches (nose/fossa/sinus) at default values and
read pressure at FIRST_PHARYNX_SECTION rather than at the lips. The
spurious ~1270 Hz peak that was hijacking F2/F3 assignment is most
plausibly a fossa or source-side boundary mode.

This script isolates the supraglottal oral tube:
- Glottis sections 23-24 fixed open at 0.05 cm² (neutral phonation
  posture, not closed).
- Nasal + fossa + sinus sections 65-92 collapsed to MIN_AREA_CM2
  (effective ablation of side branches).
- Output is mouth flow (step.mouth_flow), not pharynx pressure.
- All local maxima printed for inspection.
- Formant assignment uses target-centered windows (nearest peak
  within ±25% of expected, fallback reported explicitly).
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

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tractus_controller.body_model.anchors import iter_story_anchors  # noqa: E402
from tractus_engine_advanced.tds.constants import MIN_AREA_CM2, TIMESTEP_S  # noqa: E402
from tractus_engine_advanced.tds.mna import TubeGeometry  # noqa: E402
from tractus_engine_advanced.tds.state import TdsState  # noqa: E402
from tractus_engine_advanced.tds.step import step_one  # noqa: E402
from tractus_engine_advanced.tds.tube_topology import (  # noqa: E402
    FIRST_PHARYNX_SECTION,
    LAST_MOUTH_SECTION,
    LAST_NOSE_SECTION,
    LOWER_GLOTTIS_SECTION,
    NUM_SECTIONS,
    TOPOLOGY,
    UPPER_GLOTTIS_SECTION,
)


SAMPLE_RATE = int(round(1.0 / TIMESTEP_S))
N_SAMPLES = 16384
IMPULSE_DPA = 8000.0
GLOTTIS_OPEN_CM2 = 0.05
FREE_LO = FIRST_PHARYNX_SECTION
FREE_HI = LAST_MOUTH_SECTION + 1


# Story 2008 published computed formants.
TARGETS_STORY = {
    "/i/": (325, 2139, 2976), "/ɪ/": (413, 2103, 2626), "/e/": (484, 1887, 2404),
    "/ɛ/": (565, 1595, 2200), "/æ/": (810, 1655, 2310), "/ʌ/": (593, 836, 3099),
    "/ɑ/": (694, 942, 2948), "/ɔ/": (557, 709, 3176), "/o/": (499, 768, 2421),
    "/ʊ/": (477, 789, 2499), "/u/": (314, 702, 2298),
}


def load_default_speaker() -> dict:
    npz = np.load(ROOT / "tractus_controller/outputs/fixtures/default_tube.npz")
    return {k: np.asarray(npz[k], dtype=np.float32) for k in
            ("length_cm", "area_cm2", "wall_mass_cgs",
             "wall_resistance_cgs", "wall_stiffness_cgs")}


def build_oral_geom(default: dict, area_tea: np.ndarray,
                    length_tea: np.ndarray) -> TubeGeometry:
    """Construct geometry with side branches ablated and glottis open."""
    A_full = default["area_cm2"].copy().astype(np.float32)
    L_full = default["length_cm"].copy().astype(np.float32)

    # Splice Story shape into supraglottal slice
    A_full[FREE_LO:FREE_HI] = area_tea
    L_full[FREE_LO:FREE_HI] = length_tea

    # Open the glottis (sections 23-24) so source can radiate freely
    A_full[LOWER_GLOTTIS_SECTION] = GLOTTIS_OPEN_CM2
    A_full[UPPER_GLOTTIS_SECTION] = GLOTTIS_OPEN_CM2

    # Ablate side branches: nose 65..LAST_NOSE_SECTION + fossa + sinus
    # All collapsed to MIN_AREA_CM2 to suppress their resonances.
    A_full[LAST_MOUTH_SECTION + 1:NUM_SECTIONS] = float(MIN_AREA_CM2)

    return TubeGeometry.from_arrays(
        length_cm=L_full,
        area_cm2=A_full,
        wall_mass=default["wall_mass_cgs"],
        wall_resistance=default["wall_resistance_cgs"],
        wall_stiffness=default["wall_stiffness_cgs"],
    )


def impulse_response_mouth(geom: TubeGeometry, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Drive subglottal-pressure boundary with an impulse, capture
    mouth_flow per sample."""
    tds_state = TdsState.zeros()

    def body(carry, sample_idx):
        st = carry
        p_in = jnp.where(sample_idx == 0, IMPULSE_DPA, 0.0)
        out = step_one(
            st, geom, TOPOLOGY,
            subglottal_pressure_dPa=p_in,
            soft_walls=True,
            turbulence_losses=True,
        )
        return out.state, out.mouth_flow

    _, response = jax.lax.scan(body, tds_state, jnp.arange(n_samples))
    return np.asarray(response)


def all_peaks(impulse_resp: np.ndarray, sr: int,
              f_min: float = 150.0, f_max: float = 5000.0,
              smooth_w: int = 7) -> list[tuple[float, float]]:
    """All local maxima in the smoothed transfer-function magnitude.
    Returns list of (freq_Hz, amplitude_dB) sorted by frequency."""
    x = impulse_resp.astype(np.float64)
    window = np.hanning(len(x))
    X = np.fft.rfft(x * window)
    psd_dB = 20.0 * np.log10(np.abs(X) + 1e-12)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)

    pad = smooth_w // 2
    yp = np.pad(psd_dB, (pad, pad), mode="edge")
    smoothed = np.convolve(yp, np.ones(smooth_w) / smooth_w, mode="valid")

    is_peak = np.zeros_like(smoothed, dtype=bool)
    is_peak[1:-1] = (smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] > smoothed[2:])
    band = (freqs >= f_min) & (freqs <= f_max)
    return [(float(f), float(a)) for f, a in zip(freqs[is_peak & band], smoothed[is_peak & band])]


def assign_target_windowed(peaks: list[tuple[float, float]],
                            targets: tuple[float, float, float],
                            window_pct: float = 0.25
                            ) -> tuple[tuple[float, float, float],
                                       tuple[bool, bool, bool]]:
    """For each target formant, take the highest-amplitude local max
    within ±window_pct of target. Return measured triplet + per-formant
    found-flag. If nothing in window, fallback to nearest peak by freq
    and flag found=False."""
    measured = []
    found = []
    for tgt in targets:
        half = window_pct * tgt
        in_window = [(f, a) for f, a in peaks if abs(f - tgt) <= half]
        if in_window:
            in_window.sort(key=lambda x: -x[1])  # highest amplitude first
            measured.append(in_window[0][0])
            found.append(True)
        else:
            if peaks:
                nearest = min(peaks, key=lambda x: abs(x[0] - tgt))
                measured.append(nearest[0])
            else:
                measured.append(0.0)
            found.append(False)
    return tuple(measured), tuple(found)


def main() -> int:
    default = load_default_speaker()
    shapes = list(iter_story_anchors())
    print(f"Oral-only transfer-function gate. Glottis open ({GLOTTIS_OPEN_CM2} cm²), "
          f"side branches ablated. {N_SAMPLES} samples ({N_SAMPLES / SAMPLE_RATE * 1000:.1f} ms).\n")

    # JIT warmup
    name0, area0, length0 = shapes[0]
    geom0 = build_oral_geom(default, area0, length0)
    _ = impulse_response_mouth(geom0)

    print(f"{'shape':<6} {'F1 meas':>8} {'F1 St':>7} {'%err':>6} {'?':>2}  "
          f"{'F2 meas':>8} {'F2 St':>7} {'%err':>6} {'?':>2}  "
          f"{'F3 meas':>8} {'F3 St':>7} {'%err':>6} {'?':>2}  verdict")
    print("-" * 110)

    n_pass = 0
    n_total = 0
    failures = []
    all_peak_dump: dict[str, list[tuple[float, float]]] = {}

    for name, area, length in shapes:
        targets = TARGETS_STORY.get(name)
        if targets is None:
            continue
        n_total += 1
        geom = build_oral_geom(default, area, length)
        ir = impulse_response_mouth(geom)
        peaks = all_peaks(ir, SAMPLE_RATE)
        all_peak_dump[name] = peaks
        measured, found = assign_target_windowed(peaks, targets, window_pct=0.25)
        f1, f2, f3 = measured
        e1 = abs(f1 - targets[0]) / targets[0] * 100
        e2 = abs(f2 - targets[1]) / targets[1] * 100
        e3 = abs(f3 - targets[2]) / targets[2] * 100
        passed = (e1 <= 15.0) and (e2 <= 15.0) and found[0] and found[1]
        verdict = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            failures.append((name, e1, e2, e3, found))
        flag = lambda b: "✓" if b else "✗"
        print(f"{name:<6} {f1:>8.0f} {targets[0]:>7.0f} {e1:>5.1f}% {flag(found[0]):>2}  "
              f"{f2:>8.0f} {targets[1]:>7.0f} {e2:>5.1f}% {flag(found[1]):>2}  "
              f"{f3:>8.0f} {targets[2]:>7.0f} {e3:>5.1f}% {flag(found[2]):>2}  [{verdict}]")

    print()
    print(f"Pass count: {n_pass} / {n_total} (threshold ≥ 9)")

    print("\nAll local maxima (freq, amp_dB), per shape:")
    for name, peaks in all_peak_dump.items():
        peak_str = "  ".join(f"{f:.0f}({a:.1f})" for f, a in peaks)
        print(f"  {name:<6} {peak_str}")

    if n_pass >= 9:
        print()
        print("VERDICT: TEA's oral-tract transfer function matches Story computed "
              "formants on ≥ 9/11 vowels. Story anchors are calibrated ground truth.")
        return 0

    print()
    print(f"VERDICT: {n_total - n_pass} vowel(s) miss F1+F2 within 15% of Story targets "
          f"under oral-only isolation. Investigate per-vowel: which targets are absent "
          f"from the all-peaks dump above?")
    return 1


if __name__ == "__main__":
    sys.exit(main())
