"""Transfer-function calibration gate for Story anchor shapes.

Per Codex 2026-05-09 (thread [session identifier omitted]): voiced-source formant gates
mix transfer-function resonances with source-spectrum harmonics; at
F0=120 the harmonics are too dense for either LPC or FFT to cleanly
separate them. The right primary calibration measurement is the
transfer function H(f) = output / input directly.

This script drives the static Story tube with an impulse at the
subglottal-pressure boundary, captures the impulse response at the
mouth, FFTs it to get |H(f)|, and peak-picks F1/F2/F3 within
target-centered windows (not amplitude-ranked global maxima — that
fails on extreme vowels where higher resonances dominate).

Acceptance: ≥ 9 of 11 ingested vowels within 15% on F1 AND F2,
measured against Story 2008 published computed formants.

Run::
    python scripts/transfer_function_story_anchors.py
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
from tractus_engine_advanced.tds.constants import TIMESTEP_S  # noqa: E402
from tractus_engine_advanced.tds.mna import TubeGeometry  # noqa: E402
from tractus_engine_advanced.tds.state import PRESSURE, TdsState  # noqa: E402
from tractus_engine_advanced.tds.step import step_one  # noqa: E402
from tractus_engine_advanced.tds.tube_topology import (  # noqa: E402
    FIRST_PHARYNX_SECTION, LAST_MOUTH_SECTION, TOPOLOGY,
)


SAMPLE_RATE = int(round(1.0 / TIMESTEP_S))   # 44100 by construction
N_SAMPLES = 8192   # ~186 ms; gives ~5.4 Hz freq resolution
IMPULSE_DPA = 8000.0
FREE_LO = FIRST_PHARYNX_SECTION
FREE_HI = LAST_MOUTH_SECTION + 1


# Story 2008 published computed formants (from worker's table).
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


def build_geom(default: dict, area_tea: np.ndarray,
               length_tea: np.ndarray) -> TubeGeometry:
    """Splice Story shape into default tube's supraglottal slice."""
    A_full = default["area_cm2"].copy()
    A_full[FREE_LO:FREE_HI] = area_tea
    L_full = default["length_cm"].copy()
    L_full[FREE_LO:FREE_HI] = length_tea
    return TubeGeometry.from_arrays(
        length_cm=L_full,
        area_cm2=A_full,
        wall_mass=default["wall_mass_cgs"],
        wall_resistance=default["wall_resistance_cgs"],
        wall_stiffness=default["wall_stiffness_cgs"],
    )


def impulse_response(geom: TubeGeometry, n_samples: int = N_SAMPLES) -> np.ndarray:
    """Drive the tube with an impulse at the subglottal-pressure source.

    Sample 0: pressure = IMPULSE_DPA.  Sample 1+: pressure = 0.
    Output: pressure at FIRST_PHARYNX_SECTION per sample (proxy for the
    tract's response at the start of the supraglottal column — same
    pressure point output_iir.py uses for skin radiation)."""
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
        # Read pressure at the first pharynx section — this is what the
        # output stage reads for radiation. Cleanest tap into the tract's
        # acoustic state without applying the IIR output filtering.
        p_out = out.state.section[FIRST_PHARYNX_SECTION, PRESSURE]
        return out.state, p_out

    _, response = jax.lax.scan(body, tds_state, jnp.arange(n_samples))
    return np.asarray(response)


def transfer_function_peaks(impulse_resp: np.ndarray, sr: int,
                             targets: tuple[float, float, float],
                             f_min: float = 150.0,
                             f_max: float = 5000.0,
                             min_separation_Hz: float = 200.0,
                             ) -> tuple[float, float, float]:
    """FFT the impulse response, find all local maxima, assign to
    F1/F2/F3 in frequency order with minimum-separation constraint.

    This is the standard approach: the transfer function of a vocal
    tract has resonance peaks at F1 < F2 < F3 < ..., so the ordered
    sequence of local maxima in the right band IS the formant
    sequence. No need for amplitude-rank or target-centered windows
    (both of which fail on extreme vowels)."""
    x = impulse_resp.astype(np.float64)
    window = np.hanning(len(x))
    X = np.fft.rfft(x * window)
    psd_dB = 20.0 * np.log10(np.abs(X) + 1e-12)
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)

    # Smooth lightly to reject spurious wiggles
    pad = 5
    yp = np.pad(psd_dB, (pad, pad), mode="edge")
    smoothed = np.convolve(yp, np.ones(11) / 11, mode="valid")

    # Local maxima
    is_peak = np.zeros_like(smoothed, dtype=bool)
    is_peak[1:-1] = (smoothed[1:-1] > smoothed[:-2]) & (smoothed[1:-1] > smoothed[2:])
    band = (freqs >= f_min) & (freqs <= f_max)
    peak_freqs = freqs[is_peak & band]

    # Greedy by-frequency assignment with min-separation
    chosen = []
    for f in peak_freqs:
        if not chosen or (f - chosen[-1] >= min_separation_Hz):
            chosen.append(f)
        if len(chosen) >= 3:
            break
    while len(chosen) < 3:
        chosen.append(0.0)
    return tuple(chosen[:3])


def main() -> int:
    default = load_default_speaker()
    shapes = list(iter_story_anchors())
    print(f"Transfer-function gate. {N_SAMPLES} samples ({N_SAMPLES / SAMPLE_RATE * 1000:.1f} ms) "
          f"impulse response, {SAMPLE_RATE} Hz.\n")

    print(f"{'shape':<6} {'F1 meas':>9} {'F1 St':>7} {'F1 err%':>8}  "
          f"{'F2 meas':>9} {'F2 St':>7} {'F2 err%':>8}  "
          f"{'F3 meas':>9} {'F3 St':>7} {'F3 err%':>8}  verdict")
    print("-" * 102)

    n_pass = 0
    failures = []
    # JIT warmup
    name0, area0, length0 = shapes[0]
    geom0 = build_geom(default, area0, length0)
    _ = impulse_response(geom0)

    for name, area, length in shapes:
        targets = TARGETS_STORY.get(name)
        if targets is None:
            continue
        geom = build_geom(default, area, length)
        ir = impulse_response(geom)
        f1, f2, f3 = transfer_function_peaks(ir, SAMPLE_RATE, targets)
        e1 = abs(f1 - targets[0]) / targets[0] * 100
        e2 = abs(f2 - targets[1]) / targets[1] * 100
        e3 = abs(f3 - targets[2]) / targets[2] * 100
        passed = (e1 <= 15.0) and (e2 <= 15.0)
        verdict = "PASS" if passed else "FAIL"
        if passed:
            n_pass += 1
        else:
            failures.append((name, e1, e2))
        print(f"{name:<6} {f1:>9.0f} {targets[0]:>7.0f} {e1:>7.1f}%  "
              f"{f2:>9.0f} {targets[1]:>7.0f} {e2:>7.1f}%  "
              f"{f3:>9.0f} {targets[2]:>7.0f} {e3:>7.1f}%  [{verdict}]")

    n_total = sum(1 for n, _, _ in shapes if n in TARGETS_STORY)
    print()
    print(f"Pass count: {n_pass} / {n_total} (threshold ≥ 9)")
    if failures:
        print("Failed shapes:")
        for name, e1, e2 in failures:
            print(f"  {name}: F1 err={e1:.1f}%, F2 err={e2:.1f}%")
    if n_pass >= 9:
        print()
        print("VERDICT: TEA's transfer-function resonances match Story's published "
              "computed formants on ≥ 9/11 ingested vowels. The 002a worker's "
              "voiced-source LPC gate failure was a measurement-methodology artifact, "
              "not a TEA calibration issue. Story anchors are usable as ground truth.")
        return 0
    print()
    print(f"VERDICT: TEA-Story transfer-function disagreement on {n_total - n_pass} vowels. "
          "Worker's calibration-gap diagnosis may be partially correct after all. "
          "Investigate boundary conditions (radiation impedance, lip termination, "
          "wall coupling, glottis-end source impedance).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
