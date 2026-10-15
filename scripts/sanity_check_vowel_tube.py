"""Sanity check: forward-render a hand-constructed phonetic /a/, /i/, /u/
tube shape through TEA and measure formants. No optimization.

The question this answers: "is the TEA simulator capable, in principle,
of producing each vowel's literature formants when given a tube shape
that follows standard articulatory-phonetics rules?"

If yes → the inverse-fit failure (F2 collapses to ~1200 Hz for /i/) is
a loss-landscape / optimizer problem and is fixable upstream of the
optimizer.

If no → the simulator setup (trachea / glottis defaults / wall coupling)
has a deeper issue that no amount of optimizer work will fix.

Tube-shape rationale (Stevens 2-tube approximation, well-known in
articulatory phonetics):

  /a/:  narrow pharyngeal cavity (back) + wide mouth (front)
        → high F1, low F2   (target 730 / 1090 / 2440)

  /i/:  wide pharyngeal cavity + tight palatal constriction + open
        front cavity (lips)
        → low F1, very high F2   (target 270 / 2290 / 3010)

  /u/:  wide pharyngeal cavity + tight velar constriction +
        lip-rounded narrow front
        → low F1, low F2    (target 300 / 870 / 2240)

Run::

    cd tractus_controller
    .venv/bin/python ../scripts/sanity_check_vowel_tube.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TEA_PRECISION", "fp32")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tractus_controller"))
sys.path.insert(0, str(ROOT / "tractus_engine_advanced" / "src"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import soundfile as sf  # noqa: E402

from tractus_engine_advanced.acoustic_gates import (  # noqa: E402
    VOWEL_FORMANTS_MALE,
    formant_gate,
)
from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: E402
from tractus_engine_advanced.tds.tube_topology import (  # noqa: E402
    FIRST_PHARYNX_SECTION,   # 25
    LAST_MOUTH_SECTION,      # 64
)


SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 441
F = 30                              # 300 ms render
OUT_DIR = ROOT / "outputs/sanity_vowel_tube"

FREE_LO = FIRST_PHARYNX_SECTION
FREE_HI = LAST_MOUTH_SECTION + 1
N_FREE = FREE_HI - FREE_LO          # 40 sections


def load_default_speaker() -> dict:
    npz = np.load(ROOT / "tractus_controller/outputs/fixtures/default_tube.npz")
    return {
        "length_cm": np.asarray(npz["length_cm"], dtype=np.float32),
        "area_cm2": np.asarray(npz["area_cm2"], dtype=np.float32),
        "wall_mass_cgs": np.asarray(npz["wall_mass_cgs"], dtype=np.float32),
        "wall_resistance_cgs": np.asarray(npz["wall_resistance_cgs"], dtype=np.float32),
        "wall_stiffness_cgs": np.asarray(npz["wall_stiffness_cgs"], dtype=np.float32),
    }


def voiced_glottis_traj() -> np.ndarray:
    g = np.array(
        [120.0, 8000.0, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
        dtype=np.float32,
    )
    return np.tile(g[None, :], (F, 1))


def supraglottal_lengths_cm(default: dict) -> np.ndarray:
    """Per-section length of the 40 free sections, for cumulative-distance
    placement of constrictions."""
    return default["length_cm"][FREE_LO:FREE_HI].astype(np.float64)


def cum_dist_cm(lengths: np.ndarray) -> np.ndarray:
    """Cumulative distance from the glottis end (section 25) along the 40
    free sections, evaluated at the END of each section. For section i in
    [0, 39], cum[i] = sum(lengths[:i+1])."""
    return np.cumsum(lengths)


def _bell(cum: np.ndarray, center_cm: float, width_cm: float, depth: float) -> np.ndarray:
    """Gaussian bell, depth in (0,1); returned values in [1-depth, 1]."""
    g = np.exp(-((cum - center_cm) ** 2) / (2 * width_cm ** 2))
    return 1.0 - depth * g


def shape_a(lengths: np.ndarray, base_area: float = 4.0) -> np.ndarray:
    """/a/: narrow pharynx (back constriction at ~3 cm from glottis),
    wider mouth toward front. F1 high, F2 low."""
    cum = cum_dist_cm(lengths)
    # base of 4 cm² uniform, narrow at ~3 cm from glottis (pharyngeal
    # constriction), open up forward.
    a = base_area * _bell(cum, center_cm=3.0, width_cm=2.0, depth=0.85)
    # widen the front (~10-13 cm) up to ~7 cm²
    front_taper = 1.0 + 0.7 * np.clip((cum - 8.0) / 5.0, 0.0, 1.0)
    return (a * front_taper).clip(0.1, 9.0).astype(np.float32)


def shape_i(lengths: np.ndarray, base_area: float = 5.0) -> np.ndarray:
    """/i/: wide pharynx, tight palatal constriction at ~10-12 cm from
    glottis, wide front cavity. F1 low, F2 very high."""
    cum = cum_dist_cm(lengths)
    # wide pharynx (~5-6 cm²) for first 7 cm, then tight constriction in
    # the palate (~10-12 cm from glottis), then open front.
    a = np.full_like(cum, base_area)
    # palatal constriction
    a *= _bell(cum, center_cm=11.0, width_cm=1.2, depth=0.92)
    # slight widening at front (lips somewhat open for /i/)
    a *= 1.0 + 0.4 * np.clip((cum - 13.0) / 2.0, 0.0, 1.0)
    return a.clip(0.1, 9.0).astype(np.float32)


def shape_u(lengths: np.ndarray, base_area: float = 4.0) -> np.ndarray:
    """/u/: moderately wide pharynx, velar constriction at ~7-9 cm from
    glottis, lip-rounded narrow front. F1 low, F2 low."""
    cum = cum_dist_cm(lengths)
    a = np.full_like(cum, base_area)
    # velar constriction at mid-tract
    a *= _bell(cum, center_cm=8.0, width_cm=1.5, depth=0.85)
    # lip rounding: narrow at very front
    a *= _bell(cum, center_cm=14.5, width_cm=1.0, depth=0.75)
    return a.clip(0.1, 9.0).astype(np.float32)


SHAPES = {"a": shape_a, "i": shape_i, "u": shape_u}


def render(default: dict, supra_area: np.ndarray) -> np.ndarray:
    L_full = jnp.asarray(default["length_cm"])
    A_full = jnp.asarray(default["area_cm2"]).at[FREE_LO:FREE_HI].set(
        jnp.asarray(supra_area)
    )
    Mw = jnp.asarray(default["wall_mass_cgs"])
    Bw = jnp.asarray(default["wall_resistance_cgs"])
    Kw = jnp.asarray(default["wall_stiffness_cgs"])
    L_t = jnp.tile(L_full[None, :], (F, 1))
    A_t = jnp.tile(A_full[None, :], (F, 1))
    Mw_t = jnp.tile(Mw[None, :], (F, 1))
    Bw_t = jnp.tile(Bw[None, :], (F, 1))
    Kw_t = jnp.tile(Kw[None, :], (F, 1))
    g = jnp.asarray(voiced_glottis_traj())
    audio, *_ = synthesize_dynamic_tube(
        L_t, A_t, Mw_t, Bw_t, Kw_t,
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=g,
    )
    return np.asarray(audio)


def main() -> int:
    default = load_default_speaker()
    lengths = supraglottal_lengths_cm(default)
    cum = cum_dist_cm(lengths)
    print(f"Free section span: 0 .. {cum[-1]:.2f} cm "
          f"({N_FREE} sections, mean section length {lengths.mean():.3f} cm)")
    print(f"JIT-compiling forward pass (T={F} frames, 300 ms)...")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print()

    # Warm up jit once
    t0 = time.time()
    _ = render(default, SHAPES["a"](lengths))
    print(f"  warmup compile: {time.time() - t0:.1f}s\n")

    print(f"{'vowel':<6}{'F1 meas':>10}{'F1 tgt':>10}{'F2 meas':>10}{'F2 tgt':>10}{'F3 meas':>10}{'F3 tgt':>10}{'   verdict'}")
    print("-" * 84)
    for name, shape_fn in SHAPES.items():
        F1, F2, F3 = VOWEL_FORMANTS_MALE[name]
        supra = shape_fn(lengths)
        audio = render(default, supra)

        # Save WAV
        peak = float(np.max(np.abs(audio)))
        scale = 0.5 / peak if peak > 0 else 1.0
        sf.write(str(OUT_DIR / f"{name}_handcrafted.wav"), audio * scale, SAMPLE_RATE)

        # Measure formants
        gate = formant_gate(audio, SAMPLE_RATE,
                            target_F1_Hz=F1, target_F2_Hz=F2, target_F3_Hz=F3)
        m = gate.to_dict()["measured"]
        verdict = "PASS" if gate.passed else "FAIL"
        print(f"/{name}/   "
              f"{m['F1_Hz']:>9.0f}{F1:>10.0f}"
              f"{m['F2_Hz']:>10.0f}{F2:>10.0f}"
              f"{m['F3_Hz']:>10.0f}{F3:>10.0f}"
              f"   [{verdict}]  "
              f"(F1 {m['F1_err_pct']:.1f}%, F2 {m['F2_err_pct']:.1f}%, F3 {m['F3_err_pct']:.1f}%)")

    print()
    print(f"Audio: {OUT_DIR.relative_to(ROOT)}/{{a,i,u}}_handcrafted.wav")
    return 0


if __name__ == "__main__":
    sys.exit(main())
