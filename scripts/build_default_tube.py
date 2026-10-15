"""Reconstruct the ``default_tube.npz`` speaker fixture.

The original ``tractus_controller/outputs/fixtures/default_tube.npz`` was an
un-tracked (gitignored) fixture rendered on the original dev box and is lost on
this machine. No generator for it was ever committed. This script rebuilds a
physically-plausible, forward-stable default tube from tracked sources so that
the copyfit / Task-009 machinery (which loads ``default_tube.npz`` on both the
target-generation and the inversion side) has a consistent base tube.

Design decisions (all durable — this fixture shapes every downstream result):

* **93-section geometry** matches ``tds/tube_topology.py``: 23 trachea + 2
  glottis + 16 pharynx + 24 mouth + 19 nose + 5 fossa + 4 sinus.
* **Areas**: the supraglottal oral free range (sections 25..64, the 40 sections
  copyfit optimizes) is seeded with the Story-1996 neutral schwa-ish vowel
  (mean of /i/,/A/,/u/ anchors resampled to 40) so the base tube is a real
  neutral vowel tract, not a uniform pipe. Trachea = wide (2.5 cm²), glottis =
  narrow open (0.05 cm²), nose/fossa/sinus = closed-ish small side branches.
* **Lengths**: per-section lengths chosen so the supraglottal tract totals
  ~17.5 cm (Story male VTL) over its 42 pharynx+mouth+2 glottis sections; the
  engine test suite uses a uniform 0.4 cm/section and produces stable physics,
  so we stay in that regime (0.396 cm/section).
* **Wall params (Mw/Bw/Kw)**: uniform ``Mw=1.5, Bw=8000, Kw=1e6`` — the exact
  values the engine's own forward-physics test (``test_m6_soft.py:140-144``,
  ``test_tds_step_soft_forward_finite``) constructs and validates as finite and
  stable. These are speaker-uniform soft-wall constants; the ``_wall_alpha_beta``
  model scales them by per-section surface area, so uniform base constants are
  the correct, defensible reconstruction.

This tube is used *identically* for TEA-to-TEA target generation and for the
copyfit base tube, so the experiment's validity does not depend on the tube
matching the exact lost original — only on it being a consistent, stable,
vowel-realistic neutral tract. Both sides share it.

Run:
    PYTHONPATH=tractus_controller:tractus_engine_advanced/src \\
        python scripts/build_default_tube.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tractus_controller"))
sys.path.insert(0, str(_ROOT / "tractus_engine_advanced" / "src"))

# Section landmarks (mirror tds/tube_topology.py).
N_SECTIONS = 93
FIRST_TRACHEA, LAST_TRACHEA = 0, 22
LOWER_GLOTTIS, UPPER_GLOTTIS = 23, 24
FIRST_PHARYNX, LAST_PHARYNX = 25, 40
FIRST_MOUTH, LAST_MOUTH = 41, 64
FIRST_NOSE, LAST_NOSE = 65, 83
FIRST_FOSSA, LAST_FOSSA = 84, 88
FIRST_SINUS, LAST_SINUS = 89, 92
FREE_LO, FREE_HI = 25, 65            # the 40 supraglottal oral sections
N_FREE = FREE_HI - FREE_LO

OUT_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "default_tube.npz"
ANCHORS_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "story_1996_anchors.npz"

# Wall constants — the engine's validated stable forward-physics values.
WALL_MASS = 1.5        # g / cm^2  (test_m6_soft.py)
WALL_RESISTANCE = 8000.0
WALL_STIFFNESS = 1.0e6

# Per-section length. 42 supra+glottis sections * 0.396 ~= 16.6 cm supraglottal;
# stays inside the 0.4 cm/section regime the engine tests validate.
SECTION_LEN_CM = 0.396


def _neutral_vowel_40() -> np.ndarray:
    """Mean of the /i/, /A/, /u/ Story anchors -> a neutral supraglottal
    area function over the 40 free sections."""
    s = np.load(ANCHORS_PATH, allow_pickle=False)
    keys = [str(k) for k in s["shape_keys"]]
    area = np.asarray(s["area_cm2_tea"], dtype=np.float64)   # (N, 40)
    picks = []
    for k in ("i", "A", "u"):
        if k in keys:
            picks.append(area[keys.index(k)])
    if not picks:
        raise RuntimeError("no /i//A//u/ anchors found in story_1996_anchors.npz")
    return np.mean(np.stack(picks, axis=0), axis=0).astype(np.float32)


def build() -> dict:
    area = np.empty(N_SECTIONS, dtype=np.float32)
    # Trachea: wide subglottal tube.
    area[FIRST_TRACHEA:LAST_TRACHEA + 1] = 2.5
    # Glottis: narrow open glottis (overwritten per-sample by GeometricGlottis
    # at synth time anyway; a small positive value keeps the base tube stable).
    area[LOWER_GLOTTIS] = 0.05
    area[UPPER_GLOTTIS] = 0.05
    # Supraglottal oral free range: neutral Story vowel.
    area[FREE_LO:FREE_HI] = _neutral_vowel_40()
    # Nose / fossa / sinus side branches: small closed-ish cavities. The velar
    # port is effectively shut for oral vowels.
    area[FIRST_NOSE:LAST_NOSE + 1] = 0.3
    area[FIRST_FOSSA:LAST_FOSSA + 1] = 0.3
    area[FIRST_SINUS:LAST_SINUS + 1] = 0.5
    # Clamp to a sane floor.
    area = np.maximum(area, 1e-3).astype(np.float32)

    length = np.full(N_SECTIONS, SECTION_LEN_CM, dtype=np.float32)

    return {
        "length_cm": length,
        "area_cm2": area,
        "wall_mass_cgs": np.full(N_SECTIONS, WALL_MASS, dtype=np.float32),
        "wall_resistance_cgs": np.full(N_SECTIONS, WALL_RESISTANCE, dtype=np.float32),
        "wall_stiffness_cgs": np.full(N_SECTIONS, WALL_STIFFNESS, dtype=np.float32),
    }


def main() -> int:
    d = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_PATH, **d)
    print(f"Wrote {OUT_PATH}")
    for k, v in d.items():
        print(f"  {k:24s} shape={v.shape} min={v.min():.4g} max={v.max():.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
