"""correctness/002 — generate 30-utterance synthetic stress-edge test corpus.

The corpus stresses parameter-space edges (extreme F0, wide formants,
near-clamp constrictions, anchor-shape transitions) — explicitly NOT a
re-use of smoke/001's template-gesture set, which exercises typical
phones.

Each utterance is a JAX-ready NPZ saved under
``outputs/fixtures/correctness_corpus/`` with the schema:

* ``length_cm``       (F, 93)  per-frame tube section lengths (cm)
* ``area_cm2``        (F, 93)  per-frame tube section areas (cm²)
* ``wall_mass``       (F, 93)  per-frame wall mass (cgs)
* ``wall_resistance`` (F, 93)
* ``wall_stiffness``  (F, 93)
* ``glottis_traj``    (F, 11)  per-frame GeometricGlottis control params
* ``samples_per_frame`` ()     scalar int — driver uses 110 (= TEA-native)
* ``sample_rate``    ()        scalar int = 44100 Hz
* ``utt_id``         ()        str
* ``family``         ()        str — generator family
* ``description``    ()        str — short human-readable tag

Generator families (see ``UTTERANCES`` below for the full list of 30):

1. **f0_extreme**       — sweep F0 at the lower / upper clamps + glide
2. **formant_wide**     — uniform-tube vowel with extreme area asymmetry
3. **near_clamp**       — pharynx/mouth area → MIN_AREA (constrictions)
4. **anchor_transitions**  — cardinal-vowel-like area-trajectory glides
5. **glottis_sweeps**   — XB/XT, CA chink-area, RA, DP, FL, AS sweeps

The samples_per_frame=110 / sample_rate=44100 contract matches
``scripts/render_m5_oracle.py`` so this corpus is consistent with the
infra/005 oracle convention. Each utterance lasts ~0.5 s (≈ 200 frames).

The generator is deterministic (np.random seeded by utterance index)
and re-runnable. Output dir is gitignored.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Pin fp32 precision so DTYPE consistency holds for the saved NPZ.
os.environ.setdefault("TEA_PRECISION", "fp32")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TEA_SRC = REPO_ROOT / "tractus_engine_advanced" / "src"
sys.path.insert(0, str(TEA_SRC))

CONTROLLER = REPO_ROOT / "tractus_controller"
sys.path.insert(0, str(CONTROLLER))

OUT_DIR = REPO_ROOT / "outputs" / "fixtures" / "correctness_corpus"

# --- corpus-wide constants ---
SAMPLES_PER_FRAME = 110
SAMPLE_RATE = 44100
NUM_FRAMES = 200            # ~0.5 s at 44100 / 110
NUM_SECTIONS = 93
MIN_AREA = 1e-4

# Section index conventions (mirror tube_topology):
TRACHEA = slice(0, 23)              # 23 sections
LOWER_GLOTTIS = 23
UPPER_GLOTTIS = 24
PHARYNX_MOUTH = slice(25, 65)        # 40 sections
NOSE = slice(65, 84)
FOSSA = slice(84, 89)
SINUS = slice(89, 93)


def _load_default_tube() -> dict:
    """Load the default tube fixture; serves as the speaker-anatomy base."""
    from tractus_controller.adapters.r1_inference import load_default_tube
    d = load_default_tube()
    return {k: np.asarray(v, dtype=np.float32) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Corpus families
# ---------------------------------------------------------------------------


def _glottis_neutral() -> np.ndarray:
    """Default 11-vector glottis params; quasi-neutral phonation."""
    # F0=120, PR=8000, XB=0.01, XT=0.02, CA=0.05, PL=1.22, RA=1.0,
    # DP=0.05, PS=0.0, FL=25.0, AS=-10.0
    return np.array(
        [120.0, 8000.0, 0.01, 0.02, 0.05, 1.22, 1.0, 0.05, 0.0, 25.0, -10.0],
        dtype=np.float32,
    )


def _make_tube_traj(default: dict, area_overrides_pm: np.ndarray) -> tuple[np.ndarray, ...]:
    """Build (length, area, walls) trajectories of shape (F, 93) given a
    pharynx-mouth area-trajectory override (F, 40).

    Length is held at the default; only `area_cm2[:, 25:65]` varies frame
    to frame. Wall arrays are broadcast from the default (per-frame copy).
    """
    F = area_overrides_pm.shape[0]
    L = np.broadcast_to(default["length_cm"][None, :], (F, NUM_SECTIONS)).copy()
    A = np.broadcast_to(default["area_cm2"][None, :], (F, NUM_SECTIONS)).copy()
    A[:, PHARYNX_MOUTH] = np.maximum(area_overrides_pm, MIN_AREA)
    Mw = np.broadcast_to(default["wall_mass_cgs"][None, :], (F, NUM_SECTIONS)).copy()
    Bw = np.broadcast_to(default["wall_resistance_cgs"][None, :], (F, NUM_SECTIONS)).copy()
    Kw = np.broadcast_to(default["wall_stiffness_cgs"][None, :], (F, NUM_SECTIONS)).copy()
    return L, A, Mw, Bw, Kw


def _vowel_template(name: str) -> np.ndarray:
    """Return a (40,) cardinal-vowel-like pharynx-mouth area template (cm²).

    These are not anatomically-accurate area functions; they're crude
    parameterizations intended to span the formant space without being
    near-identical. Areas held >= 0.4 cm² (well above MIN_AREA).
    """
    base = np.full(40, 4.0, dtype=np.float32)
    if name == "uniform":
        return base
    if name == "ah":
        # Wide pharynx, narrow mouth.
        base[:20] = 5.0
        base[20:] = 1.5
        return base
    if name == "ee":
        # Narrow pharynx, wide mouth.
        base[:20] = 1.0
        base[20:] = 5.0
        return base
    if name == "oo":
        # Narrow at lips and pharynx, wider middle.
        base[:5] = 4.0
        base[5:25] = 5.0
        base[25:35] = 2.0
        base[35:] = 0.4
        return base
    if name == "ae":
        # Front-low: open pharynx, mid mouth.
        base[:15] = 5.0
        base[15:30] = 3.0
        base[30:] = 0.7
        return base
    raise ValueError(name)


def _glide(a: np.ndarray, b: np.ndarray, F: int) -> np.ndarray:
    """Linear glide from area-template ``a`` to ``b`` over F frames."""
    t = np.linspace(0.0, 1.0, F, dtype=np.float32)[:, None]
    return (1.0 - t) * a[None, :] + t * b[None, :]


# Family 1: F0 extremes ------------------------------------------------------
def _family_f0_extreme(rng: np.random.Generator, idx: int) -> dict:
    """F0 sweeps: low clamp, high clamp, octave glide, jitter at edges."""
    F = NUM_FRAMES
    g = np.broadcast_to(_glottis_neutral()[None, :], (F, 11)).copy()
    if idx == 0:
        # Hold F0 at lower clamp (40 Hz).
        g[:, 0] = 42.0
        desc = "f0_low_clamp_42Hz"
    elif idx == 1:
        # Hold F0 at upper clamp (600 Hz).
        g[:, 0] = 580.0
        desc = "f0_high_near_clamp_580Hz"
    elif idx == 2:
        # 80 → 500 Hz log-glide.
        g[:, 0] = np.geomspace(80.0, 500.0, F, dtype=np.float32)
        desc = "f0_glide_80_to_500Hz"
    elif idx == 3:
        # 500 → 80 Hz log-glide (descending).
        g[:, 0] = np.geomspace(500.0, 80.0, F, dtype=np.float32)
        desc = "f0_glide_500_to_80Hz"
    elif idx == 4:
        # F0=120 with extreme flutter (FL=80%) + low pressure.
        g[:, 0] = 120.0
        g[:, 9] = 80.0  # flutter percent
        g[:, 1] = 4000.0
        desc = "f0_flutter_high"
    else:
        raise IndexError(idx)
    pm = np.broadcast_to(_vowel_template("ah")[None, :], (F, 40)).copy()
    L, A, Mw, Bw, Kw = _make_tube_traj(default_tube, pm)
    return {
        "utt_id": f"f0_extreme_{idx:02d}",
        "family": "f0_extreme",
        "description": desc,
        "length_cm": L, "area_cm2": A,
        "wall_mass": Mw, "wall_resistance": Bw, "wall_stiffness": Kw,
        "glottis_traj": g,
    }


# Family 2: wide-formant vowels ---------------------------------------------
def _family_formant_wide(rng: np.random.Generator, idx: int) -> dict:
    """Vowel templates that produce widely separated formants."""
    F = NUM_FRAMES
    g = np.broadcast_to(_glottis_neutral()[None, :], (F, 11)).copy()
    g[:, 0] = 110.0
    if idx == 0:
        tpl = _vowel_template("ah")
        desc = "vowel_ah_steady"
    elif idx == 1:
        tpl = _vowel_template("ee")
        desc = "vowel_ee_steady"
    elif idx == 2:
        tpl = _vowel_template("oo")
        desc = "vowel_oo_steady"
    elif idx == 3:
        tpl = _vowel_template("ae")
        desc = "vowel_ae_steady"
    elif idx == 4:
        tpl = _vowel_template("uniform")
        desc = "uniform_tube_4cm2"
    else:
        raise IndexError(idx)
    pm = np.broadcast_to(tpl[None, :], (F, 40)).copy()
    L, A, Mw, Bw, Kw = _make_tube_traj(default_tube, pm)
    return {
        "utt_id": f"formant_wide_{idx:02d}",
        "family": "formant_wide",
        "description": desc,
        "length_cm": L, "area_cm2": A,
        "wall_mass": Mw, "wall_resistance": Bw, "wall_stiffness": Kw,
        "glottis_traj": g,
    }


# Family 3: near-clamp constrictions ----------------------------------------
def _family_near_clamp(rng: np.random.Generator, idx: int) -> dict:
    """Constrictions of varying severity; min area approaches MIN_AREA_CM2.

    These do NOT model frication noise (M7 not yet shipped) — they only
    test the MNA solver's behavior under near-zero-area sections.
    """
    F = NUM_FRAMES
    g = np.broadcast_to(_glottis_neutral()[None, :], (F, 11)).copy()
    g[:, 0] = 110.0
    base = _vowel_template("uniform")
    pm = np.broadcast_to(base[None, :], (F, 40)).copy()
    if idx == 0:
        # Lip constriction (sec 60..64). Min 0.05 cm² (well above MIN_AREA).
        constriction = 0.05 + 0.45 * (1.0 - np.exp(-np.arange(F) / 30.0))
        pm[:, 35:40] = constriction[:, None]
        desc = "lip_constriction_decay"
    elif idx == 1:
        # Tongue-back constriction (sec 35..40). Min 0.02 cm².
        constriction = 0.02 + 0.5 * np.abs(np.sin(np.linspace(0, 2 * np.pi, F)))
        pm[:, 10:15] = constriction[:, None]
        desc = "tongue_back_oscillation"
    elif idx == 2:
        # Two simultaneous constrictions.
        c1 = 0.03 + 0.3 * np.linspace(0, 1, F)
        c2 = 0.04 + 0.2 * np.linspace(1, 0, F)
        pm[:, 8:12] = c1[:, None]
        pm[:, 28:32] = c2[:, None]
        desc = "double_constriction_glide"
    elif idx == 3:
        # Slow opening from near-clamp (0.01 → 1.0 cm²).
        constriction = np.linspace(0.011, 1.0, F, dtype=np.float32)
        pm[:, 20:25] = constriction[:, None]
        desc = "very_narrow_open"
    elif idx == 4:
        # Random constriction-noise floor at fixed 0.05 cm².
        rng_fluc = rng.standard_normal((F, 5)).astype(np.float32) * 0.005
        pm[:, 18:23] = np.clip(0.05 + rng_fluc, MIN_AREA, None)
        desc = "narrow_jitter_at_005cm2"
    else:
        raise IndexError(idx)
    L, A, Mw, Bw, Kw = _make_tube_traj(default_tube, pm)
    return {
        "utt_id": f"near_clamp_{idx:02d}",
        "family": "near_clamp",
        "description": desc,
        "length_cm": L, "area_cm2": A,
        "wall_mass": Mw, "wall_resistance": Bw, "wall_stiffness": Kw,
        "glottis_traj": g,
    }


# Family 4: anchor transitions ----------------------------------------------
def _family_anchor_transitions(rng: np.random.Generator, idx: int) -> dict:
    """Glides between cardinal vowel templates."""
    F = NUM_FRAMES
    g = np.broadcast_to(_glottis_neutral()[None, :], (F, 11)).copy()
    g[:, 0] = 110.0
    pairs = [
        ("ah", "ee"),
        ("ah", "oo"),
        ("ee", "oo"),
        ("ah", "ae"),
        ("oo", "ae"),
    ]
    a, b = pairs[idx]
    pm = _glide(_vowel_template(a), _vowel_template(b), F)
    L, A, Mw, Bw, Kw = _make_tube_traj(default_tube, pm)
    return {
        "utt_id": f"anchor_transitions_{idx:02d}",
        "family": "anchor_transitions",
        "description": f"glide_{a}_to_{b}",
        "length_cm": L, "area_cm2": A,
        "wall_mass": Mw, "wall_resistance": Bw, "wall_stiffness": Kw,
        "glottis_traj": g,
    }


# Family 5: glottis-axis sweeps ---------------------------------------------
def _family_glottis_sweeps(rng: np.random.Generator, idx: int) -> dict:
    """Sweep glottis-axis params across their dynamic range, fixed tube."""
    F = NUM_FRAMES
    g = np.broadcast_to(_glottis_neutral()[None, :], (F, 11)).copy()
    g[:, 0] = 110.0
    if idx == 0:
        # CA chink-area sweep (0 → 0.5 cm² near upper clamp).
        g[:, 4] = np.linspace(-0.2, 0.2, F, dtype=np.float32)
        desc = "ca_sweep_-0.2_to_0.2"
    elif idx == 1:
        # RA (rest amplitude) full range.
        g[:, 6] = np.linspace(-1.0, 1.0, F, dtype=np.float32)
        desc = "ra_sweep_-1_to_1"
    elif idx == 2:
        # DP (double pulsing) full range.
        g[:, 7] = np.linspace(0.0, 1.0, F, dtype=np.float32)
        desc = "dp_sweep_0_to_1"
    elif idx == 3:
        # AS (aspiration) full range -40 to 0.
        g[:, 10] = np.linspace(-40.0, 0.0, F, dtype=np.float32)
        desc = "aspiration_sweep"
    elif idx == 4:
        # XB / XT (lower/upper rest gap) push-pull near clamps.
        g[:, 2] = np.linspace(-0.05, 0.3, F, dtype=np.float32)  # XB
        g[:, 3] = np.linspace(0.3, -0.05, F, dtype=np.float32)  # XT
        desc = "xb_xt_pushpull"
    elif idx == 5:
        # PR (subglottal pressure) sweep 0 → 20000 dPa.
        g[:, 1] = np.linspace(0.0, 18000.0, F, dtype=np.float32)
        desc = "pr_sweep"
    elif idx == 6:
        # PS (skewness) full range.
        g[:, 8] = np.linspace(-0.5, 0.5, F, dtype=np.float32)
        desc = "ps_sweep"
    elif idx == 7:
        # PL (phase-lag) full range 0 → π.
        g[:, 5] = np.linspace(0.0, 3.1, F, dtype=np.float32)
        desc = "pl_sweep"
    elif idx == 8:
        # FL (flutter) hold at high.
        g[:, 9] = np.linspace(0.0, 100.0, F, dtype=np.float32)
        desc = "flutter_sweep"
    elif idx == 9:
        # F0 jitter via rng (random walk inside [60, 250]).
        f0 = 110.0 + np.cumsum(rng.standard_normal(F).astype(np.float32) * 4.0)
        g[:, 0] = np.clip(f0, 60.0, 250.0)
        desc = "f0_random_walk"
    else:
        raise IndexError(idx)
    pm = np.broadcast_to(_vowel_template("ah")[None, :], (F, 40)).copy()
    L, A, Mw, Bw, Kw = _make_tube_traj(default_tube, pm)
    return {
        "utt_id": f"glottis_sweeps_{idx:02d}",
        "family": "glottis_sweeps",
        "description": desc,
        "length_cm": L, "area_cm2": A,
        "wall_mass": Mw, "wall_resistance": Bw, "wall_stiffness": Kw,
        "glottis_traj": g,
    }


# 5 + 5 + 5 + 5 + 10 = 30
UTTERANCES = (
    [(_family_f0_extreme, i) for i in range(5)]
    + [(_family_formant_wide, i) for i in range(5)]
    + [(_family_near_clamp, i) for i in range(5)]
    + [(_family_anchor_transitions, i) for i in range(5)]
    + [(_family_glottis_sweeps, i) for i in range(10)]
)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


default_tube: dict | None = None  # filled in main()


def main() -> int:
    global default_tube
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    default_tube = _load_default_tube()

    manifest = []
    for ix, (fn, sub_idx) in enumerate(UTTERANCES):
        rng = np.random.default_rng(seed=1000 + ix)
        rec = fn(rng, sub_idx)
        out_path = OUT_DIR / f"{rec['utt_id']}.npz"
        np.savez(
            out_path,
            length_cm=rec["length_cm"].astype(np.float32),
            area_cm2=rec["area_cm2"].astype(np.float32),
            wall_mass=rec["wall_mass"].astype(np.float32),
            wall_resistance=rec["wall_resistance"].astype(np.float32),
            wall_stiffness=rec["wall_stiffness"].astype(np.float32),
            glottis_traj=rec["glottis_traj"].astype(np.float32),
            samples_per_frame=np.int32(SAMPLES_PER_FRAME),
            sample_rate=np.int32(SAMPLE_RATE),
            utt_id=np.array(rec["utt_id"]),
            family=np.array(rec["family"]),
            description=np.array(rec["description"]),
        )
        manifest.append({
            "utt_id": rec["utt_id"],
            "family": rec["family"],
            "description": rec["description"],
            "frames": int(rec["length_cm"].shape[0]),
            "samples": int(SAMPLES_PER_FRAME * rec["length_cm"].shape[0]),
            "path": str(out_path.relative_to(REPO_ROOT)),
        })
        print(f"[{ix+1:02d}/30] {rec['utt_id']}: {rec['description']}")

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {len(manifest)} utterances under {OUT_DIR}")
    print(f"manifest: {OUT_DIR / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
