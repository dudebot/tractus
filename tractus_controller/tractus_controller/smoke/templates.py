"""Per-phone GestureEvent templates and the 30-d neutral trajectory.

The 30-d (T, 30) trajectory layout (per
``copy_synth/parameterization.py`` lines 22-26 and TEA G3a glottis
docs) is:

    cols 0:11   glottis params  (F0, PR, XB, XT, CA, PL, RA, DP, PS,
                                 FL, AS)
    cols 11:30  tract params    (HX, HY, JX, JA, LP, LD, VS, VO, TCX,
                                 TCY, TTX, TTY, TBX, TBY, TRX, TRY,
                                 TS1, TS2, TS3)

The neutral baseline is taken from ``natasha_shape.npz`` frame 0 — a
known-good speaker baseline already exercised by the JAX gradient
fitter and copy-synth tests. We then override a small set of channels
per phone to get phone-specific targets. The set of channels per
phone is intentionally minimal (one-to-three anchors); coarticulation
emerges later via gesture overlap in pretraining/003.

This is NOT a phonetics-grade articulator inventory — it's a smoke
floor. The named-shape catalog from the legacy upstream XML was
deleted (ARCH §1051); pulling those values back in is a separate
fixture task. For smoke we hand-tune offsets within the trained R1
NN's tract_min/tract_max ranges (see r1_inference.py: norm.tract_min /
tract_max bounds).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 30-d trajectory column indices.
GLOTTIS_F0 = 0
GLOTTIS_PR = 1     # subglottal lung pressure (dPa)
GLOTTIS_XB = 2     # vocal-fold lower-mass displacement
GLOTTIS_XT = 3     # vocal-fold upper-mass displacement
GLOTTIS_CA = 4     # adduction (chink area, cm^2)
GLOTTIS_PL = 5
GLOTTIS_RA = 6
GLOTTIS_DP = 7
GLOTTIS_PS = 8
GLOTTIS_FL = 9     # flutter / jitter
GLOTTIS_AS = 10    # aspiration

TRACT_HX = 11
TRACT_HY = 12
TRACT_JX = 13
TRACT_JA = 14    # jaw angle (negative = open)
TRACT_LP = 15    # lip protrusion
TRACT_LD = 16    # lip distance / aperture (cm)
TRACT_VS = 17
TRACT_VO = 18    # velar opening (nasals)
TRACT_TCX = 19   # tongue body horizontal
TRACT_TCY = 20   # tongue body vertical
TRACT_TTX = 21   # tongue tip horizontal
TRACT_TTY = 22   # tongue tip vertical
TRACT_TBX = 23
TRACT_TBY = 24
TRACT_TRX = 25
TRACT_TRY = 26
TRACT_TS1 = 27
TRACT_TS2 = 28
TRACT_TS3 = 29


# Default named channel (string) -> 30-d column. Used by GestureEvent
# emit code so the sidecar JSON stays human-legible (channel_id
# strings rather than naked column indices).
CHANNEL_BY_NAME: dict[str, int] = {
    "F0": GLOTTIS_F0,
    "lung_pressure": GLOTTIS_PR,
    "glottal_adduction": GLOTTIS_CA,
    "aspiration": GLOTTIS_AS,
    "jaw_angle": TRACT_JA,
    "lip_protrusion": TRACT_LP,
    "lip_aperture": TRACT_LD,
    "velum_opening": TRACT_VO,
    "tongue_body_x": TRACT_TCX,
    "tongue_body_y": TRACT_TCY,
    "tongue_tip_x": TRACT_TTX,
    "tongue_tip_y": TRACT_TTY,
}


# Neutral baseline trajectory frame (30,) — taken from natasha_shape.npz
# frame 0, a known-good speaker baseline. Loaded lazily so the module
# imports without numpy file I/O.
_NEUTRAL_FRAME: np.ndarray | None = None


def _natasha_path() -> Path:
    """Locate the natasha_shape fixture.

    Lives in the repo's top-level ``outputs/fixtures/`` (not in
    ``tractus_controller/outputs/``). Falls back to an env-var override
    for relocated worktrees.
    """
    import os

    env = os.environ.get("TRACTUS_NATASHA_SHAPE")
    if env:
        p = Path(env)
        if p.exists():
            return p

    # ``__file__`` -> .../tractus_controller/tractus_controller/smoke/templates.py
    # workspace_root() in paths.py is .../tractus_controller/.. (the repo).
    # outputs/fixtures lives at the repo root.
    here = Path(__file__).resolve()
    for parent in (here.parents[3], here.parents[4]):
        cand = parent / "outputs" / "fixtures" / "natasha_shape.npz"
        if cand.exists():
            return cand
    raise FileNotFoundError(
        "natasha_shape.npz not found under outputs/fixtures. Set "
        "TRACTUS_NATASHA_SHAPE to override."
    )


def neutral_frame() -> np.ndarray:
    """Return the (30,) neutral baseline frame."""
    global _NEUTRAL_FRAME
    if _NEUTRAL_FRAME is None:
        d = np.load(_natasha_path(), allow_pickle=True)
        traj = np.asarray(d["trajectory"], dtype=np.float64)
        _NEUTRAL_FRAME = traj[0].copy()
    return _NEUTRAL_FRAME.copy()


# A "silent" frame: copy neutral, then zero out lung pressure and
# aspiration so the simulator coasts (no driving energy).
def silent_frame() -> np.ndarray:
    f = neutral_frame()
    f[GLOTTIS_PR] = 0.0
    f[GLOTTIS_AS] = 0.0
    return f


# A "voiced rest" frame: same neutral, set PR to a steady-voicing
# pressure and F0 to a relaxed mid-male pitch.
def voiced_rest_frame(f0_hz: float = 110.0, lung_pa: float = 8000.0) -> np.ndarray:
    f = neutral_frame()
    f[GLOTTIS_F0] = f0_hz
    f[GLOTTIS_PR] = lung_pa
    return f


@dataclass(frozen=True)
class PhoneTemplate:
    """One ARPAbet phone's anchor gesture targets.

    Each entry in ``overrides`` is ``(column_index, target_value,
    strength)`` — strength scales how strongly the trajectory is pulled
    toward the target during the phone's window (0.0 -> no movement,
    1.0 -> full target).

    ``voiced`` flips the lung-pressure / F0 envelope on or off. Stops
    take ``release_s_default`` (a brief burst window), fricatives a
    short fade.
    """

    arpabet: str
    overrides: tuple[tuple[int, float, float], ...]
    voiced: bool
    aspiration: float = 0.0          # 0..1 (AS column)
    release_s_default: float = 0.02


# Tract-channel "named shapes" used below. These values are
# hand-picked within the per-channel min/max bounds tracked by
# ``copy_synth/parameterization.DEFAULT_FREE_CHANNELS`` and the R1 NN
# norm (see r1_inference.tract_min/tract_max). A real named-shape
# inventory would come from the deleted upstream XML — that's a fixture
# task, not smoke-blocking.
#
# Bounds reminder (from FreeChannel + R1Norm):
#   JA   (col 14)  range [-7.0, 0.0]    (0 = closed, -6 = wide open)
#   LP   (col 15)  range [-1.0, 1.0]
#   LD   (col 16)  range [-2.0, 4.0]    (negative = pressed shut)
#   VO   (col 18)  range [-0.1, 1.0]    (0 = closed, 1 = wide nasal)
#   TCX  (col 19)  range [-3.0, 4.0]    (front/back)
#   TCY  (col 20)  range [-3.0, 1.0]    (low/high)
#
# Strength 0.6-0.9 is enough to nudge the simulator into phonetic
# territory without slamming bounds (which would expose R1 NN
# extrapolation problems on top of an undertrained body model).

# Anchor offsets keyed by phone class.
_S = 0.7  # default override strength


def _vowel(arpa: str, ja: float, ld: float, tcx: float, tcy: float, lp: float = 0.0) -> PhoneTemplate:
    return PhoneTemplate(
        arpabet=arpa,
        overrides=(
            (TRACT_JA,  ja,  _S),
            (TRACT_LD,  ld,  _S),
            (TRACT_LP,  lp,  _S),
            (TRACT_TCX, tcx, _S),
            (TRACT_TCY, tcy, _S),
        ),
        voiced=True,
    )


def _stop(arpa: str, *, voiced: bool, place: str) -> PhoneTemplate:
    """Stop closure target: very small lip aperture (LD) for bilabials,
    or tongue-tip raise (TTY) for alveolars, or tongue-body raise (TCY)
    for velars. Smoke is loose — we just need the simulator to *do*
    something class-appropriate.
    """
    if place == "bilabial":  # P, B
        overrides = (
            (TRACT_LD, -1.5, 0.9),
            (TRACT_LP,  0.3, _S),
            (TRACT_JA, -0.2, _S),
        )
    elif place == "alveolar":  # T, D
        overrides = (
            (TRACT_TTY,  3.5, 0.9),
            (TRACT_TCX,  0.5, _S),
            (TRACT_LD,   1.0, _S),
        )
    elif place == "velar":  # K, G
        overrides = (
            (TRACT_TCY,  0.8, 0.9),
            (TRACT_TCX, -1.0, _S),
            (TRACT_LD,   1.2, _S),
        )
    elif place == "post_alveolar":  # CH, JH (affricates, treated as stops)
        overrides = (
            (TRACT_TTY,  2.5, 0.85),
            (TRACT_TCX,  0.0, _S),
            (TRACT_LD,   1.0, _S),
        )
    else:
        raise ValueError(f"unknown stop place: {place}")
    aspir = 0.0 if voiced else 0.6
    return PhoneTemplate(
        arpabet=arpa,
        overrides=overrides,
        voiced=voiced,
        aspiration=aspir,
        release_s_default=0.03 if not voiced else 0.015,
    )


def _fricative(arpa: str, *, voiced: bool, place: str) -> PhoneTemplate:
    if place == "labio_dental":  # F, V
        overrides = (
            (TRACT_LD,  -0.4, 0.85),
            (TRACT_LP,   0.2, _S),
            (TRACT_JA,  -0.4, _S),
        )
    elif place == "dental":  # TH, DH
        overrides = (
            (TRACT_TTY,  2.5, 0.8),
            (TRACT_TTX,  0.5, _S),
            (TRACT_LD,   0.5, _S),
        )
    elif place == "alveolar":  # S, Z
        overrides = (
            (TRACT_TTY,  2.8, 0.85),
            (TRACT_TCX,  0.5, _S),
            (TRACT_LD,   0.7, _S),
        )
    elif place == "post_alveolar":  # SH, ZH
        overrides = (
            (TRACT_TTY,  2.0, 0.85),
            (TRACT_TCX, -0.3, _S),
            (TRACT_LP,   0.4, _S),
            (TRACT_LD,   0.9, _S),
        )
    elif place == "glottal":  # HH (frication-like aspiration)
        overrides = (
            (TRACT_LD,   1.5, _S),
        )
    else:
        raise ValueError(f"unknown fricative place: {place}")
    # Voiceless frics need turbulent flow even with small subglottal
    # pressure — smoke pre-frication relies on AS (aspiration) channel
    # for noise excitation; M7 frication (correctness/004) will replace
    # this with per-section gated frication.
    aspir = 0.0 if voiced else 0.5
    if arpa == "HH":
        aspir = 0.7
    return PhoneTemplate(
        arpabet=arpa,
        overrides=overrides,
        voiced=voiced,
        aspiration=aspir,
        release_s_default=0.01,
    )


def _nasal(arpa: str, place: str) -> PhoneTemplate:
    if place == "bilabial":  # M
        overrides = (
            (TRACT_LD,  -1.5, 0.9),
            (TRACT_VO,   0.6, 0.9),
            (TRACT_JA, -0.2, _S),
        )
    elif place == "alveolar":  # N
        overrides = (
            (TRACT_TTY,  3.0, 0.9),
            (TRACT_VO,   0.6, 0.9),
            (TRACT_LD,   1.0, _S),
        )
    elif place == "velar":  # NG
        overrides = (
            (TRACT_TCY,  0.7, 0.9),
            (TRACT_VO,   0.6, 0.9),
            (TRACT_TCX, -1.0, _S),
        )
    else:
        raise ValueError(f"unknown nasal place: {place}")
    return PhoneTemplate(
        arpabet=arpa,
        overrides=overrides,
        voiced=True,
    )


def _approximant(arpa: str) -> PhoneTemplate:
    if arpa == "L":
        overrides = (
            (TRACT_TTY,  2.2, 0.8),
            (TRACT_TCY, -0.8, _S),
            (TRACT_LD,   1.2, _S),
        )
    elif arpa == "R":
        overrides = (
            (TRACT_TTY,  1.0, 0.7),
            (TRACT_TCX, -0.5, _S),
            (TRACT_LP,   0.4, _S),
            (TRACT_LD,   1.2, _S),
        )
    elif arpa == "W":
        overrides = (
            (TRACT_LP,   0.8, 0.85),
            (TRACT_LD,  -0.5, 0.85),
            (TRACT_TCX, -1.0, _S),
            (TRACT_TCY,  0.5, _S),
        )
    elif arpa == "Y":
        overrides = (
            (TRACT_TCY,  0.6, 0.85),
            (TRACT_TCX,  1.5, _S),
            (TRACT_LD,   1.0, _S),
        )
    else:
        raise ValueError(f"unknown approximant: {arpa}")
    return PhoneTemplate(arpabet=arpa, overrides=overrides, voiced=True)


# Vowel quadrant by IPA (front/back, low/high). Tongue body coords
# are RELATIVE offsets onto the natasha-baseline TCX/TCY values; the
# absolute targets stay inside R1 NN's tract_min/tract_max range.
PHONE_TEMPLATES: dict[str, PhoneTemplate] = {
    # Vowels — args: arpa, JA, LD, TCX, TCY, [LP]
    # JA (jaw): more negative = more open. LD (lip aperture): larger = open.
    # TCX: front +, back -. TCY: high +, low -.
    "IY": _vowel("IY", -1.0, 0.6,  2.0,  0.5),       # high front
    "IH": _vowel("IH", -1.5, 0.8,  1.5,  0.2),
    "EY": _vowel("EY", -1.5, 1.0,  1.5,  0.0),
    "EH": _vowel("EH", -2.0, 1.2,  1.0, -0.5),
    "AE": _vowel("AE", -3.0, 1.3,  0.5, -1.5),
    "AA": _vowel("AA", -3.5, 1.4, -1.5, -2.0),       # low back
    "AH": _vowel("AH", -2.0, 1.0, -0.5, -0.5),
    "AO": _vowel("AO", -2.5, 1.0, -1.0, -0.5, lp=0.4),
    "UH": _vowel("UH", -1.5, 0.5, -1.5,  0.3, lp=0.5),
    "UW": _vowel("UW", -1.0, 0.3, -2.0,  0.7, lp=0.7),
    "OW": _vowel("OW", -2.0, 0.7, -1.5,  0.0, lp=0.6),
    "AW": _vowel("AW", -3.0, 1.0, -1.0, -0.5, lp=0.4),
    "AY": _vowel("AY", -3.0, 1.2,  0.5, -1.0),
    "OY": _vowel("OY", -2.0, 0.8, -1.0,  0.0, lp=0.5),
    "ER": _vowel("ER", -2.0, 1.0, -0.5, -0.3, lp=0.3),

    # Stops
    "P": _stop("P",  voiced=False, place="bilabial"),
    "B": _stop("B",  voiced=True,  place="bilabial"),
    "T": _stop("T",  voiced=False, place="alveolar"),
    "D": _stop("D",  voiced=True,  place="alveolar"),
    "K": _stop("K",  voiced=False, place="velar"),
    "G": _stop("G",  voiced=True,  place="velar"),
    "CH": _stop("CH", voiced=False, place="post_alveolar"),
    "JH": _stop("JH", voiced=True,  place="post_alveolar"),

    # Fricatives
    "F":  _fricative("F",  voiced=False, place="labio_dental"),
    "V":  _fricative("V",  voiced=True,  place="labio_dental"),
    "TH": _fricative("TH", voiced=False, place="dental"),
    "DH": _fricative("DH", voiced=True,  place="dental"),
    "S":  _fricative("S",  voiced=False, place="alveolar"),
    "Z":  _fricative("Z",  voiced=True,  place="alveolar"),
    "SH": _fricative("SH", voiced=False, place="post_alveolar"),
    "ZH": _fricative("ZH", voiced=True,  place="post_alveolar"),
    "HH": _fricative("HH", voiced=False, place="glottal"),

    # Nasals
    "M":  _nasal("M",  "bilabial"),
    "N":  _nasal("N",  "alveolar"),
    "NG": _nasal("NG", "velar"),

    # Approximants & lateral
    "L":  _approximant("L"),
    "R":  _approximant("R"),
    "W":  _approximant("W"),
    "Y":  _approximant("Y"),
}


# Pre-flatten the inventory for callers that just want the column->target
# pairs by phone.
def template_for(arpabet_token: str) -> PhoneTemplate:
    """Return the smoke template for an ARPAbet token (stress digit ok)."""
    if arpabet_token and arpabet_token[-1].isdigit():
        bare = arpabet_token[:-1]
    else:
        bare = arpabet_token
    if bare not in PHONE_TEMPLATES:
        raise KeyError(
            f"no smoke template for ARPAbet phone {bare!r} (token "
            f"{arpabet_token!r}); supported: {sorted(PHONE_TEMPLATES)}"
        )
    return PHONE_TEMPLATES[bare]


# Re-export the neutral frame as a callable-result module attribute.
# Lazy because numpy/load is module-import-side-effect avoidant.
class _NeutralProxy:
    def __call__(self) -> np.ndarray:
        return neutral_frame()

    def __array__(self) -> np.ndarray:
        return neutral_frame()


NEUTRAL_TRAJECTORY = _NeutralProxy()
