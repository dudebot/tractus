"""Hand-crafted phoneme ground-truth test for TEA + glottis + frication.

Bypasses everything upstream of TEA (planner, controller, smoke pipeline,
R1 NN body model). Hand-builds known-correct tube geometry + glottis
params + frication inputs for a small canonical phoneme set, renders each
through ``synthesize_dynamic_tube`` directly, writes WAVs.

The point is to find out what TEA can produce given INPUTS WE KNOW ARE
RIGHT. If hand-crafted /a/ /i/ /u/ sound like vowels and /s/ sounds like a
sibilant, the audio simulator + source models work — and the body model
(R1 NN at 26K rows; ARCH §1049) is the bottleneck for the smoke pipeline.
If they don't, TEA itself or the source/frication modules are broken.

Run::

    cd tractus_controller
    .venv/bin/python ../scripts/phoneme_test.py
"""
from __future__ import annotations

import os
os.environ.setdefault("TEA_PRECISION", "fp32")

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tractus_controller"))
sys.path.insert(0, str(ROOT / "tractus_engine_advanced" / "src"))

import jax
import jax.numpy as jnp
import soundfile as sf

from tractus_engine_advanced.pipeline import synthesize_dynamic_tube
from tractus_engine_advanced.frication import (
    Frication,
    PLACE_NONE,
    PLACE_LABIO_DENTAL,
    PLACE_DENTAL,
    PLACE_ALVEOLAR,
    PLACE_POST_ALVEOLAR,
    PLACE_VELAR,
    PLACE_GLOTTAL,
    PLACE_BURST,
)
from tractus_engine_advanced.tds.tube_topology import (
    FIRST_PHARYNX_SECTION,
    LAST_PHARYNX_SECTION,
    FIRST_MOUTH_SECTION,
    LAST_MOUTH_SECTION,
    FIRST_NOSE_SECTION,
    LAST_NOSE_SECTION,
)


SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 441   # 10 ms / frame
DURATION_S = 0.5
OUT_DIR = ROOT / "outputs/phoneme_test"


# --------------------------------------------------------------------------
# Default speaker baseline.
# --------------------------------------------------------------------------

def load_default_speaker() -> dict:
    npz = np.load(ROOT / "tractus_controller/outputs/fixtures/default_tube.npz")
    return {
        "length_cm": np.asarray(npz["length_cm"], dtype=np.float32),
        "area_cm2": np.asarray(npz["area_cm2"], dtype=np.float32),
        "wall_mass_cgs": np.asarray(npz["wall_mass_cgs"], dtype=np.float32),
        "wall_resistance_cgs": np.asarray(npz["wall_resistance_cgs"], dtype=np.float32),
        "wall_stiffness_cgs": np.asarray(npz["wall_stiffness_cgs"], dtype=np.float32),
    }


def baseline(default: dict) -> dict:
    return {k: np.asarray(v).copy() for k, v in default.items()}


# --------------------------------------------------------------------------
# Glottis control params (length 11): F0, PR, XB, XT, CA, PL, RA, DP, PS, FL, AS
# Cribbed from outputs/fixtures/natasha_shape.npz mean values.
# --------------------------------------------------------------------------

def voiced(F0_Hz: float = 120.0, PR_dPa: float = 8000.0) -> np.ndarray:
    """Voiced glottis: full fold vibration."""
    return np.array(
        [F0_Hz, PR_dPa, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
        dtype=np.float32,
    )


def voiceless(PR_dPa: float = 6000.0, aspiration_dB: float = -5.0) -> np.ndarray:
    """Voiceless glottis: abducted, no phonation amplitude, pressure still drives turbulence."""
    return np.array(
        [80.0, PR_dPa, 0.15, 0.15, 0.20, 1.22, 0.0, 0.05, 0.0, 0.0, aspiration_dB],
        dtype=np.float32,
    )


# --------------------------------------------------------------------------
# Per-phoneme tube shapes (modify pharynx + mouth from default).
# Pharynx idx 25-40 (16 sections), Mouth idx 41-64 (24 sections).
# --------------------------------------------------------------------------

def shape_a(default: dict) -> dict:
    """/a/ open vowel — narrow pharynx, wide mouth."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 1.0
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION + 1] = 4.0
    return t


def shape_i(default: dict) -> dict:
    """/i/ front high vowel — wide pharynx, palatal constriction in mid-mouth."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 4.0
    mid = (FIRST_MOUTH_SECTION + LAST_MOUTH_SECTION) // 2
    t["area_cm2"][FIRST_MOUTH_SECTION:mid] = 2.5
    t["area_cm2"][mid:mid + 8] = 0.3
    t["area_cm2"][mid + 8:LAST_MOUTH_SECTION + 1] = 2.0
    return t


def shape_u(default: dict) -> dict:
    """/u/ back high vowel — wide back, narrow lips."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 3.5
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION - 5] = 3.0
    t["area_cm2"][LAST_MOUTH_SECTION - 5:LAST_MOUTH_SECTION + 1] = 0.3
    return t


def shape_schwa(default: dict) -> dict:
    """/ə/ schwa — neutral-ish, mostly default."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 2.5
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION + 1] = 2.5
    return t


def shape_s(default: dict) -> dict:
    """/s/ alveolar fricative — narrow constriction at very front."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 3.0
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION - 3] = 3.0
    t["area_cm2"][LAST_MOUTH_SECTION - 3:LAST_MOUTH_SECTION] = 0.05
    t["area_cm2"][LAST_MOUTH_SECTION] = 0.5
    return t


def shape_sh(default: dict) -> dict:
    """/ʃ/ post-alveolar fricative — wider, slightly more retracted constriction."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 3.0
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION - 7] = 3.0
    t["area_cm2"][LAST_MOUTH_SECTION - 7:LAST_MOUTH_SECTION - 3] = 0.10
    t["area_cm2"][LAST_MOUTH_SECTION - 3:LAST_MOUTH_SECTION + 1] = 0.5
    return t


def shape_f(default: dict) -> dict:
    """/f/ labio-dental — narrow at lips."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 3.0
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION] = 3.0
    t["area_cm2"][LAST_MOUTH_SECTION - 1:LAST_MOUTH_SECTION + 1] = 0.10
    return t


def shape_m(default: dict) -> dict:
    """/m/ bilabial nasal — closed lips, open velum, voiced."""
    t = baseline(default)
    t["area_cm2"][FIRST_PHARYNX_SECTION:LAST_PHARYNX_SECTION + 1] = 3.0
    t["area_cm2"][FIRST_MOUTH_SECTION:LAST_MOUTH_SECTION] = 2.0
    t["area_cm2"][LAST_MOUTH_SECTION - 1:LAST_MOUTH_SECTION + 1] = 0.0
    t["area_cm2"][FIRST_NOSE_SECTION:LAST_NOSE_SECTION + 1] = 1.5
    return t


# --------------------------------------------------------------------------
# Render helper.
# --------------------------------------------------------------------------

def render(name: str, tube: dict, glottis: np.ndarray,
           place_id: int = PLACE_NONE, fric_enabled: bool = False,
           fric_section_lo: int | None = None, fric_section_hi: int | None = None,
           duration_s: float = DURATION_S):
    n_samples = int(duration_s * SAMPLE_RATE)
    F = max(1, n_samples // SAMPLES_PER_FRAME)

    L = np.tile(tube["length_cm"][None, :], (F, 1))
    A = np.tile(tube["area_cm2"][None, :], (F, 1))
    Mw = np.tile(tube["wall_mass_cgs"][None, :], (F, 1))
    Bw = np.tile(tube["wall_resistance_cgs"][None, :], (F, 1))
    Kw = np.tile(tube["wall_stiffness_cgs"][None, :], (F, 1))
    g_traj = np.tile(glottis[None, :], (F, 1))

    if fric_enabled:
        frication = Frication.default()
        # place_index is (F, 93). Tag only the constriction sections with the
        # given place_id; rest stay PLACE_NONE so frication is silent there.
        pi = np.full((F, 93), PLACE_NONE, dtype=np.int32)
        if fric_section_lo is None:
            fric_section_lo = LAST_MOUTH_SECTION - 3
        if fric_section_hi is None:
            fric_section_hi = LAST_MOUTH_SECTION + 1
        pi[:, fric_section_lo:fric_section_hi] = place_id
        place_idx = jnp.asarray(pi)
        voicing = jnp.full((F,), 1.0 if glottis[6] > 0.0 else 0.0, dtype=jnp.float32)
        obstacle = jnp.zeros((F, 93), dtype=jnp.float32)
        color = jnp.zeros((F, 93), dtype=jnp.float32)
        noise_key = jax.random.PRNGKey(42)
    else:
        frication = None
        place_idx = voicing = obstacle = color = noise_key = None

    audio, _, _, _ = synthesize_dynamic_tube(
        jnp.asarray(L), jnp.asarray(A),
        jnp.asarray(Mw), jnp.asarray(Bw), jnp.asarray(Kw),
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=jnp.asarray(g_traj),
        frication=frication,
        place_index=place_idx,
        voicing=voicing,
        obstacle_proxy=obstacle,
        frication_color=color,
        noise_key=noise_key,
    )
    audio = np.asarray(audio, dtype=np.float32)
    peak_pre = float(np.max(np.abs(audio)))
    rms_pre = float(np.sqrt(np.mean(audio ** 2)))

    # Peak-normalize to 0.5 (so user's playback isn't clipping or whisper-quiet).
    if peak_pre > 0:
        audio = audio * (0.5 / peak_pre)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.wav"
    sf.write(str(out), audio, SAMPLE_RATE)
    print(f"  {name}: F={F} N={audio.size} peak={peak_pre:.4f} rms={rms_pre:.4f} → {out.name}")
    return out


def main():
    default = load_default_speaker()
    print(f"Loaded default speaker tube: total length={default['length_cm'].sum():.2f} cm, "
          f"mean area={default['area_cm2'].mean():.3f} cm²\n")

    print("Rendering canonical phonemes (0.5 s each, 44.1 kHz):")
    print()
    print("=== VOWELS (voiced glottis, no frication) ===")
    render("01_a_vowel",     shape_a(default),     voiced())
    render("02_i_vowel",     shape_i(default),     voiced())
    render("03_u_vowel",     shape_u(default),     voiced())
    render("04_schwa",       shape_schwa(default), voiced())
    render("05_a_low_F0",    shape_a(default),     voiced(F0_Hz=85.0))
    render("06_a_high_F0",   shape_a(default),     voiced(F0_Hz=200.0))

    print()
    print("=== NASAL (voiced glottis, velum open) ===")
    render("07_m_nasal",     shape_m(default),     voiced())

    print()
    print("=== FRICATIVES (voiceless glottis, frication module on) ===")
    render("08_s_no_fric",   shape_s(default),   voiceless(),
           place_id=PLACE_NONE, fric_enabled=False)
    render("09_s_with_fric", shape_s(default),   voiceless(),
           place_id=PLACE_ALVEOLAR, fric_enabled=True)
    render("10_sh_with_fric", shape_sh(default), voiceless(),
           place_id=PLACE_POST_ALVEOLAR, fric_enabled=True)
    render("11_f_with_fric",  shape_f(default),  voiceless(),
           place_id=PLACE_LABIO_DENTAL, fric_enabled=True)

    print()
    print("=== EDGE CASES (sanity checks) ===")
    # Default tube + voiced — what does the unmodified speaker baseline sound like?
    render("12_default_tube_voiced", baseline(default), voiced())
    # /h/-aspiration through /a/ shape
    render("13_h_aspiration_a",      shape_a(default), voiceless(aspiration_dB=0.0),
           place_id=PLACE_GLOTTAL,   fric_enabled=True)
    # /a/ but voiceless drive — should sound like whispered /a/
    render("14_whispered_a",         shape_a(default), voiceless(),
           place_id=PLACE_NONE,      fric_enabled=False)

    print()
    print(f"All WAVs in {OUT_DIR}/")


if __name__ == "__main__":
    main()
