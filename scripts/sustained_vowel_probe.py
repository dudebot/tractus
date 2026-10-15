"""Sustained-vowel hand-tune probe — does TEA produce realistic resonance?

Hand-tuned sanity check for the developer's "does it sound like a trumpet
or a vowel, and are the sinuses contributing?" question. Renders ~11 short
WAV files the developer can listen to. No optimization, no learning — just
Story-anchor tubes + voiced source + TEA.

Outputs (under ``outputs/hand_tune_vowels/``):

  vowels/
    a.wav, e.wav, i.wav, o.wav, u.wav         — Story-anchor vowels
  nasals/
    m.wav, n.wav                              — hand-crafted nasals
  nasal_ab/
    i_velum_closed.wav, i_velum_open.wav      — does opening velum matter?
  sinus_ablation/
    m_intact.wav, m_no_sinus.wav              — do sinus side-branches matter?

  spectrograms/                               — one PNG per WAV
  README.md                                   — what to listen for

The probe deliberately uses the simplest possible source schedule
(constant F0=120, PR=8000 dPa, voiced for the full 1.5 s) so any timbral
difference between renders attributes to the **tract**, not the source.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make engine + controller imports work.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tractus_engine_advanced" / "src"))
sys.path.insert(0, str(_ROOT / "tractus_controller"))

import jax.numpy as jnp  # noqa: E402
import soundfile as sf  # noqa: E402

from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: E402
from tractus_engine_advanced.tds.tube_topology import (  # noqa: E402
    FIRST_PHARYNX_SECTION,   # 25
    LAST_MOUTH_SECTION,      # 64
    FIRST_NOSE_SECTION,      # 65
    LAST_NOSE_SECTION,       # 83
    FIRST_SINUS_SECTION,     # 89
    LAST_SINUS_SECTION,      # 92
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SR = 22050
SAMPLES_PER_FRAME = 441         # 20 ms / frame at 22.05 kHz
DURATION_S = 1.5
N_FRAMES = int(DURATION_S * SR / SAMPLES_PER_FRAME)

DEFAULT_TUBE_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "default_tube.npz"
STORY_ANCHORS_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "story_1996_anchors.npz"

OUT = _ROOT / "outputs" / "hand_tune_vowels"


# --------------------------------------------------------------------------
# Source: constant voiced glottis state — F0=120 Hz, PR=8000 dPa
# --------------------------------------------------------------------------

def voiced_glottis(n_frames: int) -> np.ndarray:
    """11-channel sustained voiced source, matches inverse_fit_demo baseline."""
    # Channel order: F0, PR, XB, XT, CA, PL, RA, DP, PS, FL, AS
    g = np.array(
        [120.0, 8000.0, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
        dtype=np.float32,
    )
    return np.tile(g[None, :], (n_frames, 1))


# --------------------------------------------------------------------------
# Tube helpers
# --------------------------------------------------------------------------

def load_default() -> dict:
    d = np.load(DEFAULT_TUBE_PATH, allow_pickle=True)
    return {k: np.asarray(d[k], dtype=np.float32) for k in d.files
            if d[k].dtype.kind in "fi"}


def load_story_anchors() -> dict[str, np.ndarray]:
    """Return {shape_key: area_cm2_tea[40]} for ingested shapes only."""
    s = np.load(STORY_ANCHORS_PATH, allow_pickle=True)
    keys = [str(k) for k in s["shape_keys"]]
    status = [str(x) for x in s["shape_status"]]
    areas = np.asarray(s["area_cm2_tea"], dtype=np.float32)
    return {k: areas[i] for i, (k, st) in enumerate(zip(keys, status)) if st == "ingested"}


def splice_story_into_default(default: dict, story_area_40: np.ndarray) -> np.ndarray:
    """Return area_cm2[93] with sections [25:65] replaced by Story area."""
    a = default["area_cm2"].copy()
    a[FIRST_PHARYNX_SECTION:LAST_MOUTH_SECTION + 1] = story_area_40
    return a


def render(
    area_cm2: np.ndarray,
    default: dict,
    n_frames: int = N_FRAMES,
    *,
    velum_section_area: float | None = None,
    sinus_sections_area: float | None = None,
    label: str = "",
) -> np.ndarray:
    """Render a single sustained tube to WAV. Optional knobs:

    - ``velum_section_area``: if given, overrides ``area[65]`` (first nose
      section) on every frame. Small (e.g. 1e-3) = closed velum; larger
      (e.g. 0.3) = open velum.
    - ``sinus_sections_area``: if given, overrides ``area[89:93]`` (sinus
      side-branches) on every frame. Tiny value effectively ablates sinus
      side-branches; default-tube value is the "intact" baseline.
    """
    a93 = area_cm2.copy()
    if velum_section_area is not None:
        a93[FIRST_NOSE_SECTION] = velum_section_area
    if sinus_sections_area is not None:
        a93[FIRST_SINUS_SECTION:LAST_SINUS_SECTION + 1] = sinus_sections_area

    length_93 = default["length_cm"]
    Mw_93 = default["wall_mass_cgs"]
    Bw_93 = default["wall_resistance_cgs"]
    Kw_93 = default["wall_stiffness_cgs"]

    L = jnp.tile(jnp.asarray(length_93)[None, :], (n_frames, 1))
    A = jnp.tile(jnp.asarray(a93)[None, :], (n_frames, 1))
    Mw = jnp.tile(jnp.asarray(Mw_93)[None, :], (n_frames, 1))
    Bw = jnp.tile(jnp.asarray(Bw_93)[None, :], (n_frames, 1))
    Kw = jnp.tile(jnp.asarray(Kw_93)[None, :], (n_frames, 1))
    g = jnp.asarray(voiced_glottis(n_frames))

    print(f"  rendering {label} (T={n_frames} frames, ~{n_frames * SAMPLES_PER_FRAME / SR:.2f} s)...")
    audio, *_ = synthesize_dynamic_tube(
        L, A, Mw, Bw, Kw,
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=g,
        frame_chunk_size=10,
    )
    return np.asarray(audio)


def rms_normalize(x: np.ndarray, target_rms: float = 0.1) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return (x * (target_rms / rms)).astype(np.float32)


def save_wav(audio: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), rms_normalize(audio), SR, subtype="PCM_16")
    print(f"    wrote {path.relative_to(_ROOT)}")


def save_spectrogram(audio: np.ndarray, path: Path, label: str) -> None:
    """Optional spectrogram PNG for visual inspection. Lazy import so the
    script still runs if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    Pxx, freqs, _, _ = ax.specgram(audio, NFFT=1024, Fs=SR, noverlap=512, scale="dB")
    ax.set_ylim(0, 6000)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Hz")
    ax.set_title(label)
    plt.tight_layout()
    plt.savefig(path, dpi=80)
    plt.close()


# --------------------------------------------------------------------------
# Probe sequences
# --------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"=== sustained-vowel probe → {OUT.relative_to(_ROOT)} ===")
    print(f"  SR={SR} Hz, duration={DURATION_S}s, voiced F0=120 Hz, PR=8000 dPa")

    default = load_default()
    story = load_story_anchors()

    print(f"  default tube: {default['area_cm2'].shape[0]} sections")
    print(f"  story shapes ingested: {sorted(story.keys())}")

    # 1) Render every ingested Story vowel (velum closed, oral-only).
    print("\n[1] Story-anchor vowels (velum closed)")
    for key in sorted(story.keys()):
        area = splice_story_into_default(default, story[key])
        audio = render(area, default,
                       velum_section_area=1e-3, label=f"vowel '{key}'")
        # Filename-safe: keep the key as-is (already safe — letters only).
        save_wav(audio, OUT / "vowels" / f"{key}.wav")
        save_spectrogram(audio, OUT / "spectrograms" / f"vowel_{key}.png", f"vowel '{key}'")

    # 2) Hand-crafted nasals: oral closure + open velum.
    print("\n[2] Hand-crafted nasals (velum open, oral cavity closed)")
    nasals = {
        "m": (LAST_MOUTH_SECTION, 1e-3),       # bilabial closure at last mouth section
        "n": (LAST_MOUTH_SECTION - 4, 1e-3),   # alveolar (rough; ~60)
    }
    # Use 'A' (open back vowel, Story key) as the base posture for nasals.
    base_vowel_key = "A" if "A" in story else sorted(story.keys())[0]
    for name, (closure_idx, closure_area) in nasals.items():
        base_area = splice_story_into_default(default, story[base_vowel_key])
        base_area[closure_idx:closure_idx + 2] = closure_area  # 2-section closure
        audio = render(base_area, default,
                       velum_section_area=0.3, label=f"nasal /{name}/")
        save_wav(audio, OUT / "nasals" / f"{name}.wav")
        save_spectrogram(audio, OUT / "spectrograms" / f"nasal_{name}.png", f"nasal /{name}/")

    # 3) Nasal A/B on 'i' (Story key): velum closed vs velum open.
    print("\n[3] Velum A/B on 'i' — does opening velum change the timbre?")
    i_key = "i" if "i" in story else ("I" if "I" in story else None)
    if i_key is not None:
        i_area = splice_story_into_default(default, story[i_key])
        for cond, velum in [("velum_closed", 1e-3), ("velum_open", 0.3)]:
            audio = render(i_area, default,
                           velum_section_area=velum, label=f"'{i_key}' {cond}")
            save_wav(audio, OUT / "nasal_ab" / f"i_{cond}.wav")
            save_spectrogram(audio, OUT / "spectrograms" / f"i_{cond}.png", f"'{i_key}' {cond}")

    # 4) Sinus ablation A/B on /m/.
    print("\n[4] Sinus ablation on /m/ — do the sinus side-branches matter?")
    base_area = splice_story_into_default(default, story[base_vowel_key])
    base_area[LAST_MOUTH_SECTION:LAST_MOUTH_SECTION + 2] = 1e-3  # lip closure
    sinus_default = float(default["area_cm2"][FIRST_SINUS_SECTION])
    for cond, sinus_a in [("intact", sinus_default), ("no_sinus", 1e-5)]:
        audio = render(base_area, default,
                       velum_section_area=0.3,
                       sinus_sections_area=sinus_a,
                       label=f"/m/ sinus_{cond}")
        save_wav(audio, OUT / "sinus_ablation" / f"m_{cond}.wav")
        save_spectrogram(audio, OUT / "spectrograms" / f"m_sinus_{cond}.png", f"/m/ sinus_{cond}")

    # 5) README.
    readme = OUT / "README.md"
    readme.write_text(_README_TEXT, encoding="utf-8")
    print(f"\nwrote {readme.relative_to(_ROOT)}")
    print("\ndone.")


_README_TEXT = """# Sustained-vowel hand-tune probe — what to listen for

All clips are 1.5 s sustained sounds. Source schedule is identical
across every clip: voiced glottis, F0 = 120 Hz, PR = 8000 dPa. So any
timbral difference between clips attributes to the **tract**, not the
source. All clips RMS-normalized to the same loudness, so apparent
volume isn't a tell.

## 1. `vowels/` — Story-anchor vowels

Five vowels rendered from Story 1996 published area functions, velum
closed (oral-only resonance).

**Listen for:** Does each clip sound like a *recognizable* vowel? Do
you hear the characteristic resonances (you should hear at least two
formants stacked, not a single dominant tone)? If they all sound like
the same buzz with different pitch shadings, the tract isn't shaping
the source — that's the "trumpet" failure mode.

Expected character:
- `/a/` — open / mid-low formants, bright
- `/i/` — high F2 ("eee" quality), low F1
- `/u/` — low + clustered, dark and round
- `/e/` — intermediate front
- `/o/` — intermediate back, rounded

## 2. `nasals/` — hand-crafted nasals (velum open)

`/m/` is bilabial (closed at the lips); `/n/` is alveolar (closed
~mid-mouth). Both with velum open (0.3 cm²) so the nose participates.

**Listen for:** Is there an audible nasal hum? Nasals are characterized
by a strong low frequency component, attenuated higher formants, and
anti-formants (notches in the spectrum). They should NOT sound like
vowels with closed mouth.

## 3. `nasal_ab/` — velum A/B on `/i/`

Same tract shape, different velum opening.

**Listen for:** `i_velum_closed.wav` should sound like a clear `/i/`.
`i_velum_open.wav` should sound nasalized (think speaking with a
cold) — same vowel quality, but with a subtle nasal coloration.
If they sound *identical*, the velar coupling isn't working.

## 4. `sinus_ablation/` — sinus side-branches on `/m/`

Same nasal `/m/` posture, sinus sections (89-92) either at default
area (intact) or clamped to ~zero (ablated).

**Listen for:** A difference between the two. Sinuses contribute
anti-formants (spectral notches) and weak resonances in the 200-500 Hz
range. The intact version should have slightly different timbre
than the ablated. If they sound identical, sinus side-branches
aren't contributing meaningfully to the output.

## 5. `spectrograms/` — visual cross-check

PNG per WAV, log magnitude, 0-6 kHz. Use this to see formant
structure if you can't tell by ear.

**For vowels:** you should see 3-4 horizontal bands (formants).
**For nasals:** wider low-frequency energy + notches.
**For ablations:** look for spectrum-shape differences between the
two `/m/` conditions.

## Verdict signals

- **All vowels sound like the same buzz → TEA isn't shaping source.**
  This would block the rest of the project; either the area function
  isn't getting through or the wall losses are too aggressive.
- **Vowels distinguishable, velum A/B sounds identical → nasal**
  coupling broken. Architecture says nose is a side-branch from
  section 40 → 65; the wire is there but if there's no audible effect,
  the coupling is silenced somewhere.
- **Sinus ablation has no audible effect → expected.** Sinus effects
  are subtle by design. Check the spectrograms for notches around
  300-500 Hz in the intact `/m/` vs ablated.
- **Everything sounds plausible → green light to dispatch Task 001**
  (the worker's `scripts/copyfit_neural_field_phrase.py` is already
  on branch `worktree-agent-af90ebf604d72cf51` and gates as YELLOW —
  ready to commit a 3-4 hour Gate-1 run when you're ready).
"""


if __name__ == "__main__":
    main()
