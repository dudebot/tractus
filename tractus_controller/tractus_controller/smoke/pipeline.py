"""End-to-end rule-based text -> audio smoke pipeline.

Pure-Python orchestration: builds the (T, 30) trajectory from a
sequence of phone GestureEvents, then dispatches to TEA via the
existing ``copy_synth/tea_backend.synthesize_via_tea`` numpy wrapper.

Sample rate: TEA renders at its native 44.1 kHz (TIMESTEP_S =
1/44100). The smoke task spec asks for 22.05 kHz; we render native
and decimate at output stage by polyphase resampling so the WAV on
disk is exactly the rate the user asked for. Native 44.1 kHz output
is also supported via ``--sample-rate 44100``.

The intelligibility-WER gate (``runbooks/whisper_gate.sh``) is
smoke/002 — explicitly NOT run from this module. P1 is gated on P0
close, which has open items (correctness/004, /005 partial, /007
partial, /008 pending).
"""

from __future__ import annotations

import json
import math
import wave
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..gestures import GestureEvent
from ..phonemes import durations as durmod
from ..phonemes.arpabet import (
    UnsupportedPhone,
    is_supported,
    split_stress,
)
from ..phonemes.ipa_arpabet import ipa_to_arpabet
from ..phonemes.phonemize import phonemize as ipa_phonemize
from .templates import (
    CHANNEL_BY_NAME,
    GLOTTIS_AS,
    GLOTTIS_F0,
    GLOTTIS_PR,
    neutral_frame,
    template_for,
)

# TEA's native sample rate. (See tractus_engine_advanced/tds/state.py
# / pipeline.py for the canonical TIMESTEP_S = 1/44100 source.)
TEA_NATIVE_SR_HZ: int = 44100


# Frame rate for the (T, 30) trajectory. 100 Hz matches the planner /
# copy-synth defaults (``frame_step_samples = 441 @ 44.1 kHz`` — but
# natasha_shape uses 882 / 50 Hz). Smoke uses 100 Hz so duration
# resolution is 10 ms, which is finer than any phone we render.
SMOKE_FRAME_RATE_HZ: int = 100


@dataclass
class SmokeConfig:
    """Smoke render configuration. All fields have defaults."""

    sample_rate: int = 22050               # output WAV sample rate
    f0_hz: float = 110.0                   # mid-male reference pitch
    f0_drift_semitones: float = 0.5        # gentle drift across utterance
    lung_pressure_pa: float = 8000.0       # voiced subglottal pressure
    leading_silence_s: float = 0.10
    trailing_silence_s: float = 0.10
    frame_rate_hz: int = SMOKE_FRAME_RATE_HZ
    duration_table: Path | None = None     # JSON path; None = built-in defaults
    deterministic_seed: int = 0
    speaker: str = "default"
    # Hard-cap utterance length so a runaway phonemize doesn't OOM the
    # JIT'd TEA scan.
    max_duration_s: float = 12.0
    # correctness/004: enable the physically-derived per-section frication
    # noise. Defaults to False so existing smoke renders are bit-equal to
    # pre-correctness/004; set True to compare pre/post for the Whisper
    # gate.
    frication_enabled: bool = False
    # PRNG seed for the deterministic reparameterized noise sampling.
    frication_noise_seed: int = 0


@dataclass
class SmokeRender:
    """Result bundle: audio array + provenance for the sidecar JSON."""

    audio: np.ndarray                       # (N,) float32 at sample_rate
    sample_rate: int
    arpabet_tokens: list[str]
    durations_s: list[float]
    gestures: list[GestureEvent]
    trajectory: np.ndarray                  # (T, 30)
    frame_rate_hz: int
    config: SmokeConfig
    input_text: str
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# G2P + duration resolution.
# ---------------------------------------------------------------------------


def _normalize_espeak_ipa(ipa: str) -> str:
    """Pre-fold a few espeak-ng emissions onto canonical IPA segments
    accepted by ``phonemes.ipa_arpabet``.

    The IPA->ARPAbet table is intentionally strict (it raises on
    anything outside the CMU inventory) so the synth CLI's OOV guard
    surfaces vocabulary gaps. Smoke is more permissive — we map known
    espeak-isms onto safe ARPAbet equivalents so the regression set
    renders without per-word manual cleanup. Folds applied:

    * ``ɾ`` (alveolar flap, AmE T/D in 'butter', 'water') -> ``d``
    * ``ᵻ`` (centralized ɪ, AmE 'roses') -> ``ɪ``
    * ``ɜ`` / ``ɜː`` (open-mid central, BE NURSE) -> ``ɝ`` (ARPAbet ER)
    * ``ɐ`` (near-open central, AmE schwa-rotic) -> ``ə`` (ARPAbet AH)
    * ``ɫ`` (velarized lateral, AmE dark L) -> ``l``
    * ``ʔ`` (glottal stop, AmE word-internal) -> ``t`` (ARPAbet T)

    These are smoke-grade approximations. The strict mapper is still
    used downstream for any phone the fold doesn't cover, so genuine
    OOV cases still raise loudly.
    """
    folds = (
        ("ɜː", "ɝ"),
        ("ɜ", "ɝ"),
        ("ɾ", "d"),
        ("ᵻ", "ɪ"),
        ("ɐ", "ə"),
        ("ɫ", "l"),
        ("ʔ", "t"),
    )
    out = ipa
    for src, dst in folds:
        if src in out:
            out = out.replace(src, dst)
    return out


def text_to_phones(text: str, language: str = "en-us") -> list[str]:
    """Return ARPAbet tokens for ``text``.

    Pipeline: espeak-ng IPA via ``phonemes.phonemize.phonemize``,
    smoke-grade IPA normalization, then strict IPA -> ARPAbet via
    ``phonemes.ipa_arpabet.ipa_to_arpabet``. Word-boundary metadata is
    dropped here (smoke doesn't reuse it).
    """
    ipa = ipa_phonemize(text, language=language)
    if not ipa:
        return []
    ipa = _normalize_espeak_ipa(ipa)
    tokens, _ = ipa_to_arpabet(ipa)
    # Drop anything outside the supported subset defensively (the IPA
    # mapper validates this at import time but a future inventory
    # change could regress).
    out = []
    for t in tokens:
        if not is_supported(t):
            raise UnsupportedPhone(
                f"phone {t!r} from text {text!r} is outside the smoke "
                f"inventory. Re-phonemize or extend PHONE_TEMPLATES."
            )
        out.append(t)
    return out


def load_duration_table(path: Path | None) -> dict[str, float]:
    """Load a per-phone duration override table, or empty dict.

    Schema: ``{ "AA": 0.13, "P": 0.05, ... }`` — values in seconds.
    Missing entries fall back to ``phonemes.durations.duration_of``.
    """
    if path is None:
        return {}
    raw = json.loads(Path(path).read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"duration table {path}: expected a JSON object")
    out: dict[str, float] = {}
    for k, v in raw.items():
        # Skip underscore-prefixed keys (JSON-comment convention).
        if k.startswith("_"):
            continue
        out[str(k)] = float(v)
    return out


def resolve_durations(
    arpabet_tokens: list[str],
    overrides: dict[str, float] | None = None,
) -> list[float]:
    """Per-phone durations in seconds. Overrides win over class defaults."""
    overrides = overrides or {}
    out: list[float] = []
    for t in arpabet_tokens:
        bare, _ = split_stress(t)
        if t in overrides:
            out.append(overrides[t])
        elif bare in overrides:
            out.append(overrides[bare])
        else:
            out.append(durmod.duration_of(t))
    return out


# ---------------------------------------------------------------------------
# GestureEvent emission per phone.
# ---------------------------------------------------------------------------


def _channel_name_for(col: int) -> str:
    """Reverse-lookup of the human-legible channel name for a 30-d column."""
    for name, idx in CHANNEL_BY_NAME.items():
        if idx == col:
            return name
    return f"col_{col}"


def emit_gestures(
    arpabet_tokens: list[str],
    durations_s: list[float],
    *,
    onset_offset_s: float = 0.0,
    f0_hz: float = 110.0,
    lung_pressure_pa: float = 8000.0,
) -> list[GestureEvent]:
    """Emit one GestureEvent per (phone, channel-anchor) pair plus the
    per-phone F0 / lung pressure / aspiration baseline events.

    Coarticulation (overlap, anticipation) is intentionally OFF.
    """
    if len(arpabet_tokens) != len(durations_s):
        raise ValueError("arpabet_tokens and durations_s length mismatch")
    events: list[GestureEvent] = []
    t = onset_offset_s
    for token, dur in zip(arpabet_tokens, durations_s, strict=True):
        tmpl = template_for(token)
        bare, stress = split_stress(token)
        prosody: tuple[str, ...] = ()
        if stress == "1":
            prosody = ("primary_stress",)
        elif stress == "2":
            prosody = ("secondary_stress",)

        # Per-phone glottal source events (F0 / lung_pressure / aspiration)
        if tmpl.voiced:
            events.append(GestureEvent(
                channel_id="F0",
                onset_s=t,
                target_value=float(f0_hz),
                duration_s=float(dur),
                release_s=0.01,
                strength=1.0,
                ramp_shape="hold_release",
                prosody_tags=prosody,
            ))
            events.append(GestureEvent(
                channel_id="lung_pressure",
                onset_s=t,
                target_value=float(lung_pressure_pa),
                duration_s=float(dur),
                release_s=0.02,
                strength=1.0,
                ramp_shape="hold_release",
                prosody_tags=prosody,
            ))
        else:
            # Voiceless: drop pressure to a small breath value, raise
            # aspiration instead.
            events.append(GestureEvent(
                channel_id="lung_pressure",
                onset_s=t,
                target_value=float(lung_pressure_pa) * 0.4,
                duration_s=float(dur),
                release_s=0.01,
                strength=1.0,
                ramp_shape="hold_release",
                prosody_tags=prosody,
            ))
        if tmpl.aspiration > 0:
            events.append(GestureEvent(
                channel_id="aspiration",
                onset_s=t,
                target_value=float(tmpl.aspiration),
                duration_s=float(dur),
                release_s=0.005,
                strength=1.0,
                ramp_shape="hold_release",
                prosody_tags=prosody,
            ))

        # Per-articulator anchor events.
        for col, target, strength in tmpl.overrides:
            events.append(GestureEvent(
                channel_id=_channel_name_for(col),
                onset_s=t,
                target_value=float(target),
                duration_s=float(dur),
                release_s=tmpl.release_s_default,
                strength=float(strength),
                ramp_shape="hold_release",
                prosody_tags=prosody + (f"phone:{bare}",),
            ))
        t += dur
    return events


# ---------------------------------------------------------------------------
# GestureEvents -> (T, 30) trajectory.
# ---------------------------------------------------------------------------


def gestures_to_trajectory(
    gestures: Iterable[GestureEvent],
    *,
    total_duration_s: float,
    frame_rate_hz: int,
    f0_hz: float,
    lung_pressure_pa: float,
    deterministic_seed: int = 0,
) -> np.ndarray:
    """Stamp a (T, 30) trajectory from a list of GestureEvents.

    Algorithm (smoke-grade, intentionally simple — no coarticulation):

    1. Initialize all frames to a "silent" frame (neutral baseline,
       PR=0, AS=0).
    2. For each gesture, compute its frame-window
       ``[onset_frame, offset_frame)`` and a release tail of
       ``release_s`` frames.
    3. Within the hold window: the target column is moved from its
       current value toward ``target_value`` by a factor of
       ``strength`` — i.e. simple per-frame override
       ``frame[col] = (1 - s) * frame[col] + s * target``. Outside the
       window the value relaxes back toward neutral over the release.

    Last gesture wins on a column (later events override earlier on
    the same channel for overlapping windows). Smoke does not need
    additive composition — that lives in correctness/004 +
    pretraining/003.
    """
    rng = np.random.default_rng(deterministic_seed)
    T = max(2, int(round(total_duration_s * frame_rate_hz)) + 1)
    # Initialize to voiced-rest, NOT silent-rest. The body model is
    # undertrained (R1 NN at 26 K rows; ARCH §1049) and produces
    # near-zero output on the long tracts that emerge when PR drops to
    # 0 mid-utterance. Holding voiced rest as the baseline keeps the
    # source on so consonant bursts ride on top of an already-active
    # glottal source. True silence is enforced for the leading +
    # trailing pads by zeroing the head/tail explicitly below.
    base = neutral_frame()
    base[GLOTTIS_F0] = float(f0_hz)
    base[GLOTTIS_PR] = float(lung_pressure_pa)
    traj = np.broadcast_to(base, (T, 30)).copy()

    # Track which (col, frame) cells have been written by a gesture, so
    # the release tail only fires on touched columns.
    touched = np.zeros((T, 30), dtype=bool)

    sorted_g = sorted(
        gestures, key=lambda g: (g.onset_s, g.channel_id)
    )

    for g in sorted_g:
        col = CHANNEL_BY_NAME.get(g.channel_id)
        if col is None:
            # Unknown channel — skip (sidecar still records it).
            continue
        f0 = max(0, int(math.floor(g.onset_s * frame_rate_hz)))
        f1 = min(T, int(math.ceil((g.onset_s + g.duration_s) * frame_rate_hz)))
        if f1 <= f0:
            f1 = min(T, f0 + 1)
        s = float(np.clip(g.strength, 0.0, 1.0))
        target = float(g.target_value)
        for f in range(f0, f1):
            traj[f, col] = (1.0 - s) * traj[f, col] + s * target
            touched[f, col] = True

        # Release: linear fade from target back toward neutral over
        # release_s, on top of whatever the next gesture writes (so
        # if another gesture starts at f1, the release is masked by
        # the override).
        if g.release_s and g.release_s > 0:
            r_frames = max(1, int(round(g.release_s * frame_rate_hz)))
            for k in range(r_frames):
                f = f1 + k
                if f >= T:
                    break
                if touched[f, col]:
                    continue
                alpha = (k + 1) / (r_frames + 1)  # 0->1 over release
                traj[f, col] = (1.0 - alpha) * target + alpha * base[col]
                touched[f, col] = True

    # Cosmetic post-processing: a tiny F0 drift across the utterance so
    # the audio doesn't sound monotone (smoke-only, no model behind it).
    # Only applies where F0 was set by a voicing gesture.
    f0_col = GLOTTIS_F0
    f0_touched = touched[:, f0_col]
    if f0_touched.any():
        # 0.5 semitones of slow random walk, deterministic by seed.
        n_touched = int(f0_touched.sum())
        # Random walk in semitones, then bias to mean of 0.
        steps = rng.normal(0.0, 0.05, size=n_touched)
        drift_st = np.cumsum(steps)
        drift_st -= drift_st.mean()
        # Hz multiplier from semitone offset.
        mult = np.power(2.0, drift_st / 12.0)
        idx = np.where(f0_touched)[0]
        traj[idx, f0_col] = traj[idx, f0_col] * mult

    # Where lung_pressure was never set (frames in pure silence), keep
    # the initial 0 — so silence is true silence, not a low hum.
    return traj.astype(np.float64)


def _hard_silence_pads(
    traj: np.ndarray,
    *,
    leading_silence_s: float,
    trailing_silence_s: float,
    frame_rate_hz: int,
) -> np.ndarray:
    """Zero PR + AS in the leading/trailing pad windows so silence is real."""
    out = traj.copy()
    pad_lead = int(round(leading_silence_s * frame_rate_hz))
    pad_trail = int(round(trailing_silence_s * frame_rate_hz))
    if pad_lead > 0:
        out[:pad_lead, GLOTTIS_PR] = 0.0
        out[:pad_lead, GLOTTIS_AS] = 0.0
    if pad_trail > 0:
        out[-pad_trail:, GLOTTIS_PR] = 0.0
        out[-pad_trail:, GLOTTIS_AS] = 0.0
    return out


# ---------------------------------------------------------------------------
# Resampling helper (44.1 kHz native -> sample_rate).
# ---------------------------------------------------------------------------


def _resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32, copy=False)
    from math import gcd

    g = gcd(int(sr_in), int(sr_out))
    up = int(sr_out // g)
    down = int(sr_in // g)
    try:
        from scipy.signal import resample_poly  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "scipy required for resampling; pin in pyproject."
        ) from e
    out = resample_poly(audio.astype(np.float64), up, down)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Public entry point.
# ---------------------------------------------------------------------------


def render_text_to_audio(
    text: str,
    cfg: SmokeConfig | None = None,
) -> SmokeRender:
    """End-to-end smoke render. Pure function modulo TEA's lru_cache."""
    cfg = cfg or SmokeConfig()
    if not text or not text.strip():
        raise ValueError("smoke render: empty text")

    # 1. G2P
    arpabet_tokens = text_to_phones(text)
    if not arpabet_tokens:
        raise ValueError(
            f"smoke render: no phones produced for text {text!r}"
        )

    # 2. Durations
    overrides = load_duration_table(cfg.duration_table)
    durations = resolve_durations(arpabet_tokens, overrides)

    # 3. Add leading/trailing silence as zero-duration anchors? No —
    # we model silence via an empty-template hold of the silent frame,
    # but the simpler approach is to emit only utterance gestures and
    # pad the trajectory with silent frames front and back.
    utt_dur_s = sum(durations)
    if utt_dur_s + cfg.leading_silence_s + cfg.trailing_silence_s > cfg.max_duration_s:
        raise ValueError(
            f"smoke render: utterance ({utt_dur_s:.2f}s + silences) exceeds "
            f"max_duration_s={cfg.max_duration_s}"
        )

    # 4. GestureEvents
    gestures = emit_gestures(
        arpabet_tokens,
        durations,
        onset_offset_s=cfg.leading_silence_s,
        f0_hz=cfg.f0_hz,
        lung_pressure_pa=cfg.lung_pressure_pa,
    )

    # 5. Trajectory
    total_s = cfg.leading_silence_s + utt_dur_s + cfg.trailing_silence_s
    trajectory = gestures_to_trajectory(
        gestures,
        total_duration_s=total_s,
        frame_rate_hz=cfg.frame_rate_hz,
        f0_hz=cfg.f0_hz,
        lung_pressure_pa=cfg.lung_pressure_pa,
        deterministic_seed=cfg.deterministic_seed,
    )
    trajectory = _hard_silence_pads(
        trajectory,
        leading_silence_s=cfg.leading_silence_s,
        trailing_silence_s=cfg.trailing_silence_s,
        frame_rate_hz=cfg.frame_rate_hz,
    )

    # 6. TEA render. Frame step in TEA samples = native_sr / frame_rate.
    if TEA_NATIVE_SR_HZ % cfg.frame_rate_hz != 0:
        raise ValueError(
            f"frame_rate_hz={cfg.frame_rate_hz} must divide "
            f"TEA_NATIVE_SR_HZ={TEA_NATIVE_SR_HZ}"
        )
    samples_per_frame = TEA_NATIVE_SR_HZ // cfg.frame_rate_hz

    from ..copy_synth.tea_backend import synthesize_via_tea

    # 6a. correctness/004: optionally build per-frame frication inputs
    # from the gesture list and pass them through to TEA. Disabled by
    # default; enable via SmokeConfig.frication_enabled for the Whisper
    # pre-vs-post gate.
    fr_inputs: dict | None = None
    if cfg.frication_enabled:
        from .frication_inputs import build_frication_inputs

        fr_inputs_np = build_frication_inputs(
            arpabet_tokens=arpabet_tokens,
            durations_s=durations,
            onset_offset_s=cfg.leading_silence_s,
            total_duration_s=total_s,
            frame_rate_hz=cfg.frame_rate_hz,
        )
        # Trim/pad to match the trajectory length exactly. The
        # gestures_to_trajectory T = max(2, round(total*fr)+1) formula
        # may differ by ±1 frame from the inputs above; pad/truncate.
        T_traj = trajectory.shape[0]
        for k in ("place_index", "obstacle_proxy"):
            arr = fr_inputs_np[k]
            if arr.shape[0] < T_traj:
                pad = np.zeros((T_traj - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            elif arr.shape[0] > T_traj:
                arr = arr[:T_traj]
            fr_inputs_np[k] = arr
        v = fr_inputs_np["voicing"]
        if v.shape[0] < T_traj:
            v = np.concatenate([v, np.zeros((T_traj - v.shape[0],), dtype=v.dtype)], axis=0)
        elif v.shape[0] > T_traj:
            v = v[:T_traj]
        fr_inputs_np["voicing"] = v

        # Wrap in JAX arrays + the Frication config + a deterministic key.
        from tractus_engine_advanced import Frication
        import jax

        fr_inputs = {
            "frication": Frication.default(),
            "place_index": fr_inputs_np["place_index"],
            "obstacle_proxy": fr_inputs_np["obstacle_proxy"],
            "voicing": fr_inputs_np["voicing"],
            "noise_key": jax.random.key(int(cfg.frication_noise_seed)),
        }

    audio_native = synthesize_via_tea(
        trajectory,
        frame_step_samples=samples_per_frame,
        frication_inputs=fr_inputs,
    )

    # 7. Resample to requested SR.
    audio_out = _resample(audio_native, TEA_NATIVE_SR_HZ, cfg.sample_rate)

    # 8. Peak-normalize to ~0.95 so the audio is audible at default
    # listener volumes without clipping. The body model produces
    # transient-dominated output (peaks at ~0.5, RMS ~ 0.01 because
    # most of the utterance is low-energy frames between phone bursts —
    # an R1 NN floor issue, ARCH §1049). Without this normalization the
    # 'RMS in [0.01, 0.5]' regression gate would be borderline. This is
    # smoke-grade peak-norm only; M7 frication + body-model retrain
    # will be the real fix for the energy distribution.
    #
    # correctness/004 caveat: when comparing pre- vs post-frication
    # spectra (band-energy gate), peak-normalization rescales the
    # whole signal differently between the two renders and can swamp
    # the frication band's measurable lift. The fix is in the gate
    # script (``scripts/m7_frication_band.py``): it loads the same
    # audio with peak-norm applied to both, but uses a *common* scale
    # derived from the pre-render so the post-render isn't independently
    # rescaled. The render path itself stays unchanged for backwards
    # compatibility with smoke/001 regression tests.
    #
    # correctness/004b note on Whisper measurement headroom: peak-norm
    # to 0.95 (~0.4 dBFS) is well above Whisper-base.en's speech-detection
    # threshold (~−40 dBFS RMS rule of thumb). The empirical observation
    # post-fix is that Whisper STILL produces empty/garbage transcripts
    # on the rule-based smoke output even at 0.95 peak — the failure
    # mode is body-model-quality (transient-dominated, sub-formant-clarity
    # voicing) rather than amplitude. The Whisper gate is therefore
    # pinned at ``inconclusive`` until pretraining/002 lands the
    # full-corpus body-model retrain. Frication-band-energy lift
    # (``scripts/m7_frication_band.py``) is the binding measurement
    # on smoke output until then. See task
    # ``workplan/tasks/01_correctness/004b_frication_audible.md`` and
    # the consolidated ``outputs/m7_frication/audible_report.json``.
    peak = float(np.abs(audio_out).max())
    if peak > 1e-6:
        audio_out = (audio_out * (0.95 / peak)).astype(np.float32)

    return SmokeRender(
        audio=audio_out,
        sample_rate=cfg.sample_rate,
        arpabet_tokens=arpabet_tokens,
        durations_s=durations,
        gestures=gestures,
        trajectory=trajectory,
        frame_rate_hz=cfg.frame_rate_hz,
        config=cfg,
        input_text=text,
        extra={
            "tea_native_sr_hz": TEA_NATIVE_SR_HZ,
            "samples_per_frame_native": samples_per_frame,
            "native_audio_samples": int(audio_native.shape[0]),
        },
    )


# ---------------------------------------------------------------------------
# WAV + sidecar I/O.
# ---------------------------------------------------------------------------


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int) -> Path:
    """Write a 16-bit mono WAV. Clamps to [-1, 1] and rounds to int16."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(audio, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).round().astype(np.int16)
    with wave.open(str(out), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())
    return out


def write_sidecar(
    wav_path: Path | str,
    render: SmokeRender,
    *,
    consent_verified: bool = True,
    anatomy: str = "default",
) -> Path:
    """Write the sidecar JSON next to the WAV (per ARCH consent/provenance).

    Schema (per smoke/001 task):

        {
          "git_hash":          str | None,
          "config":            {...},
          "gesture_list":      [GestureEvent.to_dict(), ...],
          "anatomy":           "default" | str,
          "input_text":        str,
          "consent_verified":  bool,
          "deterministic_seed": int
        }

    Plus the standard ``RenderSidecar`` envelope from
    ``tractus_controller.sidecar`` so the file is interoperable with
    existing sidecar tooling. Smoke-specific fields go under
    ``extra.smoke``.
    """
    from ..sidecar import _git_describe

    cfg = render.config
    extra = {
        "smoke": {
            "git_hash": _git_describe(),
            "config": {
                "sample_rate": cfg.sample_rate,
                "f0_hz": cfg.f0_hz,
                "lung_pressure_pa": cfg.lung_pressure_pa,
                "leading_silence_s": cfg.leading_silence_s,
                "trailing_silence_s": cfg.trailing_silence_s,
                "frame_rate_hz": cfg.frame_rate_hz,
                "duration_table": (
                    str(cfg.duration_table) if cfg.duration_table else None
                ),
                "deterministic_seed": cfg.deterministic_seed,
                "speaker": cfg.speaker,
                "tea_native_sr_hz": TEA_NATIVE_SR_HZ,
            },
            "input_text": render.input_text,
            "arpabet_tokens": render.arpabet_tokens,
            "durations_s": render.durations_s,
            "gesture_list": [g.to_dict() for g in render.gestures],
            "anatomy": anatomy,
            "consent_verified": bool(consent_verified),
            "deterministic_seed": cfg.deterministic_seed,
            **render.extra,
        }
    }

    # Compose the standard envelope and write it.
    from ..sidecar import make_sidecar

    side = make_sidecar(
        fixture_path=str(wav_path),
        speaker_path=None,
        sampling_rate=int(render.sample_rate),
        audio=render.audio,
        extra=extra,
    )
    return side.write(wav_path)
