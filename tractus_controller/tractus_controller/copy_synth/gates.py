"""Multi-gate evaluation panel for copy-synth fits.

A single-axis gate (Whisper jaccard alone, or acoustic loss alone) is too
noisy to use as a stop-loss. Real failure modes — e.g. low-loss local
minima that produce unintelligible output — slip through any one
channel. ARCH §10 lesson 1 codifies this; the panel implements it.

This module runs three independent panels and returns a structured
``GateReport``:

1. **Whisper-decode similarity**: jaccard word-set overlap of fit vs target
   transcripts (reuses ``whisper_sanity.py`` logic).
2. **WER/CER vs known target text**: anchored against the .lab transcript,
   not against Whisper's reading of the target. Catches the case where
   Whisper hears the target correctly AND the fit produces a different but
   recognizable phrase (jaccard would be ~OK but the WER vs ground truth is
   atrocious).
3. **Trajectory plausibility**: per-channel velocity / acceleration / jerk /
   drift on the knot values, all normalized by bound width so the caps are
   comparable across channels. Plus a phone-boundary alignment score that
   measures whether trajectory inflections (sign changes in velocity) line
   up with phone boundaries from the MFA alignment json.

Composite verdicts:
    - ``speech_recognized``: jaccard >= 0.30 OR WER < 0.50
    - ``physically_plausible``: every per-channel velocity/jerk/drift cap
      respected
    - ``overall_pass``: both of the above
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tractus_controller.copy_synth.parameterization import (
    DEFAULT_FREE_CHANNELS,
    NO_VELUM_FREE_CHANNELS,
)

# ---------------------------------------------------------------------------
# Text normalization + WER/CER
# ---------------------------------------------------------------------------


_PUNCT = ",.!?;:\"'()[]{}—–-…"


def _norm_words(s: str) -> list[str]:
    return [w.strip(_PUNCT).lower() for w in s.split() if w.strip(_PUNCT)]


def word_jaccard(a: str, b: str) -> float:
    aw = set(_norm_words(a))
    bw = set(_norm_words(b))
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def _levenshtein(a: list[str] | str, b: list[str] | str) -> int:
    """Token-level edit distance. Works for words (lists) or chars (strings)."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """WER = edit_distance(ref_words, hyp_words) / len(ref_words). Empty ref → 0/1."""
    ref = _norm_words(reference)
    hyp = _norm_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        import jiwer  # type: ignore[import-not-found]

        return float(jiwer.wer(" ".join(ref), " ".join(hyp) or " "))
    except ImportError:
        return _levenshtein(ref, hyp) / len(ref)


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER on alpha+space normalized text. Empty ref → 0/1."""
    ref = "".join(c for c in reference.lower() if c.isalpha() or c == " ").strip()
    hyp = "".join(c for c in hypothesis.lower() if c.isalpha() or c == " ").strip()
    if not ref:
        return 0.0 if not hyp else 1.0
    try:
        import jiwer  # type: ignore[import-not-found]

        return float(jiwer.cer(ref, hyp or " "))
    except ImportError:
        return _levenshtein(ref, hyp) / len(ref)


# ---------------------------------------------------------------------------
# Whisper transcription
# ---------------------------------------------------------------------------


_WHISPER_MODEL_CACHE: dict[str, Any] = {}


def _load_whisper(model_name: str = "small"):
    if model_name in _WHISPER_MODEL_CACHE:
        return _WHISPER_MODEL_CACHE[model_name]
    import whisper  # type: ignore[import-not-found]

    model = whisper.load_model(model_name)
    _WHISPER_MODEL_CACHE[model_name] = model
    return model


def transcribe(wav_path: Path, model_name: str = "small") -> str:
    model = _load_whisper(model_name)
    result = model.transcribe(str(wav_path), language="en", fp16=False, verbose=False)
    return str(result["text"]).strip()


# ---------------------------------------------------------------------------
# Trajectory plausibility
# ---------------------------------------------------------------------------


def normalized_diffs(
    knot_values: np.ndarray,
    bounds: np.ndarray,
    knot_times_s: np.ndarray,
    *,
    n_frames: int | None = None,
    knot_rate_hz: float | None = None,
    interp_kind: str = "cubic",
    channels=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Velocity / acceleration / jerk on the trajectory frame grid.

    Returns (velocity, acceleration, jerk), each in **fraction of bound
    width per second** (and per second-squared / cubed). Caps calibrated
    in those units are then K-invariant: doubling the number of knots no
    longer doubles the measured velocity at fixed underlying motion. This
    closes R-V4-2 — see plan/70-handoff-2026-05-03.md.

    Implementation: interpolate ``knot_values`` to a uniformly-spaced
    trajectory grid ``T = n_frames`` at ``frame_dt = 1 / knot_rate_hz``
    (matching the renderer), then take finite differences on that grid.

    Args:
        knot_values: (K, C) per-knot residual values.
        bounds: (C, 2) per-channel residual bounds (lo, hi).
        knot_times_s: (K,) knot positions in seconds.
        n_frames: T frames to interpolate to. Defaults to
            ``round(knot_times_s[-1] * knot_rate_hz) + 1`` if
            ``knot_rate_hz`` is given, else falls back to legacy K-grid
            behavior (kept only for callers that genuinely want knot-grid
            stats; the gates panel always supplies frame-grid args).
        knot_rate_hz: trajectory frame rate. If both this and ``n_frames``
            are None, falls back to the legacy K-grid path.
        interp_kind: ``"linear"`` or ``"cubic"`` — must match the kind used
            to build the trajectory; otherwise the kinematics here won't
            agree with the kinematics actually rendered.
        channels: ignored; accepted for forward-compatibility (callers may
            pass channel objects).
    """
    width = bounds[:, 1] - bounds[:, 0]
    width = np.where(width <= 0, 1.0, width)
    K, C = knot_values.shape

    # Legacy K-grid fallback: only used when caller has not migrated. The
    # gates evaluator + the anti-cheat harness both pass the frame-grid
    # args. Kept so that diagnostic callers don't break silently.
    if n_frames is None and knot_rate_hz is None:
        norm_knots = knot_values / width[None, :]
        if K < 2:
            empty = np.zeros((0, C))
            return empty, empty, empty
        dt = np.diff(knot_times_s)
        dt = np.where(dt <= 0, 1e-6, dt)
        vel = np.diff(norm_knots, axis=0) / dt[:, None]
        if vel.shape[0] < 2:
            empty = np.zeros((0, C))
            return vel, empty, empty
        dt_acc = (knot_times_s[2:] - knot_times_s[:-2]) / 2.0
        dt_acc = np.where(dt_acc <= 0, 1e-6, dt_acc)
        acc = np.diff(vel, axis=0) / dt_acc[:, None]
        if acc.shape[0] < 2:
            empty = np.zeros((0, C))
            return vel, acc, empty
        dt_jrk = (knot_times_s[3:] - knot_times_s[:-3]) / 3.0
        dt_jrk = np.where(dt_jrk <= 0, 1e-6, dt_jrk)
        jrk = np.diff(acc, axis=0) / dt_jrk[:, None]
        return vel, acc, jrk

    # Frame-grid path (the K-invariant fix).
    from tractus_controller.copy_synth.parameterization import _interp_residual

    if knot_rate_hz is None or knot_rate_hz <= 0:
        raise ValueError("knot_rate_hz must be positive on the frame-grid path")
    if n_frames is None:
        # Cover [0, knot_times_s[-1]] with frame_dt = 1/knot_rate_hz.
        n_frames = int(round(float(knot_times_s[-1]) * knot_rate_hz)) + 1
    if n_frames < 2:
        empty = np.zeros((0, C))
        return empty, empty, empty

    frame_dt = 1.0 / float(knot_rate_hz)
    times_s = np.arange(n_frames, dtype=np.float64) * frame_dt

    if K < 2:
        empty = np.zeros((0, C))
        return empty, empty, empty

    # Interpolate each channel to the frame grid using the same kind as
    # the renderer. Then normalize to bound width and finite-difference.
    interp = np.empty((n_frames, C), dtype=np.float64)
    for c in range(C):
        interp[:, c] = _interp_residual(
            knot_times_s, knot_values[:, c], times_s, interp_kind
        )
    norm = interp / width[None, :]                              # (T, C)

    vel = np.diff(norm, axis=0) / frame_dt                      # (T-1, C)
    if vel.shape[0] < 2:
        empty = np.zeros((0, C))
        return vel, empty, empty
    acc = np.diff(vel, axis=0) / frame_dt                       # (T-2, C)
    if acc.shape[0] < 2:
        empty = np.zeros((0, C))
        return vel, acc, empty
    jrk = np.diff(acc, axis=0) / frame_dt                       # (T-3, C)
    return vel, acc, jrk


def per_channel_drift(
    knot_values: np.ndarray, init_knots: np.ndarray, bounds: np.ndarray
) -> np.ndarray:
    """Mean |knot - init| / width per channel. Shape (C,)."""
    width = bounds[:, 1] - bounds[:, 0]
    width = np.where(width <= 0, 1.0, width)
    return np.abs(knot_values - init_knots).mean(axis=0) / width


def phone_boundary_alignment(
    trajectory: np.ndarray,
    knot_rate_hz: float,
    phones: list[dict],
    *,
    free_idxs: tuple[int, ...] = tuple(ch.idx for ch in DEFAULT_FREE_CHANNELS),
    tol_ms: float = 30.0,
) -> float:
    """Score in [0, 1] for how well per-channel velocity sign-changes align with
    phone boundaries.

    For each phone boundary t_b, look across all free channels for any velocity
    sign-change within ±tol_ms. Score = (boundaries with at least one nearby
    inflection) / (total interior boundaries).

    Returns 0.0 if there are no interior phone boundaries.
    """
    if not phones or trajectory.shape[0] < 3:
        return 0.0

    boundaries = sorted({p["t_start"] for p in phones} | {p["t_end"] for p in phones})
    if len(boundaries) < 3:
        return 0.0
    boundaries = boundaries[1:-1]  # drop sentence start/end

    n_frames = trajectory.shape[0]
    times_s = np.arange(n_frames, dtype=np.float64) / knot_rate_hz

    sign_change_times: list[float] = []
    for idx in free_idxs:
        col = trajectory[:, idx]
        v = np.diff(col)
        # sign changes (strict): product of consecutive velocities is < 0
        sc = np.where(v[:-1] * v[1:] < 0)[0]
        for k in sc:
            sign_change_times.append(times_s[k + 1])
    if not sign_change_times:
        return 0.0
    sc_arr = np.array(sorted(sign_change_times))

    tol_s = tol_ms / 1000.0
    hit = 0
    for tb in boundaries:
        # binary search for closest sign-change
        idx = np.searchsorted(sc_arr, tb)
        candidates = []
        if idx < len(sc_arr):
            candidates.append(sc_arr[idx])
        if idx > 0:
            candidates.append(sc_arr[idx - 1])
        if any(abs(c - tb) <= tol_s for c in candidates):
            hit += 1
    return hit / len(boundaries)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class GateReport:
    # Whisper
    whisper_target_text: str
    whisper_fit_text: str
    whisper_jaccard: float
    whisper_wer: float           # WER vs KNOWN target text (.lab file)
    whisper_cer: float           # CER vs KNOWN target text
    # Trajectory plausibility
    max_velocity_per_channel: dict[str, float]
    max_acceleration_per_channel: dict[str, float]
    max_jerk_per_channel: dict[str, float]
    drift_per_channel: dict[str, float]
    phone_boundary_alignment_score: float
    # Composite verdicts
    speech_recognized: bool
    physically_plausible: bool
    overall_pass: bool
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_fit(
    fit_npz_path: Path,
    fit_wav_path: Path,
    target_wav_path: Path,
    target_text: str,
    alignment_json_path: Path | None,
    *,
    velocity_cap: float = 0.20,
    acceleration_cap: float = 2.00,
    jerk_cap: float = 20.00,
    drift_cap: float = 0.50,
    boundary_alignment_min: float = 0.20,
    jaccard_min: float = 0.30,
    wer_max: float = 0.50,
    knot_rate_hz: float = 49.0,
    whisper_model: str = "small",
    skip_whisper: bool = False,
    channels=None,
) -> GateReport:
    """Run the multi-gate panel on a fit. See module docstring for details.

    ``knot_rate_hz`` is the trajectory frame rate (49 Hz for renderer standard); only
    used for the phone-boundary alignment metric.
    """
    fit_npz_path = Path(fit_npz_path)
    fit_wav_path = Path(fit_wav_path)
    target_wav_path = Path(target_wav_path)

    flags: list[str] = []

    # --- Whisper -----------------------------------------------------------
    if skip_whisper:
        target_text_w = ""
        fit_text_w = ""
        jaccard = 0.0
    else:
        target_text_w = transcribe(target_wav_path, model_name=whisper_model)
        fit_text_w = transcribe(fit_wav_path, model_name=whisper_model)
        jaccard = word_jaccard(target_text_w, fit_text_w)

    wer = word_error_rate(target_text, fit_text_w) if not skip_whisper else 1.0
    cer = character_error_rate(target_text, fit_text_w) if not skip_whisper else 1.0

    speech_recognized = (jaccard >= jaccard_min) or (wer < wer_max)
    if not speech_recognized:
        if jaccard < jaccard_min:
            flags.append(f"jaccard_too_low:{jaccard:.2f}")
        if wer >= wer_max:
            flags.append(f"wer_too_high:{wer:.2f}")

    # --- Trajectory --------------------------------------------------------
    d = np.load(fit_npz_path)
    knot_values = d["knot_values"]
    init_knots = d["init_knots"]
    bounds = d["bounds"]
    knot_times_s = d["knot_times_s"]
    trajectory = d["trajectory"]

    # Auto-detect channel layout from npz width if caller didn't override.
    # Default is 8-ch, NO_VELUM is 7-ch; if width mismatches both, fall back to
    # generic synthetic channel names so the rest of the pipeline still works.
    if channels is None:
        n_ch = knot_values.shape[1]
        if n_ch == len(DEFAULT_FREE_CHANNELS):
            channels = DEFAULT_FREE_CHANNELS
        elif n_ch == len(NO_VELUM_FREE_CHANNELS):
            channels = NO_VELUM_FREE_CHANNELS
        else:
            from tractus_controller.copy_synth.parameterization import FreeChannel

            channels = tuple(
                FreeChannel(f"ch{i}", i, 0.0, 1.0) for i in range(n_ch)
            )

    # Compute kinematics on the trajectory frame grid (K-invariant; closes
    # R-V4-2). interp_kind defaults to "cubic" — matches the v4 fitter; if
    # the npz was produced with linear, max-vel is identical either way
    # (linear segments have constant velocity = knot secant).
    vel, acc, jrk = normalized_diffs(
        knot_values, bounds, knot_times_s,
        n_frames=int(trajectory.shape[0]),
        knot_rate_hz=float(knot_rate_hz),
        interp_kind="cubic",
    )
    drift = per_channel_drift(knot_values, init_knots, bounds)

    def _per_ch(arr: np.ndarray) -> dict[str, float]:
        if arr.size == 0:
            return {ch.name: 0.0 for ch in channels}
        m = np.abs(arr).max(axis=0)
        return {ch.name: float(m[i]) for i, ch in enumerate(channels)}

    max_vel = _per_ch(vel)
    max_acc = _per_ch(acc)
    max_jrk = _per_ch(jrk)
    drift_map = {ch.name: float(drift[i]) for i, ch in enumerate(channels)}

    # Caps
    physically_plausible = True
    for name, v in max_vel.items():
        if v > velocity_cap:
            flags.append(f"velocity_exceeded:{name}={v:.2f}>{velocity_cap}")
            physically_plausible = False
    for name, v in max_acc.items():
        if v > acceleration_cap:
            flags.append(f"acceleration_exceeded:{name}={v:.2f}>{acceleration_cap}")
            physically_plausible = False
    for name, v in max_jrk.items():
        if v > jerk_cap:
            flags.append(f"jerk_exceeded:{name}={v:.2f}>{jerk_cap}")
            physically_plausible = False
    for name, v in drift_map.items():
        if v > drift_cap:
            flags.append(f"drift_exceeded:{name}={v:.2f}>{drift_cap}")
            physically_plausible = False

    # --- Phone boundary alignment -----------------------------------------
    boundary_score = 0.0
    if alignment_json_path is not None and Path(alignment_json_path).exists():
        with open(alignment_json_path) as fh:
            alignment = json.load(fh)
        phones = alignment.get("phones", [])
        boundary_score = phone_boundary_alignment(
            trajectory, knot_rate_hz, phones, tol_ms=30.0
        )
        if boundary_score < boundary_alignment_min and phones:
            flags.append(
                f"boundary_alignment_low:{boundary_score:.2f}<{boundary_alignment_min}"
            )

    return GateReport(
        whisper_target_text=target_text_w,
        whisper_fit_text=fit_text_w,
        whisper_jaccard=float(jaccard),
        whisper_wer=float(wer),
        whisper_cer=float(cer),
        max_velocity_per_channel=max_vel,
        max_acceleration_per_channel=max_acc,
        max_jerk_per_channel=max_jrk,
        drift_per_channel=drift_map,
        phone_boundary_alignment_score=float(boundary_score),
        speech_recognized=bool(speech_recognized),
        physically_plausible=bool(physically_plausible),
        overall_pass=bool(speech_recognized and physically_plausible),
        flags=flags,
    )


def evaluate_corpus(
    fits_dir: Path,
    *,
    chunks_dir: Path | None = None,
    alignments_dir: Path | None = None,
    transcripts_dir: Path | None = None,
    **kwargs,
) -> list[GateReport]:
    """Evaluate every ``*_fit.wav`` under ``fits_dir`` whose sibling NPZ + chunk
    target audio + .lab file all resolve.

    Convention (matches ``outputs/fits``):
        ``<fits_dir>/<utt>_fit.wav``     — fit audio
        ``<fits_dir>/<utt>.npz``         — fit artifact
        ``<fits_dir>/<utt>_alignment.json`` — MFA alignment
        ``<chunks_dir>/<utt>.wav``       — target audio
        ``<chunks_dir>/<utt>.lab``       — target transcript
    """
    fits_dir = Path(fits_dir)
    chunks_dir = Path(chunks_dir) if chunks_dir else fits_dir.parent / "natural_audio" / "natasha" / "chunks"
    alignments_dir = Path(alignments_dir) if alignments_dir else fits_dir
    transcripts_dir = Path(transcripts_dir) if transcripts_dir else chunks_dir

    reports: list[GateReport] = []
    for fit_wav in sorted(fits_dir.glob("*_fit.wav")):
        utt = fit_wav.stem.replace("_fit", "")
        npz = fits_dir / f"{utt}.npz"
        target_wav = chunks_dir / f"{utt}.wav"
        lab = transcripts_dir / f"{utt}.lab"
        align = alignments_dir / f"{utt}_alignment.json"
        if not (npz.exists() and target_wav.exists() and lab.exists()):
            continue
        text = lab.read_text().strip()
        report = evaluate_fit(
            npz, fit_wav, target_wav, text, align if align.exists() else None, **kwargs
        )
        reports.append(report)
    return reports


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m tractus_controller.copy_synth.gates",
        description="Multi-gate evaluation panel for copy-synth fits.",
    )
    p.add_argument("fit_npz", type=Path, help="Path to fit .npz artifact")
    p.add_argument("fit_wav", type=Path, help="Path to synthesized fit .wav")
    p.add_argument("target_wav", type=Path, help="Path to natural target .wav")
    p.add_argument("--text", required=True, help="Known target transcript")
    p.add_argument("--alignment", type=Path, default=None, help="MFA alignment JSON")
    p.add_argument("--whisper-model", default="small")
    p.add_argument("--velocity-cap", type=float, default=0.20)
    p.add_argument("--acceleration-cap", type=float, default=2.00)
    p.add_argument("--jerk-cap", type=float, default=20.00)
    p.add_argument("--drift-cap", type=float, default=0.50)
    p.add_argument("--knot-rate-hz", type=float, default=49.0)
    p.add_argument("--skip-whisper", action="store_true", help="Skip Whisper (debugging)")
    p.add_argument("--json", action="store_true", help="Emit JSON to stdout")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)
    report = evaluate_fit(
        fit_npz_path=args.fit_npz,
        fit_wav_path=args.fit_wav,
        target_wav_path=args.target_wav,
        target_text=args.text,
        alignment_json_path=args.alignment,
        velocity_cap=args.velocity_cap,
        acceleration_cap=args.acceleration_cap,
        jerk_cap=args.jerk_cap,
        drift_cap=args.drift_cap,
        knot_rate_hz=args.knot_rate_hz,
        whisper_model=args.whisper_model,
        skip_whisper=args.skip_whisper,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(f"speech_recognized:    {report.speech_recognized}")
        print(f"physically_plausible: {report.physically_plausible}")
        print(f"overall_pass:         {report.overall_pass}")
        print(f"  jaccard:           {report.whisper_jaccard:.3f}")
        print(f"  wer (vs target):   {report.whisper_wer:.3f}")
        print(f"  cer (vs target):   {report.whisper_cer:.3f}")
        print(f"  boundary align:    {report.phone_boundary_alignment_score:.3f}")
        print(f"  whisper target:    {report.whisper_target_text!r}")
        print(f"  whisper fit:       {report.whisper_fit_text!r}")
        if report.flags:
            print("  flags:")
            for f in report.flags:
                print(f"    - {f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
