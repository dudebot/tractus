"""correctness/004 Whisper-gate driver.

Renders the fricative-rich smoke corpus twice — once with frication
disabled (master pre-correctness/004 behaviour reproduced via
``frication_enabled=False``) and once with the new physical frication
module on. Runs the runbook Whisper transcriber on each and computes
fricative-character substitution rate before/after.

**Note on the Whisper gate:** the rule-based smoke pipeline produces
audio that Whisper-base.en classifies as English but cannot transcribe
(empty output) on either pre or post conditions — the R1 NN body model
emits a transient-dominated waveform with peaks ~0.5 and RMS ~0.03 that
falls below Whisper's speech-detection floor (smoke/001 + ARCH §1049 +
smoke/002 intelligibility gate is explicitly deferred to P1 close).
This means the **pre rate** is 100% (all fricatives "deleted" because
Whisper produces no transcript) and there is no headroom for the post
rate to demonstrate the spec'd ≥50% reduction.

The Whisper gate as written is therefore inconclusive given the
upstream intelligibility floor, NOT a frication-module failure. Per
the task's escalation clause the secondary automated gate
``runbooks/source_tract_sep.py`` (frication-band-energy difference)
is the binding measurement when Whisper can't resolve. The companion
script ``scripts/m7_frication_band.py`` runs that measurement on the
same pre/post WAVs.

Output: JSON report at ``outputs/m7_frication/whisper_gate.json`` plus
WAVs at ``outputs/m7_frication/audio/{pre,post}/<id>.wav``.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

# Ensure the worktree's editable packages take priority over the master
# venv installs. Without this the MetaPathFinder wires
# ``tractus_controller`` to a different checkout and the
# new SmokeConfig.frication_enabled field is missing.
WORKTREE = Path(__file__).resolve().parents[1]
for p in (
    str(WORKTREE / "tractus_controller"),
    str(WORKTREE / "tractus_engine_advanced" / "src"),
    str(WORKTREE),
):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault(
    "TRACTUS_ENGINE_ADVANCED_SRC", str(WORKTREE / "tractus_engine_advanced" / "src")
)

# Ten fricative-rich sentences spanning /s/ /ʃ/ /f/ /θ/ /z/ /v/ + /h/
# aspiration + stop bursts. Hand-picked to load each fricative class at
# least twice so a single bad render doesn't dominate the metric.
FRICATIVE_RICH: list[dict] = [
    {"id": "s_seashells",   "text": "She sells seashells by the seashore."},
    {"id": "s_sister",      "text": "Sister Susie sews socks for soldiers."},
    {"id": "sh_pushed",     "text": "Pushed sushi shops shut shop yesterday."},
    {"id": "f_fluffy",      "text": "Fluffy feathers fall on fresh fields."},
    {"id": "th_three",      "text": "The three thieves thought through their theft."},
    {"id": "z_lazy",        "text": "Lazy bees buzz noisily in the breeze."},
    {"id": "v_vivid",       "text": "Vivid violets vie above velvet vines."},
    {"id": "h_high",        "text": "How high are the hopeful happy hounds?"},
    {"id": "burst_peter",   "text": "Peter Piper picked a peck of pickled peppers."},
    {"id": "burst_kindly",  "text": "Kindly cookies came quick to the kitchen."},
]


# Fricative phone classes (lowercase). Used for char-level substitution
# rate. Whisper output is normalized to lowercase + strip punctuation;
# we count any of these characters appearing in the *target* and check
# whether they were preserved in the ASR output. A "substitution" is
# when the target char count for a fricative class is X but the ASR
# count is < X (i.e., the fricative was deleted or replaced).
FRICATIVE_CHARS: list[str] = ["s", "z", "f", "v", "h", "t"]
# 'sh' / 'th' / 'ch' digraphs handled via substring counts below.
FRICATIVE_DIGRAPHS: list[str] = ["sh", "th", "ch"]


def _normalize(text: str) -> str:
    out = text.lower()
    # Strip punctuation other than apostrophes.
    return "".join(c for c in out if c.isalpha() or c == " " or c == "'")


def fricative_substitution_rate(target: str, asr: str) -> dict:
    """Per-fricative-class deletion rate and aggregate.

    Returns a dict with per-class ``deleted``, ``target_count``, and the
    aggregate ``rate`` = ``sum(deleted)/sum(target_count)`` over all
    classes.
    """
    t = _normalize(target)
    a = _normalize(asr)
    classes = {c: c for c in FRICATIVE_CHARS}
    classes.update({c: c for c in FRICATIVE_DIGRAPHS})

    per_class: dict[str, dict] = {}
    total_target = 0
    total_deleted = 0
    for label, pattern in classes.items():
        target_count = t.count(pattern)
        asr_count = a.count(pattern)
        deleted = max(target_count - asr_count, 0)
        per_class[label] = {
            "target_count": target_count,
            "asr_count": asr_count,
            "deleted": deleted,
        }
        total_target += target_count
        total_deleted += deleted
    rate = total_deleted / max(total_target, 1)
    return {"per_class": per_class, "rate": rate, "total_target": total_target, "total_deleted": total_deleted}


def render_one(text: str, frication_enabled: bool) -> "tuple[any, any, dict]":
    """Render ``text`` and return (audio, sample_rate, render_meta).

    Bypasses the standard ``render_text_to_audio`` peak-normalization
    so pre/post comparison sees the actual frication contribution
    (peak-norm rescales pre and post on independent peaks, swamping
    the frication-band lift). For Whisper transcription a separate
    peak-norm is applied AFTER the comparison-friendly raw audio is
    captured.
    """
    import numpy as np
    from tractus_controller.smoke.pipeline import (
        SmokeConfig, text_to_phones, resolve_durations, load_duration_table,
        emit_gestures, gestures_to_trajectory, _hard_silence_pads,
        _resample, TEA_NATIVE_SR_HZ,
    )

    cfg = SmokeConfig(
        sample_rate=22050,
        frication_enabled=frication_enabled,
        frication_noise_seed=42,  # fixed for reproducibility
    )

    t0 = time.time()
    arpabet_tokens = text_to_phones(text)
    overrides = load_duration_table(cfg.duration_table)
    durations = resolve_durations(arpabet_tokens, overrides)
    utt_dur_s = sum(durations)
    gestures = emit_gestures(
        arpabet_tokens, durations,
        onset_offset_s=cfg.leading_silence_s,
        f0_hz=cfg.f0_hz, lung_pressure_pa=cfg.lung_pressure_pa,
    )
    total_s = cfg.leading_silence_s + utt_dur_s + cfg.trailing_silence_s
    trajectory = gestures_to_trajectory(
        gestures, total_duration_s=total_s, frame_rate_hz=cfg.frame_rate_hz,
        f0_hz=cfg.f0_hz, lung_pressure_pa=cfg.lung_pressure_pa,
        deterministic_seed=cfg.deterministic_seed,
    )
    trajectory = _hard_silence_pads(
        trajectory,
        leading_silence_s=cfg.leading_silence_s,
        trailing_silence_s=cfg.trailing_silence_s,
        frame_rate_hz=cfg.frame_rate_hz,
    )
    samples_per_frame = TEA_NATIVE_SR_HZ // cfg.frame_rate_hz

    fr_inputs = None
    if cfg.frication_enabled:
        from tractus_controller.smoke.frication_inputs import build_frication_inputs
        from tractus_engine_advanced import Frication
        import jax

        inp_np = build_frication_inputs(
            arpabet_tokens=arpabet_tokens, durations_s=durations,
            onset_offset_s=cfg.leading_silence_s,
            total_duration_s=total_s, frame_rate_hz=cfg.frame_rate_hz,
        )
        T_traj = trajectory.shape[0]
        for k in ("place_index", "obstacle_proxy"):
            arr = inp_np[k]
            if arr.shape[0] < T_traj:
                pad = np.zeros((T_traj - arr.shape[0], arr.shape[1]), dtype=arr.dtype)
                arr = np.concatenate([arr, pad], axis=0)
            elif arr.shape[0] > T_traj:
                arr = arr[:T_traj]
            inp_np[k] = arr
        v = inp_np["voicing"]
        if v.shape[0] < T_traj:
            v = np.concatenate([v, np.zeros((T_traj - v.shape[0],), dtype=v.dtype)], axis=0)
        elif v.shape[0] > T_traj:
            v = v[:T_traj]
        inp_np["voicing"] = v
        fr_inputs = {
            "frication": Frication.default(),
            "place_index": inp_np["place_index"],
            "obstacle_proxy": inp_np["obstacle_proxy"],
            "voicing": inp_np["voicing"],
            "noise_key": jax.random.key(int(cfg.frication_noise_seed)),
        }

    from tractus_controller.copy_synth.tea_backend import synthesize_via_tea
    audio_native = synthesize_via_tea(
        trajectory, frame_step_samples=samples_per_frame,
        frication_inputs=fr_inputs,
    )
    audio_out = _resample(audio_native, TEA_NATIVE_SR_HZ, cfg.sample_rate)
    elapsed = time.time() - t0

    # NO peak-norm here — the raw audio is what the band-energy gate
    # measures. For Whisper input we apply a *shared* gain in main()
    # so pre and post are scaled identically.
    meta = {
        "render_seconds": float(elapsed),
        "tokens": arpabet_tokens,
        "durations_s": durations,
        "audio_samples": int(audio_out.shape[0]),
        "audio_rms": float((audio_out**2).mean() ** 0.5),
        "audio_peak": float(abs(audio_out).max()),
    }
    return audio_out, cfg.sample_rate, meta


def write_wav(path: Path, audio, sample_rate: int) -> None:
    """Write 16-bit mono WAV."""
    import wave
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(audio, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).round().astype("int16")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def transcribe_one(wav_path: Path, model: str = "base") -> str:
    """Run faster-whisper directly on a WAV. Returns the transcript."""
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        # Fall back to the shell wrapper if the import isn't available;
        # the wrapper handles its own import failure messaging.
        return _transcribe_via_runbook(wav_path, model)

    # CPU-only — frication-rendered audio is ~2 s clips; CPU faster-whisper
    # base.en model handles 10 clips in ~30 s without GPU contention.
    asr = WhisperModel(model, device="cpu", compute_type="int8")
    segments, _ = asr.transcribe(str(wav_path), language="en")
    return " ".join(seg.text.strip() for seg in segments).strip()


def _transcribe_via_runbook(wav_path: Path, model: str) -> str:
    """Fallback: invoke ``runbooks/whisper_gate.sh`` on each WAV.

    Returns just the asr_text from the JSON. The threshold args are
    passed as wide-open so the gate always 'passes' — we ignore the
    pass/fail and use the asr_text only.
    """
    out = subprocess.run(
        [
            "bash",
            str(WORKTREE / "workplan" / "runbooks" / "whisper_gate.sh"),
            "--audio", str(wav_path),
            "--target", "placeholder",
            "--threshold-wer", "1.0",
            "--threshold-jaccard", "0.0",
            "--model", model,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    try:
        report = json.loads(out.stdout)
        return report.get("asr_text", "")
    except json.JSONDecodeError:
        return ""


def main() -> int:
    p = argparse.ArgumentParser(prog="m7_whisper_gate")
    p.add_argument("--out", type=Path, default=WORKTREE / "outputs" / "m7_frication")
    p.add_argument("--model", type=str, default="base")
    p.add_argument("--limit", type=int, default=0, help="Render only first N (0=all).")
    args = p.parse_args()

    out_dir: Path = args.out
    audio_pre = out_dir / "audio" / "pre"
    audio_post = out_dir / "audio" / "post"
    audio_pre.mkdir(parents=True, exist_ok=True)
    audio_post.mkdir(parents=True, exist_ok=True)

    items = FRICATIVE_RICH if args.limit <= 0 else FRICATIVE_RICH[: args.limit]

    per_utt: list[dict] = []
    pre_total_target = 0
    pre_total_deleted = 0
    post_total_target = 0
    post_total_deleted = 0

    import numpy as np

    for ix, item in enumerate(items):
        uid = item["id"]
        text = item["text"]
        print(f"[{ix+1}/{len(items)}] {uid}: {text!r}", flush=True)
        try:
            a_pre, sr_pre, meta_pre = render_one(text, frication_enabled=False)
        except Exception as exc:  # pragma: no cover — record failure, keep going
            per_utt.append({"id": uid, "text": text, "error_pre": str(exc)})
            continue
        try:
            a_post, sr_post, meta_post = render_one(text, frication_enabled=True)
        except Exception as exc:
            per_utt.append({"id": uid, "text": text, "error_post": str(exc), "meta_pre": meta_pre})
            continue

        # SHARED peak-norm: scale both pre and post by the SAME factor,
        # picked from max(|pre_peak|, |post_peak|) so the louder of the
        # two clips at 0.95 and the relative spectrum stays intact. The
        # band-energy gate on these WAVs sees the actual frication
        # contribution rather than two independently-rescaled signals.
        peak = float(max(np.abs(a_pre).max(), np.abs(a_post).max()))
        if peak > 1e-6:
            a_pre = (a_pre * (0.95 / peak)).astype(np.float32)
            a_post = (a_post * (0.95 / peak)).astype(np.float32)
        wav_pre = audio_pre / f"{uid}.wav"
        wav_post = audio_post / f"{uid}.wav"
        write_wav(wav_pre, a_pre, sr_pre)
        write_wav(wav_post, a_post, sr_post)

        try:
            asr_pre = transcribe_one(wav_pre, model=args.model)
        except Exception as exc:
            asr_pre = ""
            print(f"  ASR pre failed: {exc}", flush=True)
        try:
            asr_post = transcribe_one(wav_post, model=args.model)
        except Exception as exc:
            asr_post = ""
            print(f"  ASR post failed: {exc}", flush=True)

        rate_pre = fricative_substitution_rate(text, asr_pre)
        rate_post = fricative_substitution_rate(text, asr_post)
        pre_total_target += rate_pre["total_target"]
        pre_total_deleted += rate_pre["total_deleted"]
        post_total_target += rate_post["total_target"]
        post_total_deleted += rate_post["total_deleted"]

        print(f"  ASR pre  : {asr_pre!r}  rate={rate_pre['rate']:.3f}", flush=True)
        print(f"  ASR post : {asr_post!r}  rate={rate_post['rate']:.3f}", flush=True)

        per_utt.append({
            "id": uid,
            "text": text,
            "asr_pre": asr_pre,
            "asr_post": asr_post,
            "rate_pre": rate_pre["rate"],
            "rate_post": rate_post["rate"],
            "per_class_pre": rate_pre["per_class"],
            "per_class_post": rate_post["per_class"],
            "meta_pre": meta_pre,
            "meta_post": meta_post,
        })

    pre_rate = pre_total_deleted / max(pre_total_target, 1)
    post_rate = post_total_deleted / max(post_total_target, 1)
    ratio = post_rate / max(pre_rate, 1e-6)
    passed = bool(ratio <= 0.5)

    report = {
        "model": args.model,
        "n_utterances": len(items),
        "aggregate": {
            "pre_rate": pre_rate,
            "post_rate": post_rate,
            "ratio": ratio,
            "pre_total_target": pre_total_target,
            "pre_total_deleted": pre_total_deleted,
            "post_total_target": post_total_target,
            "post_total_deleted": post_total_deleted,
        },
        "pass": passed,
        "pass_criterion": "post_rate / pre_rate <= 0.5",
        "utterances": per_utt,
    }
    out_json = out_dir / "whisper_gate.json"
    out_json.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {out_json}", flush=True)
    print(f"PRE  fric-deletion rate: {pre_rate:.3f}  ({pre_total_deleted}/{pre_total_target})")
    print(f"POST fric-deletion rate: {post_rate:.3f}  ({post_total_deleted}/{post_total_target})")
    print(f"Ratio (post/pre): {ratio:.3f}   {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
