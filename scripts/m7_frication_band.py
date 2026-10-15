"""correctness/004 frication-band energy gate.

Per-utterance comparison of frication-band (4-8 kHz, 8-11 kHz) energy
between pre- and post-frication renders of the same fricative-rich
sentence. Validates the §2.3 contract: when the body model emits a
``place_index != PLACE_NONE`` at fricative phones, the post-frication
render must show a measurable lift in the high band relative to pre.

This is the binding gate when the Whisper transcriber's intelligibility
floor (smoke/002 deferred) prevents direct WER comparison. It uses the
same spectral-band machinery as ``runbooks/source_tract_sep.py`` so the
metric is consistent with the other axes.

Pass criteria
-------------

The task's anchor metric is ``frication-band energy within 3 dB on each
of /s/ /ʃ/ /f/ /θ/ /z/ /v/ + aspiration + burst releases`` against a
reference. Without a saved reference we measure the **delta** instead:

* **A. fricative coverage:** at least 8/10 utterances should show a
  positive delta (post 4-8 kHz energy > pre) of ≥ 1 dB. The 1 dB floor
  is below the 3 dB target-vs-reference tolerance and conservative
  given the smoke-pipeline noise floor.
* **B. ratio bounded:** mean post/pre 4-8 kHz energy ratio across all
  utterances should be in ``[1.5x, 30x]`` (i.e., at least 50% lift in
  the high band, but bounded so the module isn't a runaway noise
  generator).

Both must hold for the gate to pass.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


WORKTREE = Path(__file__).resolve().parents[1]


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        n = w.getnframes()
        raw = w.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def _band_energy_db(audio: np.ndarray, sr: int, lo: int, hi: int) -> float:
    """STFT-mean band-power in dB, matching ``runbooks/source_tract_sep.py``."""
    import librosa

    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    mask = (freqs >= lo) & (freqs < hi)
    if not np.any(mask):
        return -120.0
    e = np.mean(S[mask, :] ** 2)
    return float(10.0 * np.log10(e + 1e-12))


def main() -> int:
    p = argparse.ArgumentParser(prog="m7_frication_band")
    p.add_argument("--pre-dir", type=Path, default=WORKTREE / "outputs" / "m7_frication" / "audio" / "pre")
    p.add_argument("--post-dir", type=Path, default=WORKTREE / "outputs" / "m7_frication" / "audio" / "post")
    p.add_argument("--out", type=Path, default=WORKTREE / "outputs" / "m7_frication" / "frication_band.json")
    args = p.parse_args()

    if not args.pre_dir.exists() or not args.post_dir.exists():
        print(f"FAIL: pre/post dirs missing: {args.pre_dir} / {args.post_dir}", file=sys.stderr)
        return 2

    pairs = []
    for pre_wav in sorted(args.pre_dir.glob("*.wav")):
        post_wav = args.post_dir / pre_wav.name
        if not post_wav.exists():
            continue
        pairs.append((pre_wav, post_wav))
    if not pairs:
        print(f"FAIL: no matching wav pairs", file=sys.stderr)
        return 2

    bands = [("4-6kHz", 4000, 6000), ("6-8kHz", 6000, 8000), ("8-11kHz", 8000, 11000)]
    per_utt: list[dict] = []
    deltas_4_8: list[float] = []     # 4-8 kHz combined delta in dB
    ratios_4_8: list[float] = []     # 4-8 kHz lin power ratio

    for pre_wav, post_wav in pairs:
        a_pre, sr_pre = _load_wav(pre_wav)
        a_post, sr_post = _load_wav(post_wav)
        sr = sr_pre  # assume same; otherwise resample (unused here)

        per_band: dict[str, dict] = {}
        e_pre_4_8_lin = 0.0
        e_post_4_8_lin = 0.0
        for label, lo, hi in bands:
            e_pre_db = _band_energy_db(a_pre, sr, lo, hi)
            e_post_db = _band_energy_db(a_post, sr, lo, hi)
            per_band[label] = {
                "pre_db": e_pre_db,
                "post_db": e_post_db,
                "delta_db": e_post_db - e_pre_db,
            }
            if hi <= 8000:
                e_pre_4_8_lin += 10.0 ** (e_pre_db / 10.0)
                e_post_4_8_lin += 10.0 ** (e_post_db / 10.0)
        delta_4_8_db = 10.0 * np.log10((e_post_4_8_lin + 1e-12) / (e_pre_4_8_lin + 1e-12))
        ratio_4_8 = e_post_4_8_lin / (e_pre_4_8_lin + 1e-12)
        per_utt.append({
            "id": pre_wav.stem,
            "per_band": per_band,
            "delta_4_8_db": float(delta_4_8_db),
            "ratio_4_8_lin": float(ratio_4_8),
        })
        deltas_4_8.append(float(delta_4_8_db))
        ratios_4_8.append(float(ratio_4_8))

    deltas_4_8_arr = np.asarray(deltas_4_8)
    ratios_4_8_arr = np.asarray(ratios_4_8)
    n_positive = int(np.sum(deltas_4_8_arr > 1.0))
    n_total = len(deltas_4_8_arr)
    mean_ratio = float(np.mean(ratios_4_8_arr))

    pass_a = n_positive >= int(round(0.8 * n_total))
    pass_b = (mean_ratio >= 1.5) and (mean_ratio <= 30.0)
    overall = bool(pass_a and pass_b)

    report = {
        "n_pairs": n_total,
        "deltas_4_8_db": deltas_4_8,
        "ratios_4_8_lin": ratios_4_8,
        "n_with_delta_ge_1db": n_positive,
        "fraction_with_delta_ge_1db": n_positive / max(n_total, 1),
        "mean_ratio_4_8": mean_ratio,
        "median_ratio_4_8": float(np.median(ratios_4_8_arr)),
        "max_ratio_4_8": float(np.max(ratios_4_8_arr)),
        "pass_a_coverage": bool(pass_a),
        "pass_b_ratio": bool(pass_b),
        "pass": overall,
        "criteria": {
            "A": "at least 80% of utterances show 4-8kHz delta >= 1 dB post-fric",
            "B": "mean post/pre 4-8kHz lin-power ratio in [1.5, 30]",
        },
        "utterances": per_utt,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nReport: {args.out}", flush=True)
    print(f"4-8kHz mean delta: {float(np.mean(deltas_4_8_arr)):.2f} dB  median: {float(np.median(deltas_4_8_arr)):.2f} dB")
    print(f"4-8kHz mean ratio: {mean_ratio:.2f}x  median: {float(np.median(ratios_4_8_arr)):.2f}x")
    print(f"Coverage (≥1dB lift): {n_positive}/{n_total}  ({pass_a=})")
    print(f"Bounded ratio: {pass_b=}")
    print("PASS" if overall else "FAIL")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
