#!/usr/bin/env python
"""Render the 20-utterance smoke regression corpus.

Reads ``tractus_controller/configs/smoke_utterances.json``, renders
each utterance through ``tractus_controller.smoke.pipeline``, and
writes WAV + sidecar JSON to ``outputs/smoke/<id>.wav``. Per-utterance
pass/fail is judged by the smoke/001 acceptance gate ("no NaN, no
Inf, RMS in [0.01, 0.5]").

NOT a Whisper-WER intelligibility gate — that's smoke/002 and is
gated on P0 close (correctness/004 + correctness/008 still open).

Invocation:
    JAX_PLATFORMS=cpu .venv/bin/python scripts/smoke_render_corpus.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def _add_repo_paths() -> None:
    """Make the controller / engine packages importable from a fresh venv."""
    here = Path(__file__).resolve().parent.parent
    for sub in ("tractus_controller", "tractus_engine_advanced/src", "tractus_shared/src"):
        p = (here / sub).resolve()
        if p.exists() and str(p) not in sys.path:
            sys.path.insert(0, str(p))


def main() -> int:
    _add_repo_paths()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus", type=Path,
        default=Path("tractus_controller/configs/smoke_utterances.json"),
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("outputs/smoke"),
    )
    parser.add_argument("--sample-rate", type=int, default=22050)
    parser.add_argument(
        "--durations", type=Path,
        default=Path("tractus_controller/configs/smoke_durations.json"),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from tractus_controller.smoke.pipeline import (
        SmokeConfig,
        render_text_to_audio,
        write_sidecar,
        write_wav,
    )

    corpus = json.loads(args.corpus.read_text())
    items = corpus["utterances"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SmokeConfig(
        sample_rate=args.sample_rate,
        duration_table=args.durations if args.durations and args.durations.exists() else None,
        deterministic_seed=args.seed,
    )

    n_pass = 0
    rows = []
    for it in items:
        uid = it["id"]
        text = it["text"]
        wav_path = args.out_dir / f"{uid}.wav"
        t0 = time.time()
        try:
            r = render_text_to_audio(text, cfg)
            wav = write_wav(wav_path, r.audio, r.sample_rate)
            write_sidecar(wav, r)
            elapsed = time.time() - t0
            audio = r.audio
            finite = bool(np.isfinite(audio).all())
            has_nan = bool(np.isnan(audio).any())
            has_inf = bool(np.isinf(audio).any())
            rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
            peak = float(np.abs(audio).max())
            ok = finite and not has_nan and not has_inf and 0.01 <= rms <= 0.5
            rows.append({
                "id": uid, "text": text, "wav": str(wav),
                "elapsed_s": round(elapsed, 2),
                "duration_s": round(len(audio) / r.sample_rate, 3),
                "rms": round(rms, 4), "peak": round(peak, 4),
                "finite": finite, "nan": has_nan, "inf": has_inf,
                "pass": ok,
            })
            if ok:
                n_pass += 1
            print(
                f"{'PASS' if ok else 'FAIL'} {uid:14s}  RMS={rms:.4f}  "
                f"peak={peak:.3f}  dur={len(audio)/r.sample_rate:.2f}s  "
                f"render={elapsed:.1f}s  text={text[:40]!r}"
            )
        except Exception as e:  # noqa: BLE001
            elapsed = time.time() - t0
            rows.append({"id": uid, "text": text, "error": repr(e),
                         "elapsed_s": round(elapsed, 2), "pass": False})
            print(f"ERROR {uid:14s}  {e!r}")

    report_path = args.out_dir / "corpus_report.json"
    report_path.write_text(json.dumps({
        "n_total": len(items), "n_pass": n_pass, "rows": rows,
    }, indent=2))
    print(f"\n{n_pass}/{len(items)} pass; report -> {report_path}")
    return 0 if n_pass == len(items) else 1


if __name__ == "__main__":
    sys.exit(main())
