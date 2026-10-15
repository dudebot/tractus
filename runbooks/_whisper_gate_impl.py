"""whisper_gate Python implementation. Called from whisper_gate.sh."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Local imports relative to the runbooks/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _lib.asr import transcribe
from _lib.json_io import emit
from _lib.text_metrics import jaccard, wer


def main() -> int:
    p = argparse.ArgumentParser(prog="whisper_gate")
    p.add_argument("--audio", required=True, type=Path)
    p.add_argument("--target", required=True, type=str)
    p.add_argument("--threshold-wer", type=float, default=0.25)
    p.add_argument("--threshold-jaccard", type=float, default=0.30)
    p.add_argument("--model", type=str, default="base")
    args = p.parse_args()

    schema = Path(__file__).resolve().parent / "schemas" / "whisper_gate.schema.json"

    if not args.audio.exists():
        report = {
            "wer": 1.0,
            "jaccard": 0.0,
            "asr_text": "",
            "pass": False,
            "error": f"audio file not found: {args.audio}",
        }
        emit(report, f"FAIL whisper_gate: missing audio {args.audio}", schema)
        return 1

    try:
        asr_text = transcribe(args.audio, args.model)
    except RuntimeError as exc:
        report = {
            "wer": 1.0,
            "jaccard": 0.0,
            "asr_text": "",
            "pass": False,
            "error": str(exc),
        }
        emit(report, f"FAIL whisper_gate: ASR backend unavailable: {exc}", schema)
        return 1

    w = wer(args.target, asr_text)
    j = jaccard(args.target, asr_text)
    passed = (w < args.threshold_wer) and (j > args.threshold_jaccard)

    report = {
        "wer": float(w),
        "jaccard": float(j),
        "asr_text": asr_text,
        "pass": bool(passed),
        "threshold_wer": float(args.threshold_wer),
        "threshold_jaccard": float(args.threshold_jaccard),
    }
    if passed:
        emit(report, f"PASS whisper_gate: WER={w:.3f} jaccard={j:.3f}", schema)
        return 0
    failed_gate = []
    if not (w < args.threshold_wer):
        failed_gate.append(f"WER {w:.3f} >= {args.threshold_wer}")
    if not (j > args.threshold_jaccard):
        failed_gate.append(f"jaccard {j:.3f} <= {args.threshold_jaccard}")
    emit(report, "FAIL whisper_gate: " + "; ".join(failed_gate), schema)
    return 1


if __name__ == "__main__":
    sys.exit(main())
