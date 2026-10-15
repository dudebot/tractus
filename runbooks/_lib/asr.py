"""ASR backend dispatcher for the whisper_gate runbook.

Resolves a transcription backend in this order:

1. ``WHISPER_GATE_MOCK_TRANSCRIPT`` env var — if set, returned verbatim.
   Used by smoke fixtures to keep the runbook hermetic in CI.
2. ``faster_whisper`` — preferred wheel; lighter than whisperx and
   single-file inference.
3. ``whisperx``.

Raises ``RuntimeError`` if no backend is reachable. Callers should
surface the message in their failure JSON.
"""
from __future__ import annotations

import os
from pathlib import Path


def transcribe(audio_path: Path, model_size: str = "base") -> str:
    mock = os.environ.get("WHISPER_GATE_MOCK_TRANSCRIPT")
    if mock is not None:
        return mock

    # faster_whisper preferred.
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        WhisperModel = None  # type: ignore

    if WhisperModel is not None:
        # CPU + int8 is the minimum-friction default; users with
        # GPUs can wrap their own caller.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, _info = model.transcribe(str(audio_path), language="en")
        return " ".join(seg.text for seg in segments).strip()

    try:
        import whisperx  # type: ignore
    except ImportError:
        whisperx = None  # type: ignore

    if whisperx is not None:
        model = whisperx.load_model(model_size, device="cpu", compute_type="int8")
        result = model.transcribe(str(audio_path))
        return " ".join(seg["text"] for seg in result.get("segments", [])).strip()

    raise RuntimeError(
        "No ASR backend installed. Install faster_whisper or whisperx, or "
        "set WHISPER_GATE_MOCK_TRANSCRIPT for fixture testing."
    )
