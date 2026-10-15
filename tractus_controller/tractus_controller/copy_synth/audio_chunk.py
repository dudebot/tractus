"""Split a long monologue WAV + transcript into per-sentence utterance pairs.

Strategy
--------
1. Sentence-split the transcript on terminal punctuation.
2. RMS-frame the audio; find low-energy gaps long enough to be sentence pauses.
3. Pick the (N-1) most prominent gaps to partition the audio into N sentences,
   with a small bias toward gaps whose centers are near uniform spacing
   (proportional to per-sentence character count). This keeps us robust when
   the transcript has more or fewer sentences than there are detected gaps.
4. Emit one ``<stem>_s<NN>.wav`` + ``<stem>_s<NN>.lab`` pair per sentence in
   ``out_dir``, plus a ``<stem>_chunks.json`` manifest.

The output layout matches MFA's corpus-directory format (wav + .lab side-by-side).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import soundfile as sf


_SENT_TERMINATORS = re.compile(r"[.!?]+(?=\s+|$)")
_RUNS_OF_WS = re.compile(r"\s+")


@dataclass
class Chunk:
    chunk_id: str
    text: str
    start_s: float
    end_s: float
    wav_path: str
    lab_path: str


def split_sentences(text: str) -> list[str]:
    """Split on terminal . ! ? followed by whitespace or EOF.

    Keeps the terminator with the sentence. Collapses internal whitespace.
    """
    text = text.strip()
    if not text:
        return []
    parts: list[str] = []
    last = 0
    for m in _SENT_TERMINATORS.finditer(text):
        end = m.end()
        sent = text[last:end].strip()
        if sent:
            parts.append(_RUNS_OF_WS.sub(" ", sent))
        last = end
    tail = text[last:].strip()
    if tail:
        parts.append(_RUNS_OF_WS.sub(" ", tail))
    return parts


def rms_envelope(
    audio: np.ndarray, sr: int, frame_ms: float = 25.0, hop_ms: float = 10.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return (rms_db, frame_centers_s). audio is 1-D float."""
    win = max(1, int(round(sr * frame_ms / 1000.0)))
    hop = max(1, int(round(sr * hop_ms / 1000.0)))
    if audio.ndim != 1:
        audio = audio.mean(axis=-1)
    n_frames = max(0, 1 + (len(audio) - win) // hop)
    if n_frames <= 0:
        return np.empty(0), np.empty(0)
    starts = np.arange(n_frames) * hop
    frames = np.stack([audio[s : s + win] for s in starts])
    rms = np.sqrt(np.mean(frames * frames, axis=1) + 1e-12)
    rms_db = 20.0 * np.log10(rms + 1e-12)
    centers = (starts + win / 2.0) / sr
    return rms_db, centers


def find_silence_gaps(
    rms_db: np.ndarray,
    frame_centers_s: np.ndarray,
    min_gap_s: float = 0.30,
    db_threshold: float | None = None,
) -> list[tuple[float, float, float]]:
    """Find contiguous below-threshold runs >= min_gap_s.

    Returns list of (start_s, end_s, depth_db). ``depth_db`` is how far below
    the threshold the gap dipped on average — used to rank gap prominence.

    If ``db_threshold`` is None, choose 12 dB below the median RMS.
    """
    if rms_db.size == 0:
        return []
    if db_threshold is None:
        db_threshold = float(np.median(rms_db) - 12.0)
    below = rms_db < db_threshold
    gaps: list[tuple[float, float, float]] = []
    in_run = False
    run_start_idx = 0
    for i, b in enumerate(below):
        if b and not in_run:
            in_run = True
            run_start_idx = i
        elif not b and in_run:
            in_run = False
            gap_dur = frame_centers_s[i - 1] - frame_centers_s[run_start_idx]
            if gap_dur >= min_gap_s:
                depth = float(db_threshold - rms_db[run_start_idx:i].mean())
                gaps.append((frame_centers_s[run_start_idx], frame_centers_s[i - 1], depth))
    if in_run:
        gap_dur = frame_centers_s[-1] - frame_centers_s[run_start_idx]
        if gap_dur >= min_gap_s:
            depth = float(db_threshold - rms_db[run_start_idx:].mean())
            gaps.append((frame_centers_s[run_start_idx], frame_centers_s[-1], depth))
    return gaps


def assign_gaps_to_boundaries(
    gaps: list[tuple[float, float, float]],
    sentences: list[str],
    total_duration_s: float,
    pad_s: float = 0.10,
) -> list[tuple[float, float]]:
    """Pick (N-1) gaps as sentence boundaries given N sentences.

    Estimate ideal boundaries from per-sentence character lengths (each
    sentence's audio span is proportional to its char count) and pick
    the closest detected gap to each ideal time. Pad sentence ends/starts
    inward by ``pad_s`` so we don't clip onsets.

    Returns list of (start_s, end_s) per sentence.
    """
    n = len(sentences)
    if n <= 0:
        return []
    if n == 1:
        return [(0.0, total_duration_s)]

    char_counts = np.array([max(1, len(s)) for s in sentences], dtype=np.float64)
    cum = np.cumsum(char_counts)
    # Ideal end positions for sentences 0..n-2 (sentence n-1 ends at total).
    ideal_ends = (cum[:-1] / cum[-1]) * total_duration_s

    boundaries: list[float] = []
    if not gaps:
        boundaries = [float(x) for x in ideal_ends]
    else:
        gap_centers = sorted(float((g[0] + g[1]) / 2.0) for g in gaps)
        used = [False] * len(gap_centers)
        last = 0.0
        for ideal in ideal_ends:
            best_idx = -1
            best_dist = float("inf")
            for i, c in enumerate(gap_centers):
                if used[i]:
                    continue
                if c <= last:
                    continue
                d = abs(c - float(ideal))
                if d < best_dist:
                    best_dist = d
                    best_idx = i
            if best_idx >= 0:
                used[best_idx] = True
                last = gap_centers[best_idx]
                boundaries.append(last)
            else:
                # No remaining gap is after `last`. Pick something monotonic.
                last = max(last + 0.20, float(ideal))
                boundaries.append(last)

    # Build (start, end) intervals; pad inward.
    spans: list[tuple[float, float]] = []
    prev = 0.0
    for b in boundaries:
        start = max(0.0, prev + (pad_s if prev > 0.0 else 0.0))
        end = max(start + 0.05, b - pad_s)
        spans.append((start, end))
        prev = b
    spans.append((max(0.0, prev + pad_s), total_duration_s))
    return spans


def chunk_monologue(
    wav_path: Path,
    text: str,
    out_dir: Path,
    stem: str,
    *,
    target_sr: int | None = None,
) -> list[Chunk]:
    """Split a wav + transcript into per-sentence pairs.

    Writes ``out_dir/<stem>_s<NN>.wav`` + ``.lab`` for each sentence, plus a
    summary ``<stem>_chunks.json``. Returns the list of Chunk records.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, sr = sf.read(str(wav_path), always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=-1)
    if target_sr is not None and target_sr != sr:
        # Avoid extra deps; only resample on demand later. MFA accepts 22050.
        raise NotImplementedError(
            f"resample requested {sr} -> {target_sr} not supported here"
        )
    total_dur = len(audio) / sr

    sentences = split_sentences(text)
    if not sentences:
        raise ValueError(f"no sentences split out of transcript for {wav_path}")

    rms_db, centers = rms_envelope(audio, sr)
    gaps = find_silence_gaps(rms_db, centers)
    spans = assign_gaps_to_boundaries(gaps, sentences, total_dur)

    chunks: list[Chunk] = []
    for i, ((start_s, end_s), sent) in enumerate(zip(spans, sentences)):
        s_idx = int(round(start_s * sr))
        e_idx = int(round(end_s * sr))
        seg = audio[s_idx:e_idx]
        chunk_id = f"{stem}_s{i:02d}"
        wav_out = out_dir / f"{chunk_id}.wav"
        lab_out = out_dir / f"{chunk_id}.lab"
        sf.write(str(wav_out), seg.astype(np.float32), sr, subtype="PCM_16")
        lab_out.write_text(sent.strip() + "\n", encoding="utf-8")
        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=sent,
                start_s=float(start_s),
                end_s=float(end_s),
                wav_path=str(wav_out.relative_to(out_dir.parent)),
                lab_path=str(lab_out.relative_to(out_dir.parent)),
            )
        )

    manifest = {
        "source_wav": str(wav_path.resolve()),
        "stem": stem,
        "sample_rate": sr,
        "total_duration_s": total_dur,
        "n_sentences": len(sentences),
        "n_gaps_detected": len(gaps),
        "chunks": [asdict(c) for c in chunks],
    }
    (out_dir / f"{stem}_chunks.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return chunks
