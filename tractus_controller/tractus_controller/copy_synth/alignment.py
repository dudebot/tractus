"""MFA alignment plumbing.

Given a corpus of pre-chunked WAV + .lab pairs already aligned by MFA into
``output_format=json``, normalize each utterance into the canonical
phone-aligned schema:

    {
      "utterance_id": "<chunk_id>",
      "duration_s": float,
      "phones": [{"phoneme": str, "t_start": float, "t_end": float}, ...],
      "words":  [{"word": str, "t_start": float, "t_end": float}, ...],
      "qc": {"flags": [...], "metrics": {...}},
    }

Phones are emitted using the ``english_us_arpa`` dictionary's stressed
ARPAbet (e.g. ``AA0``, ``AA1``, ``AA2``). Stress is **lexical** (from the
dictionary), not acoustic.

Quality flags raised here:
- ``low_snr``: SNR (per MFA's ``alignment_analysis.csv``) below ``min_snr_db``.
- ``high_duration_deviation``: alignment quality score above ``max_dev``.
- ``phone_too_short``: any phone shorter than ``min_phone_s`` (HMM minimum-
  state artifact — likely the aligner squeezed past a phone the audio didn't
  realize).
- ``phone_too_long``: any phone longer than ``max_phone_s`` (alignment
  over-extended across what should be silence or another segment — suggests
  transcript mismatch in the chunk).

Codex flagged that boundary-precision is least trustworthy at stops,
affricates, and rhotics in TTS-derived audio. The ``too_short`` / ``too_long``
flags are a coarse first line of defence; finer per-phone confidence comes
from MFA's per-utterance log-likelihood and is not currently surfaced.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AlignmentMetrics:
    overall_log_likelihood: float
    speech_log_likelihood: float
    phone_duration_deviation: float
    snr_db: float


@dataclass
class NormalizedAlignment:
    utterance_id: str
    duration_s: float
    phones: list[dict]
    words: list[dict]
    metrics: AlignmentMetrics
    flags: list[str] = field(default_factory=list)


def _load_metrics(analysis_csv: Path) -> dict[str, AlignmentMetrics]:
    out: dict[str, AlignmentMetrics] = {}
    with analysis_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["file"]] = AlignmentMetrics(
                overall_log_likelihood=float(row["overall_log_likelihood"]),
                speech_log_likelihood=float(row["speech_log_likelihood"]),
                phone_duration_deviation=float(row["phone_duration_deviation"]),
                snr_db=float(row["snr"]),
            )
    return out


def _normalize_one(
    mfa_json_path: Path,
    metrics: AlignmentMetrics,
    *,
    min_phone_s: float,
    max_phone_s: float,
    min_snr_db: float,
    max_duration_deviation: float,
) -> NormalizedAlignment:
    raw = json.loads(mfa_json_path.read_text())
    utt_id = mfa_json_path.stem

    phones = [
        {"phoneme": entry[2], "t_start": float(entry[0]), "t_end": float(entry[1])}
        for entry in raw["tiers"]["phones"]["entries"]
    ]
    words = [
        {"word": entry[2], "t_start": float(entry[0]), "t_end": float(entry[1])}
        for entry in raw["tiers"]["words"]["entries"]
    ]

    flags: list[str] = []
    if metrics.snr_db < min_snr_db:
        flags.append("low_snr")
    if metrics.phone_duration_deviation > max_duration_deviation:
        flags.append("high_duration_deviation")
    short_count = sum(
        1 for p in phones if (p["t_end"] - p["t_start"]) < min_phone_s and p["phoneme"] != ""
    )
    long_count = sum(
        1 for p in phones if (p["t_end"] - p["t_start"]) > max_phone_s and p["phoneme"] != ""
    )
    if short_count:
        flags.append(f"phone_too_short:{short_count}")
    if long_count:
        flags.append(f"phone_too_long:{long_count}")

    duration = float(raw["end"]) - float(raw["start"])
    return NormalizedAlignment(
        utterance_id=utt_id,
        duration_s=duration,
        phones=phones,
        words=words,
        metrics=metrics,
        flags=flags,
    )


def normalize_corpus(
    mfa_output_dir: Path,
    out_dir: Path,
    *,
    min_phone_s: float = 0.008,
    max_phone_s: float = 0.40,
    min_snr_db: float = 12.0,
    max_duration_deviation: float = 8.0,
) -> dict:
    """Normalize all MFA JSON outputs in ``mfa_output_dir`` and write
    one ``<utt_id>_alignment.json`` per utterance into ``out_dir``.
    Also writes ``out_dir/alignments_qc.json`` summarizing flagged chunks.

    Returns the QC summary dict.
    """
    mfa_output_dir = Path(mfa_output_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    analysis_csv = mfa_output_dir / "alignment_analysis.csv"
    if not analysis_csv.exists():
        raise FileNotFoundError(f"missing {analysis_csv}")
    metrics_by_id = _load_metrics(analysis_csv)

    summary: list[dict] = []
    for jpath in sorted(mfa_output_dir.glob("*.json")):
        utt_id = jpath.stem
        if utt_id not in metrics_by_id:
            continue
        norm = _normalize_one(
            jpath,
            metrics_by_id[utt_id],
            min_phone_s=min_phone_s,
            max_phone_s=max_phone_s,
            min_snr_db=min_snr_db,
            max_duration_deviation=max_duration_deviation,
        )
        out_path = out_dir / f"{utt_id}_alignment.json"
        out_path.write_text(
            json.dumps(
                {
                    "utterance_id": norm.utterance_id,
                    "duration_s": norm.duration_s,
                    "phones": norm.phones,
                    "words": norm.words,
                    "qc": {
                        "flags": norm.flags,
                        "metrics": {
                            "overall_log_likelihood": norm.metrics.overall_log_likelihood,
                            "speech_log_likelihood": norm.metrics.speech_log_likelihood,
                            "phone_duration_deviation": norm.metrics.phone_duration_deviation,
                            "snr_db": norm.metrics.snr_db,
                        },
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        summary.append(
            {
                "utterance_id": norm.utterance_id,
                "duration_s": norm.duration_s,
                "n_phones": len(norm.phones),
                "n_words": len(norm.words),
                "flags": norm.flags,
                "snr_db": norm.metrics.snr_db,
                "phone_duration_deviation": norm.metrics.phone_duration_deviation,
            }
        )

    summary_doc = {
        "n_utterances": len(summary),
        "n_clean": sum(1 for s in summary if not s["flags"]),
        "n_flagged": sum(1 for s in summary if s["flags"]),
        "thresholds": {
            "min_phone_s": min_phone_s,
            "max_phone_s": max_phone_s,
            "min_snr_db": min_snr_db,
            "max_duration_deviation": max_duration_deviation,
        },
        "utterances": summary,
    }
    (out_dir / "alignments_qc.json").write_text(
        json.dumps(summary_doc, indent=2), encoding="utf-8"
    )
    return summary_doc
