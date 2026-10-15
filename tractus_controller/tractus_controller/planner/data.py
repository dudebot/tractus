"""Dataset assembly for the planner.

Two ingestion paths:

1. **Real corpus path** (LibriTTS-R / VCTK + MFA TextGrid + PyWorld F0).
   ``CorpusManifest`` reads a JSONL of ``{wav, textgrid, transcript,
   speaker_id, f0_range_hz, affect}`` entries and produces ``Sample``
   tensors. F0 + loudness are computed lazily from the waveform.

2. **Synthetic path** (``synthetic_dataset``). Generates plausible
   phone sequences from the supported ARPAbet inventory with random
   durations, sinusoidal F0 contours, and random affect vectors.
   Smokes the full loss path without external data — used by
   ``tests/test_planner.py`` and the in-tree smoke training run. NOT a
   substitute for real-corpus training; documented as such.

Both paths emit ``CollatedBatch`` (token + stress + affect + f0_range +
duration_frames + log_f0 + voiced + loud_db + phone_mask + phrase_break).
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from ..phonemes.arpabet import DAISY_SUBSET
from .f0 import DEFAULT_FRAME_PERIOD_MS, extract_f0
from .mfa import parse_textgrid, phrase_break_mask
from .tokens import (
    AFFECT_DIM,
    PAD_ID,
    PHONE_VOCAB,
    PhoneTokenizer,
    SIL_ID,
    pad_sequence,
)


# ---- shared dataclasses ---------------------------------------------------


@dataclass
class Sample:
    """Per-utterance planner sample (CPU tensors / numpy arrays)."""

    token_ids: list[int]
    stress: list[int]
    durations_frames: list[int]  # frames at 100 Hz
    log_f0: np.ndarray
    voiced: np.ndarray
    loud_db: np.ndarray
    phrase_break: list[bool]
    affect: np.ndarray  # [AFFECT_DIM]
    speaker_f0_range_hz: tuple[float, float]


@dataclass
class CollatedBatch:
    token_ids: torch.Tensor
    stress: torch.Tensor
    affect: torch.Tensor
    speaker_f0_range_hz: torch.Tensor
    durations_frames: torch.Tensor
    log_dur_target: torch.Tensor
    phone_mask: torch.Tensor
    phrase_break_target: torch.Tensor
    log_f0_target: torch.Tensor
    voiced_target: torch.Tensor
    loud_db_target: torch.Tensor
    frame_mask: torch.Tensor

    def to(self, device: torch.device | str) -> "CollatedBatch":
        return CollatedBatch(
            **{k: v.to(device) for k, v in self.__dict__.items()}
        )


# ---- collation ------------------------------------------------------------


def collate(samples: list[Sample], log_dur_eps: float = 1e-3) -> CollatedBatch:
    """Pad a batch of variable-length samples into a CollatedBatch."""
    L = max(len(s.token_ids) for s in samples)
    T = max(len(s.log_f0) for s in samples)
    B = len(samples)
    token_ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    stress = torch.zeros((B, L), dtype=torch.long)
    durations_frames = torch.zeros((B, L), dtype=torch.long)
    log_dur_target = torch.zeros((B, L), dtype=torch.float32)
    phone_mask = torch.zeros((B, L), dtype=torch.bool)
    phrase_break_target = torch.zeros((B, L), dtype=torch.float32)
    log_f0_target = torch.zeros((B, T), dtype=torch.float32)
    voiced_target = torch.zeros((B, T), dtype=torch.float32)
    loud_db_target = torch.full((B, T), -120.0, dtype=torch.float32)
    frame_mask = torch.zeros((B, T), dtype=torch.bool)
    affect = torch.zeros((B, AFFECT_DIM), dtype=torch.float32)
    f0_range = torch.zeros((B, 2), dtype=torch.float32)
    for b, s in enumerate(samples):
        Lp = len(s.token_ids)
        token_ids[b, :Lp] = torch.tensor(s.token_ids, dtype=torch.long)
        stress[b, :Lp] = torch.tensor(s.stress, dtype=torch.long)
        durs_t = torch.tensor(s.durations_frames, dtype=torch.long)
        durations_frames[b, :Lp] = durs_t
        # log-duration target on seconds
        dur_s = durs_t.float() / 100.0
        log_dur_target[b, :Lp] = torch.log(dur_s + log_dur_eps)
        phone_mask[b, :Lp] = True
        phrase_break_target[b, :Lp] = torch.tensor(s.phrase_break, dtype=torch.float32)
        Tf = len(s.log_f0)
        log_f0_target[b, :Tf] = torch.tensor(s.log_f0, dtype=torch.float32)
        voiced_target[b, :Tf] = torch.tensor(s.voiced.astype(np.float32))
        loud_db_target[b, :Tf] = torch.tensor(s.loud_db, dtype=torch.float32)
        frame_mask[b, :Tf] = True
        affect[b] = torch.tensor(s.affect, dtype=torch.float32)
        f0_range[b] = torch.tensor(s.speaker_f0_range_hz, dtype=torch.float32)
    return CollatedBatch(
        token_ids=token_ids,
        stress=stress,
        affect=affect,
        speaker_f0_range_hz=f0_range,
        durations_frames=durations_frames,
        log_dur_target=log_dur_target,
        phone_mask=phone_mask,
        phrase_break_target=phrase_break_target,
        log_f0_target=log_f0_target,
        voiced_target=voiced_target,
        loud_db_target=loud_db_target,
        frame_mask=frame_mask,
    )


# ---- real-corpus loader ---------------------------------------------------


@dataclass
class CorpusEntry:
    wav: str
    textgrid: str
    transcript: str
    speaker_id: str
    f0_range_hz: tuple[float, float] = (75.0, 350.0)
    affect: list[float] = field(default_factory=lambda: [0.0] * AFFECT_DIM)


def load_manifest(path: str) -> list[CorpusEntry]:
    """Read a JSONL manifest of corpus entries.

    Per ARCH consent/provenance rule (and Codex C #9), every entry MUST
    carry ``consent_verified: true``. Reuse the shared ``manifest_natural``
    schema rather than rolling our own loose loader. The planner-specific
    fields (``textgrid``, ``f0_range_hz``, ``affect``, ``transcript``)
    travel through ``manifest_natural``'s ``extra`` payload, which is
    preserved opaquely.
    """
    from tractus_shared.manifest_natural import read as _read

    entries: list[CorpusEntry] = []
    for raw in _read(path):  # raises ManifestError on missing/false consent
        extra = raw.extra or {}
        # Planner-required fields. ``audio_path`` is the manifest's wav.
        textgrid = extra.get("textgrid")
        if not textgrid:
            raise ValueError(
                f"manifest entry for {raw.audio_path!r} missing 'textgrid' "
                "(planner needs MFA-aligned phone durations)"
            )
        entries.append(
            CorpusEntry(
                wav=raw.audio_path,
                textgrid=str(textgrid),
                transcript=raw.text,
                speaker_id=raw.speaker_id or "unk",
                f0_range_hz=tuple(extra.get("f0_range_hz", [75.0, 350.0])),
                affect=list(extra.get("affect", [0.0] * AFFECT_DIM)),
            )
        )
    return entries


def load_sample_from_entry(entry: CorpusEntry, tokenizer: PhoneTokenizer) -> Sample:
    """Materialise a Sample from a CorpusEntry. Reads wav + TextGrid."""
    import soundfile as sf  # type: ignore

    utt = parse_textgrid(entry.textgrid)
    # MFA labels map directly to ARPAbet (English models). Drop stress,
    # since MFA dictionaries usually omit it; defer stress supervision
    # to a later pass where we cross-reference CMUdict.
    raw_phones = [p.label.upper() for p in utt.phones]
    # Replace MFA's silence labels (usually "" → "<sil>" via parser, or
    # "spn"/"sp" for spoken/short pauses) with our <sil> token.
    norm = []
    for p in raw_phones:
        if p in ("", "<SIL>", "SIL", "SP", "SPN"):
            norm.append("<sil>")
        elif p in DAISY_SUBSET:
            norm.append(p)
        else:
            norm.append("<unk>")  # downstream embedding will absorb it
    # Wrap with BOS/EOS in the tokenizer; durations for those are 0.
    ids, stress = tokenizer.encode(norm)
    durs_s = [p.dur_s for p in utt.phones]
    # BOS/EOS get 0-duration slots:
    pad_dur_s: list[float] = []
    if tokenizer.add_bos:
        pad_dur_s.append(0.0)
    pad_dur_s.extend(durs_s)
    if tokenizer.add_eos:
        pad_dur_s.append(0.0)
    durations_frames = [int(round(d * 100.0)) for d in pad_dur_s]
    # phrase-break mask, padded with False for BOS/EOS
    pb = phrase_break_mask(utt)
    pad_pb = [False] * (1 if tokenizer.add_bos else 0) + pb + [False] * (1 if tokenizer.add_eos else 0)
    # F0 + loudness from the waveform
    wav, sr = sf.read(entry.wav, dtype="float64", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    f0 = extract_f0(wav, sr)
    return Sample(
        token_ids=ids,
        stress=stress,
        durations_frames=durations_frames,
        log_f0=f0.log_hz,
        voiced=f0.voiced,
        loud_db=f0.loud_db,
        phrase_break=pad_pb,
        affect=np.asarray(entry.affect, dtype=np.float32),
        speaker_f0_range_hz=tuple(entry.f0_range_hz),
    )


# ---- synthetic generator (for the smoke run) ------------------------------


def _supported_phones() -> list[str]:
    return list(sorted(DAISY_SUBSET))


def synthetic_sample(
    rng: random.Random,
    n_phones: int = 24,
    speaker_f0_range_hz: tuple[float, float] | None = None,
    affect_dim: int = AFFECT_DIM,
) -> Sample:
    """One synthetic planner sample. Roughly utterance-shaped:

    - 20–40 phones from the supported inventory + occasional <sil>.
    - Per-phone durations 60–250 ms (vowels longer); 100 Hz frame grid.
    - Log-F0 contour: speaker-mean (in log-Hz) ± 0.15 sin-modulation.
    - Voiced mask: vowels + voiced consonants.
    - Loudness: -25 dBFS sustain ± 5 dB random walk.
    - Phrase break ~ every 8–12 phones.
    """
    phones_inv = _supported_phones()
    tokens: list[str] = []
    durations_s: list[float] = []
    voiced_per_phone: list[bool] = []
    pb_per_phone: list[bool] = []
    next_break_at = rng.randint(8, 12)
    for i in range(n_phones):
        # Occasional silence
        if i > 0 and rng.random() < 0.05:
            tokens.append("<sil>")
            durations_s.append(rng.uniform(0.05, 0.15))
            voiced_per_phone.append(False)
            pb_per_phone.append(False)
            continue
        ph = rng.choice(phones_inv)
        # primary stress on a few vowels
        if DAISY_SUBSET[ph].phone_class == "vowel" and rng.random() < 0.3:
            ph = ph + "1"
        tokens.append(ph)
        cls = DAISY_SUBSET[ph[:2] if ph[-1].isdigit() else ph].phone_class
        if cls == "vowel":
            d = rng.uniform(0.09, 0.20)
        elif cls == "stop":
            d = rng.uniform(0.05, 0.09)
        elif cls == "fricative":
            d = rng.uniform(0.07, 0.13)
        else:
            d = rng.uniform(0.06, 0.10)
        durations_s.append(d)
        voiced_per_phone.append(DAISY_SUBSET[ph[:2] if ph[-1].isdigit() else ph].voiced)
        pb_per_phone.append((i + 1) >= next_break_at)
        if (i + 1) >= next_break_at:
            next_break_at = i + 1 + rng.randint(8, 12)

    tokenizer = PhoneTokenizer()
    ids, stress = tokenizer.encode(tokens)
    durs_f = [int(round(d * 100.0)) for d in durations_s]
    if tokenizer.add_bos:
        durs_f = [0] + durs_f
        voiced_per_phone = [False] + voiced_per_phone
        pb_per_phone = [False] + pb_per_phone
    if tokenizer.add_eos:
        durs_f = durs_f + [0]
        voiced_per_phone = voiced_per_phone + [False]
        pb_per_phone = pb_per_phone + [False]
    total_frames = max(int(sum(durs_f)), 1)
    # Build per-frame log_f0/voiced/loud_db
    if speaker_f0_range_hz is None:
        # broadly scatter male/female ranges
        if rng.random() < 0.5:
            speaker_f0_range_hz = (75.0, 200.0)
            f0_mean_hz = rng.uniform(95.0, 145.0)
        else:
            speaker_f0_range_hz = (130.0, 350.0)
            f0_mean_hz = rng.uniform(170.0, 260.0)
    else:
        lo, hi = speaker_f0_range_hz
        f0_mean_hz = (lo + hi) * 0.5
    log_f0 = np.zeros(total_frames, dtype=np.float32)
    voiced = np.zeros(total_frames, dtype=bool)
    loud_db = np.full(total_frames, -25.0, dtype=np.float32)
    cursor = 0
    log_mean = math.log(f0_mean_hz)
    for ph_idx, df in enumerate(durs_f):
        if df <= 0:
            continue
        end = min(cursor + df, total_frames)
        if voiced_per_phone[ph_idx]:
            t = np.arange(end - cursor, dtype=np.float32) / 100.0
            log_f0[cursor:end] = log_mean + 0.15 * np.sin(2 * math.pi * 0.7 * t).astype(np.float32)
            voiced[cursor:end] = True
        # loud random walk
        loud_db[cursor:end] = -25.0 + 5.0 * rng.gauss(0.0, 1.0)
        cursor = end
    affect = np.zeros(affect_dim, dtype=np.float32)
    if rng.random() < 0.3:
        # random "labelled" emotion vector — sparse, mean-zero
        affect = np.array(
            [rng.gauss(0.0, 0.5) for _ in range(affect_dim)], dtype=np.float32
        )
    return Sample(
        token_ids=ids,
        stress=stress,
        durations_frames=durs_f,
        log_f0=log_f0,
        voiced=voiced,
        loud_db=loud_db,
        phrase_break=pb_per_phone,
        affect=affect,
        speaker_f0_range_hz=speaker_f0_range_hz,
    )


def synthetic_dataset(
    n: int,
    seed: int = 0,
    n_phones_range: tuple[int, int] = (20, 40),
) -> list[Sample]:
    """``n`` independent synthetic samples; deterministic given ``seed``."""
    rng = random.Random(seed)
    out: list[Sample] = []
    for _ in range(n):
        n_ph = rng.randint(*n_phones_range)
        out.append(synthetic_sample(rng, n_phones=n_ph))
    return out
