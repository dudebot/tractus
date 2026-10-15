"""Montreal Forced Aligner (MFA) TextGrid → planner targets.

MFA produces a TextGrid with a ``phones`` tier containing intervals of
the form ``(start_s, end_s, label)``. We map labels back to ARPAbet
(MFA's English models use ARPAbet without stress digits in some
dictionaries; we accept both) and emit ``(token_ids, durations_s,
phrase_break_mask)``.

Phrase boundaries are derived from word-tier silences ≥ 200 ms — a
defensible heuristic; deeper prosody-break supervision can land later.
"""

from __future__ import annotations

from dataclasses import dataclass

PHRASE_BREAK_MIN_SILENCE_S: float = 0.20


@dataclass(frozen=True)
class MfaPhone:
    label: str
    start_s: float
    end_s: float

    @property
    def dur_s(self) -> float:
        return self.end_s - self.start_s


@dataclass(frozen=True)
class MfaUtterance:
    phones: list[MfaPhone]
    word_silences: list[tuple[float, float]]  # (start, end) for sil between words


def parse_textgrid(path: str) -> MfaUtterance:
    """Parse an MFA TextGrid, returning the phone intervals + word silences."""
    try:
        import textgrid  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "textgrid required to parse MFA output; "
            "install with `pip install textgrid`."
        ) from e
    tg = textgrid.TextGrid.fromFile(path)
    phone_tier = None
    word_tier = None
    for tier in tg.tiers:
        name = tier.name.lower()
        if "phone" in name:
            phone_tier = tier
        elif "word" in name:
            word_tier = tier
    if phone_tier is None:
        raise ValueError(f"{path}: no phone tier found")
    phones: list[MfaPhone] = []
    for iv in phone_tier:
        label = (iv.mark or "").strip()
        if not label:
            label = "<sil>"
        phones.append(MfaPhone(label=label, start_s=float(iv.minTime), end_s=float(iv.maxTime)))
    silences: list[tuple[float, float]] = []
    if word_tier is not None:
        for iv in word_tier:
            if not (iv.mark or "").strip():
                silences.append((float(iv.minTime), float(iv.maxTime)))
    return MfaUtterance(phones=phones, word_silences=silences)


def phrase_break_mask(utt: MfaUtterance) -> list[bool]:
    """One bool per phone: True if a phrase boundary follows this phone.

    A boundary follows a phone iff a word-tier silence ≥
    ``PHRASE_BREAK_MIN_SILENCE_S`` begins within 1 ms of that phone's end
    time.
    """
    out: list[bool] = [False] * len(utt.phones)
    for s_start, s_end in utt.word_silences:
        if (s_end - s_start) < PHRASE_BREAK_MIN_SILENCE_S:
            continue
        for i, ph in enumerate(utt.phones):
            if abs(ph.end_s - s_start) <= 1e-3:
                out[i] = True
                break
    return out
