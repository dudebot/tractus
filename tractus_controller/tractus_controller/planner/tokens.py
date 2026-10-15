"""Phone tokenizer + special tokens for the planner.

Vocabulary = ARPAbet inventory (DAISY_SUBSET, 39 phones) + control tokens
(`<pad>`, `<bos>`, `<eos>`, `<unk>`, `<sil>`, `<phrase_break>`). Stress
is carried as a separate per-token integer (0/1/2), not folded into the
vocab — that keeps the embedding matrix small and lets the model learn
stress as an additive bias.

Affect dimension is fixed at 8 per task spec; unlabeled samples pass a
zero vector. Any per-corpus emotion/register/intent label is projected
through ``AffectEmbedder`` (in ``model.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..phonemes.arpabet import DAISY_SUBSET, split_stress

# ---- vocabulary --------------------------------------------------------

_SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>", "<sil>", "<phrase_break>"]

PHONE_VOCAB: tuple[str, ...] = tuple(_SPECIAL) + tuple(sorted(DAISY_SUBSET))
"""Ordered phone vocabulary; index 0 reserved for <pad>."""

PAD_ID: int = PHONE_VOCAB.index("<pad>")
BOS_ID: int = PHONE_VOCAB.index("<bos>")
EOS_ID: int = PHONE_VOCAB.index("<eos>")
UNK_ID: int = PHONE_VOCAB.index("<unk>")
SIL_ID: int = PHONE_VOCAB.index("<sil>")
PHRASE_BREAK_ID: int = PHONE_VOCAB.index("<phrase_break>")

VOCAB_SIZE: int = len(PHONE_VOCAB)

AFFECT_DIM: int = 8
"""D_affect (task spec): 8-dim affect projection. Unlabeled → zeros."""

STRESS_LEVELS: int = 3  # 0 = unstressed/none, 1 = primary, 2 = secondary


# ---- tokenizer ---------------------------------------------------------


@dataclass(frozen=True)
class PhoneTokenizer:
    """Map ARPAbet phone strings (with optional stress digits) to ids.

    The tokenizer is stateless and intentionally tiny — the inventory is
    fixed at module-import time. ``encode`` ignores commas (consistent
    with ``arpabet.parse_arpabet_string``) and drops empty tokens.
    """

    add_bos: bool = True
    add_eos: bool = True

    def encode(self, tokens: list[str]) -> tuple[list[int], list[int]]:
        """Return ``(ids, stress)``.

        ``ids[i]`` is the vocab index. ``stress[i] ∈ {0, 1, 2}``. BOS/EOS
        get stress=0. Unknown phones map to ``<unk>`` and stress=0.
        """
        ids: list[int] = []
        stress: list[int] = []
        if self.add_bos:
            ids.append(BOS_ID)
            stress.append(0)
        for raw in tokens:
            phone, s = split_stress(raw)
            if phone in DAISY_SUBSET:
                ids.append(PHONE_VOCAB.index(phone))
            elif phone == "<sil>":
                ids.append(SIL_ID)
            elif phone == "<phrase_break>":
                ids.append(PHRASE_BREAK_ID)
            else:
                ids.append(UNK_ID)
            try:
                stress.append(int(s)) if s.isdigit() else stress.append(0)
            except ValueError:
                stress.append(0)
        if self.add_eos:
            ids.append(EOS_ID)
            stress.append(0)
        return ids, stress

    def decode(self, ids: list[int]) -> list[str]:
        return [PHONE_VOCAB[i] for i in ids]


def pad_sequence(seqs: list[list[int]], pad_value: int = PAD_ID) -> list[list[int]]:
    """Right-pad a list of int sequences to a common length. In-place safe."""
    if not seqs:
        return []
    L = max(len(s) for s in seqs)
    return [s + [pad_value] * (L - len(s)) for s in seqs]


def is_phrase_boundary_id(idx: int) -> bool:
    return idx == PHRASE_BREAK_ID
