"""Word-level WER and Jaccard similarity for runbook ASR gates.

Standalone — no imports from tractus_controller. Implementation
follows the standard Levenshtein-on-word-tokens definition so the
runbook contract stays portable across environments.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable


_PUNCT_RE = re.compile(r"[^\w\s']", flags=re.UNICODE)


def normalize_text(s: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace, drop punctuation."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = _PUNCT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokenize(s: str) -> list[str]:
    return normalize_text(s).split()


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate. Returns 0.0 when reference is empty AND
    hypothesis is empty; 1.0 when reference is empty but hypothesis
    is non-empty (insertions only)."""
    ref_words = tokenize(reference)
    hyp_words = tokenize(hypothesis)
    if not ref_words and not hyp_words:
        return 0.0
    if not ref_words:
        return 1.0
    n, m = len(ref_words), len(hyp_words)
    # Levenshtein on words.
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == hyp_words[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[n][m] / float(n)


def jaccard(reference: str, hypothesis: str) -> float:
    """Word-set Jaccard. Empty / empty = 1.0; either empty alone = 0.0."""
    a = set(tokenize(reference))
    b = set(tokenize(hypothesis))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))
