# research/002 — Anatomy basis enrichment beyond 30–60

**Phase:** research    **Status:** pending    **Effort:** M
**User-input:** none
**Blocked by:** P4 closed; cloning/005 results inform whether this is needed
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §0 risk #4. v1 anatomy is 30–60 scalars
(inversion/005). If cloning/005 shows speaker identity plateaus
short of target quality and the cluster analysis from inversion/005
suggests we're hitting basis-rank limitations (anatomies of
distinct speakers compress onto the same axis), enriching beyond
60 is the next move.

## Deliverable

1. Cluster analysis on cloning/002's 5-speaker fits: are anatomies
   well-separated in basis space, or are they getting projected
   onto a low-rank subspace?
2. If basis-limited: extend to 60–120 scalars. Candidates:
   - Higher-rank area residual (rank 32 → 64).
   - Fine-grained source/tissue (e.g. per-section glottal pulse
     asymmetry).
   - Frication-color basis per articulator class.
3. Re-run inversion/005 + cloning/002 with extended basis.
4. Compare: held-out reconstruction, identity cosine,
   anti-cheat refusal.

## Acceptance gates (automated)

- ✅ Cluster analysis report produced.
- ✅ If basis-limited diagnosed: extended basis trained.
- ✅ Held-out log-mel improves vs 30–60 baseline.
- ✅ Anti-cheat refusal stays ≥ 80% — extended basis is not
  making cheat-channels easier.
- ✅ Identity cosine improves on 5-speaker test set.

## Anti-deliverable / scope guard

- Do NOT extend basis without cluster-analysis evidence. "More
  parameters" is not the answer to every quality gap.
- Do NOT exceed 120 scalars in this task. Above 120 is a separate
  decision-point task with explicit user review (free-residual
  identifiability failure mode territory); INDEX records this task
  with `User-input: none` because the cap stays at 120 here.

## Notes

Pairs with research/001 logically: if (b) is the right fork but
the basis is the bottleneck, this task fixes it without the
multi-month procedural investment.
