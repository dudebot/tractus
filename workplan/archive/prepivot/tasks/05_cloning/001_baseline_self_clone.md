# cloning/001 — Default-baseline self-clone end-to-end smoke

**Phase:** cloning    **Status:** pending    **Effort:** S
**User-input:** none
**Blocked by:** P3 closed
**Blocks:** cloning/002
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §4 (reference-fit protocol). End-to-end
smoke for the cloning pipeline. Use default-baseline-rendered audio
as the "30s reference," fit Module 5 + motor adapter, render
held-out text in the resulting "speaker" voice. Verify the system
reproduces the default baseline correctly.

This is "does the cloning pipeline complete end-to-end?" before
attempting real speakers in cloning/002.

## Deliverable

1. CLI: `tractus-controller clone --ref-audio <30s.wav>
   --eval-text "..." --out <path>`. Add a pointer to the new
   `clone` verb in ARCH §8.2 in the same commit.
2. Pipeline:
   - Capture 30s of default-baseline-rendered audio with the
     phonetic-coverage prompt set per `ARCHITECTURE.md` §4 step 1.
   - Encoder bootstrap (Module 5).
   - Anatomy / source / channel inverse refinement on fit set
     (inversion/001's loop).
   - Motor-style adapter fit.
   - Eval render: held-out text → planner → controller (with motor
     adapter) → body model → TEA → audio.
3. Report:
   - Held-out reconstruction log-mel distance.
   - Identity (ECAPA-TDNN cosine) vs reference.
   - Per-axis source/tract separation.
   - WER + jaccard on a 5-utterance held-out test.

## Acceptance gates (automated)

- ✅ Pipeline completes without exception.
- ✅ Held-out log-mel L1 ≤ 1.5 dB vs reference.
- ✅ Identity cosine ≥ 0.85 (we're cloning ourselves; should be
  high).
- ✅ Held-out WER ≤ 25%, jaccard ≥ 0.40.
- ✅ Anti-cheat refusal rate ≥ 80% on a 10-target unreachable set.
  Residual + `wall_dynamic_residual` + `frication_color` are all
  asserted disabled for self-clone (no cheat-channel use).
- ✅ Total wall-clock for the 30s-→clone pipeline ≤ **15 min** on
  consumer GPU. "Consumer GPU" = RTX 3060 / 4060 8 GB class.
- ✅ Per-speaker output object size ≤ 50 KB.

## Anti-deliverable / scope guard

- Do NOT skip held-out evaluation in favor of fit-set.
- Do NOT enable `wall_dynamic_residual` or `frication_color` to
  chase quality numbers. Anti-cheat-gated.

## Notes

If this fails at the smoke level, something architectural is wrong
upstream. Don't paper over with hyperparameter tuning; escalate to
the validator session for diagnosis.
