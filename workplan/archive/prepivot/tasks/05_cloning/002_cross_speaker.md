# cloning/002 — Cross-speaker generalization (5 speakers, held-out, listener A/B)

**Phase:** cloning    **Status:** pending    **Effort:** M
**User-input:** review (listener A/B suite)
**Blocked by:** cloning/001
**Blocks:** cloning/005
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §1.4 (quality bar), §4. Real test of
cloning generalization. 5 reference speakers (consented, with
phonetic-coverage prompts), held-out phrases, listener A/B suite vs
commercial baseline.

## Deliverable

1. Speaker corpus: 5 consented reference speakers, each providing
   30s of prompted-text audio per `ARCHITECTURE.md` §4 phonetic
   coverage requirements.
2. Per-speaker: run cloning/001's pipeline. Produce per-speaker
   `.npz` object + 5 held-out-phrase render WAVs.
3. Listener A/B suite:
   - 5 speakers × 5 held-out phrases × 2 systems (tractus,
     commercial baseline at matched reference length).
   - Prepare blinded WAV pairs.
   - Listener test: 5 listeners, blinded single forced-choice
     ("which sounds more like the reference speaker?"). Record
     per-listener responses for the report.
4. Objective metrics:
   - Identity cosine vs reference (ECAPA-TDNN), per speaker.
   - WER vs ground-truth held-out transcript.
   - `runbooks/source_tract_sep.py` per-axis on each rendered
     held-out.
   - `runbooks/anticheat_unreachable.py` on each speaker.

## Acceptance gates (automated for objective metrics; informal for listener A/B)

- ✅ 5 speakers fit successfully (consent verified, phonetic
  coverage met, encoder + inverse refinement converge).
- ✅ Held-out WER ≤ 30% averaged across all speaker × phrase
  combinations.
- ✅ Identity cosine ≥ 0.70 averaged across speakers (lower than
  cloning/001's self-clone gate because we're now generalizing to
  speakers we never trained on).
- ✅ Anti-cheat refusal ≥ 80% on each speaker.
- ⏸️ **User review (listener A/B)**: ≥ 2 of 3 listeners can pick
  the correct speaker identity from a 4-way set including the
  reference and 2 distractors. (Generous threshold for v0; tighter
  in cloning/005's commercial comparison.)

## Anti-deliverable / scope guard

- Do NOT cherry-pick speakers. Pick 5 with diverse **vocal
  characteristics** — F0 mean (low / mid / high), tract length
  (short / mid / long), plus a mix of accents. Diversity is on
  acoustic/anatomical axes, not on demographic categories.
- Phonetic-coverage validation per ARCH §4 step 1 is a hard
  per-speaker gate; speakers failing coverage do not enter the test
  set.
- Do NOT skip the consent + phonetic-coverage validation.
- Do NOT report aggregate-only metrics; per-speaker breakdowns
  matter (an outlier failure points at body-model coverage gaps).

## Notes

This is the first "real" external eval. Outcomes here drive
decisions in pretraining/002 (do we need richer body model?),
inversion/005 (do we need richer anatomy basis?), and surrogate
fidelity tuning.
