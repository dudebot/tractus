# pretraining/001 — Planner pretrain

**Phase:** pretraining    **Status:** pending    **Effort:** M
**User-input:** review (corpus mix sign-off)
**Blocked by:** — (Day-1 parallel per INDEX dispatch; sees no audio
loss so it's orthogonal to TEA correctness work).
**Blocks:** P3 close
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §2.1. Module 1 is the **shared
linguistic / prosody planner**: text → phones, durations, stress,
phrase boundaries, target F0/loudness envelopes, affect-conditioned
prosody plan. Speaker-agnostic (only sees F0 *range*, not contour).
Size target ≤ 10 M params.

This task can run **in parallel with TEA-correctness work** because
it sees no audio loss — pure text + alignment + F0 corpora.

## Deliverable

1. Corpus assembly:
   - LibriTTS-R or VCTK as primary text-audio corpus (license
     review noted in INFRA queue).
   - MFA-aligned phone-level durations.
   - PyWorld-DIO F0 contours.
   - (Optional) loudness envelopes.
2. Architecture: small transformer encoder–decoder, ≤ 10 M params.
   (NAT-style duration predictor + flow F0 head is the alternative;
   pick one for v0 — comparative-architecture work is out of scope.)
3. Training:
   - Duration regression (Huber on log-dur).
   - F0 regression (MSE on log-Hz, voiced-mask weighted).
   - Loudness regression.
   - Phrase-break classification.
   - Alignment loss against MFA targets.
4. Affect conditioning input (first-class — see ARCHITECTURE.md
   §2.1 interface). v0: `D_affect = 8` (e.g., a learned 8-dim
   projection of any available emotion/register/intent labels;
   unlabeled samples → zero vector = "neutral"). Wire the input
   dimension regardless of corpus label availability.
5. Eval: held-out test split. Per-phone duration RMSE,
   F0 cents-RMSE on voiced segments, loudness MSE,
   phrase-break F1.

## Acceptance gates (automated)

- ✅ Phone-duration RMSE ≤ 25 ms on held-out.
- ✅ F0 cents-RMSE ≤ 50 cents on voiced.
- ✅ Phrase-break F1 ≥ 0.85.
- ✅ Model size ≤ 10 M params.
- ✅ Inference: text → planner output for a 5-second utterance in
  ≤ 100 ms on consumer GPU.
- ⏸️ **User review**: corpus mix sign-off (LibriTTS-R vs VCTK vs
  Common Voice mixture). One-time decision; cosmetic going forward.

## Anti-deliverable / scope guard

- Do NOT make this an 80M Kokoro-class model. Target ≤ 10 M; we'll
  scale up post-Phase-4 if quality demands it.
- Do NOT train against audio loss in this module. Planner emits
  symbolic plan; audio comes from the renderer downstream. Zero
  audio gradients flow to Module 1.
- Do NOT condition on speaker_embed. Module 1 is shared across
  speakers by design.

## Notes

If the user picks "borrow Misaki/Kokoro G2P + duration heads as
warm-start," that's fine and likely faster than from-scratch
training. Document the choice.
