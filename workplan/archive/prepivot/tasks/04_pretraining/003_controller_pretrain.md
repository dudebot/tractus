# pretraining/003 — Controller pretrain

**Phase:** pretraining    **Status:** pending    **Effort:** M
**User-input:** none
**Blocked by:** inversion/006 (D_body_effective for body model output
contract); can run parallel with pretraining/002 once both have
their input contracts.
**Blocks:** pretraining/004
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §2.2. Module 2 emits **asynchronous
overlapping `GestureEvent`** objects (NOT fixed-rate knots). ~12
articulator + ~6–8 source channels. Average density ~30 events/s.
Conditioned on speaker motor-style embedding (from Module 5) +
affect (from Module 1). Size target ~5 M params.

## Deliverable

1. Architecture: small autoregressive transformer OR NAT-style
   duration-conditioned conv-transformer that emits a list of
   `GestureEvent` tuples.
2. Training corpus:
   - Synthetic gesture-event sequences from procedural templates
     (one per phone, then perturbed for coarticulation).
   - Inversion/005-fitted gesture sequences (real-distribution).
   - For each: paired (text + prosody plan, gesture-event list).
3. Loss:
   - Gesture-knot supervision (MSE on target_value) when GT fits
     exist.
   - Acoustic loss through **short-window TEA** (≤ 0.5s,
     gradient-checkpointed) on small batches. Per ARCH §888 this is
     the canonical training-time gradient path; surrogate is
     reserved for pretraining/005 and is not used here.
   - Smoothness penalty in continuous-time units.
   - Phone-contrast margin loss between confusable pairs:
     /i/ vs /ɪ/, /ɛ/ vs /æ/, /u/ vs /ʊ/, /s/ vs /ʃ/, /b/ vs /p/,
     /d/ vs /t/, /g/ vs /k/, /m/ vs /n/, /f/ vs /θ/, /v/ vs /ð/, plus
     a curated 20-pair set of confusable English minimal pairs.
   - F0/energy consistency vs planner targets.
4. Affect propagation: gesture distributions biased by affect
   embedding (gesture duration, strength, adduction, breathiness,
   etc., per `ARCHITECTURE.md` §2.2).
5. Eval:
   - `runbooks/plausibility_panel.py` on emitted gesture
     trajectories.
   - `runbooks/source_tract_sep.py` on rendered audio.
   - Phone-contrast accuracy on a small confusable-pair test set.

## Acceptance gates (automated)

- ✅ Plausibility panel passes on 95% of generated utterances.
- ✅ Phone-contrast accuracy ≥ 90% on 20-pair test set.
- ✅ `runbooks/source_tract_sep.py` per-axis thresholds pass on
  90% of generated utterances at this stage. (Final tightening
  comes in cloning/* phases.)
- ✅ Model size ≤ 5 M params.
- ✅ Affect-conditioning ablation: rendering with vs without
  affect embedding produces a log-mel L1 difference ≥ 0.5 dB
  averaged over the test set (audibly different) while keeping
  WER within 5 pp of the no-affect baseline (intelligibility
  preserved).
- ✅ CI guardrail (infra/002) passes: no full-utterance backprop
  through TEA in the training loop.

## Anti-deliverable / scope guard

- Do NOT switch back to fixed-rate knots. The async GestureEvent
  representation is the architectural choice; if it doesn't train
  well, fix the training, don't switch the representation.
- Do NOT use the surrogate (5A) for the primary acoustic loss
  here. TEA short-window IS the canonical gradient path per ARCH
  §888; surrogate replacement is the pretraining/005 deliverable.

## Notes

Pairs naturally with pretraining/002 — body model and controller
co-evolve. Recommend training them in lockstep with shared anatomy
prior + shared regression set.
