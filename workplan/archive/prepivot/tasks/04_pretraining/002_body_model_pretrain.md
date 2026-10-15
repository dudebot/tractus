# pretraining/002 — Body model pretrain (anchor-grounded)

**Phase:** pretraining    **Status:** pending
**Effort:** L
**User-input:** none
**Blocked by:** pretraining/002a (Story anchors), pretraining/002b
(training infra precursor); inversion/006 latent-dim decision is a soft
prerequisite — start with D_body=64 placeholder and refactor later.
**Blocks:** pretraining/004
**How to dispatch:** fresh Claude Code session, after 002a+002b land

**Re-scope note (2026-05-08):** the original plan assumed the 40%
"from inversion/005-fitted speakers" corpus slice would be available
at dispatch time. It isn't. Per Codex review (thread
`[session identifier omitted]`), the 60% / 30% / 10%
synthetic-only mix with R1 NN as teacher does NOT meet this task's
intent — it produces a smoother R1, not a better body model. That
work is now tracked separately as `pretraining/002b` (R1-distill
baseline; useful as training infra). This task — pretraining/002 —
is reserved for the **anchor-grounded** body model that uses real
articulatory data (Story 1996 via 002a, plus optionally SPARC via 006,
plus inversion-fitted speakers when available).

---

## Context

`../../../ARCHITECTURE.md` §2.3 (Module 3 body model, learned
variant (b) for v0). The v0 body model maps composed `GestureEvent`
sets + anatomy → body-state latent at `D_body_effective` (chosen by
inversion/006) → expanded TEA renderer interface (~500–600 fields,
mostly derived).

The R1 NN existing in `tractus_controller/adapters/r1_nn.py` is the
v0 baseline; this task scales it to the full body-model contract +
trains on a much larger corpus.

**Non-circularity rule (load-bearing):** body model trained on
**explicit gesture/body targets or physically constrained synthetic
generation**, NEVER on TEA-rendered audio loss.

## Deliverable

0. Prerequisite: any controller-side data tooling that this task
   relies on must already be aligned with the GestureEvent contract
   (ARCH §2.2). The legacy `controller/dataset.py` `.ges → tract-
   sequence` framing was deleted; do not resurrect it. If a piece is
   missing, file a follow-up task.
1. Corpus generation: provenance-tagged, anchor-grounded mix. The
   sampling proportions in the original spec assumed inversion/005
   fits would be available; until they are, the working v1 mix is:
   - **anchor** (Story 1996 via 002a): high weight, held-out anchor
     gates. Required.
   - **anchor_interp** (vowel-manifold interpolation between Story
     anchors): medium weight. Required.
   - **r1_fixture** (frozen `r1_train/val/test.npz` fixtures from
     `outputs/fixtures/MANIFEST.md` — NOT R1 NN re-inferred): low /
     medium weight. Legacy coverage prior.
   - **uniform_reachability**: low weight; for clamp / refusal
     testing only, NOT primary supervision.
   - **unreachable**: refusal loss only, not target imitation.
   - **sparc** (real-audio articulatory features via 006 if 006
     unblocks): medium / high weight. Optional v1; required by v2.
   - **inversion_fitted** (when inversion/005 lands): high weight.
     Adds the original 40% slice.

   Each shard MUST tag samples by provenance source. The training
   loop weights by source. Do NOT mix sources without tags.
2. Body-model architecture upgrade: input is the composed gesture
   field at 50 Hz frame rate (linear interpolation between
   GestureEvent knots, evaluated at frame centers; the interpolator
   is "load-bearing" per ARCH §2.2 — pin its contract here);
   output is full ~600-field TEA renderer interface; latent is
   `D_body_effective`-dim.
3. Training:
   - Gesture-supervision loss (MSE) when ground-truth fits exist.
   - Reachability constraints (hard clamps on areas, lengths,
     glottis state).
   - Smoothness penalty in continuous-time units (post-
     interpolation grid, NOT knot grid).
   - Anti-cheat: include unreachable-target samples; body must
     refuse to find impossible postures.
   - **NO audio loss through TEA in this module's training loop.**
4. Validation: inversion/004's holdout, plus
   `runbooks/anticheat_unreachable.py` ≥ 80% refusal.

## Acceptance gates (automated)

- ✅ Gesture-target RMSE on held-out fit-set body trajectories
  within 10% of per-channel std (i.e., RMSE/std ≤ 0.10 channel-wise).
- ✅ **Anchor area-function fidelity** (Codex review 2026-05-09):
  the body model's predicted `area_cm2_tea` + `length_cm_tea` for
  each Story anchor input must match the ingested Story values
  within tight per-channel normalized error (recommended target:
  RMSE/std ≤ 0.10 channel-wise on the 40 supraglottal sections).
  Primary gate. The body model's job is to reproduce real area
  functions; whether TEA's audio matches Story's separate simulator
  is TEA's calibration concern (out of scope for this task; tracked
  separately as `correctness/009 — TEA subglottal/source-side mode
  characterization`).
- ✅ **Anchor TEA-regression parity**: render the body model's
  predicted area function through TEA via
  `scripts/transfer_function_oral_only.py` and compare formant
  triplets to the SAME TEA-rendered Story anchor baseline (NOT to
  Story's published computed formants). Tolerance: same normalized
  error as Story-vs-TEA-baseline on canonical anchors (i.e., model
  errors should not exceed TEA's own measurement noise on the same
  geometry). Catches geometry mistakes without making Story-vs-TEA
  simulator mismatch block the body model.
- 🔍 **Optional audit** (not gating): report Story-published formant
  deltas for the rendered-model output as informational metric. Do
  not promote to hard gate without an absolute-formant-truth source.
- ✅ Reachability constraints never violated on 10K random
  in-envelope inputs.
- ✅ Anti-cheat refusal ≥ 80%.
- ✅ Model size ≤ 5 M params.
- ✅ inversion/004 gates pass when this body model replaces R1 NN
  baseline.
- ✅ CI guardrail: non-circularity lint passes — no `synthesize_dynamic_tube`
  in the loss path.
- ✅ Checkpoint manifest carries teacher-provenance metadata listing
  which sources contributed and at what weight. Required so future
  artifacts can be compared by data lineage, not just by metric.

## Anti-deliverable / scope guard

- Do NOT add a free `wall_dynamic_residual` channel here without
  explicit ablation evidence. Default OFF; cheat-channel CI gate
  (infra/002) will flag it if enabled lazily.
- Do NOT train against TEA audio. **Non-circularity rule.** This
  is the highest-risk drift — engineers will be tempted to "just
  add a small audio reconstruction loss." Don't.
- Do NOT bake in a specific `D_body` if inversion/006 hasn't
  closed. The architecture-design step can proceed in parallel; only
  corpus generation and training are blocked on the decision.

## Notes

This is the heaviest pretraining task. ~3 sessions across corpus
gen (1), training (1–2), eval (1). Worker should split if needed
and file follow-ups.

Quality of the gesture supervision corpus directly determines
final cloning quality. Worth investing time in corpus diversity.
