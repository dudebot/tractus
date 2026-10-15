# pretraining/002b — R1-distill body baseline (training infra precursor)

**Phase:** pretraining    **Status:** in-progress (sessions 1+2 dispatched 2026-05-08)
**Effort:** L (already mid-flight)
**User-input:** none
**Blocked by:** —
**Blocks:** pretraining/002 (real anchor-grounded body model — needs the
training infra this task builds)
**How to dispatch:** already dispatched — sessions 1+2 worker active

---

## Context

This is the **infrastructure precursor** to `pretraining/002`. It does
NOT deliver the real anchor-grounded body model; it delivers the
training plumbing (architecture, corpus loader, training loop,
checkpointing, eval gates, non-circularity guard) under a known-weak
data signal.

The data signal is weak by construction: the corpus is procedurally
generated trajectories around hand-crafted phone templates, with R1 NN
inference as the supervision target. R1 NN's own gate file
(`tractus_controller/outputs/checkpoints/r1_nn/r1_v1_gates.json`)
reports `all_pass: false` — so this is distillation from a documented-
imperfect teacher, not real ground truth.

**The artifact this produces is named `r1_distill_body_baseline`,
NOT pretraining/002 v0.** Quarantine framing per Codex review,
2026-05-08 (thread `[session identifier omitted]`).

## What this task builds

(In progress; the dispatched workers are doing this. Documenting for
the record so the next session has the contract.)

1. **Body-model architecture** — `tractus_controller/tractus_controller/body_model/`
   - `model.py` — D_body=64 latent, ≤5M params
   - `interp.py` — 50 Hz frame interpolator (load-bearing per ARCH §2.2)
   - `channels.py` — gesture / renderer-interface schemas
   - `reachability.py` — hard clamps on areas, lengths, glottis state
2. **Corpus generator** — `scripts/gen_body_corpus.py`
3. **Training script** — `scripts/train_body_model.py`
4. **Eval script** — `scripts/eval_body_model.py`
5. **Tests** — `tractus_controller/tests/body_model/`
6. **Checkpoint** — `outputs/body_model_ckpt/` (gitignored)

## Required provenance discipline

Per Codex review 2026-05-08, any artifact this task produces MUST:

- Carry teacher-provenance metadata in checkpoint files (which corpus
  slices, which teacher per slice).
- Be labeled `r1_distill_body_baseline` in the checkpoint manifest, NOT
  `pretraining/002_v0`.
- Include a comparison probe in eval: render canonical /a, i, u/ +
  5 smoke utterances through the trained model AND through direct R1
  inference; report formant deltas + audible differences. If the
  trained model is "R1 but smoother" with no other improvements,
  the run is recorded as that — not promoted to a body-model baseline.
- Support teacher-source weighting in the trainer config (so a future
  Story-anchor-incorporating run can swap in real anchor data with
  high weight without rebuilding the trainer).

## Acceptance gates

- ✅ Training loop runs end-to-end; checkpoint is loadable.
- ✅ Reachability constraints never violated on 10K random in-envelope
  inputs.
- ✅ Anti-cheat refusal ≥ 80% on unreachable samples.
- ✅ Non-circularity lint passes (no `synthesize_dynamic_tube` in
  loss path).
- ✅ Comparison probe (this-model vs direct R1, on /a i u/ + 5 smoke
  utterances) is committed to the checkpoint metadata.
- ❌ Gesture-target RMSE within 10% of per-channel std — NOT a gate
  here. RMSE-to-R1 is a sanity check, not progress; hitting it just
  confirms successful distillation. The real RMSE gate is on
  pretraining/002 against real anchors.
- ❌ inversion/004 holdout improves — NOT a gate; this artifact is
  not expected to improve over R1.

## Anti-deliverable / scope guard

- Do NOT label the checkpoint as the project's body-model baseline.
- Do NOT report intelligibility wins from this artifact — it's
  expected to perform similarly to R1 on the smoke pipeline.
- Do NOT delete the checkpoint when pretraining/002 lands; the
  comparison ("did real anchors actually help?") needs both artifacts.
- Do NOT spend session-3-style polish effort here (eval against
  inversion/004 holdouts, hyperparameter sweeps, distributed training).
  This task's value is the plumbing, not the weights.

## Notes

This task exists because sessions 1+2 of pretraining/002 were
dispatched before the data-source limitation was fully scoped. Rather
than throw away the training infra (loaders, masking, checkpointing,
lint, eval scaffolding), we keep it and re-label the artifact for what
it actually is.

The next task in the chain is `pretraining/002a` (Story 1996 anchor
ingestion), which produces the real anchor data needed for
`pretraining/002` to deliver on its actual spec.
