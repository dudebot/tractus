# inversion/001 — Short-window TEA inverse refinement loop

**Phase:** inversion    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** inversion/003
**Blocked by:** P0+P1 closed, infra/003
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §4 (reference-fit protocol), §6 (memory
gates G2 + G3), §10 lessons 1, 2, 9. The 30s-cloning recipe
processes reference audio as **independent short windows**
(≤ 0.5s each, gradient-checkpointed) — never full-utterance
backprop through TEA (G4 is out-of-scope as a product path).

This task lands the inverse-refinement loop as a reusable harness:
given a reference audio file + an initial body state, refine the
body state on short windows to better reconstruct the audio.

## Deliverable

1. CLI: `tractus-controller inverse-refine --ref-audio <path>
   --init-anatomy <path> --out <path> [--window-s 0.5
   --batch 4]`. The `--init-anatomy` file is an `.npz` with the
   schema declared in pretraining/004 (anatomy_vec, source_defaults,
   speaker_embed, channel_model); the CLI fails fast on schema
   mismatch.
2. Audio chunker that:
   - Slices ref audio into ≤ 0.5s windows.
   - **Preserves MFA-anchored silences** (transferred lesson per
     ARCH §10 — non-negotiable). If MFA alignment is provided, use
     it; else fall back to VAD-based silence preservation.
   - Outputs (window, start_t, end_t, transcript) tuples.
3. Per-window inverse fitter:
   - Adam optimizer over body-state latent for that window.
   - Loss: 0.5 · log-mel + 0.3 · MR-STFT + 0.2 · per-axis
     source/tract terms (one scalar per axis, mean over time).
   - Gradient through `synthesize_dynamic_tube` with
     `jax.checkpoint`.
   - Anatomy frozen during inner loop; outer loop on anatomy.
4. Outer loop: alternate per-window gesture refits (anatomy frozen)
   with anatomy gradient steps (gestures frozen). Per
   `ARCHITECTURE.md` §4 step 3. Halting: max 20 outer iterations or
   anatomy-update L2-norm Δ < 1e-3 of the anatomy-vec scale,
   whichever comes first.
5. Logging: `outputs/inverse_refine/<utt-id>/{report.json, fit_traces.npz}`
   per utterance.

## Acceptance gates (automated)

- ✅ Per-window fit fits in **G2** budget: ≤ **2 GB** at fp32 batch 4
  (matches ARCH §6 G2). Anything larger is out-of-spec.
- ✅ Inverse on **perturbed** smoke/001 audio (perturb the source
  body trajectory by Gaussian noise, σ specified by deterministic
  seed; render; refit) recovers the original trajectory within
  5% RMSE — non-trivial sanity check that the loop closes.
- ✅ `runbooks/plausibility_panel.py` passes on the recovered
  trajectory.
- ✅ Fit time per 5s utterance ≤ 5 minutes on consumer GPU at
  batch 4. (If much slower, infra/003 timing is wrong; escalate.)

## Anti-deliverable / scope guard

- Do NOT attempt full-utterance backprop. G4 is out-of-scope.
- Do NOT use a TEA-rendered audio loss to train the body model.
  This is **inverse refinement of state given a fixed body model**,
  not body-model training. Non-circularity rule.
- Do NOT skip silence preservation. The transferred lesson per ARCH
  §10 (originally D8 in the historical decision log) shows what
  happens when you do.

## Notes

This is the workhorse for inversion/* and cloning/*. Quality of
implementation here directly determines reference-fit quality in
P4. Worth taking the extra time to get the chunker + outer/inner
loop semantics right.
