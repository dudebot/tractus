# inversion/004 — Anatomy fitting on synthetic-baseline holdout improves ≥ 1 dB

**Phase:** inversion    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** P2 close
**Blocked by:** inversion/001, inversion/002, inversion/003
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §4 (reference-fit protocol), §10 phase
exit P2. Synthetic-but-realistic test: take the default-baseline-
rendered audio, perturb anatomy minimally, fit anatomy back on a
fit-set, verify held-out reconstruction improves by ≥ 1 dB log-mel
without degrading WER.

Sanity check before scaling to real-speaker reference audio.

## Deliverable

1. Generate test corpus: 30 utterances rendered through TEA with
   the default anatomy baseline. Save
   `(audio, body_traj, anatomy_vec)` tuples.
2. Perturb anatomy_vec by independent Gaussian, σ = 0.5 × per-scalar
   hard-clamp range (single deterministic seed; ground-truth
   perturbation on the smoke basis, or v1 basis if inversion/005
   already landed).
3. Re-render the 30 utterances with perturbed anatomy. This is the
   "speaker we're cloning."
4. 80/20 split: 24 fit-set, 6 held-out.
5. Run inversion/001's outer loop on the fit-set with anatomy
   trainable (initialized at default baseline, NOT at ground-truth
   perturbation). Inner loop refits gestures per window.
6. Evaluate fitted anatomy on held-out:
   - Log-mel L1 vs perturbed-anatomy targets.
   - WER preserved relative to fit-set.
   - `runbooks/source_tract_sep.py` per-axis on held-out.
7. Compare to default-baseline-no-fit reconstruction on the same
   held-out audio (the floor).

## Acceptance gates (automated)

- ✅ Held-out log-mel improves by **≥ 1 dB** over default-baseline
  no-fit floor.
- ✅ Fit-set log-mel improves by **≥ 1.5 dB** over the same floor
  (overfitting check: the fit-set/held-out gap should be ≤ 0.5 dB).
- ✅ Held-out WER does not degrade by more than 5 percentage
  points relative to fit-set.
- ✅ Fitted anatomy is within 50% of the ground-truth perturbation
  in L2 norm (i.e., `||fitted − truth||₂ ≤ 0.5 · ||truth||₂`).
- ✅ `runbooks/anticheat_unreachable.py` ≥ 80% refusal — anatomy
  fit didn't compensate by absorbing cheat-channel structure.
- ✅ JSON report.

## Anti-deliverable / scope guard

- Do NOT use the same audio for fit-set and held-out.
- Do NOT compare to a no-anatomy-fit baseline that uses different
  gestures. Hold gestures fixed; anatomy is the only variable.
- Do NOT skip the anti-cheat check.
