# correctness/006 — Backward NaN bisect

**Phase:** correctness    **Status:** pending    **Effort:** S
**User-input:** none
**Blocks:** P0 close
**Blocked by:** correctness/001
**How to dispatch:** fresh Claude Code session

---

## Context

Per the dependency-removal handoff (historical, retained per ARCH
§1093): at T = 20 frames (fits GPU comfortably), backward returns
**NaN gradients** even though forward is finite at 0.81. Forward at
T = 20 is fine on synthetic-input smoke; the NaN is in the chained
`R1 NN → compose → G3a → log-mel → multi-resolution STFT` path on a
natasha-shape input.

"natasha-shape" = the (T = 20, 30) synthetic input fixture under
`outputs/fixtures/natasha_shape.npz` (a representative 30-d
trajectory tile derived from the natasha test speaker; the file
defines the exact shape). Worker should pin both the fixture path
and the JAX version in the regression test metadata.

Bisect candidates (priority order from the deletion-handoff):
1. `features_jax.log` near zero on quiet samples — clamp to
   `log(eps + x²)` instead of `log(x²)`.
2. softplus saturation in R1 NN heads on natasha-distribution tract
   params; rerun with fp32 forward to check fp32 overflow.
3. Numerical instability inside G3a (geometric glottis) under
   autodiff at certain glottis trajectories.

correctness/001 (fp32 audit) may already resolve some/all of this —
the NaN was detected at fp64 originally; behavior at fp32 may be
different. Run this task **after** correctness/001 lands.

## Deliverable

1. Reproduce the NaN at the exact natasha-shape input (T = 20).
   Pin a deterministic seed.
2. Run with `jax.debug.print` instrumented at backward through the
   chained path. Localize which sample / which intermediate first
   produces NaN.
3. Apply the appropriate fix from candidates above (log clamp /
   softplus stabilizer / G3a Tikhonov bump). Smallest possible
   change.
4. Re-run with the fix; confirm gradients finite on 1000 random
   in-envelope inputs at the same shape.
5. Add a regression test under
   `tractus_engine_advanced/tests/test_backward_nan_regression.py`
   that runs the exact natasha-shape forward+backward and asserts
   finite gradients.

## Acceptance gates (automated)

- ✅ `jax.grad` of the chained path at T = 20 on the original
  natasha-shape input produces finite gradients (no NaN, no Inf).
- ✅ Per-intermediate-gradient finite-check at each layer of the
  chained `R1 → compose → G3a → log-mel → MR-STFT` path passes.
- ✅ 1000 random in-envelope inputs at T = 20 also finite.
- ✅ Forward output unchanged within 1e-6 RMSE post-fix (the fix
  must not perturb forward output meaningfully).
- ✅ Regression test added and passing; metadata includes the
  pinned JAX version + fixture path.
- ✅ JSON report `outputs/backward_nan/report.json` with the
  bisect trace, identified root cause, applied fix, and post-fix
  stats.

## Anti-deliverable / scope guard

- Do NOT add `jax.lax.stop_gradient` to make the NaN go away.
  That's silencing the symptom.
- Do NOT loosen the `eps` in the log clamp beyond what's needed.
  Larger `eps` distorts the loss surface meaningfully on quiet
  segments.
- Do NOT alter G3a's algorithm in this task. If G3a is the
  culprit, a small Tikhonov / clamp at the unstable point is the
  in-scope fix; algorithmic rewrite is a separate task with
  separate review.

## Notes

If correctness/001 already eliminated the NaN by switching to fp32
(plausible — fp32 has different overflow / underflow envelope than
fp64), this task closes quickly with a "verified no longer
reproducible at fp32 default" note + the regression test.
