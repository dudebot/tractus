# correctness/003 — M6 soft-mode gradient stability

**Phase:** correctness    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** correctness/004
**Blocked by:** correctness/002
**How to dispatch:** fresh Claude Code session

---

## Context

`tractus_engine_advanced/SCOPE.md` §3.C (smooth surrogates inventory)
+ `ARCHITECTURE.md` §10 (validation discipline). M6 closes when:
soft-mode gates active, gradients NaN-free over 10 K random forward
passes, finite-difference gradients agree with autograd within
tolerance per module.

The smooth surrogates already implemented (soft area clamp,
tanh-blended Bernoulli direction, soft glottis warp, per-section
gated noise stub for M7) need a stability sweep at fp32, with
explicit gradient FD verification.

## Deliverable

1. Implement / wire the `Smoothing` config (`SCOPE.md` §2) as a
   test-fixture-level config (NOT a CLI knob — public API exposure
   is a separate task): `Smoothing.hard()` for parity tests,
   `Smoothing.default_soft()` for training, plus per-stage τ knobs.
2. Run a 10K-seed forward sweep at fp32 with `default_soft()`. Each
   seed: random in-envelope tract trajectory + random glottis
   trajectory. Report: NaN count, Inf count, max abs audio sample,
   audio RMS distribution.
3. Per-module gradient finite-difference checks (fp64 baseline,
   fp32 main path) — this task owns the canonical implementation;
   infra/001 only registers the `gradient_fd` marker:
   - `glottis/geometric.py`: `jax.grad` of glottal flow w.r.t.
     control params vs FD.
   - `tds/step.py`: `jax.grad` of per-sample state w.r.t. tube
     geometry vs FD.
   - `output/iir_chebyshev.py`: `jax.grad` of audio sample w.r.t.
     skin-flow input vs FD.
   - `tractus_controller/adapters/r1_nn.py`: `jax.grad` of body-state
     output w.r.t. articulator inputs vs FD (chained gradient path
     to TEA).
4. Record gradient-magnitude distribution across the 10K seeds.
   Flag outliers > 100× median (likely numerical instability hot
   spots).
5. Add a soft-mode test slice: `pytest -m m6_soft` exercises the
   above on a small representative seed set.
6. Soft-mode → hard-mode equivalence smoke: at τ → 0 the soft
   surrogates must reproduce the hard-mode forward output within
   numerical noise (a few lines; identity-property check).

## Acceptance gates (automated)

- ✅ NaN count = 0 over 10K random forward passes at fp32.
- ✅ Inf count = 0.
- ✅ Per-module FD agreement: relative error < 1e-4 fp64,
  < 1e-2 fp32 on at least 99/100 random in-envelope inputs per
  module.
- ✅ Gradient-magnitude tail: 99.9th percentile / median < 100×.
- ✅ `pytest -m m6_soft` passes.
- ✅ JSON report `outputs/m6_soft/report.json` with sweep stats +
  per-module FD agreement breakdown.

## Anti-deliverable / scope guard

- Do NOT implement the per-section gated noise (that's M7 =
  correctness/004). Soft-mode here is the surrogates already
  designed in `SCOPE.md` §3.C, plus the public `Smoothing` config.
- Do NOT loosen the FD tolerance to make the sweep pass.
  Numerical-instability hot spots get fixed at the source (Tikhonov
  regularization, log-space clamps, softplus thresholds), not by
  loosening the test.

## Notes

This task surfaces the backward-NaN issue from the dependency-removal
handoff if it isn't already resolved by correctness/006. They share
root-cause candidates (log clamp, softplus saturation, G3a stability
under autodiff). The two tasks may run in parallel under
correctness/001; whichever lands first publishes findings to the
other.
