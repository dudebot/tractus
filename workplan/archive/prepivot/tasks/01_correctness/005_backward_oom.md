# correctness/005 — Backward OOM fix (jax.checkpoint on per-sample scan)

**Phase:** correctness    **Status:** pending    **Effort:** S
**User-input:** none
**Blocks:** P0 close
**Blocked by:** correctness/001
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §6 (memory gates G1–G4), §8.2 (autodiff
gotchas), `tractus_controller/plan/90-dependency-removal.md`. Backward
through R1 NN + TEA G3a + features_jax at production T = 118 frames
OOMs ~20 GiB on 16 GiB GPU at fp64. Per-sample `lax.scan`
materializes per-sample state during autodiff (~104K samples × ~3 KB
carry × tape-blowup factor).

Fix: wrap the scan body in `jax.checkpoint` (recompute strategy) so
backward rebuilds state in chunks. ~2× forward time cost.

correctness/001 (fp32 audit) likely shrinks the tape ~2×, so this
task in combination with 001 should comfortably fit.

## Deliverable

1. Add `jax.checkpoint` (with appropriate policy — likely
   `dots_with_no_batch_dims_saveable`) to the per-sample scan body
   in `pipeline.py:synthesize_dynamic_tube`.
2. Verify forward audio output is unchanged (bit-equal) after the
   checkpoint wrap.
3. Verify backward fits in target VRAM budget at production T at
   fp32.
4. Benchmark forward and backward time before/after.
5. Smoke: backward at T = 20 still produces finite gradients
   post-checkpoint (regression check against correctness/006).
6. Report any deviation from ARCH §6 G2 / §8.2 in
   `outputs/backward_oom/report.json`. Architecture-doc updates are
   a separate validator-session task, not silently rolled in here.

## Acceptance gates (automated)

- ✅ Forward output bit-equal pre/post checkpoint wrap (max abs
  diff = 0).
- ✅ Backward through `synthesize_dynamic_tube` at T = 118 fits
  ≤ **2 GB** at fp32, batch 4 (matches ARCH §6 G2). Anything larger
  is out-of-spec; do not loosen the gate, file a follow-up for
  chunked-scan instead.
- ✅ Forward time post-checkpoint within **1.3×** baseline.
  Backward time post-checkpoint within **2.5×** what it would be
  without rematerialization (rough budget; we're paying for the
  recompute).
- ✅ JSON report `outputs/backward_oom/report.json` with timings +
  peak memory per (T, batch_size) pair from a small grid:
  T ∈ {20, 50, 118}, batch ∈ {1, 2, 4}.

## Anti-deliverable / scope guard

- Do NOT change the scan body's algorithm. Pure rematerialization
  policy change.
- Do NOT precompute or cache anything outside the scan to skip the
  backward — that defeats the autodiff correctness contract.
- Do NOT mark this task done if NaN gradients persist. Those are
  correctness/006; that task may need to run first depending on
  bisection results.

## Notes

Standard JAX `jax.checkpoint` pattern; ~10 lines of code change.
The bench grid is the load-bearing artifact — gives the validator
session a clean before/after table.
