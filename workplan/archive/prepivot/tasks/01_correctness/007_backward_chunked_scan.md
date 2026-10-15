# correctness/007 — Backward chunked scan / custom_vjp on MNA solve

**Phase:** correctness    **Status:** in-progress    **Effort:** M
**User-input:** none
**Blocks:** P0 close (replaces correctness/005 b=4 acceptance gate)
**Blocked by:** correctness/005 (jax.checkpoint scaffolding landed)
**How to dispatch:** fresh Claude Code session

> **Iteration 1 landed (chunked outer scan).** A
> `frame_chunk_size: int = 1` kwarg now reshapes the per-frame inputs
> into `(n_chunks, K)` and wraps the per-chunk body in
> `jax.checkpoint(policy=nothing_saveable)`. Verifier confirms the
> kwarg is wired and bit-equality + T=20 grad-finite still pass, but
> T=118 backward STILL OOMs because the per-frame scan-output transpose
> just gains a unit `K`-dim and is otherwise unchanged. **Iteration 2
> needs the `jax.lax.custom_linear_solve` path** below — the dense
> per-sample MNA-solve VJP is the actual structural cost; chunking
> alone can't bound it.

---

## Context

`correctness/005` landed `jax.checkpoint(policy=nothing_saveable)` on the
per-sample scan body (`pipeline.py:synthesize_dynamic_tube`). That fix
gives:

- bit-equal forward pre/post wrap,
- finite gradients at T = 20 (smoke),
- but **T = 118 b = 1 backward still OOMs** on a 16 GB GPU because XLA's
  static analysis shows peak ≥ 17.7 GB. The unavoidable peak is a
  4-D transpose `f32[118, 882, 97, 97]` ≈ 3.92 GB just for one
  intermediate, materialized by the scan-of-MNA-solves backward pass.

The 005 anti-deliverable explicitly anticipated this:

> ✅ Backward through `synthesize_dynamic_tube` at T = 118 fits ≤ **2 GB**
> at fp32, batch 4. Anything larger is out-of-spec; do not loosen the
> gate, file a follow-up for chunked-scan instead.

This is that follow-up.

The structural issue is autograd through 882 × 118 ≈ 104 K dense
97 × 97 MNA solves. Even with full rematerialization, XLA materializes
a per-sample Jacobian array at peak. The fix needs to bypass autograd
through the MNA solve itself.

## Deliverable

Two complementary changes (worker may pick the cheaper-first one and
re-evaluate after benchmarking):

1. **`jax.custom_vjp` on `step_one`** (or its inner solver).
   The MNA system is `M(state, geom) · x = b(state, geom)`. The
   VJP of `solve(M, b)` w.r.t. `(M, b)` is the standard implicit
   formula — solve `M^T · y = cotangent`, accumulate into `M` and `b`
   tangents. Replaces the dense per-sample Jacobian materialization
   with one extra transposed solve per backward step. Reference:
   JAX docs §"Custom derivative rules", `lax.custom_linear_solve`.
2. **Chunked outer scan** if needed: split the per-frame scan into
   chunks of K frames, applying `jax.checkpoint` per chunk so the
   inner-iter tape footprint is bounded by K rather than n_frames.
   Pure rematerialization, no algorithm change.

## Acceptance gates (automated)

Re-use `outputs/backward_oom/_run_verify.py` (or the moved
`scripts/verify_backward_oom.py`) for the gate. Updated thresholds:

- ✅ Forward output bit-equal pre/post custom_vjp + chunked-scan
  (max abs diff = 0). Same as 005's gate.
- ✅ Backward at T = 118 batch = 1 finishes; finite grads.
- ✅ Backward at T = 118 batch = 4 fp32 fits ≤ **2 GB peak VRAM**
  (the original ARCH §6 G2 budget).
- ✅ Forward time post-change within **1.3×** baseline.
- ✅ Backward time within **3×** the baseline forward time at the
  matched (T, batch) point.
- ✅ Updated JSON report `outputs/backward_oom/report.json` with the
  T = 118 grid filled in.

## Anti-deliverable / scope guard

- Do NOT implement a custom rule that changes solver semantics
  (e.g. iterative solve in place of dense). Pure-VJP swap only.
- Do NOT skip the bit-equality re-check; algebraic identities for the
  custom rule are easy to derive wrong.
- Do NOT bury the chunked-scan unroll factor in code; expose it as
  a kwarg with a sensible default.
- **Do NOT slip into the G4 / full-utterance training-time backprop
  path.** ARCHITECTURE.md §6 (G4) is explicit: end-to-end full-
  utterance backprop is structurally out-of-scope; the surrogate
  (Module 5A) owns that gradient path. This task is about closing
  G2 (≤ 2 GB at batch 4) for the **production-T short-window**
  backward; do not add machinery (e.g. gradient checkpointing across
  a 5 s utterance, full-utterance scan partitioning) that would
  enable G4 as a side effect.

## Notes

`jax.experimental.host_callback` and `jax.lax.custom_linear_solve`
are both candidates; `custom_linear_solve` is the more JAX-native
path. If `step_one` does extra non-linear logic around the solve,
target the linear solve specifically.

The 005 verification harness already has the b=1 + b=4 measurement
scaffolding; just delete the `b1_status != "ok"` skip-guard once b=1
fits.
