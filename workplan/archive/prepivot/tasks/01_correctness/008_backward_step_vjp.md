# correctness/008 — Custom VJP on `step_one` (closes 007's b=4 gate)

**Phase:** correctness    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** P0 close (closes the production-T backward gate G2)
**Blocked by:** correctness/007 (iter 2 `custom_linear_solve` landed)
**How to dispatch:** fresh Claude Code session

---

## Context

`correctness/007` iter 2 (commit `1a11e93`) wrapped the per-sample
MNA Cholesky in `jax.lax.custom_linear_solve`. Effect: the dense
per-sample-Jacobian transpose `f32[F, samples_per_frame, 97, 97]`
(~3.9 GB) is gone. **T=118 b=1 backward now completes** at 11.2 GB
peak VRAM (was OOM pre-iter-2).

The remaining cost: at b=4 the verifier still OOMs needing 14.6 GB
for an intermediate transpose `f32[118, 1, 882, 4, 97, 97]`. The
explanation per Codex pre-mortem: `custom_linear_solve` fixed the
**solve VJP** but not the **cotangent-of-M** cost. The implicit-
function-theorem rule still needs `dM = -y · x^T` per sample, where
`x` is the solve output and `y` is the cotangent of the solve. For a
per-sample 97×97 matrix M built fresh each step from state, this
forces materializing per-sample `x` and `y` arrays — which JAX
collects into a leading-axis transpose at the outer-scan boundary.

The structural fix is to bypass autodiff at a wider granularity than
the bare solve: a **custom VJP on `step_one`** that hand-writes the
backward without ever building M as a dense 97×97. The forward path
keeps building M (no algorithm change), but the VJP receives the
cotangent of `step_one`'s output and computes input cotangents via
direct sparse multiplication against the same MNA structure used in
the forward, never materializing the full M cotangent.

## Deliverable

1. `jax.custom_vjp` decorator on `step_one`
   (`tractus_engine_advanced/src/tractus_engine_advanced/tds/step.py`).
   Forward path computes the same `(state, audio_sample)` output and
   stashes only the small residuals needed for backward (state at
   step boundary + the few scalars the VJP needs — NOT the M
   matrix).
2. Backward path computes input cotangents using the implicit-
   function-theorem rule expressed against the **sparse MNA
   structure** in `assemble_matrix` (read it before starting; the
   matrix is ~10× sparser than 97×97 dense). Use `jnp.einsum` /
   masked operations rather than dense `M.T @ y`.
3. Forward output is bit-equal pre/post (`max abs diff = 0`).
4. T=118 b=4 backward fits ≤ 2 GB peak VRAM at fp32 and yields
   finite gradients.
5. Forward time post-change within 1.3× baseline.
6. Backward time within 3× the forward baseline at matched (T, b).

## Acceptance gates (automated)

Re-use `scripts/verify_backward_oom.py` (already includes the b=4
measurement). The gates are encoded there:

- ✅ `bit_equal_pre_post_wrap.answer == "Y"` (max_abs_diff = 0).
- ✅ `forward_speed.gate_passed == true`.
- ✅ `backward_smoke_T20.grad_finite == true`.
- ✅ `backward_T118_batch1.status == "ok"` (already passes post-007
  iter 2, must not regress).
- ✅ `backward_T118_batch4_fp32.gate_passed == true` — VRAM peak ≤
  2048 MB, finite grads. **This is the new gate.**

## Anti-deliverable / scope guard

- Do NOT change the forward solver semantics. The whole point of
  `custom_vjp` is that forward stays plain Cholesky; only the VJP
  changes.
- Do NOT skip the bit-equality re-check; hand-rolled VJP rules are
  easy to derive wrong. If the bit-equality fails, the rule is
  wrong, not the test.
- Do NOT use `host_callback` / `jax.pure_callback` / iterative
  forward solver. Pure-VJP swap only.
- **Do NOT slip into G4 / full-utterance training-time backprop.**
  ARCH §6 G4 explicitly out-of-scope. This task is for production-T
  short-window (T=118) backward only.
- Do NOT remove the `frame_chunk_size` kwarg from
  `synthesize_dynamic_tube`. If the gate passes without chunking
  (K=1 = no chunk), document that finding; if K=1 is now over-
  scaffolding, leave it as a no-op default but keep the kwarg for
  compatibility.

## Notes

The most likely shape of the implementation:

```python
@jax.custom_vjp
def step_one(state, geom, topology, *, ...):
    # Same as today: assemble M+b, solve, update state, compute audio.
    ...

def _step_one_fwd(state, geom, topology, **kwargs):
    out = step_one(state, geom, topology, **kwargs)
    # Stash only the scalars the VJP needs — flow_vector, prep, geom.area.
    # Crucially, NOT the full 97×97 M.
    return out, (flow_vector, ...)

def _step_one_bwd(residuals, cotangent_out):
    # Recompute M's effect on the cotangent without materializing M itself.
    # Use the same sparse-structure traversal as `assemble_matrix`.
    ...
    return (cotangent_state, cotangent_geom, ...)

step_one.defvjp(_step_one_fwd, _step_one_bwd)
```

This is real engineering — budget M effort. The reward is closing
the P0-blocking memory gate at the right level.

If iter 2's 11.2 GB at b=1 is still puzzling: the per-sample state
saved is `(tds_state.section, tds_state.branch, output_state, glottis_state)`,
each ~few KB × 882 samples × 118 frames × 4 batch ≈ 1-2 GB total.
The b=4 OOM is dominated by the M-cotangent transpose, not by carry
state. Carry state will become the dominant cost at b=4 ≤ 2 GB.

Codex pre-mortem fallback (e) — Python-loop per-frame outside the
monolithic JIT — is a viable Plan B if `custom_vjp` proves too
intricate. It has a speed cost but bounds peak memory by
construction. Try `custom_vjp` first; fall back to Plan B if the
backward time multiplier exceeds 5× forward.
