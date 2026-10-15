# correctness/001 — fp32 audit (drop fp64-by-default)

**Phase:** correctness    **Status:** pending    **Effort:** S
**User-input:** none
**Blocks:** correctness/002, correctness/005, correctness/006
**Blocked by:** —
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §0 (TL;DR), §7 (Precision policy), §8.1
(TEA current state). TEA was originally fp64-by-default for bit-near
parity with the deleted upstream C++ articulator-synthesis oracle. Oracle is
gone; fp64 is now just slow + 2× VRAM cost, and contributed to the
JIT-compile OOM at production T (`plan/01-decisions.md` D15). fp32
is the production default; fp64 is parity/debug only.

Single-day spike to validate fp32 forward.

## Deliverable

1. Audit `tractus_engine_advanced/src/tractus_engine_advanced/`:
   inventory every `jax_enable_x64`, every `dtype=jnp.float64`,
   every hard-coded fp64.
2. Make fp32 the default. Add a `precision` parameter or env var
   `TEA_PRECISION=fp64` that opts back into fp64 for parity tests.
3. Run TEA forward at fp32 on saved 30-utterance fixture set under
   `outputs/fixtures/`. Compare audio to the saved-fixture audio
   (the surviving oracle per ARCH §8.1, §9 entry 6).
4. Ensure parity-test infrastructure runs fp64 when explicitly
   requested, in an isolated subprocess so it cannot flip the
   default fp32 state of any other test.
5. Re-run existing test suite. fp32 default; document any test that
   breaks. Tests that genuinely require fp64 (parity tests against
   saved fp64 fixtures) get `@pytest.mark.fp32_default` flipped off
   via the marker convention from infra/001 — never silent.

## Acceptance gates (automated)

- ✅ Forward at fp32 within **1e-3 RMSE** of saved-fixture audio on
  ≥ 27/30 utterances OR log-mel L1 ≤ 0.5 dB on the failing
  utterances. Per-utterance RMSE + log-mel L1 in JSON.
- ✅ JIT compile + forward at production T (~118 frames × 882
  samples/frame) runs without OOM on a ≤16 GB GPU at fp32. Peak GPU
  memory recorded in JSON. (Prior fp64 path OOM'd at JIT compile —
  the gate is "no longer OOMs", not a hard MB target.)
- ✅ Forward speed at fp32 ≥ 1.5× faster than fp64 on GPU OR fp64
  OOMs at production T (the fp64 baseline isn't always measurable).
- ✅ Pure-math tests still pass at fp32 (`test_tube_topology`,
  `test_tds_state_layout` non-bridge cases, `test_output_iir` Layers
  1+2).
- ✅ Smoke import: `python -c "from tractus_engine_advanced.pipeline
  import synthesize_dynamic_tube"` runs at fp32 default and produces
  finite audio on a static-tube test.
- ✅ Run `runbooks/source_tract_sep.py` on the 30 fp32-rendered
  fixtures vs saved-fixture audio. Per-axis distances within bounds
  (F0 RMSE < 50 cents, tilt within ±2 dB/oct, formants within 5%).
- ✅ Backward sanity at short window (T ≤ 500) fits ≤ 2 GB at
  fp32 batch 1; finite gradients (smoke against ARCH §6 G2 budget;
  full backward path is correctness/005).

## Anti-deliverable / scope guard

- Do NOT touch the per-section MNA solver's algorithm. Precision
  change only.
- Do NOT change `parity/bridge.py` semantics; it's historical.
- Do NOT delete the parity tests. Tag them with the
  fp32_default-off marker so they're skipped in the default slice
  but runnable on demand.
- Do NOT add a third precision mode (mixed-precision per stage).
  Out of scope; separate optimization task.

## Notes

If OOM doesn't resolve from precision alone, correctness/005
(jax.checkpoint) is the next move. Complementary, not alternative.
