# cloning/003 — 8 GB inference benchmark on consumer GPU @ 22.05 kHz

**Phase:** cloning    **Status:** pending    **Effort:** S
**User-input:** none
**Blocked by:** cloning/002
**Blocks:** cloning/005
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §6 (memory gates G1), §1.4. The
load-bearing compute claim: text → 22.05 kHz audio in single-utterance
forward inference, ≤ 8 GB VRAM, ≤ 1× realtime on consumer hardware.

## Deliverable

1. Configure inference path:
   - Trained planner (pretraining/001).
   - Trained controller (pretraining/003).
   - Trained body model (pretraining/002).
   - Per-speaker object loaded from a cloning/002 fit.
   - TEA at fp32, with `jax.checkpoint` from correctness/005.
   - Surrogate **disabled** (CI guardrail).
   - Residual **disabled** initially; cloning/004 covers ablation.
2. Benchmark suite:
   - 5 utterance lengths: 1s, 2s, 5s, 10s, 30s.
   - Each at 16 kHz, 22.05 kHz, 24 kHz.
   - Each with batch=1 (typical inference).
   - Measure: peak GPU memory, JIT compile time, per-token
     forward time, end-to-end wall-clock.
3. Compare against the budget table from `ARCHITECTURE.md` §6.1.
   Document any gap.

## Acceptance gates (automated)

- ✅ Single-utterance forward at 22.05 kHz fits ≤ **3 GB** peak
  GPU memory for a 10s utterance (ARCH §6 G1 budgets ≤2.5 GB; 3 GB
  here is headroom for variance — record the actual peak).
- ✅ Single-utterance forward at 16 kHz fits ≤ **2 GB**.
- ✅ Wall-clock per second of audio ≤ **1×** realtime at 22.05 kHz
  on consumer GPU (RTX 3060 / 4060 8 GB class).
- ✅ JIT compile time recorded per-stage; report cold-cache and
  warm-cache numbers separately. No portable absolute target.
- ✅ JSON benchmark report at
  `outputs/inference_bench/report.json` with all combinations AND
  per-stage time breakdown (planner / controller / body / TEA) so
  the bottleneck-driven escalation in Notes is actionable.

## Anti-deliverable / scope guard

- Do NOT fall back to surrogate to make the benchmark pass. CI
  guardrail will fail; treat as hard block.
- Do NOT use fp64 anywhere in the inference path. Production is
  fp32 by `ARCHITECTURE.md` §7.
- Do NOT batch multiple utterances. Single-utterance inference is
  the contract.

## Notes

If gates fail: investigate which stage dominates from the per-stage
breakdown. The escalation paths (TEA M3 banded LDLᵀ + custom_vjp; or
fp4 weight quantization for the neural modules) are not yet
workplan tasks — file follow-ups with concrete scope when the bench
report identifies the bottleneck. Do not leave speculative
escalation in this task's worker prompt.
