# infra/005 — Audio-fixture regeneration

**Phase:** infra    **Status:** pending    **Effort:** S
**User-input:** none
**Blocks:** correctness/001 (deferred gates), correctness/002 (M5 retool)
**Blocked by:** —
**How to dispatch:** fresh Claude Code session

---

## Context

Several P0 acceptance gates compare TEA forward output against
**saved-fixture audio** as the surviving oracle (per
`../../../ARCHITECTURE.md` §8.1, §9 entry 6, and the handoff
document `../../handoff/2026-05-04_session.md` known-hole #3).

Specifically:

- **correctness/001 deferred gates:**
  - "Forward at fp32 within 1e-3 RMSE of saved-fixture audio on
    ≥ 27/30 utterances OR log-mel L1 ≤ 0.5 dB on the failing
    utterances"
  - "Run `runbooks/source_tract_sep.py` on the 30 fp32-rendered
    fixtures vs saved-fixture audio. Per-axis distances within
    bounds (F0 RMSE < 50 cents, tilt within ±2 dB/oct, formants
    within 5%)."
- **correctness/002:** assumes saved-fixture audio under
  `outputs/fixtures/m5_oracle/` exists.

Both currently sit on a directory that does not exist on disk.

The handoff explicitly flagged this:

> **No fixture-regeneration plan.** correctness/002 (M5 retool)
> assumes saved-fixture audio under `outputs/fixtures/m5_oracle/`
> exists. If it doesn't, that's a separate fixture-regeneration
> task — file it before correctness/002 dispatches.

That separate task is this one.

## Deliverable

1. Render a 30-utterance audio set through TEA forward at fp32
   (post-correctness/001) using a deterministic seed, default
   speaker anatomy, and a fixed gesture-driver harness.
2. Inputs come from existing `outputs/fixtures/_oracle_work/`
   `.ges` / `.tract` / `.seg` files (30 utterances already drawn
   from the natasha_1 corpus). If 30 isn't divisible from what's
   there, take the first 30 by lexical filename order and
   document which.
3. Save as `outputs/fixtures/m5_oracle/<utt-id>.npz` with
   schema: `audio: float32[T]`, `sample_rate: int32 = 22050`,
   `length_cm: float32[F, 93]`, `area_cm2: float32[F, 93]`,
   `glottis_traj: float32[F, 11]`, `git_hash: str`,
   `tea_precision: str = "fp32"`,
   `deterministic_seed: int32`. One NPZ per utterance.
4. Add a `MANIFEST.md` row per utterance: filename, source
   `.ges`, frame count, sample count, audio RMS, render time,
   peak GPU MB.
5. Update `outputs/fixtures/MANIFEST.md` to point to the new
   `m5_oracle/` subdirectory and document its provenance.

## Acceptance gates (automated)

- ✅ 30 NPZ files under `outputs/fixtures/m5_oracle/` with the
  documented schema.
- ✅ Each NPZ loads, audio is finite (no NaN/Inf), RMS in
  `[0.005, 0.5]`.
- ✅ Deterministic-seed re-render of a single utterance produces
  bit-equal audio (regression-prevention).
- ✅ Manifest entries match files on disk.
- ✅ correctness/001's two deferred gates run end-to-end against
  the new oracle (both pass — fp32 self-comparison is trivially
  zero-distance, source_tract_sep just establishes a baseline).

## Anti-deliverable / scope guard

- Do NOT regenerate the existing `r1_*.npz` body-model corpora
  — those are the R1 NN training data and are independent of
  this oracle.
- Do NOT include audio rendered from any deleted upstream C++
  driver. The render must come from the current Python TEA path
  at fp32. Document that explicitly in the MANIFEST.
- Do NOT make this oracle a fp64 reference. fp32 is the
  production default; the oracle reflects that.

## Dispatch notes

This is a small, mechanical task once correctness/001 lands. Run
on the smallest available GPU (or CPU — 30 utterances at
production T is ~30 min on CPU, ~3 min on GPU). Dispatch as a
fresh Claude Code session with this file as input.
