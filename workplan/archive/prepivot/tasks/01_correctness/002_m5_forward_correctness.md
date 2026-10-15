# correctness/002 — M5 forward-correctness gate

**Phase:** correctness    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** correctness/003
**Blocked by:** infra/001, infra/003, correctness/001
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §10 (validation discipline). The
forward-correctness gate establishes that TEA's per-sample MNA solve
+ output stage produce stable, physically plausible audio at fp32
across the 30-utterance test corpus. There is no live external
oracle post-removal; correctness is established by:

1. **fp32-vs-fp64 self-consistency** on synthetic inputs.
2. **Per-axis source/tract separation** within physical-plausibility
   bounds (`runbooks/source_tract_sep.py`).
3. **Energy-conservation invariant**: total acoustic power radiated
   ≤ total power injected at the source over a rolling window.
4. **No NaN / Inf** over a 10K-seed random-input sweep.

## Deliverable

1. Generate the 30-utterance synthetic test corpus from procedurally
   varied control trajectories (sweeps over the parameter space, plus
   anchor-shape transitions). Save under
   `outputs/fixtures/correctness_corpus/`. Stress edge cases (extreme
   F0, wide formants, near-clamp constrictions) — do NOT reuse
   smoke/001's template-gesture generator (those exercise typical
   phones, not edge cases).
2. Run TEA forward at fp32 and fp64 on each utterance; record
   audio-sample RMSE (the fp32-vs-fp64 self-consistency check on
   the new corpus).
3. Run TEA forward at fp32 on the pre-deletion saved-fixture
   utterances; compare to saved-fixture audio (the surviving oracle
   per ARCH §1029-1030, §1135-1137). This is a separate gate from
   the fp32-vs-fp64 self-consistency.
4. Run `runbooks/source_tract_sep.py` on each utterance; assert per-
   axis bounds by name: F0 (cents-RMSE), spectral tilt (dB/octave),
   frication-band energy (dB), nasalization metric, formants F1–F3
   (% deviation).
5. Implement an energy-conservation invariant check, split per
   segment class (voiced steady-state and frication separately) since
   wall + radiation losses differ by phone class. Helper integrates
   mouth_flow + nostril_flow vs subglottal pressure × area_at_source
   over a rolling window.
6. Implement a tube-state physical-plausibility check: per-frame
   areas > 0, finite pressures, finite per-section velocities.
7. 10K-seed random-in-envelope sweep at fp32; record NaN/Inf counts.
   "Envelope" is the procedural-corpus parameter ranges in (1).

## Acceptance gates (automated)

- ✅ fp32-vs-fp64 audio RMSE ≤ 1e-3 on ≥ 27/30 utterances of the
  new synthetic corpus (self-consistency).
- ✅ fp32 audio vs saved-fixture audio RMSE ≤ 1e-3 on ≥ 27/30
  utterances of the pre-deletion fixture set (parity to the
  surviving oracle).
- ✅ All 30 corpus utterances pass `runbooks/source_tract_sep.py`
  per-axis thresholds (F0 RMSE < 50 cents, tilt within ±2 dB/oct,
  formants within 5%). Frication-band energy gate is **conditional**
  on M7 (correctness/004) shipping; until then, assert frication
  energy ≈ 0 in absence of constriction (physical derivation).
- ✅ Energy-conservation invariant within 5% on voiced steady-state
  segments. The frication-segment threshold is deferred until M7
  lands.
- ✅ Tube-state physical-plausibility check passes on 30/30 corpus
  utterances (areas > 0, pressures finite, velocities finite).
- ✅ NaN/Inf count = 0 over 10K seeds at fp32.
- ✅ JSON report at `outputs/correctness_m5/report.json`.

## Anti-deliverable / scope guard

- Do NOT introduce a live external oracle. The corpus is synthetic
  and self-consistent; quality is established by physical
  invariants, not external comparison.
- Do NOT loosen acceptance thresholds without diagnosis.
- Do NOT skip per-axis reporting. Aggregate RMSE alone is a
  transferred-lesson trap.
