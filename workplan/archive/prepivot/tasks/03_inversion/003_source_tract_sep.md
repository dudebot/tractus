# inversion/003 — Source/tract separation metrics on every fit

**Phase:** inversion    **Status:** pending    **Effort:** S
**User-input:** none
**Blocks:** P2 close
**Blocked by:** inversion/001
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §10 lesson 7. Aggregate audio loss
(log-mel, MR-STFT) hides compensations between F0 and length scale,
between adduction and constriction. Every fit must report F0,
spectral tilt, frication band energy, nasalization, and formants
**separately**.

`runbooks/source_tract_sep.py` is the implementation; this task
ensures every inverse-fit run invokes it as part of the report
pipeline.

## Deliverable

1. Modify the inversion/001 fit pipeline so each completed fit
   automatically runs `runbooks/source_tract_sep.py` on
   `(synthesized_audio, reference_audio)` and stores the per-axis
   report alongside the fit-trace artifacts.
2. The fit output JSON gains a `source_tract_sep` block:
   `{"f0_rmse_cents": ..., "tilt_db_per_oct_diff": ...,
   "frication_band_energy_db_diff": [...], "nasal_metric_diff":
   ..., "formant_pct_diff": [F1, F2, F3]}`.
3. Add per-fit acceptance thresholds to inversion/001's gating
   config (default values from `ARCHITECTURE.md` §10 + smoke/002):
   F0 RMSE ≤ 50 cents; tilt within ±2 dB/oct; formants within 5%;
   frication band energy within 3 dB; nasalization metric: nasal-band
   energy ratio within ±3 dB of reference (band defined as 250-450
   Hz integrated; matches the ARCH §2.3 velum-coupling acoustic
   signature).
4. Failures on any per-axis threshold flag the fit for review
   (logged to a `flagged_fits.jsonl` file the validator session
   reads).

## Acceptance gates (automated)

- ✅ Every inverse fit produces a per-axis report alongside the
  aggregate loss.
- ✅ Per-axis thresholds applied in CI; failing fits flagged but
  not silently dropped.
- ✅ Aggregate threshold and per-axis thresholds disagree on at
  least 3 utterances in the smoke/001 corpus, demonstrating the
  per-axis machinery catches what aggregate loss misses (synthetic
  compensation fixtures if needed; the 3-utterance bar makes this
  unfakeable by a single hand-crafted case).
- ✅ `pytest -m source_tract_sep` covers the threshold logic.

## Anti-deliverable / scope guard

- Do NOT use per-axis thresholds as a substitute for the aggregate
  audio loss. Both fire; they're orthogonal evidence.
- Do NOT silently round up per-axis distances to make a fit pass.
- Do NOT add new metrics without justifying which compensation
  failure mode they catch.

## Notes

This is the cheap infra task that sharpens every inversion result.
Pairs with inversion/002 — anti-cheat catches WER trades; per-axis
catches anatomy/source/coloration trades.
