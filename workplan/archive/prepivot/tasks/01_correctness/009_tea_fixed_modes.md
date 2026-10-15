# correctness/009 — TEA fixed-mode characterization (subglottal / source-side)

**Phase:** correctness    **Status:** pending    **Effort:** S
**User-input:** none
**Blocked by:** —
**Blocks:** — (does not block pretraining/002; characterization only)
**How to dispatch:** fresh Claude Code session

---

## Context

While building the Story 1996 anchor calibration gate (pretraining/002a +
the transfer-function probes 2026-05-09), several parasitic peaks
appeared in EVERY vowel's TEA-rendered transfer function spectrum at
fixed frequencies that don't move with the supraglottal area function:

```
~1287, 2719, 3400, 4054, 4667 Hz
```

Under oral-only isolation (glottis open, side branches ablated, mouth
flow output) these peaks persist. They are most plausibly:

- Subglottal closed-closed standing waves (sections 0-22 are not
  overridden by the Story splice)
- Wall vibration modes (wall_mass × wall_stiffness resonances)
- Numerical artifacts from MNA solver / Cholesky reconditioning
- Source-side boundary-condition artifacts at the impulse injection
  point

They DON'T block pretraining/002 because the revised acceptance gate
grades the body model against the TEA-rendered Story baseline (which
has the same fixed peaks) rather than against absolute literature
formants. But they ARE a real measurement noise source whenever
formant analysis is downstream of TEA.

## Deliverable

1. **Reproduce + characterize** the fixed peaks. Run `scripts/transfer_function_oral_only.py`
   (already exists). Confirm the peaks at 1287 / 2719 / 3400 / 4054 /
   4667 Hz are present across all vowels.
2. **Ablate to source.** Build a probe variant that:
   - Sets subglottal sections (0-22) to MIN_AREA_CM2. Re-run.
   - Sets all wall_mass / wall_stiffness to extreme values (e.g.,
     ×1000 to eliminate wall vibration). Re-run.
   - Drives the impulse at upper-glottis section (24) instead of
     subglottal pressure source. Re-run.
   - For each variant, report which fixed peaks disappear.
3. **Diagnose**: which fixed peaks come from where. Document in a
   short report under `outputs/tea_fixed_modes/report.md` (gitignored).
4. **Decide whether to fix** any of them. If a peak is unambiguously
   a numerical artifact (e.g., MNA discretization at Nyquist) and a
   cheap filter / damping change removes it, file the fix as a
   follow-up correctness task. If they're physical (wall vibration
   genuinely resonates near 2700 Hz), document and accept.

## Acceptance gates (automated)

- ✅ Variant probe scripts exist and produce per-variant peak lists.
- ✅ Report identifies the source of each fixed peak (subglottal /
  wall / numerical / source-side).
- ✅ Each peak has a verdict: `accept` (physical, document only) or
  `fix` (with a follow-up task ID for the actual fix).

## Anti-deliverable / scope guard

- Do NOT modify TEA's MNA solver, wall coupling, or boundary
  conditions in this task. Characterization only. Fixes go in
  follow-up tasks.
- Do NOT introduce new TEA calibration knobs unless the fix is
  unambiguous and cheap.
- Do NOT block pretraining/002 on this. The revised gate semantics
  (model area function vs Story area function, not formant matching)
  are designed to bypass this concern.

## Notes

This task exists because the transfer-function calibration work in
pretraining/002a surfaced these peaks as a measurement confound. The
peaks are stable across vowels (good for relative comparisons) but
prevent absolute formant validation against external reference
simulators. Codex review (thread `[session identifier omitted]`,
2026-05-09) recommended characterizing these as a separate task rather
than blocking body-model work on a TEA fix.
