# smoke/002 — Intelligibility + acoustic-sanity gate on smoke set

**Phase:** smoke    **Status:** pending    **Effort:** S
**User-input:** none
**Blocks:** P2 starts
**Blocked by:** smoke/001, infra/003
**How to dispatch:** fresh Claude Code session

---

## Context

smoke/001 produced 20 rule-based-rendered utterances. This task
runs the multi-axis evaluation harness on them so we can say
**objectively** whether the simulator + body-model baseline is
intelligible enough to gate Phase 2 (inverse fitting) on it.

Per `../../../ARCHITECTURE.md` §10 (validation discipline + P1
gate): rule-based path produces ≤ 20% word-WER on the smoke set,
F0 contour follows planner targets within 50 cents, "Daisy, Daisy"
regression anchor renders.

## Deliverable

1. Run `runbooks/whisper_gate.sh` on each of the 20 smoke
   utterances. Aggregate **word-WER** + jaccard. (Whisper is
   word-level; phone-WER would require an MFA aligner pass and is
   out of scope for v0 smoke.)
2. Calibration anchor: run Whisper on the source-recorded ground
   truth of the same 20 utterances first; the calibration WER bounds
   what we can credit our system for.
3. Run `runbooks/source_tract_sep.py` per-utterance: F0 contour,
   formants, frication band energy. Compare to template-driven
   targets where applicable.
4. Run `runbooks/plausibility_panel.py` on the GestureEvent +
   resulting body-state trajectories: continuous-time velocity /
   accel / jerk caps + drift bounds.
5. Compose a one-page report
   `outputs/smoke/intelligibility_report.md` with per-utterance
   metrics + aggregate summary.
6. **Decide gate.** This is the validator-session decision in real
   time, but the report should make the call obvious — pass or
   fail per acceptance gates below.

## Acceptance gates (automated)

- ✅ Whisper-gated: **word-WER** ≤ **20%** averaged over 20
  utterances. (Per-utterance variance allowed; outliers
  characterized.) Calibration: must be within 15 pp of the
  source-recording WER from deliverable 2.
- ✅ Whisper jaccard ≥ **0.50** averaged.
- ✅ "Daisy, Daisy" specifically renders intelligibly to Whisper
  (recognized as the song title or close approximation).
- ✅ If smoke/001's rule-based path emits explicit F0 targets, F0
  contour RMSE ≤ 50 cents on voiced segments. Otherwise SKIP the
  F0-target gate and report contour shape stats only.
- ✅ Frication band energy ≥ 20 dB above the estimated noise floor
  (or ≥ 10 dB below in-utterance vowel-band energy) on all
  fricative-class phones in the set.
- ✅ Plausibility panel: 0 utterances exceed continuous-time
  velocity caps; ≤ 2 utterances exceed accel/jerk caps (smoke is
  rule-based; some templating artifacts expected).
- ✅ JSON report at `outputs/smoke/report.json`.

If gates fail: the candidate bottlenecks are body-model floor,
template-gesture inventory, rule-based durations, and G2P. Use the
runbook breakdowns to localize — do not pre-name the body model as
the diagnosis. The runbook reports together (WER + per-axis source/
tract sep + plausibility) should pinpoint which layer fails.

## Anti-deliverable / scope guard

- Do NOT train anything to make the gate pass. If we're failing,
  failing is information.
- Do NOT loosen thresholds. The 20% WER target is generous for a
  rule-based-template system; if we can't clear it, the
  architecture's smoke premise is wrong and we need to know.
- Do NOT skip "Daisy, Daisy" — that's the named regression anchor.

## Notes

This task sets the floor for everything downstream. If the smoke
path doesn't pass, no learned-module training will save it; we
need to fix the body-model floor first. That's a real possibility
given the R1 NN's known undertraining (`ARCHITECTURE.md` §8.2).
