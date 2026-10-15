# Task 004 — Controller with SPARC pseudo-labels (Week 3)

> **Blocked by:** Task 003 GREEN. **Read first:**
> `workplan/handoff/2026-05-09_course_correction.md`.

## Goal

Train the neuromotor controller (Module 2) to map planner output
(phones + prosody) to gesture trajectories that, when fed through
the body model + TEA, reproduce held-out utterances.

The bridge here is **SPARC labels as pseudo-targets**: SPARC's EMA
channels become the controller's intermediate supervision target,
since EMA is closer to "gesture" than waveform.

## Architecture

Reuse the existing controller skeleton from `pretraining/001` if
salvageable; otherwise replace. ~5 M params target. NAT-style or
small autoregressive transformer that consumes:

```
inputs:
  segments[N]:    (phone_id, dur_s)
  stress[N]:      int
  f0_env[T_f0]:   log-Hz
  loud_env[T_f0]: log-RMS
  speaker_motor_embed[D_motor]: from speaker fit (Task 003)

outputs:
  gestures: list[GestureEvent]   (asynchronous, ~30 events/s)
```

Then composed via `body_model/interp.py` to a 50 Hz trajectory and
fed to the body model from Task 003.

## Training

Two-stage:

1. **Kinematic supervision** (warm-start). Train controller to
   produce gesture trajectories whose 50 Hz interp matches SPARC's
   EMA + F0 + loudness directly. Loss: MSE on EMA + Huber on F0 +
   MSE on loudness.

2. **Acoustic refinement.** Switch to acoustic loss through the
   frozen body model + TEA (short windows; `jax.checkpoint`). Hold
   kinematic loss as auxiliary at 0.3 weight.

Use the same SPARC corpus from Task 002 plus 10 developer-recorded
utterances.

## Acceptance gates

### Gate 1 — 10 held-out utterances reproduce

Same 10 utterances as Task 003 Gate 2. After full Week-3 training:

- ≥ 7 of 10 produce non-empty Whisper transcripts.
- WER ≤ 0.65 averaged over the 10.
- Improvement over Task 003 Gate 2 (the body-model-with-SPARC-labels
  baseline): the controller-driven path should be no worse than
  the SPARC-driven path.

### Gate 2 — End-to-end intelligibility

5 unseen Harvard sentences (developer hasn't recorded these). Render
text → planner → controller → body → TEA. WER ≤ 0.65 average.

### Gate 3 — Gesture-event sanity

The controller's emitted `GestureEvent` lists should:

- Average ~20-40 events/s (matches the architectural ~30 events/s).
- Be physically plausible (no overlapping events on the same channel
  with conflicting target_value; no events shorter than 30 ms).
- Render through `interp.py` without clamping warnings.

### Gate 4 — No cheat

Run `runbooks/source_tract_sep.py` on 5 of the held-out renders.
Per-axis source vs tract attribution should be roughly balanced —
if the controller is dumping all its information into one channel
(e.g., source absorbing what should be tract), this gate flags it.

## Kill criterion

If Gate 1 fails after Week-3 + 2 debug days:

- Check whether body model from Task 003 still passes its gates on
  the same 10 utterances. If not: Task 003 regressed; investigate
  drift.
- Check whether SPARC pseudo-labels are dense and clean for the
  failure-mode utterances.
- Consider whether the planner output is mis-aligned (durations way
  off; check against MFA forced-alignments if available).

If after diagnosis the controller still fails, the constrained-demo
scope shrinks: Week 4 demos via SPARC-labels-direct rather than
text→controller→body. Document the regression honestly.

## Anti-deliverables

- No retraining the body model. It's frozen this week.
- No new corpus. Reuse Task 002's SPARC-labeled set.
- No phoneme inventory expansion. Stay within the existing G2P.
- No multi-language work. English only for v0.

## Reporting

`workplan/LOG.md`:

```
## YYYY-MM-DD — Week 3 controller

- Gate 1 (held-out 10):     __ / 10, avg WER __
- Gate 2 (end-to-end 5):    __ / 5, avg WER __
- Gate 3 (gesture sanity):  pass | fail
- Gate 4 (no cheat):        pass | fail
- Verdict:                   GREEN | RED-DEMO-DEGRADED
```

GREEN → Task 005 unblocked.
RED-DEMO-DEGRADED → Task 005 ships SPARC-driven demo, not
controller-driven.

## References

- Existing planner skeleton: `tractus_controller/.../planner/`
- `interp.py` interpolation contract: `…/body_model/interp.py`
- `runbooks/source_tract_sep.py` for cheat detection.
- Course correction: `workplan/handoff/2026-05-09_course_correction.md`
