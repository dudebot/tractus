# CHECKIN — developer's pocket reference

> One file. For the developer, not for worker sessions. Two jobs:
> (1) "are we actually making progress?" — milestone table + PM-
> session protocol. (2) "are we accidentally neutering TEA again?"
> — explicit cross-check.

## What we're building (3 sentences)

A physics-first, differentiable TTS where a small learned controller
emits sparse gesture intentions, a continuous neural-field body model
turns them into a tract area function `A_θ(s, t)`, and a JAX
transmission-line simulator (TEA) renders that to audio. Cloning is
fitting a kilobyte-scale per-speaker object from 30 s of consented
reference. v0 ships when 5 unseen Harvard sentences pass Whisper at
WER ≤ 0.5 in the developer's voice with no post-filter rescuing.

## Where we are (one paragraph, replace as work lands)

**2026-05-10.** Pivoted yesterday from a generic conv-MLP body model
to neural-field + SPARC kinematics. Five active task files seeded;
prepivot artifacts archived. **Next gate:** Task 001 (Week 1 phrase
probe) — neural-field copy-synthesis of one developer-recorded
Harvard sentence through TEA. **Hard prerequisite:** record
`data/probes/juice_lemons_ref.wav` first.

## How we know we're making progress

| Week | Gate that proves the week worked | Gate that proves the project worked |
|---|---|---|
| 1 | Whisper transcribes the reconstructed phrase with WER ≤ 0.75 and ≥ 3 of 5 content words recognized. Field doesn't collapse (clamp occupancy < 5 %). | TEA + differentiable optimization can recover intelligible speech. Foundation OK. |
| 2 | Trained body model reproduces Week-1 phrase AND ≥ 7 of 10 held-out short utterances are non-empty in Whisper. | The body model *generalizes* — not just memorizes one fit. |
| 3 | 5 unseen sentences through `text → planner → controller → body → TEA` average WER ≤ 0.65. | The full controller path beats SPARC-driven direct rendering. |
| 4 | 5 Harvard sentences in developer's voice, WER ≤ 0.5 avg, no phrase > 0.7, listener picks correct voice on at least 1 of 3 A/B. | Cloning works at the v0 constrained scope. |

**If a week's gate fails:** every task file has a kill criterion.
Read it. Exercise it. The kill criteria are the project's
sanity-preserving check valves — they exist so we don't keep tuning
a dead path. See course-correction Risk C for the ship-on-DDSP
fallback if TEA itself can't speak.

## The PM-session check (how to validate a worker's verdict)

You're right that specialists self-validate against their own gates
and usually say "done" or "partial." The fix is a separate review
pass that doesn't trust the worker's pass/fail claim. Two ways to
run it:

### Option A — interactive PM session in Claude Code

After a worker reports verdict on Task NNN:

1. Open a **fresh** Claude Code session (not a continuation of the
   worker session — bias risk).
2. Paste this prompt, filling in `<NNN>`:

   ```
   You are a project-manager review of worker output on Task <NNN>.
   The worker reported its verdict in `workplan/LOG.md` and may
   have written artifacts under `outputs/`. Independently verify
   each acceptance gate in `workplan/tasks/00_active/<NNN>_*.md`.
   Do not trust the worker's pass/fail claim — re-run gates where
   you can (Whisper, smoke tests, file inspection); for ones you
   can't re-run, spot-check the artifacts. Report KEEP/DROP per
   gate with one sentence of evidence. End with: should the project
   move on to the next task, or does Task <NNN> need rework?
   ```

3. Read the response. If the PM disagrees with the worker, dig.

### Option B — dispatch as Agent subagent

If you don't want to switch sessions:

```
subagent_type: general-purpose
prompt: <the same template above>
isolation: (none — read-only check)
```

### What the PM session is NOT

- Not a re-implementation. It checks; it doesn't redo.
- Not asking the worker for clarification. The worker's report is
  the artifact.
- Not gentle. Defaults to skeptical. "Mostly works" is a fail.
- Not for trivial tasks. Skip the PM step on edits < ~50 LOC.

## The anti-neutering cross-check

**The fear:** the C++ VTL backend deletion stripped a working
articulator→tube mapping with no replacement; the conv-MLP body
model was a poor substitute; now the SPARC pivot might do the same
thing — replace something with less than what came before. We want
a guarantee that's not "trust me."

**The probe.** Render the surviving R1 frozen fixtures through TEA
*directly* — bypassing any body model. This is a baseline of "what
TEA already produces given a known-good tube sequence." If the new
neural-field + SPARC pipeline can't match this baseline on the same
content, we're neutering.

### Setup (~1 hour, runs once)

```bash
# Pick 5 R1 fixture rows (sustained vowels or short sequences).
# r1_natasha_oracle.npz is a dynamic sequence (full utterance);
# r1_train.npz has many static-tube examples.
tractus_controller/.venv/bin/python scripts/probe_r1_baseline.py \
  --fixtures tractus_controller/outputs/fixtures/r1_natasha_oracle.npz \
  --out outputs/anti_neutering/r1_baseline/ \
  --source-mode voiced_sustained \
  --frames 50 \
  --rms-normalize
```

(`scripts/probe_r1_baseline.py` doesn't exist yet — Task 001 worker
should write it as part of Week-1 implementation, or we add it as
Task 001.5. The shape is simple: load the npz, take 5 frames, pass
each frame's `(tube_area, tube_length)` directly to TEA with a
sustained voiced source, save audio + report.)

### What the baseline tells you

For each rendered R1-fixture clip:
- Is it recognizable as a speech-like sound (vowel character if
  voiced; fricative character if unvoiced)?
- Does Whisper transcribe it as non-empty when fed a sustained
  vowel concatenation?
- Are F1/F2/F3 in the right ballpark for what the tube *should*
  produce?

This is **what TEA can already do today**, given that the
articulator→tube mapping problem is solved (by the deleted oracle,
in frozen form). It's the floor. Any new body model has to clear it.

### The gate that prevents neutering

**Before locking in SPARC as the kinematic backbone** (Task 002
landing), the trained body model from Task 003 must:

1. Take the same gesture trajectory that the R1-baseline tubes
   correspond to (or as close as the gesture-channel inventory
   permits).
2. Render through the new pipeline.
3. Match or exceed the R1-baseline on: phone intelligibility,
   formant correctness on sustained vowels, frication audibility
   on sustained fricatives.

**If the new pipeline can't match the R1 baseline, we are
neutering.** That's the trigger to: pause Task 003, return to
Codex (`[session identifier omitted]`) for re-deliberation, and possibly fall back
to keeping the R1 frozen mapping as the body-model spine with a
small differentiable residual layer on top.

### Why this is qualitatively different from the VTL deletion

The VTL deletion removed code with no replacement; the pivot
*replaces* with a different kind of model. But "different kind" is
not automatically "better" — that's why this cross-check exists.
The probe is cheap; the consequence of skipping it is months of
work on a regression.

## Red flags to halt on

If you see any of these, stop and surface — do not let the worker
self-correct past them:

1. **"Partial done" with the deferred bit named imprecisely.** If a
   worker writes "done with calibration TBD," ask what the
   calibration value is and what setting it correctly would change.
   Vague deferrals stack.
2. **Whisper transcripts that are empty AND the worker calls it a
   pass on other gates.** Whisper-empty means the audio isn't
   speech. Every other gate is downstream of "is this even speech."
3. **Loss curves that drop but gates that don't pass.** The optimizer
   found a degenerate solution. Inspect the audio, not the loss.
4. **Worker rationalizes away one of the kill criteria.** ("The
   spec said 7/10 but 6/10 is close enough, and the failing one was
   a hard utterance.") No. The kill criterion exists because
   "close enough" already cost us a week last time.
5. **New files appear that aren't in the task's deliverable list.**
   Scope creep. Either the spec was wrong (update it) or the worker
   went off-piste (revert).
6. **The fix is to add a feature flag / fallback / configuration
   option rather than fix the root cause.** Pattern from past
   sessions — feature-flagging a half-working path produces a
   codebase full of half-working paths.

## When this file is wrong

This file is a snapshot. The course-correction document
(`workplan/handoff/2026-05-09_course_correction.md`) and
`ARCHITECTURE.md` are the source of truth on *why* and *what*. If
they disagree with this file, they win.

Rewrite this file when the week changes — replace the "where we
are" paragraph and update the milestone row. Keep it under ~200
lines forever.
