# NOW — active truth

> Single source of "what should we be doing right now." Updated as
> work lands. If this file disagrees with anything else in the repo,
> this file wins.

## Current direction

**4-week sprint to a constrained articulatory-TTS demo.** Pivoted
2026-05-09 from a generic conv-MLP body model to a continuous
neural-field area function `A_θ(s, t)` conditioned on SPARC
kinematics + small speaker vector.

Read order for a cold session:
1. [`ARCHITECTURE.md`](../ARCHITECTURE.md) (~600 lines, freshly rewritten).
2. [`workplan/handoff/2026-05-09_course_correction.md`](handoff/2026-05-09_course_correction.md)
   for the *why* — the bridge record from the previous architecture.
3. This file.
4. [`tasks/00_active/001_neural_field_phrase_probe.md`](tasks/00_active/001_neural_field_phrase_probe.md)
   to dispatch.

## Immediate actions for the developer

In order:

1. **Kill the `pretraining/002b` R1-distill worker** if it's still
   running anywhere. The handoff lists three worktree agents
   (`[private worktree identifiers omitted]`); none are visible in `ps -ef` on this box,
   but they may be on another machine or stuck in a worktree. To
   find + kill:
   ```bash
   ps -ef | grep -E "train_body_model|gen_body_corpus" | grep -v grep
   # if any → kill -TERM <PID>
   ```
   The artifact has zero load-bearing role in the new plan and
   consumes GPU that Week 1 needs. Archive any partial checkpoint
   as `outputs/body_model_ckpt/r1_distill_aborted_2026-05-09/`
   with `promotable: false` in metadata.

2. **Record the Week-1 reference audio.** One Harvard sentence in
   a quiet room, mono 22.05 kHz, ~2-3 seconds. Recommended:
   *"The juice of lemons makes fine punch."* (vowels, fricatives,
   stops, voiced/unvoiced contrast). Save to
   `data/probes/juice_lemons_ref.wav`.

3. **Dispatch [`tasks/00_active/001_neural_field_phrase_probe.md`](tasks/00_active/001_neural_field_phrase_probe.md).**
   Either as a fresh Claude Code session or as an Agent worker
   (`subagent_type: general-purpose`, `isolation: worktree`) with
   the task file as the prompt.

4. **Decide on SPARC adoption** before Week 2. Read
   [`docs/references/core/2024_sparc_vocal_tract_kinematics.pdf`](../docs/references/core/2024_sparc_vocal_tract_kinematics.pdf).
   The decision is training-only, so the inference architecture
   isn't held hostage.

## Active tasks

Five tasks, one per Week. Strict serialization — Week N+1 is blocked
on Week N's GREEN verdict. Do not parallelize; Week 1's result
determines the rest.

| # | File | Week | Status |
|---|---|---|---|
| 001 | [neural_field_phrase_probe.md](tasks/00_active/001_neural_field_phrase_probe.md) | 1 | dispatchable |
| 002 | [sparc_extraction_pipeline.md](tasks/00_active/002_sparc_extraction_pipeline.md) | 2 prep | blocked-on-001 |
| 003 | [speaker_conditioned_body_model.md](tasks/00_active/003_speaker_conditioned_body_model.md) | 2 | blocked-on-002 |
| 004 | [controller_with_sparc.md](tasks/00_active/004_controller_with_sparc.md) | 3 | blocked-on-003 |
| 005 | [constrained_demo.md](tasks/00_active/005_constrained_demo.md) | 4 | blocked-on-004 |

## Blocked on developer decisions

None block Week 1.

- **SPARC adoption.** Training-time teacher only, separate PyTorch
  env (`tools/sparc/.venv/`), zero JAX-side dep. License + WavLM
  weight access verification before bulk extraction.
- **TEA-non-cooperative fallback policy.** If Week-1 + 2 debug days
  fail the gate: ship demo on DDSP-articulatory and reframe the
  project as "physics-anchored articulatory TTS with neural
  production rendering." Surfaced as Risk C in the
  course-correction doc.
- **Cloning ambition for v0.** The new plan ships at
  "anatomy + source fitting from 30-s reference," not full zero-shot
  cloning. Full zero-shot is post-v0.

## What was killed in this pivot (executed 2026-05-09)

- **Body-model design** — generic conv-MLP → 385 renderer fields.
  Code moved to [`legacy/body_model_conv_mlp.py`](../legacy/body_model_conv_mlp.py).
- **Conv-MLP training scaffolding** — `scripts/{gen_body_corpus,
  train_body_model, eval_body_model}.py` and
  `tractus_controller/tests/body_model/` moved to `legacy/`.
- **Primary loss = area-MSE against Story 1996.** Demoted to
  held-out sanity gate.
- **5-phase serialization** (P0 → P1 → P2 → P3 → P4). Replaced by
  the 4-week sprint.
- **Creator/orchestrator/worker three-role dispatch model.** Retired.
- **33 prepivot task files + INDEX.md.** Moved to
  [`workplan/archive/prepivot/`](archive/prepivot/) — preserved
  for archaeology, not for execution.
- **`pretraining/002b` R1-distill artifact** — see action #1 above.

## What was kept

- TEA simulator + JAX backend + custom_vjp / checkpoint backward.
- 20-channel gesture inventory (`channels.py`).
- 50 Hz frame interpolator (`interp.py`).
- Reachability bounds (`reachability.py`, repurposed as neural-field
  barrier priors).
- Story anchor loader (`anchors.py`, repurposed as sanity gate).
- `runbooks/whisper_gate.sh` + `runbooks/source_tract_sep.py`.
- Story anchor data + transfer-function probes
  (`scripts/transfer_function_oral_only.py`).
- Inverse-fit demo (`scripts/inverse_fit_demo.py`) — kept as the
  empirical evidence for the parameterization pivot.
- `pretraining/001` planner module + smoke-train CLI — partially
  compatible with the new direction (planner emitting phones + F0
  + loudness is unchanged).
- `correctness/004b` frication path — TEA-internal, unchanged.

## Latest gate results (pre-pivot)

- TEA forward correctness: **simulator works.** Story F2 within
  1.7 % for /i/ via oral-only transfer-function probe.
- TEA backward at T=118 b=4: **708 MB peak** (passes
  ARCHITECTURE.md §5.2 G2).
- Smoke-pipeline Whisper transcription: **empty/garbage.**
  Triggered the pivot. Result is consistent with `infra/006`'s
  "discrete-bin parameterization can't place 3 formants jointly"
  finding.

## How to dispatch

Either:

- **Fresh Claude Code session.** Paste the task file as the first
  message. The worker self-validates against the task's gates
  before reporting.
- **Agent subagent.** `subagent_type: general-purpose`, prompt is
  the task body, `isolation: worktree`.

## Daily progress log

Append to [`workplan/LOG.md`](LOG.md) when a task gates green or
red. Keep entries short — date, task, verdict, link to artifacts.

## Codex thread (continue if needed)

`[session identifier omitted]` — multi-turn deliberation
that produced this direction. Use `codex-reply` to continue;
**don't** start a fresh thread for body-model questions unless cold
review is specifically wanted (the previous Story-anchor thread
`[session identifier omitted]` is superseded).
