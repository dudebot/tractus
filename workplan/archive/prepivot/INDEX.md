# tractus workplan — INDEX

> ⚠️ **SUPERSEDED 2026-05-09.** The 5-phase serialization and the
> creator/orchestrator/worker dispatch model below are no longer the
> active plan. **Read [`workplan/NOW.md`](NOW.md) and
> [`workplan/handoff/2026-05-09_course_correction.md`](handoff/2026-05-09_course_correction.md) first.**
> The status table below remains as historical record of where
> prepivot tasks landed; the dependency graph is archived. Active
> tasks live under [`tasks/00_active/`](tasks/00_active/).

> Living index of dispatchable tasks, dependencies, decision points,
> and roles. **Single source of truth** for "what should we do next?"
> Update in the same commit when a task lands; do not let it drift.

## Roles

This is a three-session model:

1. **Creator session.** Has full architectural context. Authored
   the task prompts, runbooks, and acceptance gates. Reactivated
   whenever a phase closes to **validate**: spawn Explore + Codex
   stress-tests on what landed, decide pass/fail, write the handoff
   for the next phase.
2. **Orchestrator session(s).** Reads `INDEX.md`, picks the next
   unblocked task, hands its prompt file to a worker. Updates
   status. Does NOT itself implement; routes only.
3. **Worker sessions.** Fresh Claude Code sessions handed a single
   task prompt. Each task file is self-contained — no prior context
   assumed. Worker runs the task's automated acceptance gates
   (`runbooks/`) before reporting done.

The user is **not** part of the dispatch loop. The user spawns the
creator session at phase boundaries and answers the explicit
decision-point tasks below. Whisper handles "is this gibberish";
runbooks handle pass/fail; the creator session handles
architectural alignment.

> **Future-skill candidate.** This three-role pattern is reusable
> across projects. Worth distilling into a Claude Code skill once
> tested through one full phase-completion cycle. Tracked but not
> built this session.

## START HERE — what to dispatch first

Phase folders are numbered (`00_infra/`, `01_correctness/`, … so
`ls workplan/tasks/` gives execution order at a glance.

**Day-1 dispatchable in parallel** (no blockers):

1. **`tasks/00_infra/001_test_overhaul.md`** — testing floor.
2. **`tasks/00_infra/003_runbooks_scaffolding.md`** — fleshes out
   the 6 runbook stubs into real implementations. Acceptance gates
   throughout the workplan invoke these.
3. **`tasks/01_correctness/001_fp32_audit.md`** — cheap immediate
   win; potentially resolves OOM/NaN gotchas as a side effect.
4. **`tasks/04_pretraining/001_planner_pretrain.md`** — orthogonal
   to TEA work; sees no audio loss; can run in parallel from day 1.

**Day-2 onward**: see Status table for what unblocks as the day-1
tasks land. The dependency graph is the authoritative source.

## How a task is dispatched

Three modes:

1. **Fresh Claude Code session** — paste the task file's contents
   as the first message. Worker self-validates against acceptance
   gates and reports.
2. **Subagent (Agent tool)** — pass the task body as `prompt`. Pick
   `general-purpose` for write tasks, `Explore` for read-only audits.
3. **Local execution only** — pure data-driven runbooks
   (`bash runbooks/<name>.sh ...`). No LLM in the loop.

When a task starts, the orchestrator updates its row in the table
below (Status: `pending` → `in-progress`). When done, the worker
runs the acceptance gates; the orchestrator marks `done` and
updates any tasks the completed one unblocks.

## Phase order and gating

```
Phase           Gate
P0  correctness        ALL of correctness/* done before inversion/* starts
P1  smoke              P0 done; produces eval anchor
P2  inversion          P0 + P1 + infra/001 done
P3  pretraining        P0 + P1 + inversion/006 done
P4  cloning            P3 done
P5  research           P4 done

INFRA  infra/001 blocks correctness/002
       infra/003 (runbooks) gates all acceptance
       infra/002 (CI) lands at P0 close
```

## Dependency graph

```
infra/003 (runbooks)             ──┐
infra/001 (test overhaul)        ──┼──► correctness/002 (M5 forward correctness)
correctness/001 (fp32 audit)     ──┴──► correctness/{002,005,006}

correctness/002 (M5 forward)     ─────► correctness/003 (M6 soft)
correctness/003 (M6 soft)        ─────► correctness/004 (M7 frication)
correctness/004 (M7 frication)   ─┐
correctness/005 (backward OOM)   ─┼──► P0 closed
correctness/006 (backward NaN)   ─┘

infra/002 (CI guardrails)        ─────► lands at P0 close

smoke/001 (rule-based path)      ─────► smoke/002 (intelligibility gate)
smoke/002 (intelligibility)      ─────► P1 closed → P2 starts

inversion/001..005 (fitting,     ─────► inversion/006 (dim discovery)
                  anatomy)
inversion/006 (D_body chosen)    ─────► pretraining/{002,003} ready

pretraining/001 (planner)        ─── parallel to pretraining/{002,003}
pretraining/002a (Story anchors) ─┐
pretraining/002b (R1-distill)    ─┼──► pretraining/002 (anchor-grounded body model)
pretraining/006 (SPARC)          ─┘    (002b infra reusable; 006 optional but recommended early)
pretraining/002 (body model)     ─┐
pretraining/003 (controller)     ─┼──► pretraining/004 (speaker enc)
pretraining/005 (surrogate)      ─┘
pretraining/004 (speaker enc)    ─────► P3 closed

cloning/001 → 002 → 003 → 004 → 005    serial
research/* — no v0 deps; post-P4
```

## Status table

Effort: **S** ≤ 1 session, **M** 1–2 sessions, **L** 3+ sessions.
User-input: `none` / `decision` / `review`.

### infra — cross-cutting

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [infra/001](tasks/00_infra/001_test_overhaul.md) | Test overhaul against new architecture | M | none | done |
| [infra/002](tasks/00_infra/002_ci_guardrails.md) | CI guardrails (surrogate ban, residual cap, channel smallness, cheat-channel ablation, regression lint) | M | none | done |
| [infra/003](tasks/00_infra/003_runbooks_scaffolding.md) | Runbook scaffolding (whisper, source/tract sep, plausibility, anticheat, dim, matched-ref) | M | none | done |
| [infra/005](tasks/00_infra/005_fixture_regeneration.md) | Audio-fixture regeneration (30 NPZ at m5_oracle/, unblocks correctness/001 deferred gates + correctness/002) | S | none | pending |
| [infra/006](tasks/00_infra/006_target_sound_inverse_fit.md) | Target-sound inverse-fit demo: differentiable spectral-template loss + per-vowel gradient-descent through TEA on tube area function (settles whether simulator can produce literature-formant shapes at all) | M | none | done — partial convergence: each vowel locks ONE formant correctly (/a/ F1 4%, /i/ F1 12%, /u/ F3 1%) but not all three jointly. Simulator IS capable; area-function-only parameter space too narrow to coordinate multi-formant placement. See `workplan/handoff/2026-05-07_inverse_fit_findings.md` |

### correctness — TEA correctness (P0; gating)

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [correctness/001](tasks/01_correctness/001_fp32_audit.md) | fp32 audit: drop fp64-by-default in TEA forward | S | none | done (2 gates deferred to infra/005 fixture-regen) |
| [correctness/002](tasks/01_correctness/002_m5_forward_correctness.md) | M5 forward-correctness gate (fp32 self-consistency + per-axis sep + energy invariant) | M | none | partial (5/6 dispatchable gates ✓; saved-fixture parity deferred → infra/005) |
| [correctness/003](tasks/01_correctness/003_m6_soft_gates.md) | M6 soft-mode gates: gradient stability over 10K forwards | M | none | done |
| [correctness/004](tasks/01_correctness/004_m7_frication.md) | M7 per-section gated frication noise (CORE) | L | review | partial (004a: `tractus_engine_advanced.frication` module + 12 unit-test gates ✓; MNA `monopole_source_amp` plumbing + custom_vjp through `step_one` ✓; pipeline + smoke wiring ✓; **audible-frication gate FAIL → correctness/004b**) |
| [correctness/004b](tasks/01_correctness/004b_frication_audible.md) | M7 audible-frication: PSD shaping (sibilant 6-8 kHz peaks) + MNA injection-point stability fix (currently NaNs above 0.5 cm³/s) + smoke audio level fix (so Whisper has measurement headroom) | M | escalation-only | partial (PSD shaping via per-place SOS biquads ✓; MNA injection relocated to post-solve `update_variables` net-flow term — stable at 1/5/15 cm³/s ✓; bit-equality at `source_amp=0` ✓; 12/12 + 5/5 unit tests ✓; frication-band 4-8 kHz lift PASS 9/10 utt, mean +8.2 dB; **Whisper gate inconclusive** — empty/garbage transcripts on smoke pipeline both pre AND post → escalation per body-model floor; smoke/002 / pretraining/002 P0-blocking) |
| [correctness/005](tasks/01_correctness/005_backward_oom.md) | Backward OOM fix: jax.checkpoint on per-sample scan | S | none | partial (bit-equal + T=20 grad-finite ✓; T=118 b=4 deferred → correctness/007) |
| [correctness/006](tasks/01_correctness/006_backward_nan.md) | Backward NaN bisect | S | none | done |
| [correctness/007](tasks/01_correctness/007_backward_chunked_scan.md) | Backward chunked-scan / custom_vjp on MNA solve (closes 005's T=118 b=4 gate) | M | none | partial (iter 1 + iter 2 landed → T=118 b=1 backward PASSES at 11.2 GB; ≤ 2 GB / b=4 gate deferred to correctness/008) |
| [correctness/008](tasks/01_correctness/008_backward_step_vjp.md) | Custom VJP on `step_one` to bypass per-sample M materialization (closes 007's b=4 gate) | M | none | done (T=118 b=4 backward: 708 MB peak in subprocess ≤ 2 GB G2 gate; verifier b=4 stage isolated to avoid cumulative-peak poisoning) |
| [correctness/009](tasks/01_correctness/009_tea_fixed_modes.md) | TEA fixed-mode characterization (subglottal/source-side parasitic peaks at ~1287/2719/3400/4054/4667 Hz surfaced during 002a anchor calibration) | S | none | pending (does not block pretraining/002) |

### smoke — TEA-only speech smoke (P1)

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [smoke/001](tasks/02_smoke/001_text_audio_smoke.md) | Rule-based G2P + template gestures + TEA, regression smoke | M | none | in-progress (front-half scaffolding landed; 20/20 corpus renders pass RMS+finite gates @ 22.05 kHz; TEA renders native 44.1 kHz, resamples at output stage; intelligibility gate gated on P0 close) |
| [smoke/002](tasks/02_smoke/002_intelligibility_gate.md) | Intelligibility + acoustic-sanity gate on smoke set | S | none | pending |

### inversion — Inverse fitting + anatomy enrichment + dimensionality (P2)

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [inversion/001](tasks/03_inversion/001_short_window_inverse.md) | Short-window TEA inverse refinement loop | M | none | pending |
| [inversion/002](tasks/03_inversion/002_anticheat_harness.md) | Anti-cheat synth-recovery harness | S | none | pending |
| [inversion/003](tasks/03_inversion/003_source_tract_sep.md) | Source/tract separation metrics on every fit | S | none | pending |
| [inversion/004](tasks/03_inversion/004_anatomy_synthetic_holdout.md) | Anatomy fitting on synthetic-baseline holdout improves ≥ 1 dB | M | none | pending |
| [inversion/005](tasks/03_inversion/005_anatomy_enrichment.md) | Anatomy enrichment 9 → 30–60 scalars; multi-speaker validation | M | review | pending |
| [inversion/006](tasks/03_inversion/006_dimensionality_discovery.md) | **Body-state dimensionality discovery** | L | **decision** | pending |

### pretraining — Module pretraining (P3; gated on inversion/006)

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [pretraining/001](tasks/04_pretraining/001_planner_pretrain.md) | Pretrain planner on multi-speaker text+alignment+F0 corpus | M | review | in-progress (module + smoke-train CLI landed, ~3.5M params; full LibriTTS-R/MFA/DIO corpus + eval gates deferred) |
| [pretraining/002](tasks/04_pretraining/002_body_model_pretrain.md) | Pretrain body model — anchor-grounded (Story + R1 fixtures + optional SPARC) | L | none | pending (re-scoped 2026-05-08; blocked by 002a + 002b) |
| [pretraining/002a](tasks/04_pretraining/002a_story_anchor_ingest.md) | Story 1996 vocal-tract area-function anchor ingestion (11 of 12 vowels via Story 2008 open-access; /ɝ/ + 3 nasals + 3 plosives missing) | S | none | done (data layer). Calibration verified 2026-05-09 via `scripts/transfer_function_oral_only.py` (oral-only isolation: open glottis + ablated side branches + mouth-flow output + target-windowed peak assignment): TEA matches Story F2 within 1-25% across vowels; front-vowel F2 reaches target (/i/ F2=2175 vs 2139). Initial worker's "TEA calibration gap" diagnosis was measurement methodology artifact (LPC + voiced source at F0=120 + parasitic side-branch peaks). Fixed-mode investigation tracked separately as correctness/009. Anchor data is calibrated ground truth for pretraining/002 area-function targets. |
| [pretraining/002b](tasks/04_pretraining/002b_r1_distill_baseline.md) | R1-distill body baseline (training infra precursor — quarantined artifact, NOT pretraining/002 v0) | L | none | done (infra layer). Sessions 1+2 produced: body_model package (12 unit tests ✓) + gen_body_corpus.py + train_body_model.py (724 lines: per-channel-std-normalized MSE + smoothness + static + refusal losses) + eval_body_model.py + 4 train smoke tests (16/16 total pass). Provenance discipline (artifact label `r1_distill_body_baseline`, teacher-provenance metadata in checkpoint, R1-vs-distill comparison probe per Codex) deferred to whoever first runs full training. |
| [pretraining/003](tasks/04_pretraining/003_controller_pretrain.md) | Pretrain controller on synthetic gestures + short-window TEA | M | none | pending |
| [pretraining/004](tasks/04_pretraining/004_speaker_encoder.md) | Pretrain speaker encoder; channel + identity loss | M | none | pending |
| [pretraining/005](tasks/04_pretraining/005_surrogate_train.md) | Train surrogate (5A) on 100K (anatomy, controls)→TEA-waveform | L | review | pending |
| [pretraining/006](tasks/04_pretraining/006_sparc_bootstrap.md) | SPARC bootstrap as body-model teacher (decision deferred; Codex 2026-05-08 recommends moving earlier) | M | **decision** | pending |

### cloning — 30s clone end-to-end demo (P4)

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [cloning/001](tasks/05_cloning/001_baseline_self_clone.md) | Default-baseline self-clone end-to-end smoke | S | none | pending |
| [cloning/002](tasks/05_cloning/002_cross_speaker.md) | Cross-speaker generalization, 5 reference speakers, listener A/B | M | review | pending |
| [cloning/003](tasks/05_cloning/003_inference_8gb_bench.md) | 8 GB inference benchmark on consumer GPU @ 22.05 kHz | S | none | pending |
| [cloning/004](tasks/05_cloning/004_residual_ablation.md) | TEA-only AND TEA+residual ablation reported every eval | S | none | pending |
| [cloning/005](tasks/05_cloning/005_matched_reference_eval.md) | Matched-reference evaluation vs commercial baselines | M | review | pending |

### research — Quality push (P5; post-v0)

| ID | Title | Effort | User-input | Status |
|---|---|---|---|---|
| [research/001](tasks/06_research/001_procedural_body.md) | Procedural body model exploration | L | **decision** | pending |
| [research/002](tasks/06_research/002_anatomy_beyond_60.md) | Anatomy basis enrichment beyond 30–60 if needed | M | none | pending |
| [research/003](tasks/06_research/003_multilingual_planner.md) | Multilingual planner | L | none | pending |
| [research/004](tasks/06_research/004_long_reference_clone.md) | Long-reference (1–3 min, 30+ min) cloning vs commercial baselines | M | review | pending |
| [research/005](tasks/06_research/005_cross_species_phoneme_inference.md) | Cross-species phoneme inference (bird syrinx + cetacean source modules + info-theory layer for vocab engineering) | L | **decision** | pending |

## Decision points (only places the user is a synchronous bottleneck)

1. **inversion/006 acceptance — pick `D_body_effective`.** Runbook
   produces PCA + acoustic-Jacobian SVD + dimensionality sweep
   summary. User selects smallest D that clears
   intelligibility + identity + anti-cheat gates.
2. **pretraining/006 — SPARC bootstrap yes/no.** External-dependency
   adoption decision. ~20 min.
3. **research/001 — procedural body model multi-month investment.**
   Fires only after cloning/* shows where learned body model
   plateaus. ~1 hour with evidence in hand.

## Runbook inventory

| Script | Gates | Used by |
|---|---|---|
| [whisper_gate.sh](runbooks/whisper_gate.sh) | ASR + WER + jaccard | smoke/002, inversion/002, cloning/001, cloning/002 |
| [source_tract_sep.py](runbooks/source_tract_sep.py) | Per-axis F0 / spectral tilt / frication / nasality / formants | correctness/002, correctness/003, correctness/004, inversion/003, pretraining/* |
| [plausibility_panel.py](runbooks/plausibility_panel.py) | Continuous-time velocity/accel/jerk + drift + boundary alignment | inversion/001, inversion/003, pretraining/003 |
| [anticheat_unreachable.py](runbooks/anticheat_unreachable.py) | Transcript-preserving unreachable-target generator + refusal gate | inversion/002, inversion/004, pretraining/002 |
| [dimensionality_report.py](runbooks/dimensionality_report.py) | PCA + acoustic-Jacobian SVD + sweep summary | inversion/006 |
| [matched_reference_eval.sh](runbooks/matched_reference_eval.sh) | Commercial-baseline comparison at matched reference length | cloning/005 |

## Phase entry / exit criteria

**P0 (correctness) closed when:**
- infra/001, infra/003 done.
- correctness/001 through correctness/004 done.
- correctness/005 done OR superseded by correctness/007 close
  (chunked-scan / custom_vjp closes the production-T backward gate).
- correctness/006 done.
- fp32-vs-fp64 self-consistency on synthetic corpus passes.
- M6 gradient FD agrees with autograd over 10K seeds.
- M7 frication produces audible /s/, /ʃ/, /f/, /θ/, /z/, /v/ +
  aspiration on a fixed phone-list test.
- Backward at production T fits ≤ 2 GB at batch 4 (ARCH §6 G2); no NaN.
- Validator session signs off.

**P1 (smoke) closed when:**
- Rule-based path produces ≥ 80% phone-WER intelligibility on
  20–50 utterance smoke set (Whisper-gated).
- F0 contour follows planner targets within 50 cents RMSE.
- "Daisy, Daisy" regression anchor renders.
- Validator signs off.

**P2 (inversion) closed when:**
- Anatomy fitting on synthetic-baseline holdout improves ≥ 1 dB.
- Anti-cheat refusal rate ≥ 80% on unreachable targets.
- Anatomy basis enrichment to 30–60 scalars validated.
- `D_body_effective` selected from inversion/006 report.
- Validator signs off.

**P3 (pretraining) closed when:**
- Surrogate (5A) log-mel ≤ 1.0 dB to TEA on held-out trajectories.
- All four learned modules pretrained, eval gates passing.
- Identifiability ledger CI passes.
- Validator signs off.

**P4 (cloning) closed when:**
- 30s clone fits 5 unseen speakers; held-out reconstruction passes
  intelligibility + identity gates.
- 8 GB inference benchmark passes at 22.05 kHz.
- TEA-only ablation remains intelligible without residual.
- Matched-reference comparison report produced.
- Validator signs off.

## Glossary pointer

Canonical glossary in `../ARCHITECTURE.md` §14.
