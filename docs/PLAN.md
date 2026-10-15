> **Historical roadmap — superseded by the final archive decision (2026-10-04 UTC).** Development is stopped. There is no active backlog or resumption issue. See the [final assessment](archive-2026-10-04/assessment.md). The original material below is preserved as evidence.

# tractus — the plan

> **Front door.** If you read one file to know what this project is
> doing right now, read this one. Everything else is either an input
> to the single open question below, or archaeology that explains how
> we got here.
>
> This file is a **restructure**, not new work. It re-frames the
> existing planning corpus (`docs/`, `workplan/`) around the one
> experiment the project currently turns on. It does not supersede
> `ARCHITECTURE.md` (the *what*) — it supersedes the *ordering* of
> everything in `workplan/` and `docs/week1/`.

---

## The one question

**Can gradient descent through TEA recover a *known* articulation
from acoustic loss alone, in-model, after convergence?**

That's it. That is the question the entire roadmap hinges on. It has
a pre-registered, falsifiable answer coming out of **Task 009**
(`workplan/tasks/00_active/009_tea_to_tea_discriminating_experiment.md`),
with three possible verdicts fixed *before* the run:

| Verdict | Meaning | What it does to the roadmap |
|---|---|---|
| **PROVEN** (acoustic sufficiency) | Both targets, all 3 seeds: log-area MAE < 0.3, position err < 5 sections, max-area err < 0.2 cm² | The "kinematic supervision is structurally required" claim is **falsified**. The 3-week SPARC roadmap (Tasks 002–004) becomes *optional*. Body model can train on synthetic copyfit pairs. |
| **WEAK** | Vowel target recovers, consonant target doesn't | Claim is **narrowed** — SPARC needed for fricatives, not vowels. Partial SPARC integration. |
| **FAILED** | Both targets, all seeds diverge (MAE > 0.6, position err > 10 sections) | Claim is **strengthened**. SPARC integration (Tasks 002–004) proceeds with confidence. |

Until Task 009 reports, **Tasks 002–005 are paused** — not cancelled,
paused. Their specs are correct; their *motivation* is the thing under
test. Dispatching a SPARC worker before 009 lands is spending three
weeks on a claim rated UNDETERMINED.

---

## Why this is the front door (and wasn't before)

The project *felt* like a turbo slog because the plan was organized as
a **4-week sprint** (`workplan/NOW.md`, `workplan/CHECKIN.md`): Week 1
→ Week 2 → ... each week "closing" with a verdict that unlocked the
next. That structure produced its own failure. Week 1 closed with a
confident verdict — *"kinematic supervision is structurally required,
SPARC is non-optional"* (`docs/week1/findings.md`) — that later turned
out to rest on **three contaminating bugs**, none of which were about
articulation at all:

1. **A1 — octave-mismatched loss (the real bug).** copyfit declared
   `SAMPLE_RATE = 22050`, `SAMPLES_PER_FRAME = 441`, and fed TEA output
   (rendered at a hard-coded 44.1 kHz, `tractus_engine_advanced/tds/constants.py:20`)
   into a 22.05 kHz loss. Every rendered frequency appeared **one
   octave down** to the optimizer. It was matching a target the geometry
   physically cannot reach. All three Week-1 runs are octave-contaminated.
   Documented in `docs/assumptions_audit_tea_loss.md` §2.5 and
   `docs/incorrect_assumptions.md` A1. Worst part: `scripts/render_m5_oracle.py:17-30`
   *already documented the 44.1 kHz constraint* — the project knew and
   cargo-culted past its own note.
2. **A2 — not converged.** The 100-iter run was still descending
   (`3.404 → 3.374 → 3.366 → 3.357`) at the last logged iter. 100 was a
   script default, not a convergence criterion (`docs/incorrect_assumptions.md` A2).
3. **A3 — frication off.** copyfit called `synthesize_dynamic_tube`
   without a `frication=` kwarg → `Frication.disabled()`. The reference
   phrase has ~9 fricatives the optimizer structurally could not produce
   (`docs/incorrect_assumptions.md` A3).

A Codex adversarial review (fresh thread, no prior context, verdict-
forcing prompts) rated the load-bearing claim **UNDETERMINED**, not
proven (`docs/assumptions_audit_workplan.md` §2). The three failed
runs weren't evidence about articulation — they were evidence about a
broken sample-rate contract, an early stop, and a disabled feature.

The lesson from the audit's own closing line: *"the audit found nothing
the project didn't already, in some sense, know."* The sprint structure
biased toward **shipping the verdict over verifying the verdict**. So
the restructure removes the sprint as the organizing spine and replaces
it with **the single discriminating experiment**. When 009 reports, the
plan re-forms around its verdict — not around "what's this week's
deliverable."

---

## Narrative skeleton

The restructured plan reads in five beats. Detail lives in the linked
docs; this is the spine.

### 1. The question (this file, above)
Task 009's pre-registered PROVEN/WEAK/FAILED verdict. One experiment,
~30 GPU-hours serial (2 targets × 3 seeds), clean in-model test with no
out-of-model confounders. This is the front door.

### 2. Why the old verdict can't answer it
The Week-1 "SPARC is non-optional" closure was contaminated by A1/A2/A3
and rated UNDETERMINED by adversarial review. Task 007
(`007_frication_copyfit_retest.md`) fixes A2/A3 on *real* audio — but
real audio has out-of-model elements (mic, room, ElevenLabs vocoder,
MP3 roundtrip per A10) that TEA cannot reproduce *regardless* of
articulation, so a 007 failure is **not discriminative**. Only the
TEA-to-TEA test (009) controls every confounder. This is the crux:
**007 narrows, 009 decides.**

### 3. What the answer unlocks
The verdict table above. PROVEN collapses three weeks of roadmap to
optional; FAILED green-lights it with confidence; WEAK splits it. The
SPARC pipeline (Task 002), speaker-conditioned body model (Task 003),
controller (Task 004), and demo (Task 005) are all **downstream of this
single fork** and stay paused until it resolves.

### 4. What's already true regardless of the verdict (the floor)
Independently of 009, these are established and load-bearing:
- **TEA is a working differentiable instrument.** Forward parity vs the
  C++ oracle passes; 11 distinguishable vowel timbres from Story anchors
  (`workplan/LOG.md`, 2026-05-13 sustained-vowel probe). Story F2 within
  1.7% for /i/.
- **The optimizer works in the acoustic sense.** Backward fits in ~5 GB;
  synth-/A/ probe hit acoustic ratio 0.33 — it *converges*, it just
  converged to a structurally-wrong tube (`docs/week1/findings.md`).
- **Articulatory-acoustic non-uniqueness is real** (Atal 1971 et seq.).
  That is not in dispute. What *is* in dispute is whether it's fatal for
  acoustic-only inversion in-model after convergence — which is exactly
  what 009 measures.
- **The anti-neutering floor** (`workplan/CHECKIN.md`): any new body
  model must clear what TEA already produces from known-good R1 tubes.

### 5. The historical grind (appendixed — read only if you need the why)
Everything below is preserved for archaeology, not execution. See the
appendix inventory.

---

## Document inventory

Every planning doc in the corpus, with its role and whether it's a
**KEEP** (live, load-bearing for the current question) or **APPENDIX**
(historical / superseded / archaeology).

### KEEP — live, load-bearing

| Doc | Role | Class |
|---|---|---|
| `docs/PLAN.md` (this file) | Front door. The single open question + the restructured spine. | KEEP |
| `workplan/tasks/00_active/009_tea_to_tea_discriminating_experiment.md` | **The experiment the project turns on.** Pre-registered PROVEN/WEAK/FAILED verdict on acoustic-only articulatory recovery. | KEEP |
| `docs/assumptions_audit.md` | Audit synthesis — the 4 roadmap-altering findings, ranked. Explains *why* 009 exists. | KEEP |
| `docs/incorrect_assumptions.md` | Canonical A1–A10 bug list with wrong-claim sites. The anti-cargo-cult ledger; consult before trusting any inherited constant. | KEEP |
| `docs/assumptions_audit_tea_loss.md` | Detail on the octave-shift bug (A1/2.5) + loss-weight / memory-budget verdicts. The engine-facts source for the H100 note. | KEEP |
| `docs/assumptions_audit_workplan.md` | The full UNDETERMINED takedown of "kinematic supervision structurally required" + budget-citation gaps. | KEEP |
| `docs/future_knobs.md` | K1–K6 deferred config knobs (frame-rate, body-model rate, per-component sample rate, loss rate, place supervision, loss weights). Live deferral register. | KEEP |
| `docs/H100-MIGRATION.md` | What changes when 009 (and everything after) runs on 80 GB instead of the consumer GPU. | KEEP |
| `workplan/CHECKIN.md` | Developer pocket reference: milestone table, PM-review protocol, **anti-neutering cross-check**. The check-valve doc. | KEEP |
| `workplan/LOG.md` | Daily verdict log. Append-only ground truth of what gated green/red. | KEEP |
| `docs/week1/fixed_rerun_hypotheses.md` | Pre-registration of the fixed-copyfit reruns. Model for how 009's verdict should be read. | KEEP |
| `workplan/tasks/00_active/007_frication_copyfit_retest.md` | The real-audio retest that fixes A2/A3 but *cannot* be discriminative — the doc that motivates why 009 is needed. | KEEP |
| `docs/assumptions_audit_sparc.md` | SPARC-stack facts (periodicity 0.4, MNGU0-OLS coordinate space, WavLM-16kHz). Load-bearing *if and when* 009 returns WEAK/FAILED and SPARC proceeds. | KEEP (conditional) |
| `ARCHITECTURE.md` | The *what* / vision / five-module decomposition. Authoritative on any conflict about architecture. Not superseded by this file. | KEEP |
| `CLAUDE.md` | Operational orientation: venvs, precision policy, backward/OOM gotchas, hardware. | KEEP |

### APPENDIX — historical / superseded / archaeology

| Doc | Role | Class |
|---|---|---|
| `workplan/NOW.md` | The 4-week-sprint "active truth." **Superseded by this file** as the organizing spine — its sprint framing is the structure the audit found harmful. Keep for the pivot record (what was killed/kept 2026-05-09). | APPENDIX |
| `docs/week1/findings.md` | The Week-1 closure narrative. Its headline verdict ("SPARC non-optional") is under review per A2/A3/A4 and annotated as such inline. Read for the non-uniqueness *mechanism*, not for its verdict. | APPENDIX |
| `docs/week1/sparc_dispatch_prompt.md` | Copy-paste Week-2 dispatch prompt. Do not dispatch until 009 reports; carries A5/A7/A8 wrong claims (annotated). | APPENDIX |
| `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` | SPARC pipeline spec. Paused pending 009. Correct-but-unmotivated. | APPENDIX (paused) |
| `workplan/tasks/00_active/003_speaker_conditioned_body_model.md` | Speaker-conditioned body model. Blocked on 002. | APPENDIX (paused) |
| `workplan/tasks/00_active/004_controller_with_sparc.md` | Controller with SPARC pseudo-labels. Blocked on 003. | APPENDIX (paused) |
| `workplan/tasks/00_active/005_constrained_demo.md` | v0 demo. Blocked on 004. | APPENDIX (paused) |
| `workplan/tasks/00_active/001_neural_field_phrase_probe.md` | Week-1 deliverable spec. Done (PARTIAL); carries A1/A2/A9 origin claims (annotated). Archaeology. | APPENDIX (done) |
| `workplan/tasks/00_active/006_week1_housekeeping.md` | Pre-Week-2 hygiene sweep. Low-stakes cleanup. | APPENDIX |
| `workplan/tasks/00_active/008_assumptions_audit.md` | The task that produced the audit corpus. Complete; its *outputs* (the audit docs) are the KEEPs. | APPENDIX (done) |
| `workplan/handoff/2026-05-04_session.md` | Creator session that built `workplan/`. Pre-pivot. | APPENDIX |
| `workplan/handoff/2026-05-07_inverse_fit_findings.md` | infra/006 inverse-fit result — the empirical evidence that triggered the parameterization pivot. | APPENDIX |
| `workplan/handoff/2026-05-09_course_correction.md` | The pivot bridge record (conv-MLP → neural-field + SPARC). The *why* behind the current architecture; leans on the now-under-review SPARC claim. | APPENDIX |
| `workplan/handoff/2026-05-09_purification_and_anchors.md` | VTL/JD3 purge + Story-anchor calibration session. | APPENDIX |
| `workplan/handoff/2026-05-13_week1_throughput.md` | Week-1 throughput-bound handoff. **The memory-engineering source** for the H100 note (per-iter walls, K-vs-throughput, VRAM). | APPENDIX (but load-bearing for H100-MIGRATION) |
| `workplan/archive/prepivot/` (33 task files + INDEX) | Pre-pivot 7-phase roadmap. Frozen for archaeology per the pivot. | APPENDIX (archived) |

---

## How to use this plan

- **Cold session?** Read beats 1–4 above, then dispatch or check on
  Task 009. Ignore the sprint framing in `NOW.md`.
- **009 has reported?** Take its verdict to the table in "The one
  question," pick the roadmap branch, and re-form the plan around it.
  Un-pause the relevant tasks or archive them.
- **Tempted to trust an inherited constant** (a sample rate, a loss
  weight, a T=118/b=4 budget, an 8 GB cap)? Check
  `docs/incorrect_assumptions.md` first. That ledger exists precisely
  because "we assumed one requirement and the cargo-cult engineering
  fucks us later."
