# Assumptions audit — synthesis

> Unified deliverable for `workplan/tasks/00_active/008_assumptions_audit.md`.
> Three sub-docs hold per-category detail; this doc surfaces the
> top-level findings and the recommended actions.

## Sub-doc index

| Category | Doc | Entries | Top concern |
|---|---|---|---|
| 1 — SPARC stack | [`assumptions_audit_sparc.md`](assumptions_audit_sparc.md) | 7 | EMA z-score is training-time only; physical-mm is structurally unrecoverable from the published checkpoint |
| 2 + 4 — TEA + loss | [`assumptions_audit_tea_loss.md`](assumptions_audit_tea_loss.md) | 10 | **Sample-rate contract silently broken in copyfit; all Week-1 results are octave-mismatched** |
| 3 — Workplan / scope | [`assumptions_audit_workplan.md`](assumptions_audit_workplan.md) | 7 | "Kinematic supervision is structurally required" is UNDETERMINED, not proven (Codex adversarial review) |

## TL;DR — the four findings that change roadmap

These are the items where, if the assumption is wrong, downstream
architecture decisions need to be revisited. Ranked by severity.

### Finding 1 — Octave-mismatched acoustic loss (CRITICAL)

**What:** `scripts/copyfit_neural_field_phrase.py` declares
`SAMPLE_RATE = 22050` and `SAMPLES_PER_FRAME = 441`, then calls
`synthesize_dynamic_tube(samples_per_frame=441, ...)` without
overriding `dt`. TEA's `TIMESTEP_S` is hard-coded to `1/44100` in
`tractus_engine_advanced/tds/constants.py:20`. So TEA renders 441
samples representing 10 ms of real time at 44.1 kHz, but the loss
treats those 441 samples as 20 ms of audio at 22.05 kHz. From the
loss's perspective, all spectral content is shifted down by an
octave.

**Source:** TEA audit doc, finding 1. Confirmed by Codex thread
`[session identifier omitted]` (with sign-direction
correction). Already documented in `scripts/render_m5_oracle.py:17-30`
which explicitly calls the 22.05 kHz spec "aspirational — would
require either an engine-level re-derivation of TIMESTEP_S or a
post-render decimation." Copyfit author missed this.

**Impact on Week 1:**
- ALL three closed runs (`juice_lemons_run100`, `juice_lemons_smooth100`,
  `synth_target_static_A`) trained against an octave-mismatched
  target. The "phonetic wall" verdict is un-evaluable — Whisper
  transcribing as "I" / "Uh" is consistent with the audio being
  generated correctly by the simulator but compared against
  spectrally-incompatible features.
- `docs/week1/findings.md` closure narrative is invalidated as
  evidence for the structural-supervision claim. Doesn't mean the
  claim is wrong — means there is no evidence either way.
- The frication 200-iter run started earlier this session was killed
  before it could waste 2.5 GPU-hours on the same broken contract.

**Impact on Week 2:**
- The Week-2 SPARC dispatch we GREEN-stamped earlier this session is
  technically still correct, but its Week-1 motivation is gone.
- Codex's UNDETERMINED verdict on "kinematic supervision is
  structurally required" gets sharpened: the evidence for the claim
  was not just non-converged + frication-disabled, it was rendering
  the wrong audio entirely.

**Fix options:**
- (A) Engine-side: re-derive `TIMESTEP_S` to be a parameter, not a
  module constant. Touches `tractus_engine_advanced/tds/constants.py`
  and propagates through MNA matrices. Largest blast radius; cleanest.
- (B) Copyfit-side: update `SAMPLE_RATE = 44100`, regenerate target
  audio at 44.1 kHz, set `SAMPLES_PER_FRAME = 882`. Smallest blast
  radius; doesn't help any other consumer of the engine.
- (C) Copyfit-side: leave 22.05 kHz target, decimate TEA's 44.1 kHz
  output 2× before computing loss. Mid-blast; may have anti-aliasing
  concerns.

Codex's recommended TEA-to-TEA discriminating experiment (see
finding 4 below) MUST get this fix first or the experiment results
will be useless.

### Finding 2 — "Kinematic supervision is structurally required" is UNDETERMINED

**What:** The Week-1 closure verdict that justifies all of Weeks 2-4.
Codex adversarial review (no prior context, fresh thread, verdict-
forcing prompts) ranked the supporting evidence as VOID / WEAK /
WEAK-to-overclaimed across the three evidence streams. Final verdict
**UNDETERMINED**.

**Source:** Workplan audit doc §2. Codex thread
`[session identifier omitted]`. Now compounded by Finding 1 —
the supporting evidence wasn't just inconclusive, it was rendered on
the wrong acoustic substrate.

**Codex's recommended discriminating experiment:** TEA-to-TEA
dynamic copyfit with known ground-truth articulation, frication on,
longer optimization, multi-seed, vowel-heavy + consonant-heavy
targets. Measure both acoustic AND kinematic recovery. ~1-2 days
of compute vs ~3 weeks of SPARC integration.

**Decision:** pause Task 002 (SPARC pipeline) pending the
discriminating experiment. The Week-2 SPARC dispatch is technically
correct but its motivation needs re-establishing.

### Finding 3 — SPARC EMA outputs are in MNGU0-OLS-output-space, NOT z-scored or physical-mm

**What:** SPARC's published checkpoint outputs EMA in a coordinate
system that has no published normalization metadata. The Task 002
schema we just GREEN-stamped offers `ema_normalization` choices of
`"sparc_api_native"`, `"z_score_within_utterance"`, or
`"physical_mm"`. The latter two are wrong claims — physical mm is
structurally unrecoverable from the published checkpoint because
training z-scoring was within-utterance, so the per-utterance scale
was discarded during training and cannot be recovered with any amount
of MNGU0-statistics post-processing.

**Source:** SPARC audit doc, entry 6. Codex confirmed.

**Fix:** edit `workplan/tasks/00_active/002_sparc_extraction_pipeline.md`
to replace the three-option `ema_normalization` field with a single
`"mngu0_ols_output_space"` value + an explicit note that absolute
coordinate semantics are not recoverable. Body model in Task 003
must learn the projection rather than rely on a known coordinate
system.

### Finding 4 — Workplan budgets (≤ 8 GB / ≤ 30 M / 5x / 30 s) are uncited

**What:** None of the four budget claims in `ARCHITECTURE.md` §0.2
have a cited rationale. "Consumer hardware" and "smaller than Kokoro
82 M" are positioning statements, not constraints. The 8 GB number
specifically was used in the course-correction doc to disqualify
SPARC at inference time — if 8 GB is itself wrong or under-justified,
that disqualification weakens.

**Source:** Workplan audit doc, entries 1.1-1.5.

**Fix:** dedicated decision task `010_workplan_budget_decisions.md`
where the developer either documents the rationale or removes the
numbers. Cheap admin task; high downstream value.

## Other findings (non-roadmap-altering, but worth fixing)

From the SPARC audit:
- Periodicity threshold: paper §B.1 says 0.4, code defaults 0.0.
  Add a one-line comment in `extract_sparc_labels.py` so future
  workers don't double-mask.
- WavLM at non-16-kHz: not a refusal, silent frequency shift +
  frame-rate change. SPARC's loader resamples for you, so v0 is
  safe; risk is any future code path that bypasses `Coder.encode()`.

From the TEA + loss audit:
- ARCHITECTURE.md §5.2 line 347 propagates the wrong "22.05 kHz"
  claim. Same bug as Finding 1; fix together.
- Loss-weight defaults (`W_SMOOTH_S=1e-3`, etc.) are
  inherited-unverified from the task spec's "suggested initial
  weights — tune in Week 1 itself" line. Were never tuned. Codex
  later suggested 10-30× higher smoothness in commit `c639a73` but
  only as CLI overrides; defaults still hold the unverified values.

From the workplan audit:
- 4-week sprint structure is self-imposed and is biting us — Week 1
  closed prematurely partly because of it. Make sprint pacing soft;
  add a "verify the closure verdict before declaring closure" gate.

## Recommended action sequence

In priority order. Each is a separate decision the developer should
make explicitly.

1. **Fix Finding 1 (sample-rate contract) before any further copyfit
   experiments.** Pick option (A), (B), or (C). Recommend (B) for
   speed; (A) if the engine is going to need per-render-rate flexibility
   anyway.

2. **Re-run the Week-1 baseline + frication retest** with the fix
   applied. This re-establishes (or refutes) the Week-1 evidence on
   correct acoustic substrate. Outputs to
   `outputs/copyfit_neural_field/juice_lemons_run100_fixed/` and
   `juice_lemons_frication200_fixed/`.

3. **Run Codex's discriminating experiment** (TEA-to-TEA dynamic
   copyfit) AFTER the fix. This gives a clean answer on whether
   acoustic-only inversion can recover known articulation in-model.
   New task: `009_tea_to_tea_discriminating_experiment.md`.

4. **Pause Task 002 (SPARC pipeline)** until step 3 reports. If step
   3 says acoustic-only works in-model, SPARC moves from "structurally
   required" to "potentially useful auxiliary signal" — the Week-2
   plan changes shape.

5. **Fix Finding 3 (SPARC schema language)** in Task 002 regardless
   of pause status — this is just a doc fix and prevents a future
   worker from making a wrong assumption when Task 002 is unpaused.

6. **Surface Finding 4 (budgets)** to the developer as a separate
   doc-only task. Cheap.

7. **Housekeeping items** (sprint-pacing softening, periodicity
   comment, ARCHITECTURE.md fix, loss-weight defaults review) can
   bundle into Task 006 or get separate small tasks.

## Audit verdict

The audit caught one critical, two roadmap-altering, and one
schema-altering finding in ~30 minutes of agent + Codex time.
ROI is high; this is a healthy pattern to repeat at every closure.

The user's framing — "it's been a repeating process where we assume
one requirement and the cargo cult engineering fucks us later" — was
correct. The two cargo-culted requirements that bit us hardest were
both inherited from sources within the project itself
(`render_m5_oracle.py` already documented Finding 1; the Week-1 task
spec already documented "tune these weights" without anyone tuning).
The audit found nothing the project didn't already, in some sense,
know.

The fix going forward is not "be smarter about external requirements"
— it's "before declaring a closure, re-read the engine constraints
and confirm the experiment was well-formed."
