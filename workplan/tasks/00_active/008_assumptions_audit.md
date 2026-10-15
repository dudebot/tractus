# Task 008 — Assumptions audit (cargo-cult unfucking)

> **Blocker:** none. Documentation/research task. **Read first:**
> nothing required, but the following surface the recurring pattern
> this task is trying to break:
> - `docs/week1/findings.md` (Week-1 closure was based on a
>   non-converged optimizer + disabled frication — both inherited
>   assumptions)
> - `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` (the
>   SPARC pipeline carries a stack of "this is required" claims that
>   need verification before we cargo-cult them into Week 2)

## Why this exists

Recurring failure mode in this project: a requirement gets stated
somewhere (paper, library doc, prior session note, CLAUDE.md, a Codex
review) and downstream work treats it as ground truth without
verifying. Examples that have already bitten us or are about to:

- **"SPARC needs 16 kHz."** Stated in WavLM model card. Is the actual
  failure mode "wrong predictions" or "model refuses to run"? What
  does WavLM do at 24 kHz input?
- **"loss plateaued at iter 100."** Untrue — the curve was still
  descending. The "phonetic wall" was declared without verifying
  the convergence assumption.
- **"frication is wired in."** False — `frication=None` was passed,
  module defaulted to disabled. No one verified.
- **"22.05 kHz is the right corpus rate."** In the original Week-2
  task spec; SPARC is 16 kHz. Inherited from somewhere unrelated.
- **"≤ 8 GB inference budget, ≤ 30 M params."** Stated in
  ARCHITECTURE.md. Where did those numbers come from? Are they hard
  product constraints (target hardware), soft engineering goals, or
  cargo-culted from a Codex deliberation?

The cost of an unverified requirement is not the requirement itself
— it's the downstream architecture that gets built around it. SPARC
adoption is shaped around "we need a 1.26 GB / 316 M-param model
that requires 16 kHz input"; if any of those is wrong, the whole
training-time-only carve-out may be over-engineered (or
under-engineered).

## Goal

Produce `docs/assumptions_audit.md` cataloguing every load-bearing
requirement currently shaping the project, with a verdict per item
and a confidence level. Then the developer + future workers have a
single place to look up "is X actually required, or did we assume
that?"

This is **documentation work**. It identifies concerns; it does NOT
fix them. Fixes are separate tasks raised based on this audit.

## Categories

### Category 1 — SPARC stack assumptions

Each item: cite source, verdict, failure mode if wrong.

- **16 kHz audio input requirement.** Source: WavLM model card on HF.
  Verify: what does WavLM do at 24 kHz / 22.05 kHz / 48 kHz input?
  Hard refusal, degraded performance with measurable error, or no
  difference? Is the "must resample to 16 kHz" rule from WavLM the
  same as from SPARC's wrapper code? Do they agree?
- **50 Hz frame rate output.** Source: SPARC paper. Is this the SSL
  feature rate (fixed by transformer stride) or a downstream design
  choice? Could we get 100 Hz EMA?
- **EMA channel order TD,TB,TT,LI,UL,LL.** Source: SPARC PyPI.
  Verify against actual `Coder.encode()` output. Is there a version
  where this differs?
- **MNGU0 as canonical articulator space.** Source: SPARC paper +
  Cho et al. 2024. Is this load-bearing for our use case (we're not
  doing speaker-cross AAI), or could we get away with a different
  reference speaker?
- **Periodicity threshold = 0.4.** Source: SPARC paper §B.1. Is this
  tuned for SPARC's downstream tasks or a general CREPE convention?
  What if we keep raw periodicity?
- **EMA z-scored within utterance.** Source: SPARC paper §III.A.1.
  Does the SPARC API de-normalize at inference? If not, what's the
  right way to recover physical units?
- **Speaker-embedding FFN inside vocoder checkpoint.** Source:
  SPARC paper §III.A.3. Is this a deal-breaker for our
  training-time-only setup, or does it just mean we need to download
  the vocoder?

### Category 2 — TEA simulator assumptions

- **Precision policy fp32 default, fp64 opt-in via env var.** Source:
  `tractus_engine_advanced/precision.py`, ARCHITECTURE.md. Why fp32
  in production? Is this a CUDA-on-Blackwell constraint, a memory
  constraint, or an inherited choice? What numerical issues would
  fp64 avoid that we currently work around?
- **`frame_chunk_size=1` default.** Source: `pipeline.py:220`.
  Originally chosen for backward-memory bound at T=118 b=4. With
  the per-sample MNA Jacobian-transpose still OOM-ing, is K=1 still
  the right default for the typical inverse-fit case (T=120 b=1)?
- **T=118 batch=4 as the production memory target.** Source:
  ARCHITECTURE.md §5.2 G2. Where did "118 frames" + "batch=4" come
  from? Is that a planner-output constraint, a corpus constraint,
  or a Codex deliberation?
- **Per-sample MNA solve.** Source: TEA design from m4/m5. Is this
  required for stability, or could a per-frame-block solve work?
  The custom_linear_solve VJP fix is in flight — is it actually
  worth the complexity?
- **`samples_per_frame` calculation.** Verify the audio sample rate
  the engine internally renders at vs what the inverse-fit script
  expects. Mismatches here would silently produce garbage.

### Category 3 — Workplan / scope budgets

- **≤ 8 GB inference budget.** Source: ARCHITECTURE.md. What hardware
  was this calibrated for? Edge device, single GPU, browser? Hard
  number or soft target?
- **≤ 30 M params.** Same question.
- **4-week sprint structure.** Source: course-correction doc 2026-05-09.
  Is this a calendar deadline (external) or self-imposed pacing? If
  self-imposed, is it serving us or constraining us?
- **"kinematic supervision is structurally required."** Source:
  `docs/week1/findings.md` Week-1 closure verdict. Verify: how strong
  is the evidence? The verdict was based on (a) one synth-target run
  with disabled frication, and (b) a non-converged optimizer. Do
  those together actually establish the claim, or do they establish
  "we don't yet have evidence either way"?
- **"Story 1996 is the right area-function reference."** Source:
  prior architecture (now demoted to sanity gate). Was this load-
  bearing, and is it still load-bearing as a sanity gate?

### Category 4 — Loss / optimization assumptions

- **log-mel + MR-STFT as the acoustic loss.** Source: copyfit script
  + Week-1 task spec. Why this combo? Is MR-STFT alone insufficient,
  or is log-mel alone insufficient? Has anyone ablated?
- **Smoothness / curvature / barrier weight defaults.** Source:
  copyfit script. How were they chosen? Tuned, copied from a
  reference, or eyeballed?
- **`DELTA_MAX = 3.0`.** Source: `copyfit_neural_field_phrase.py`.
  Why 3.0 (in log-area units)? What's the physical interpretation?
- **Adam warmup + update-norm cap.** Source: e17e7b8 / 9d96c30
  commits. Were these added because Adam without them diverged, or
  as preventive measures?
- **The 25 Hz Chebyshev filter on supraglottal pressure.** Source:
  `glottis_geometric.py`. Why 25 Hz? Tuning? Anti-aliasing? Stability?

## Method

For each item:

1. **Source citation.** File + line, paper section, commit SHA, or
   Codex thread ID. If the source can't be located, that itself is
   a finding.
2. **Verdict.** One of:
   - **`verified`** — empirical evidence in this codebase or in a
     read source confirms the requirement
   - **`inherited unverified`** — stated in source but never tested
     in our context
   - **`cargo-cult`** — repeated without a clear original justification
   - **`malformed`** — the assumption itself doesn't make sense on
     close reading
3. **Failure mode if wrong.** What breaks, how loudly, and how soon
   downstream of the assumption.
4. **Suggested verification.** Either a one-line empirical test we
   could run, a doc to read, or a Codex thread to resurface.

Where Codex disagrees with the worker, record both views in the doc.
The audit's value is in being honest about uncertainty, not in
forcing a single verdict.

## Method (subagent dispatch)

Suggested parallel structure:

- **Opus subagent A:** Category 1 (SPARC stack). Read the SPARC repo
  README, the WavLM model card, run smoke tests on alternate sample
  rates if feasible.
- **Opus subagent B:** Categories 2 + 4 (TEA + loss). Grep the
  codebase for the load-bearing constants, trace their introduction
  via git blame, identify whether the introduction commit had a
  clear rationale.
- **You (orchestrator):** Category 3 (workplan budgets). Read
  ARCHITECTURE.md, the course-correction doc, recent handoff
  records. Call Codex for a fresh adversarial review of the
  "kinematic supervision is structurally required" claim
  specifically — that's the claim that, if wrong, would invalidate
  the whole Week-2 plan.

Codex usage notes:
- Use **fresh threads** for adversarial review of our own conclusions
- Use **continued threads** (e.g. `[session identifier omitted]`) only for items where
  the prior thread context is directly relevant
- **Serialize** Codex calls (concurrency=1)

## Acceptance gates

### Gate 1 — Coverage

`docs/assumptions_audit.md` has at least one entry per Category-1/2/3/4
item listed above. New items discovered during the audit are added
freely; entries should not be skipped because they're "obvious."

### Gate 2 — Honesty

For each entry: source citation OR explicit "source not findable" note.
Verdict is one of the four labels above; no "TBD" / "needs more work."
If the audit can't reach a verdict in reasonable time, the verdict is
`inherited unverified` with a follow-up suggestion.

### Gate 3 — Actionability

Each entry has a one-line "what to do if you care about this
assumption being wrong." Could be: a smoke test command, a doc URL,
a Codex prompt, or "raise a follow-up task."

### Gate 4 — Surface to developer

The findings get a one-page summary at the top of the audit doc,
not buried in the body. The developer reads this summary first;
the per-item detail is reference material.

## Reporting

Append to LOG.md:

```
## YYYY-MM-DD — Assumptions audit

- Categories covered:        4 / 4
- Total assumptions audited: __
- Verified:                  __
- Inherited unverified:      __
- Cargo-cult:                __
- Malformed:                 __
- Top-3 concerns surfaced:   1) __  2) __  3) __
- Verdict: GREEN (audit complete, follow-ups raised as needed)
```

## Anti-deliverables

- **No code changes** as part of this task. If the audit finds a bug,
  raise a follow-up task; don't fix it here.
- **No re-doing prior architecture decisions** — the audit reports
  on assumptions, the developer (or a follow-up task) decides what
  to do with each verdict.
- **No invented assumptions.** Audit what's actually load-bearing in
  the current codebase + workplan. Don't audit hypothetical
  requirements that aren't in the code or docs.
- **No exhaustive list of every constant.** Focus on the ones that,
  if wrong, change architecture decisions. A typo in a docstring is
  not an assumption to audit.

## References

- `docs/week1/findings.md` — closure narrative + open issues
- `workplan/handoff/2026-05-09_course_correction.md` — pivot
  rationale
- `ARCHITECTURE.md` — vision + budget claims
- `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` — has
  the most recent set of SPARC-stack assumptions in one place
- `tractus_engine_advanced/src/tractus_engine_advanced/precision.py`
- Codex thread `[session identifier omitted]` — recent SPARC requirements review
- Codex thread `[session identifier omitted]` — Week-1 stall + non-uniqueness
