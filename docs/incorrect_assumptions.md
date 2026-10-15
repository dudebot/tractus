# Incorrect assumptions — single source of truth

> Canonical "things to fix" list assembled from the three audit
> docs (`assumptions_audit_sparc.md`, `assumptions_audit_tea_loss.md`,
> `assumptions_audit_workplan.md`) and `assumptions_audit.md`.
>
> **Purpose:** prevent re-cargo-culting. Every entry has a wrong claim,
> a correct replacement, and the known sites where the wrong claim
> lives. The three cleanup subagents work from this list; future
> sessions land here when they spot a new instance.
>
> **Maintenance rule:** when a fix lands, leave the entry in place
> with a `**FIXED in commit <SHA>**` annotation. Do NOT delete entries.
> Future Claude needs to know "we already fixed this and how." Only
> after a category has been fully cleaned for a quarter and there are
> no recurrences should an entry be archived.

---

## Project sample-rate decision (settled this session)

**Project-wide rate: 22.05 kHz.** TEA renders internally at 44.1 kHz
(engine constraint, not changing) and decimates 2× at the boundary
before loss + final output. SPARC ingests at 16 kHz via its own
internal resample. Final TTS output: 22.05 kHz.

This is the correct claim. Wrong claims and where they live:

---

### A1 — "TEA renders at 22.05 kHz" / "the engine sample rate is 22.05 kHz"

**Wrong:** TEA renders at 44.1 kHz (`TIMESTEP_S = 1/44100`,
`tractus_engine_advanced/tds/constants.py:20`). The engine has no
parameter to render at any other rate.

**Correct claim:** TEA renders at 44.1 kHz. Consumers wanting other
rates must decimate (or, in principle, re-derive `TIMESTEP_S` and
all MNA matrices, which is a major engine change).

**Known wrong-claim sites:**
- `scripts/copyfit_neural_field_phrase.py:75` — `SAMPLE_RATE = 22050`
  (used as if it were the rate TEA renders at; loss compares
  octave-shifted audio). **R3 annotated** with A1 boundary comment
  explaining the octave-shift bug.
- `scripts/copyfit_neural_field_phrase.py` `vowel_sanity()` —
  `np.fft.rfftfreq(len(ir), d=1.0 / SAMPLE_RATE)`: TEA's impulse
  response is at 44.1 kHz, so the frequency axis is halved → formant
  reads from vowel_sanity are all shifted one octave down. **R3
  annotated** with A1 bug comment.
- `scripts/synth_target_probe.py` — `SR = 22050`, `SAMPLES_PER_FRAME = 441`
  mirrors copyfit; `sf.write(..., SR)` labels the WAV as 22.05 kHz when
  TEA produced 44.1 kHz audio. **R3 annotated.**
- `ARCHITECTURE.md:40` — table cell "Sample rate | 22.05 kHz (target),
  16 kHz (floor)" (true as a *project-wide target* but contradicts
  the engine's actual rate; needs disambiguation between "render rate"
  and "project rate") — **annotated by R1**
- `ARCHITECTURE.md:482` — "Per-frame MNA solve dominates. Expect 4-8 GB
  on a 22.05 kHz [...]" (per audit, this propagates the wrong rate
  forward to memory/cost reasoning) — **annotated by R1**
- `ARCHITECTURE.md:364` — "Per-sample MNA solve runs at 22.05 kHz
  (50 Hz frames × 441 samples)" — explicitly states wrong engine rate;
  found during R1 scan, was not in original known-sites list — **annotated by R1**
- `workplan/tasks/00_active/001_neural_field_phrase_probe.md:142` —
  `samples_per_frame = 441` for "50 Hz × 22.05 kHz" without noting
  this is post-decimation count — **annotated by R1**

**Sites that already had it RIGHT (do not touch):**
- `scripts/render_m5_oracle.py:17-30` — explicitly calls out that
  TEA's `TIMESTEP_S = 1/44100`, audio is 44100 Hz, and the spec'd
  22.05 kHz is "aspirational." This is the CORRECT account; copyfit
  author should have read it.

---

### A2 — "Loss plateaued at iter 100" / "Week 1 closed at convergence"

**Wrong:** Acoustic loss in `juice_lemons_run100/report.json` was
still descending at the last logged iter (3.404 → 3.374 → 3.366 →
3.357 — visibly downward). 100 iters was a script default, not a
convergence criterion.

**Correct claim:** Week 1 stopped at the iteration budget, not at
convergence. Whether the optimizer would have continued descending
is unknown.

**Known wrong-claim sites:**
- `docs/week1/findings.md` — closure verdict implicitly assumes
  convergence
- `workplan/NOW.md` — Task 001 marked done with a "phonetic wall"
  framing
- Possibly inline comments in `scripts/copyfit_neural_field_phrase.py`
  about iteration budgets

**Annotations to add wherever the closure narrative is referenced:**
"Note: Week-1 acoustic loss did not plateau at iter 100; the closure
verdict is under review pending the discriminating experiment
(see `docs/assumptions_audit.md` Finding 2)."

---

### A3 — "Frication is wired into copyfit"

**Wrong (until this session's fix):** copyfit called
`synthesize_dynamic_tube` without a `frication=` kwarg, which
defaults to `Frication.disabled()`. The Week-1 results were generated
with frication off.

**Correct claim:** As of this session, copyfit has a `--frication`
CLI flag. Without that flag, frication is still disabled (preserves
baseline reproducibility). The Week-1 closed runs do not have
frication.

**Known wrong-claim sites:**
- `docs/week1/findings.md` — implicitly assumes frication was
  contributing
- Any comment in `scripts/copyfit_neural_field_phrase.py` that
  predates this session and mentions frication being available

**No fix needed for the script itself** — it is now correctly wired.
Just ensure narrative docs don't claim "we tested with frication."

---

### A4 — "Kinematic supervision is structurally required"

**Wrong:** Codex adversarial review (thread `[session identifier omitted]...`)
verdict: **UNDETERMINED**. The supporting evidence (Week-1 closure)
suffered from A1 (octave-shift), A2 (non-converged), and A3
(frication off). Per Codex: *"a roadmap-grade claim that needs a
converged, in-model, frication-capable negative result before it can
carry 3 weeks of downstream work."*

**Correct claim:** Current acoustic-only inverse fitting is not yet
demonstrated to recover reliable articulation. The inverse problem
is likely ill-conditioned. Whether kinematic supervision is
structurally required (vs being one of several useful add-ons) is
**unknown** pending Codex's recommended discriminating experiment.

**Known wrong-claim sites:**
- `docs/week1/findings.md` — closure verdict (R1 scope)
- `workplan/handoff/2026-05-09_course_correction.md` — the pivot
  rationale leans on this claim repeatedly (R1 scope)
- `ARCHITECTURE.md:46-48` — "supplemented by SPARC kinematic
  supervision so the body model has a non-acoustic anchor" (frames
  SPARC as required, not auxiliary) (R1 scope)
- All Week-2-and-later task files reference "the body model needs
  SPARC labels" as if it were settled (R1 scope)
- `workplan/NOW.md` — sprint plan presumes SPARC is on the critical
  path (R1 scope)
- `scripts/synth_target_probe.py` module docstring and compare output —
  "path forward: SPARC kinematic supervision" stated as settled.
  **R3 annotated** with A4 caveat in both the docstring and the
  runtime print.

**Annotation:** "Note: this claim is under review per
`docs/assumptions_audit.md` Finding 2; verdict pending discriminating
experiment (Task 009 if dispatched)."

---

### A5 — SPARC EMA normalization options "z_score_within_utterance" / "physical_mm"

**Wrong:** Both options are inaccurate. SPARC's published checkpoint
outputs EMA in MNGU0-OLS-output-space. Training z-scoring was
within-utterance, so per-utterance scale was discarded during
training and is structurally unrecoverable.

**Correct claim:** EMA outputs are in `mngu0_ols_output_space`. No
inference-time normalization is applied; physical mm is unrecoverable.
Body-model code in Task 003 must learn the projection rather than
assume a known coordinate system.

**Known wrong-claim sites:**
- `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` —
  the `ema_normalization` field schema (added by me earlier this
  session — needs revision)
- `docs/week1/sparc_dispatch_prompt.md` — references the same schema
- `assumptions_audit_sparc.md` entry 6 — already documents this
  correctly; the contradicting Task 002 needs to come into agreement
  with that audit doc

---

### A6 — Workplan budgets (≤ 8 GB / ≤ 30 M / 5× / 30 s) stated as constraints

**Wrong:** Listed in `ARCHITECTURE.md:36-41` as targets/constraints
without citation to a target hardware profile or product decision.
"Consumer hardware" and "smaller than Kokoro 82 M" are positioning
statements, not constraints.

**Correct claim:** These numbers are currently unsourced. Either
they're real product constraints (in which case the source should be
cited) or they're soft positioning statements (in which case they
should be reframed as such, not used as architecture gates).

**Known wrong-claim sites:**
- `ARCHITECTURE.md:36-41` — the budget table itself
- `ARCHITECTURE.md:141-142` — second instance of the same table with
  brief annotations ("Consumer hardware", "Far smaller than Kokoro")
- `workplan/handoff/2026-05-09_course_correction.md:253` — uses the
  8 GB number to disqualify SPARC at inference time

**Annotation:** "Source not findable as of this audit; treat as soft
target until Task 010 (`workplan/tasks/00_active/010_workplan_budget_decisions.md`,
if created) resolves. See `docs/assumptions_audit_workplan.md`
entries 1.1-1.5."

---

### A7 — "WavLM at non-16-kHz refuses to run"

**Wrong:** WavLM does not refuse. It silently consumes audio at any
rate, applying its convolutional front-end to whatever it gets. At
non-16-kHz input, downstream features are frequency-shifted and
frame-rate-shifted from what the model was trained on. Outputs
look plausible but are wrong.

**Correct claim:** WavLM expects 16 kHz mono input; non-16-kHz input
silently produces distorted features. SPARC's `Coder.encode()` API
resamples internally for you, so the v0 SPARC pipeline is safe.
Risk surface is any future code path that bypasses `Coder.encode()`
and feeds WavLM directly.

**Known wrong-claim sites:**
- `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` —
  references "16 kHz requirement" without documenting the failure
  mode of non-16-kHz input
- `docs/week1/sparc_dispatch_prompt.md` — same

---

### A8 — SPARC periodicity threshold = 0.4

**Wrong:** Paper §B.1 says SPARC thresholds CREPE periodicity at 0.4.
The published source code defaults `periodicity_threshold=0.0` and
the loader does not override.

**Correct claim:** SPARC source code uses 0.0 by default at inference.
The 0.4 figure is from a training-time configuration described in
the paper but not enforced in the published checkpoint pipeline.

**Known wrong-claim sites:**
- `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` — has
  "f0[T] ... NaN where f0_periodicity < 0.4" claim, presented as
  SPARC's behavior
- `docs/week1/sparc_dispatch_prompt.md` — same

**Fix:** Reframe as "downstream tractus convention: mask f0 where
periodicity < 0.4, with a one-line comment that this is OUR
convention, not SPARC's default."

---

### A9 — "Loss weights are tuned" / loss-weight defaults are validated

**Wrong:** `W_SMOOTH_S=1e-3`, `W_SMOOTH_T=1e-3`, `W_CURV=1e-4` etc.
in `scripts/copyfit_neural_field_phrase.py` were copied verbatim from
the Week-1 task spec line "Suggested initial weights — tune in Week 1
itself." They were never tuned. Commit `c639a73` later added CLI
overrides because Codex suggested 10-30× higher smoothness, but the
defaults in the code still hold the unverified values.

**Correct claim:** The default loss weights are unverified initial
values, not tuned. Codex's recommended values (10-30× higher
smoothness) are available via CLI but not the defaults.

**Known wrong-claim sites:**
- `scripts/copyfit_neural_field_phrase.py` — `W_SMOOTH_S` etc.
  defaults presented without "unverified" caveat. **R3 annotated**
  with A9 unverified-defaults comment above the weight constants.
- `workplan/tasks/00_active/001_neural_field_phrase_probe.md` — task
  spec presents these weights as the recipe (R1 scope)

**Fix:** add a docstring above the weight constants noting "These
are initial-guess defaults, never tuned. Codex recommended 10-30×
higher smoothness in commit c639a73 (CLI override only)."

---

### A10 — "Reference audio for Week 1 is uncompressed PCM"

**Wrong (or unverified):** `data/probes/juice_lemons_ref.wav` was
generated by ElevenLabs and downloaded as MP3, then converted to
WAV. It carries lossy-compression artifacts, particularly above
~16 kHz at typical export bitrates (128-192 kbps).

**Correct claim:** Natasha reference is MP3-roundtripped. At
22.05 kHz Nyquist (11.025 kHz), MP3 attenuation is generally below
the resampled bandwidth, so artifacts are minimal for our use case
— but should NOT be claimed as pristine reference for any future
psychoacoustic comparison.

**Known wrong-claim sites:**
- Anywhere `juice_lemons_ref.wav` is referenced as a "reference" or
  "ground truth" without the MP3-roundtrip caveat
- `workplan/tasks/00_active/001_neural_field_phrase_probe.md:24` — calls
  the file "developer-recorded" when it is actually ElevenLabs "Natasha"
  synthesis (MP3-roundtripped); found during R1 scan — **annotated by R1**
- `docs/week1/findings.md` — footer section references the file without
  MP3 caveat — **annotated by R1**

**Fix:** add `data/probes/README.md` (or expand existing) with the
provenance of each probe file: source (ElevenLabs Natasha), format
(MP3 → WAV), bitrate (if known), consent status. Future
developer-recorded probes should be marked as `pristine_pcm: true`
in their manifest entries.

---

## Findings that are **NOT** items in this list (and why)

These were considered during audit but determined to be correct
or non-load-bearing:

- **TEA precision = fp32 in production.** Verified in
  `assumptions_audit_tea_loss.md`. Documented rationale in
  `tractus_engine_advanced/precision.py`. Correct.
- **`frame_chunk_size = 1` default in pipeline.py.** Verified;
  correct memory-vs-throughput tradeoff for the production T=118
  b=4 backward case. Inverse-fit overrides to K=10 explicitly.
- **Story 1996 anchors as held-out sanity gate.** Pivoted from
  primary loss; current usage is correct.
- **MNGU0 as canonical articulator space (SPARC).** Verified; this
  is what SPARC's published checkpoint produces, no alternative
  available without retraining.
- **EMA channel order TD,TB,TT,LI,UL,LL.** Verified against SPARC
  API output; documented correctly in Task 002.

---

## Cleanup-subagent dispatch protocol

Each of three subagents takes one of the partitions below. They
read this doc, scan their partition, and apply minimal edits to
bring docs/comments into agreement. They do NOT change code
behavior — only documentation, comments, and docstrings.

### R1 — Markdown / human-facing docs

Scope: `*.md` in repo root, `docs/`, `workplan/` (including all
task files in `00_active/` and `archive/`), `runbooks/*.md`.

Out of scope: `legacy/` (frozen by archaeology rule), audit docs
themselves (`assumptions_audit*.md` and this file).

### R2 — Engine inline comments + docstrings

Scope: `tractus_engine_advanced/` source tree. All `.py` files,
focusing on docstrings (module + class + function) and inline
comments. Hot-spots: `pipeline.py`, `tds/constants.py`,
`precision.py`, `frication.py`, `glottis_geometric.py`.

### R3 — Controller + scripts + runbooks inline comments

Scope: `tractus_controller/` source tree, `scripts/`, `runbooks/`.
Hot-spots: `copyfit_neural_field_phrase.py` (touched by A1, A3,
A9), `m7_frication_band.py`, runbook scripts that mention sample
rates.

### Common rules for all three subagents

1. Every fix is a doc/comment change only. Never modify executable
   code. If you find a code bug related to A1-A10, surface it as a
   follow-up task — do not fix it.
2. Every fix references this doc by entry ID (e.g., "see
   `docs/incorrect_assumptions.md` A1").
3. If you find a wrong claim NOT in A1-A10, add a new entry to this
   doc rather than silently fixing.
4. Use `Edit` tool with replace_all=false. One change per Edit; no
   bundling.
5. Report a list of files touched + new entries added (if any) at
   the end. No diff dumps.
