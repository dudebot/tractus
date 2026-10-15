# Course correction — 2026-05-09

> **Purpose.** This document is a comprehensive overhaul of the
> tractus research pipeline. Read this before doing any further
> implementation work. It supersedes the body-model design in
> `ARCHITECTURE.md §2.3` and the body-model training plan in
> `pretraining/002` / `pretraining/002b`. Authored after a
> multi-turn Codex thread (`[session identifier omitted]`).

## TL;DR

The project deleted VocalTractLab's C++ articulator backend and
replaced it with a generic conv-MLP body model that emits area+length
per section directly. **That's the wrong design and the developer was
right to call it baby steps in the wrong direction.** Two diagnostic
findings (`infra/006` partial-convergence, `correctness/004b` empty
Whisper transcripts) signaled this and were filed as "partial done"
instead of triggering a course correction. The literature line we
should be on (Chen & Sitzmann 2025 differentiable simulator + SPARC
2024 articulatory inversion + DDSP-articulatory 2024) does not pass
through Maeda/Story heritage at all. **New direction:**

- **Body model** = continuous neural-field area function `A_θ(s, t)`
  conditioned on SPARC-style kinematic features and a small speaker
  vector. ~0.3–0.8 M params.
- **Training supervision** = SPARC as a training-time teacher only
  (WavLM Large frozen, used offline to label corpora). Acoustic
  loss through TEA. Area-MSE against Story anchors is a
  sanity-check gate, not a primary loss.
- **Inference stack** = text → planner → controller → small
  neural-field body model → TEA → waveform. SPARC is *not* present
  at inference. Per-speaker object stays kilobyte-scale.
- **First experiment** = neural-field copy-synthesis of a single
  developer-recorded Harvard sentence. Tests "can TEA + differentiable
  optimization recover intelligible speech?" If yes, foundation is
  good. If no, TEA isn't the v0 product renderer and we ship on
  DDSP-articulatory or SPARC vocoders instead.

The 5-phase workplan (`P0 correctness → P1 smoke → P2 inversion →
P3 pretraining → P4 cloning`) and the creator/orchestrator/worker
three-role dispatch model are over-engineered for a 1-developer +
Claude project. They're being flattened to a single-track 4-week
sprint.

## Why we're correcting

### What the developer originally signed up for

A JAX rewrite of VocalTractLab (Birkholz lineage) with the C++
backend replaced by a learned, differentiable mapping. The implicit
frame: "VTL's design is right; we just need to make it differentiable
and replace the articulator surface with an NN."

### Why that frame is wrong

The developer's hot take, verbatim:

> *"VTL was actually really bad. The implementation was really not
> good, on top of the fact not being differentiable meant that actual
> cloning using this technique was not the case. The approach too
> closely mirrored what VTL was without actually becoming what VTL
> should be. In the process, it probably neutered it."*

This is the load-bearing reframe. **VTL's design space — articulator
parameters → midsagittal distance → tube area, with hand-tuned
geometric heuristics — is not the right ontology for a 2026
differentiable physics-first stack.** Maeda-1990 is in the same design
family. So is Story-2005 (neutral area + eigenmodes + constriction
parameters). All three are derived from MRI snapshots of static
vowels, then heuristically extended. They're useful as **diagnostic
anchors** ("does this renderer produce sane vowels?") but not as the
primary control surface.

What the modern differentiable-simulation literature actually does:

- **Chen & Sitzmann 2025** — continuous neural-field area `A(s, t)`
  over tract location and time. Their paper explicitly reports that
  optimizing discrete tube-section variables fails for static
  vowels; they switch to a continuous neural field to smooth the
  loss surface. This directly explains tractus' `infra/006`
  partial-convergence finding.
- **SPARC 2024** — pretrained articulatory inversion from real
  speech. 12 EMA-like channels + F0 + loudness + 64-d speaker
  embedding at 50 Hz. Real-data-grounded; not VTL-heritage.
- **DDSP-articulatory 2024** — 0.4 M params, articulatory-feature
  input → audio. Loss is multi-scale spectral / adversarial, not
  area-MSE.
- **rtMRI inversion 2026** — audio → tract contours / complete
  geometry directly from real MRI data. No Maeda/Story intermediate.

None of these pass through "articulator parameters → distance →
area" as the primary modeling chain.

### Two diagnostic findings the project hadn't responded to

**Finding 1 — `infra/006` partial-convergence (2026-05-07).**
Differentiable inverse-fit demo. Gradient descent through TEA on the
tube area function to hit literature-formant spectral templates for
/a, i, u/. Result:

> *Each vowel locks ONE formant correctly (/a/ F1 4 % off, /i/ F1
> 12 %, /u/ F3 1 %) but not all three jointly. Simulator IS capable;
> area-function-only parameter space too narrow to coordinate
> multi-formant placement.*

This finding is **exactly** what Chen & Sitzmann reported in the
same regime — discrete tube-section variables produce a bad loss
surface — and they solved it by switching to a continuous neural
field. The project filed this as `done — partial`.

**Finding 2 — `correctness/004b` Whisper gate inconclusive
(2026-05-07).** Audible-frication path. PSD-shaping landed, MNA
injection stable, frication-band 4-8 kHz lift PASSES at +8.2 dB on
9/10 utterances. **But:** Whisper transcribes the smoke-pipeline
audio as empty/garbage both pre- and post-frication.

> *The current end-to-end audio is not recognizably speech.*

This is a five-alarm finding. Filed as `partial — escalation per
body-model floor`.

### What this audit concluded

Combining the two findings: **the simulator may work fine, but the
mapping into the simulator is broken, and the primary parameterization
demonstrably can't place multi-formant vowels.** Continuing to train
a body model whose primary loss is area-MSE against the same
parameterization that just failed is not a course correction — it's
ignoring the signal.

## What we got right

Worth naming, since the pivot keeps these:

1. **The simulator works.** TEA's transfer-function probe matches
   Story F2 to 1.7 % for /i/. The fixed-mode parasitic peaks
   (`correctness/009`) are diagnostic, not blocking. Per-sample
   MNA solve, custom_vjp on `step_one`, `jax.checkpoint` on the
   per-sample scan — backward at T=118 b=4 fits in 708 MB. All of
   this is solid engineering and is the foundation we keep.
2. **Differentiability is preserved.** End-to-end JAX. fp32
   precision policy is correctly load-bearing. The autograd path
   through the simulator is real, not a stub.
3. **The 20-channel gesture inventory.** Even though the body-model
   layer below it is being replaced, the gesture-event abstraction
   (`channels.py`, `interp.py`) is the right user-facing IR and
   stays.
4. **The reachability bounds.** Soft/hard physical clamps on areas,
   lengths, glottis state. Not deleted — repurposed as neural-field
   barrier priors.
5. **Story anchor data.** 11 vowels of literature-grade area
   functions. Wrong as primary supervision; right as a held-out
   sanity-check gate.

## What we got wrong

1. **Body-model parameterization.** Generic conv-MLP → 385 renderer
   fields including `area_cm2[T,93]` and `length_cm[T,93]` directly.
   No physical structure in the mapping itself; "physically
   constrained" was branding via soft clamps. Inherits VTL's
   per-section ontology without inheriting any of its physical
   structure. ~1.2 M params spent on the wrong abstraction.
2. **Primary training loss.** Area-MSE against Story anchors. Tested
   directly by `infra/006` and shown to be insufficient to coordinate
   F1–F3 placement. Even if the model perfectly fits Story's 11
   static vowel area functions, it has been trained on 11 frozen
   frames with no acoustic feedback signal.
3. **R1 distill teacher.** `pretraining/002b` is training a body
   model whose teacher is the R1 NN — which itself was trained on
   the deleted C++ oracle's output (frozen 26 K-row corpus). R1's
   own gates report `all_pass: false`. Distilling a known-failing
   baseline produces a smoother failing baseline, not progress.
   Quarantined but still running.
4. **Workplan ceremony.** Five gated phases, three-role
   creator/orchestrator/worker model, 33 task files, per-session
   handoff docs, dependency graph, INDEX.md status table. For a
   1-developer + Claude project with 7 days of momentum, this is
   ~10× too much process. The "Wave 2 — slop audit" purge already
   deleted stub lints and placeholder tests authored 1–2 weeks
   earlier; that purge proves the over-scoping. More needs to go.
5. **Phase serialization.** P0 correctness → P1 smoke → P2 inversion
   → P3 pretraining → P4 cloning is engineering-textbook ordering.
   For a *research* project where it's not yet known whether the
   instrument can do its job, the right ordering is: vertical-slice
   probe → react to result → iterate. Cloning is the product;
   getting one phrase to render recognizably is the diagnostic.
6. **The "hardest unsolved part" in plain sight.** ARCHITECTURE.md
   §2.3 admits "the body model is the hardest unsolved part of the
   architecture" and "Module 3 teacher strategy is unfunded." Then
   we shipped a body model and queued it for training anyway.

## The new design

### Inference-time stack (unchanged at the top, pivoted at the bottom)

```
text
 └─► Planner                 (≤10 M params, frozen across speakers)
      └─► segments + prosody
           └─► Controller    (~5 M params, speaker-style aware)
                └─► GestureEvent[]   (sparse async, 20 channels)
                     └─► gesture_traj[T, 20]  (50 Hz interp)
                          └─► BODY MODEL: A_θ(s, t | gestures, anatomy)
                               ║   neural field, 0.3–0.8 M params
                               ║   sinusoidal positional encoding over (s, t)
                               ║   speaker-conditioned via small anatomy vector
                               ║   reachability barriers from reachability.py
                               ║
                               └─► tract_area_cm2[T, 93] + length + glottis state
                                    └─► TEA renderer  (frozen physics)
                                         └─► waveform[T·samples_per_frame]
```

**Key differences from the old design:**

- The body-model **does not directly emit** 385 renderer fields.
  Renderer interface is now a *sampled output* of a continuous field,
  not a free decoder head.
- The 20-channel gesture inventory remains, but the channels are
  treated as task-dynamic biases on the field, not as independent
  geometry DOF.
- Speaker variation lives in `base_θ(s, speaker)` — a small per-
  speaker anatomy bias on a shared field. Cloning fits the speaker
  vector, not the full field.
- The renderer's 93 sections are a numerical discretization of the
  field, not the body's ontology. (This is a hard reframe and it
  matters for cloning: the per-speaker object is a small embedding,
  not a 93-section tube snapshot.)

### Training-time supervision (this is where SPARC enters)

```
audio corpus (LibriTTS-R, etc.)
 └─► SPARC + WavLM Large       ← runs ONCE, offline, weights frozen
      └─► frozen kinematic labels
            ║ • 12 EMA-like articulator channels @ 50 Hz
            ║ • F0 / loudness
            ║ • 64-d speaker embedding (target for our speaker encoder)
            ║
            └─► used as supervision targets for our planner / controller
                 / body-model

acoustic loss (multi-scale STFT + log-mel) computed THROUGH TEA
   on short windows, with jax.checkpoint + custom_vjp already in place.

Story anchors → held-out vowel sanity gates only:
   "given a forced /a/-like gesture trajectory, does the field produce
    an area function whose TEA-rendered transfer function has F1/F2
    within ±20 % of Story?"
```

**SPARC is never present at inference.** WavLM Large (1.26 GB,
~316 M params) blows the project's ≤8 GB inference budget by itself.
(Note per `docs/incorrect_assumptions.md` A6: the ≤8 GB figure is an
aspirational positioning target without a sourced hardware or product
constraint; however, WavLM at 1.26 GB + inference overhead would
exceed any plausible consumer GPU budget, so the practical conclusion
holds regardless.)
SPARC's role is to provide articulatory ground-truth labels that
weren't otherwise available — the same role rtMRI / EMA corpora
play in the academic literature, but available now without licensing
overhead.

### Primary losses

1. **Acoustic loss** through TEA on short windows: log-mel + multi-
   resolution STFT.
2. **Kinematic supervision** from SPARC labels: MSE on the body
   model's intermediate kinematic features matching SPARC EMA.
3. **Smoothness / reachability priors** on the field: spatial
   smoothness `d(log A)/ds`, temporal smoothness `d(log A)/dt`,
   curvature penalty, clamp-occupancy barrier.
4. **Story-anchor sanity** as a held-out validation gate, not in
   the training-time loss.

Area-MSE-against-static-Story is **deleted** as a primary signal.

## "VTL done right" — five concrete divergences

Verbatim from the Codex thread, since the developer asked for what a
2026 implementation would do differently:

1. **Continuous differentiable geometry, not hand-coded articulator
   surfaces.** Use neural fields or learned contour/area fields with
   smoothness + reachability priors. Sample to TEA only at the final
   renderer boundary.
2. **Real-data articulatory conditioning, not handcrafted parameter
   semantics.** Use SPARC / AAI / rtMRI-derived features as the
   control coordinate system. The rtMRI 2026 line predicts contours
   directly from real audio — modern targets are learned geometry
   from data, not Maeda-style basis functions.
3. **Acoustic training losses through differentiable physics.**
   Optimize log-mel / MR-STFT / adversarial / ASR-informed losses
   through the simulator. Area-MSE is a constraint, not the
   objective.
4. **Speaker anatomy as a learned low-dimensional field.** Clone
   via a speaker-conditioned neutral area / length / source
   embedding, not by retuning dozens of hand parameters.
5. **Renderer as instrument, not ontology.** TEA's 93 sections are
   a numerical discretization. The model should not expose them as
   the body state.

## 4-week sequenced plan

### Week 1 — Prove TEA can speak

Goal: Render one developer-recorded Harvard sentence intelligibly via
neural-field copy-synthesis through TEA, with no learned controller
required. **This is the diagnostic vertical slice the project should
have run first.**

Deliverable: `scripts/copyfit_neural_field_phrase.py`. Inputs: 22.05
kHz mono WAV + transcript. Outputs: reconstructed WAV + report JSON.

Optimization recipe:
- Field: `log A(s, t) = log A_default(s) + δ_max · tanh(MLP(PE(s), PE(t)))`
- Length mostly fixed; one global tract-length scale + lip extension.
- Source extracted/approximated from target audio (F0, loudness,
  voicing); frozen so the tract field can't absorb everything.
- Loss: log-mel + MR-STFT + spatial smoothness + temporal smoothness
  + curvature penalty + clamp-occupancy barrier + weak deviation
  from default-area prior.
- Optimizer: Adam, 250 iterations, 3 seeds. No L-BFGS dependency
  unless trivial.

Pass gate (all required):
- Whisper transcript non-empty.
- WER ≤ 0.75 vs target.
- Jaccard token-overlap ≥ 0.30.
- ≥ 3 of 5 content words from `{juice, lemons, makes, fine, punch}`
  recognized.
- MR-STFT loss improves ≥ 40 % from initialization.
- Clamp-occupancy < 5 % of `(frames × sections)`.
- Human five-choice listening test picks the correct sentence.

Time budget: **20–30 hours** of cold-worker implementation, **4–12 GB
VRAM**, **3–10 hours wall-clock per phrase optimization**.

**Kill criterion:** if after Week-1 + 2 focused debug days the phrase
still fails the gate, *stop tuning the body model side*. The problem
is upstream — TEA itself, the source model, or the simulation
formulation — and the next move is salvage (Chen-style frequency-
domain renderer for training; ship demo on DDSP-articulatory or
SPARC vocoder; keep TEA as research). See Risk C below.

### Week 2 — Replace per-utterance fit with shared speaker-conditioned field

Goal: Move from "optimize a field per utterance" to "trained body
model that *predicts* a field from kinematic features + speaker
vector."

Deliverable: `tractus_controller/body_model/neural_field.py` —
`A_θ(s, gesture_t, speaker_vec)`. Body-model param budget 0.3–0.8 M.

Subtasks:
1. SPARC extraction pipeline — run WavLM Large + SPARC offline on a
   small corpus (e.g., LibriTTS-R subset, or developer-recorded
   utterances). Output `.npz` per utterance with EMA + F0 + loudness
   + speaker embedding.
2. SPARC isolated in a separate PyTorch env to avoid JAX dep
   collision; export labels as pure NumPy `.npz` artifacts.
3. Body model trains to predict the optimized field from Week 1 +
   any other copy-fitted utterances, conditioned on SPARC kinematics.
4. Validation: Story-anchor /a, i, u/ vowel-rendering sanity gate
   (not Story-MSE training).

Pass gate:
- All three formants land jointly within ±20 % on /a, i, u/.
- The Week-1 phrase, when rendered through the trained body model
  (not optimized per-frame), retains intelligibility.
- Held-out short utterance (10 phrases) — Whisper non-empty on
  ≥ 7 of 10.

**Kill criterion:** if the trained body model can re-render Week 1
but cannot generalize to 5+ held-out short utterances after one round
of training, the kinematic conditioning is not picking up enough
signal — diagnose between (a) SPARC labels too noisy, (b) field
capacity too low, (c) acoustic loss too dominant.

### Week 3 — Train the controller end-to-end with SPARC pseudo-labels

Goal: Bridge gestures → kinematics. The controller emits gesture
events; the body model predicts a field from kinematic features.
Week 3 connects them: train the controller to produce gesture
trajectories that, after the body model and TEA, match SPARC's
extracted kinematics + audio of held-out utterances.

Deliverable: trained controller (~5 M params).

Pass gate:
- 10 held-out short utterances reproduced more accurately than
  hand-templates.
- TEA-rendered audio passes Whisper at WER ≤ 0.65.

### Week 4 — Constrained demo + honest baseline

Goal: ship the demo at the *constrained* form ("articulatory TTS
with speaker anatomy fitting"), not the full zero-shot 30-s clone
yet. Compare honestly against SPARC and DDSP-articulatory at the
same constraint level.

Deliverable: 5 held-out phrases rendered in the developer's voice
with anatomy + source fitted from a 30-s reference audio.

Pass gate:
- All 5 phrases pass Whisper at WER ≤ 0.5.
- Audible fricatives, stops, and vowels — no post-filter rescuing.
- Listener identifies the speaker correctly on 2/3 A/B (TEA-only
  vs random baseline) on at least one phrase.

If Week 4 succeeds, the project graduates from "physics-first
prototype" to "differentiable articulatory TTS with cloning roadmap."
If it doesn't, the demo ships with the SPARC/DDSP baseline as the
runtime path and TEA is documented as a research-grade diagnostic
renderer rather than the production vocoder.

## Disposition of existing artifacts

### Code

| Path | Verdict |
|---|---|
| `tractus_controller/tractus_controller/body_model/channels.py` | **keep, modify.** 20-channel gesture inventory + renderer-interface schema stays. Rewrite comments so renderer-interface fields are sampled-field outputs, not free decoder heads. |
| `…/body_model/interp.py` | **keep.** GestureEvent → 50 Hz trajectory composer is unchanged. |
| `…/body_model/reachability.py` | **keep, repurpose.** Bounds become neural-field barriers / smoothness regularizers. |
| `…/body_model/anchors.py` | **keep as test fixture.** Story anchor loader is sanity-gate infrastructure. |
| `…/body_model/model.py` | **delete from active path.** ~480 lines of conv-MLP body model. The architecture this implements is rejected. Move to `legacy/` or rely on git history. |
| `scripts/gen_body_corpus.py`, `scripts/train_body_model.py`, `scripts/eval_body_model.py` | **archive.** ~1300 lines of training infra built around `model.py`. Keep loaders / provenance discipline / sidecar conventions for reuse, but the training loop targets the wrong objective. |
| `outputs/fixtures/r1_corpus/*.npz` | **archive as legacy oracle fixture.** Do NOT train on this except as a clearly-labeled low-trust diagnostic. Keep for "can TEA render this old tube?" tests. |
| `outputs/body_model_ckpt/r1_distill_*` | **archive.** Label `r1_distill_aborted_2026-05-09`, `promotable: false`. |
| `tractus_controller/outputs/fixtures/story_1996_anchors.npz` | **keep as sanity-gate input.** |
| `scripts/transfer_function_oral_only.py` | **keep.** Diagnostic gate for "is TEA producing literature formants?" |
| `runbooks/whisper_gate.sh`, `runbooks/source_tract_sep.py` | **keep.** Gates remain valid. |
| `scripts/inverse_fit_demo.py` (`infra/006`) | **keep, mark as "discrete-bin baseline (failed)".** This is the empirical evidence for the parameterization pivot. |

### Background workers

The 2026-05-09 handoff lists three worktree agents. As of this
audit, none are visibly running on the local machine (`ps -ef`
inspection). The user should:

1. Check each worktree (`[private worktree identifiers omitted]`) for stuck processes:
   `cd <worktree> && git status && ps -fp $(pgrep -f train_body_model)`.
2. **Kill the `pretraining/002b` R1-distill training run** if it's
   still alive anywhere. It has zero load-bearing role in the new
   plan and consumes GPU that Week-1 needs.
3. If a partial checkpoint exists, archive it with the label above
   and remove from active artifact paths.

### Workplan structure

The current `workplan/` was over-engineered relative to throughput.
Flattened 2026-05-09 (developer sign-off received, executed via
`git mv` so history is preserved):

```
workplan/
├── NOW.md                          ← single source of active truth
├── LOG.md                          ← short daily progress notes
├── handoff/
│   ├── 2026-05-04_session.md
│   ├── 2026-05-07_inverse_fit_findings.md
│   ├── 2026-05-09_purification_and_anchors.md
│   └── 2026-05-09_course_correction.md   ← THIS FILE (the bridge)
├── tasks/
│   └── 00_active/                  ← five active task files
│       ├── 001_neural_field_phrase_probe.md  (Week 1)
│       ├── 002_sparc_extraction_pipeline.md  (Week 2 prep)
│       ├── 003_speaker_conditioned_body_model.md  (Week 2)
│       ├── 004_controller_with_sparc.md  (Week 3)
│       └── 005_constrained_demo.md  (Week 4)
└── archive/
    └── prepivot/                   ← the previous 33 task files +
                                      the prepivot INDEX.md (frozen,
                                      preserved via `git mv` for
                                      archaeology)
```

(The previous top-level `INDEX.md` is now at
`workplan/archive/prepivot/INDEX.md`; `NOW.md` replaces it as the
active-truth surface.)

What to **delete**:
- The `creator session / orchestrator session / worker session`
  three-role section. For a 1-developer project, this ceremony adds
  more friction than coordination.
- The phase-gate dependency graph. The 4-week plan is the dependency
  graph.
- The status table for the 33 prepivot tasks. They go to archive
  intact; their statuses no longer matter.
- The runbook inventory that includes already-deleted runbooks
  (`anticheat_unreachable.py`, `dimensionality_report.py`, etc.)
  — this stale state has already caused confusion once.

What to **keep**:
- The `ARCHITECTURE.md` file as the architectural north star, but
  with §2.3 (Module 3) clearly marked as **superseded** by this
  document.
- The `docs/references/` library.
- The `runbooks/whisper_gate.sh` and `runbooks/source_tract_sep.py`
  as live gates.
- `CLAUDE.md` operational notes.
- The Story anchor pipeline.

## Risk register

### Risk A — Neural-field collapse

**Failure mode.** Optimized `A_θ(s, t)` finds a degenerate solution
that matches log-mel but is physically implausible: clamp occupancy
spikes, spatial gradients pathological, area function looks nothing
like a vocal tract.

**Prior structure to prevent it.** Bounded log-area output, low
network capacity (4-layer MLP @ 128), spatial smoothness, temporal
smoothness, curvature penalty, default-area deviation prior, hard
clamp-occupancy gate.

**Kill criterion.** After 3 seeds, best run passes mel loss but has
> 5 % clamp occupancy *or* spatial gradients > 3× Story/R1
diagnostic range *or* remains unintelligible. If this happens,
acoustic loss alone is insufficient supervision; require real
articulatory priors (SPARC labels) before continuing.

### Risk B — SPARC unavailable

**Failure mode.** SPARC's WavLM Large dependency creates a license
or framework-collision blocker, or the model weights are unobtainable
on the developer's machine.

**Fallback order:**
1. Run Week 1 unconditioned (SPARC isn't needed for the phrase
   probe). Don't block on it.
2. Isolate SPARC in a separate PyTorch env and export labels as
   `.npz`. Do not merge deps into the JAX env.
3. If WavLM Large is the blocker, use HuBERT-base or WavLM-base
   as a weaker articulatory teacher.
4. If no articulatory teacher lands by Week 2, continue with
   Week-1 optimized neural-field trajectories as pseudo-labels —
   but flag cloning quality risk as high in the demo writeup.

### Risk C — TEA non-cooperative

**Failure mode.** Even with a continuous neural-field
parameterization, fixed source, and smoothness priors, TEA cannot
reconstruct one real phrase intelligibly. This means the simulator
itself has a correctness gap upstream of any body model.

**Detection.** Week-1 fails its gate. Two focused debug days don't
recover it.

**Next move.** This is the kill point for TEA-as-product. Salvage
order:
1. **First salvage.** Implement a Chen-style frequency-domain /
   overlap-add differentiable transmission-line renderer for
   training. Use TEA only as validation. (Chen & Sitzmann's paper
   is the reference for this; their FD formulation is faster *and*
   has a smoother loss surface.)
2. **Second salvage.** If FD-renderer also fails or disagrees badly
   with TEA, the simulation formulation has a real bug. Fix it or
   ship the demo on DDSP-articulatory wholesale.
3. **Hard cutoff.** TEA stays as a research artifact; the v0
   product runs SPARC/DDSP-articulatory inference. The
   "physics-first" framing becomes "physics-anchored research with
   neural production rendering."

This is the most important kill criterion in the project. Make it
explicit so it gets exercised if the evidence calls for it.

## Immediate next actions for the developer

In order:

1. **Read this document.**
2. **Kill the R1-distill worker** if it's still running anywhere
   (worktree agents `a95e149a2b77a43de`, `a642d4dbaa26ae299`,
   `ac8a5f72dbbf86638`). Archive the partial checkpoint as
   `r1_distill_aborted_2026-05-09`.
3. **Decide on workplan flattening.** Sign off (or push back) on
   the proposed `workplan/NOW.md` + `workplan/archive/prepivot/`
   restructure before any files move.
4. **Record the Week-1 reference audio.** One Harvard sentence in
   a quiet room, mono 22.05 kHz, ~2-3 seconds. Recommended:
   `"The juice of lemons makes fine punch."` (vowels, fricatives,
   stops, voiced/unvoiced contrast). Save to
   `data/probes/juice_lemons_ref.wav`.
5. **Dispatch the Week-1 task** (file: `workplan/tasks/00_active/
   001_neural_field_phrase_probe.md`, see below). Either as a
   fresh Claude Code session or as an Agent worker with the task
   file as the prompt.
6. **Decide on SPARC** before Week 2. Read `docs/references/core/
   2024_sparc_vocal_tract_kinematics.pdf`. The decision is
   training-only, so the inference-time architecture isn't held
   hostage.

## Open questions for the developer

These are not implementation choices — they're architectural
decisions only the developer can make. Listed in approximate order
of urgency.

1. **Does the developer accept that VTL/Maeda/Story heritage is
   archived as historical reference rather than primary
   architecture?** If they want to retain Maeda-style articulator
   parameterization for cloning interpretability, that's a legitimate
   architecture branch but it adds 2–4 weeks of implementation cost
   and doesn't compose cleanly with the neural-field direction.
2. **Is the 6-week demo timeline real?** Codex's 4-week plan
   leaves 2 weeks of slack, but Week 1's "20-30 hours from cold
   worker" is optimistic. The kill criteria are real; the developer
   should be ready to exercise them.
3. **What's the v0 product framing if TEA fails the Week-1 gate?**
   Codex recommends shipping on SPARC / DDSP-articulatory and
   reframing the project as "physics-anchored articulatory TTS
   with research-grade differentiable simulator." Is that an
   acceptable v0 outcome, or is "TEA is the product renderer" a
   non-negotiable?
4. **Speaker cloning ambition.** The new plan ships Week-4 demo at
   "anatomy + source fitting from 30-s reference," not full
   zero-shot cloning. Full zero-shot cloning is a separate
   research push, post-v0. Is that OK?

## Codex thread to continue if needed

`[session identifier omitted]` — fresh thread opened
2026-05-09, multi-turn deliberation. Covered: parameterization
re-rank under VTL-was-bad reframe, Week-1 deliverable, code
disposition, risk register. Use `codex-reply` to continue; don't
start fresh unless cold review is needed.

The earlier Codex thread `[session identifier omitted]` (Story-anchor calibration,
"good enough, move on") is **superseded** by this thread for body-
model decisions. The Story-anchor calibration findings themselves
remain valid — TEA is in the ballpark on Story F2, the developer's
calibration probe was correct — but the framing of "Story anchors
are the body-model supervision target" is rejected here.
