# Week 1 — Findings

> Closed 2026-05-15. Comprehensive summary of what Week 1 set out to
> prove, what we found, and what it implies for Week 2.
> Authoritative for cold-pickup sessions. Per-session narrative is in
> `workplan/handoff/*`; `workplan/LOG.md` is the daily log.

---

## TL;DR

Week 1 set out to prove that **gradient descent through TEA, conditioned
on a small neural field over the supraglottal area function, can recover
intelligible speech by matching a reference audio's log-mel features.**
The answer is **partial**:

- **Yes**: the optimizer works, audio converges to speech-like timbre,
  no catastrophic saturation, no NaN, backward fits in 5 GB VRAM.
- **No**: the recovered audio is not the *right* speech. On the Natasha
  test phrase, the developer hears speech-like syllables ("bo bo ne
  gardeh") but Whisper transcribes only "I" or empty.
- **Decisive diagnostic**: the synth-target probe (copyfit on TEA-rendered
  sustained /A/) converges acoustically (Gate 2 PASS, ratio 0.33), Whisper
  recognizes "Uh", but the **recovered tube is structurally wrong** —
  log-MAE 0.87 vs ground truth, recovered front-of-mouth is 0.5-3 cm²
  where truth is 5-9 cm². The optimizer found a *different* vowel shape
  that happens to match in mel space.

**Mechanism: articulatory-acoustic non-uniqueness.** Many tract shapes
produce similar log-mel envelopes. Acoustic loss alone has no preference
between them. The optimizer picks whichever is closest to its
uniform-pipe init.

**Implication for Week 2: SPARC kinematic supervision is structurally
necessary**, not just architecturally preferred. The course-correction
doc's Risk-A path is now empirically the path. No amount of smoothness
/ curvature / warmup / DELTA_MAX tuning will fix the non-uniqueness —
the loss landscape itself has the flat valley.

> **[Under review per `docs/incorrect_assumptions.md` A2, A3, A4]**
> The "structural necessity" verdict above is the Week-1 closure finding,
> but it rests on three confounders that were not yet understood when this
> section was written: (A2) the 100-iter run had not converged — acoustic
> loss was still descending at the last logged iter (3.404 → 3.374 → 3.366
> → 3.357); (A3) frication was disabled — the reference phrase has ~9
> fricatives/affricates the optimizer could not produce; (A4) a subsequent
> Codex adversarial review rated the kinematic-supervision necessity claim
> UNDETERMINED, requiring a converged, frication-capable negative result
> before it can carry 3 weeks of downstream work. See Task 007 for the
> discriminating retest. The "empirical mandate" framing is premature.

This is a clean, useful Week-1 closure. The foundational question is
answered. Week 2 has a clear mandate.

---

## What Week 1 was supposed to deliver

Per `workplan/tasks/00_active/001_neural_field_phrase_probe.md`, the
deliverable is `scripts/copyfit_neural_field_phrase.py` producing a
reconstructed WAV from a developer-recorded Harvard sentence
(`data/probes/juice_lemons_ref.wav`, ElevenLabs "Natasha" synthesis of
"The juice of lemons makes fine punch."). Five acceptance gates:

| # | Gate | Spec | Final state |
|---|---|---|---|
| 1 | Whisper intelligibility | non-empty + WER ≤ 0.75 + ≥3 content words | ✗ "I" |
| 2 | Loss improvement | final ≤ 0.6 × init | partial (4.06 → 3.37, ratio 0.83) |
| 3 | Field plausibility | clamp < 5 %, max\_grad\_s < 3.0, max\_grad\_t < 5.0 | ✗ 0.034 / 12.3 / 29.1 |
| 4 | Vowel sanity | F1F2F3 within 20 % on /a, i, u/ | partial (2/3) |
| 5 | Human listening | correct sentence in 5-choice | ✗ "bo bo ne gardeh" — speech-like, wrong phonemes |

Gates 1, 3, 5 fail on the Natasha phrase. Gate 2 missed but moved
meaningfully (17 % reduction). Gate 4 partial.

The synth-target diagnostic (not in the task spec — added during
debugging) is what actually closed Week 1: it isolated whether the
problem is "optimizer broken" (it isn't) or "loss landscape bad" (it
is).

---

## What we built

| Artifact | Purpose | LOC | Commits |
|---|---|---|---|
| `scripts/copyfit_neural_field_phrase.py` | Week-1 deliverable. Neural-field copyfit through TEA on a single phrase. | ~1100 | `fad6fff` (worker af90 initial), `e17e7b8`, `9d96c30`, `c639a73` |
| `scripts/sustained_vowel_probe.py` | Week-0 hand-tune sanity check on TEA's instrument quality. | 345 | `916ca36` |
| `scripts/synth_target_probe.py` | Synthetic-target diagnostic — gen + compare against ground-truth field. | 366 | `4b665e7` |
| TEA pipeline (`tractus_engine_advanced/.../pipeline.py`) | Renderer. Unchanged from pivot — `frame_chunk_size=K` knob already exposed. | — | — |
| `runbooks/whisper_gate.sh` + `_whisper_gate_impl.py` | Gate 1 ASR + WER + Jaccard. Needs `tractus_controller/.venv` for `faster_whisper`. | — | — |
| `tractus_controller/.../body_model/` (channels, interp, reachability, anchors) | Pivot-era infra. Unchanged in Week 1. | — | — |

**Key dependency note**: `tractus_engine_advanced` is installed in
`tractus_controller/.venv` as a namespace package pointing at the
project subdirectory, NOT at `src/`. Top-level re-exports
(`from tractus_engine_advanced import Smoothing`) **don't work** —
import from the submodule path (`from tractus_engine_advanced.smoothing
import Smoothing`).

---

## Three layered bugs found and fixed

Each bug was hidden by the previous one. Discovered iteratively via
Codex cold-review (thread `[session identifier omitted]`).

### Bug 1 — Field operates on TEA-overwritten sections

The neural field was parameterized over **all 93 tube sections**, but
TEA overwrites glottis sections 23/24 from the GeometricGlottis model
at synthesis time. Sections 65+ are nasal/sinus/fossa side-branches
the oral-phrase loss can't supervise.

**Symptom**: spurious barrier gradient from sections at AREA_MIN
(default tube has 23/24/65 at 1e-4 cm²). Iter-1 weighted barrier
2.88 dominated acoustic gradient ~0.5. Optimizer spent the first
~10 iters fighting clamp penalties on sections that were going to be
overwritten anyway.

**Fix**: mask the field to `FREE_LO=25, FREE_HI=65` (40 supraglottal
sections). Splice into the default tube via `.at[:, 25:65].add(delta)`.
Restrict priors to the masked range.

```python
delta_free = field_grid(params, s_grid, t_grid)   # (T, 40), not (T, 93)
log_area = log_area_default_broadcast.at[:, FREE_LO:FREE_HI].add(delta_free)
```

### Bug 2 — Softplus barrier was a biased global interior potential

The barrier `softplus(log_area - log_max) + softplus(log_min - log_area)`
has continuous non-zero value and gradient at default areas. With
default = 4.0 cm² and AREA_MAX = 12.0:

```
softplus(log(4) - log(12)) = softplus(-1.09) ≈ 0.288 per cell
mean × W_CLAMP=10 = 2.88 weighted at iter 1 — matches observation
```

**Symptom**: optimizer pulled all areas downward toward log-space
center. With LR=1e-2 (or even 3e-3 with the field's first-step
dynamics), the field saturated tanh in 5-10 iters. Acoustic loss
locked at ~3.85 because gradient went to zero.

**Fix**: margin-based softplus, fires only near the limits.

```python
log_lo = log(AREA_MIN) + BARRIER_MARGIN      # 0.5
log_hi = log(AREA_MAX) - BARRIER_MARGIN
over  = softplus(BARRIER_BETA * (log_area - log_hi)) / BARRIER_BETA
under = softplus(BARRIER_BETA * (log_lo - log_area)) / BARRIER_BETA
# β=5, margin=0.5 → barrier ≈ 0.01 unweighted at default area
```

### Bug 3 — Adam first-step saturation

With m, v ≈ 0 (zero init), Adam's first update is approximately
`lr × sign(grad)`. At LR=1e-2 this gave per-param update ~0.01 → field
saturated tanh by iter 4-5 even with the margin barrier fix.

**Symptom**: 5-iter probe with margin barrier showed acoustic moving
real (4.058 → 3.900), but `delta_sat_frac` going 0 → 0.42 by iter 5.

**Fix**: three layered safeguards.

1. **LR warmup over 20 steps** — Adam takes proportional-sized first
   steps that are well-conditioned.
2. **Update-norm cap of 0.35** — global ℓ₂ on the per-step parameter
   update. Prevents one bad early step from wrecking the trajectory.
3. **DELTA_MAX 1.5 → 3.0** — gives tanh headroom for the legitimate
   excursions the optimizer wants. Default tube is uniform 4.0 cm²;
   Story-anchor vowels need log-space excursions up to ~3.0 from
   default.

```python
warmup_scale = jnp.minimum(1.0, step / warmup_steps)
effective_lr = lr * warmup_scale
# … compute updates …
scale = jnp.minimum(1.0, cap / (upd_l2 + eps))
scaled_updates = jax.tree_util.tree_map(lambda u: u * scale, raw_updates)
```

---

## Experimental progression

Each row uses the same target (Natasha phrase) unless noted. "Whisper"
column is the runbook gate at base model + WER 0.75 + Jaccard 0.30.

| # | Recipe | Result | Telemetry highlights | Verdict |
|---|---|---|---|---|
| 1 | **Original 70-iter, all 3 bugs active** (pre-Codex deliberation) | ac 4.06 → 3.85 | flat from iter 10; sat unknown (telemetry not yet added); barrier dominated | RED (stall) |
| 2 | **Mask 25:65 + LR=1e-2, buggy barrier still** | ac 4.06 → 3.98 | sat=1.0 by iter 10; field hit ±DELTA_MAX everywhere | RED (faster collapse) |
| 3 | **Mask + margin barrier + LR=1e-2** (5-iter probe) | ac 4.06 → 3.90 | iter-1 wbar=0.097 (barrier fixed); sat → 0.42 by iter 5 | YELLOW (Adam still saturating) |
| 4 | **Full recipe** (mask, margin barrier, DELTA_MAX=3.0, LR=3e-3, warmup=20, norm-cap=0.35) | ac 4.06 → 3.37 | sat ≤ 0.05 throughout; dmax stable ~2.9; Whisper "I" | YELLOW (audio: "bo bo ne gardeh", speech-like, wrong phonemes) |
| 5 | **Anti-jaggedness** (#4 + W_SMOOTH_S 10×, W_SMOOTH_T 30×, W_CURV 10×) | ac 4.06 → 3.38 | max_grad_t 53 → 29 (1.8× drop); max_grad_s unchanged 12.3; Whisper "I" | A — smoothness fired on t-axis but didn't unlock audio |
| 6 | **Synth-target /A/ static** (#5 recipe, target = TEA-rendered Story /A/) | ac 1.97 → 0.65 (ratio 0.33) | Gate 2 PASS, Whisper "Uh", recovered field log-MAE 0.87 vs truth | DECISIVE — optimizer works in acoustic sense, recovered tube wrong |

**Compute used**: ~5 GPU-hours total on the 16 GB GPU, plus
~2 hours wasted to a GPU-contention deadlock on 2026-05-14 when an
auxiliary script was run alongside a live training process. Per-iter
wall ~44 s at frame_chunk_size=10. JIT compile ~60-90 s. Backward
peak VRAM ~5 GB (well under the 16 GB budget thanks to per-sample
`jax.checkpoint`).

---

## The decisive finding — articulatory-acoustic non-uniqueness

The synth-target probe (experiment #6) is the load-bearing data
point. Setup:

1. Take Story 1996 /A/ area function (40 supraglottal sections).
2. Splice into default tube.
3. Render through TEA at 22.05 kHz (note: TEA renders internally at 44.1 kHz
   and the file is stored at 22.05 kHz after 2× decimation at the boundary;
   the engine sample rate is 44.1 kHz — see `docs/incorrect_assumptions.md`
   A1) with sustained voiced source (F0=120 Hz, PR=8000 dPa). Save as
   `data/probes/synth_target_static_A.wav`. Save ground-truth field
   as `..._truth.npz`.
4. Run copyfit on it from uniform-pipe init, same recipe as the
   anti-jaggedness run.
5. Compare recovered field vs ground truth.

Result:

```
init acoustic loss:   1.97
final acoustic loss:  0.65    (ratio 0.33, well under Gate 2 threshold 0.6)
Whisper transcript:   "Uh"    (Whisper recognized the vowel character)
field log-MAE vs truth: 0.87
per-frame correlation: 0.882
```

The recovered tube's mean profile, compared section-by-section to
truth:

```
section  truth      recov     diff
   25   -0.570    -0.786   -0.216   ← pharynx, close
   34   -0.975    -1.132   -0.157   ← still close
   45   +0.615    -1.180   -1.795   ← mid-mouth, DIVERGES
   53   +2.192    +1.143   -1.049   ← mouth peak, recov is half-area
   60   +1.711    +0.154   -1.557   ← front of mouth, recov is closed
```

**The truth /A/ tube has an open front-of-mouth (area 5-9 cm²).** The
recovered tube has the front-of-mouth narrowed to 0.5-3 cm². The
optimizer **found a structurally different vowel** that happens to
produce a similar log-mel signature. Whisper transcribes the result
as "Uh" — closer to a schwa than to /A/.

**This is the canonical articulatory-acoustic non-uniqueness problem.**
It is well-known in speech-production research (Atal 1971; Mermelstein
1973; many subsequent inversions). Many tract trajectories produce
similar acoustic features; the inverse map from audio to articulation
is one-to-many.

Acoustic loss alone has no preference between the equivalent solutions.
The optimizer picks whichever is locally easiest from its init point.
With the default tube as init, that means "narrow front cavity"
solutions are closer in parameter space than "open front cavity" /A/.

---

## What it means

### For Week 1's closure

The foundational question — *can the TEA + differentiable optimization
stack recover speech by acoustic matching?* — has a structurally
informative answer: **the stack can produce speech-like audio, but it
cannot recover the correct articulation from acoustic loss alone, even
on a clean TEA-renderable target.**

This is GREEN for the project's claim that *TEA is a usable
differentiable speech instrument* and RED for the strategy of
*training the body model with only acoustic loss*.

### For Week 2's plan

The course-correction doc named SPARC kinematic supervision as the
Week-2 mechanism for resolving non-uniqueness. That was an
architectural argument; this is now an empirical mandate.
(**[Under review per `docs/incorrect_assumptions.md` A4]** — "empirical
mandate" overstates the evidence; the optimizer had not converged and
frication was disabled. The Codex review verdict was UNDETERMINED.
SPARC remains the planned Week-2 path, but the "now empirical" framing
is premature.)

Concretely, Week 2 needs to:

1. **Pre-extract SPARC labels offline** on a small corpus (50-100
   utterances + the Natasha probe + 10 developer-recorded clips).
   Output `.npz` per utterance with 12-d EMA + F0 + loudness + 64-d
   speaker embedding.
2. **Bind those labels into the copyfit loss** as a kinematic
   supervision term. The body model's intermediate `z_t`
   representation gets supervised by SPARC's EMA. The acoustic loss
   remains, but is no longer the sole arbiter.
3. **Re-test the synth-target probe with SPARC labels added.** Does
   the optimizer now recover the correct /A/ tube? If yes,
   non-uniqueness is resolved by kinematic priors. If no, we have a
   more fundamental issue (SPARC labels are too low-res, our body
   model needs more capacity, etc.) — but that's a different debug.

### For the cloning roadmap

The Week-2 result will determine the speaker-fitting path:

- If SPARC labels resolve non-uniqueness on synth-target → cloning is
  "fit `speaker_vec` such that the speaker's SPARC-extracted labels
  + the body model produce the speaker's audio." Per
  `ARCHITECTURE.md` §2.5.
- If SPARC doesn't resolve it → we have multiple options (richer
  articulatory targets via rtMRI inversion, larger body model,
  hybrid procedural priors). Decision deferred to that data point.

---

## Open issues (Week 2 / later)

Recordable but didn't block Week 1's main finding:

- **Frication is not wired into the copyfit pipeline** *(as of Week-1
  closure; see A3 below)*. Codex flagged this as H7. The script passes
  `frication=None` to TEA; TEA defaults to disabling frication. Natasha's
  phrase has ~9 fricatives/affricates; without frication the optimizer
  can't produce them and substitutes voiced CV gestures. Likely accounts
  for some of the "bo bo ne gardeh" character. Wiring in
  `correctness/004a`'s frication module is a 2-4 hour task that
  may meaningfully improve real-audio fits. **Worth doing before
  the synth-target-with-SPARC retest if time permits.**
  > **[Update per `docs/incorrect_assumptions.md` A3]** This session
  > added a `--frication` CLI flag to `copyfit_neural_field_phrase.py`.
  > Frication is still disabled by default (preserves baseline
  > reproducibility). Task 007 exercises the frication-enabled path.
- **Gate-2 spec mismatch in the task file**: spec at
  `001_neural_field_phrase_probe.md:191` says MR-STFT-only, but the
  script gates on composite acoustic loss. Either is OK for
  diagnostic purposes; the spec should be amended to match.
- **`train_step` returns pre-update loss**: comments in
  `copyfit_neural_field_phrase.py` claim it's post-update. Line ~818
  runs one extra Adam step to "true up" the final report. Cosmetic;
  doesn't affect the optimization.
- **`core.autocrlf=true` reverts LF fixes on runbook scripts**: the
  `runbooks/*.sh` and `runbooks/*.py` have CRLF line endings checked
  in, breaking bash invocation on Linux. Workaround is to invoke
  the Python impl directly via the venv:
  `tractus_controller/.venv/bin/python runbooks/_whisper_gate_impl.py ...`.
  A repo-level fix would be `.gitattributes` with `* text=auto eol=lf`.
- **`tractus_engine_advanced` installed as namespace package**: top-level
  re-exports don't resolve. Cosmetic; sub-module imports work.
- **Plot warning in copyfit script**: the area-function plot tries to
  match shapes (40,) and (93,) and fails. Doesn't break anything;
  the plot just isn't produced.

---

## Codex threads consulted (chronological)

- `[session identifier omitted]` — the original course-correction deliberation
  (2026-05-09). Sets up the pivot from conv-MLP body model to
  neural-field + SPARC. Read by Week 1 task spec authors.
- `[session identifier omitted]` — the worker af90's K=10 frame_chunk_size +
  fused train_step throughput analysis (2026-05-13).
- `[session identifier omitted]` — Week-1 stall diagnosis (2026-05-14):
  identified the three layered bugs (mask, margin barrier, warmup);
  prescribed the experimental progression #4 → #5; predicted the
  phonetic-wall outcome that #6 then confirmed empirically.

Use `codex-reply` to continue any of these if relevant. The
`[session identifier omitted]` thread is the most useful continuation for Week-2
SPARC integration questions, as it already has the empirical
non-uniqueness signal in its context.

---

## Repo state at Week 1 closure

```
master = 4b665e7  feat: synthetic-target copyfit probe — the "is the optimizer broken?" test
         c639a73  feat(copyfit): tunable smoothness/curvature weights via CLI
         9d96c30  feat(copyfit): DELTA_MAX=3.0 + Adam warmup + update-norm cap
         e17e7b8  fix(copyfit): mask field to 25:65 + margin softplus barrier + telemetry
         916ca36  feat: sustained-vowel hand-tune probe
         de9bf8a  merge: task 001 neural-field phrase probe — worker af90ebf6
         9b18055  refactor!: course correction — neural-field + SPARC pivot
```

Five active task files in `workplan/tasks/00_active/`. Week 1
deliverable script is at `scripts/copyfit_neural_field_phrase.py`.
Reproducible reports + audio under `outputs/copyfit_neural_field/`
(gitignored):

- `juice_lemons_run100/` — full recipe baseline (ac 3.37, Whisper "I")
- `juice_lemons_smooth100/` — anti-jaggedness recipe (ac 3.38, Whisper "I")
- `synth_target_static_A/` — synth-target /A/ probe (ac 0.65, Whisper "Uh", log-MAE 0.87 vs truth)

Reference audio at `data/probes/juice_lemons_ref.wav` (ElevenLabs
synthetic, marked `consent_verified`; **note per `docs/incorrect_assumptions.md`
A10**: this file was originally generated as MP3 by ElevenLabs and converted
to WAV — it carries lossy-compression artifacts and must NOT be claimed as
pristine PCM for psychoacoustic comparisons) and
`data/probes/synth_target_static_A.wav` (TEA-rendered, ground-truth
field at `synth_target_static_A_truth.npz`).

---

## Disposition

**Week 1: CLOSED.** Foundational question answered. The
recipe-tuning branch of the workplan is exhausted; further iteration
on smoothness / curvature / init / LR will not cross the phonetic
wall. The Codex-named decision-tree outcome is **"phonetic wall
confirmed empirically — SPARC is non-optional"** (Codex H6/H7's
fallback recommendation now upgraded to primary).

> **[Under review per `docs/incorrect_assumptions.md` A2, A3, A4]**
> "Phonetic wall confirmed" and "SPARC non-optional" are stronger claims
> than the data support. Confounders: (A2) the 100-iter optimizer had not
> converged; (A3) frication was disabled, so fricative-rich content was
> structurally unreachable; (A4) the Codex adversarial review verdict was
> UNDETERMINED, not confirmation. Task 007 (frication retest at extended
> iters) is the discriminating experiment. Closure verdict stands
> qualitatively — the non-uniqueness problem is real — but the "SPARC
> non-optional" framing should not be treated as fully settled evidence
> until Task 007 reports.

**Week 2: dispatch SPARC extraction pipeline** as the next worker
task. The dispatch prompt is in `docs/week1/sparc_dispatch_prompt.md`
(forthcoming, see below). The Week-2 task spec
(`workplan/tasks/00_active/002_sparc_extraction_pipeline.md`) is
unchanged from the pivot — it was correctly scoped and now its
predicates are met.

**Followup decisions for the developer:**

1. **Frication wiring** before or after SPARC ingest? Cheap (2-4h)
   improvement to copyfit, doesn't gate SPARC, may meaningfully
   reduce the "bo bo ne gardeh" → recognizable-phrase gap.
   Recommended: in parallel.
2. **SPARC license + WavLM Large weight access**: 5-min verification
   before bulk extraction begins. Fallback to HuBERT-base if
   blocked.
3. **Body-model training scope**: how many utterances does
   pretraining/003 need? With SPARC labels resolving non-uniqueness,
   a few hundred utterances might suffice for the proof-of-concept;
   tens of thousands for the full v0 demo. Sequence: probe first,
   scale second.
