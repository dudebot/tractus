# tractus — Architecture

> **Status:** living architecture document. Greenfield rewrite
> 2026-05-09 around the neural-field + SPARC pivot. Authoritative
> on any conflict; if a planning artifact disagrees with this doc,
> reconcile here. Active work plan lives in
> [`workplan/NOW.md`](workplan/NOW.md). Pivot rationale is in
> [`workplan/handoff/2026-05-09_course_correction.md`](workplan/handoff/2026-05-09_course_correction.md).

---

## 0. TL;DR

A physics-first, **differentiable** text-to-speech system. Five
modules above a 1-D transmission-line acoustic simulator (TEA):

1. **Planner** — text → phones, durations, prosody envelopes.
2. **Controller** — phones + prosody + speaker style → sparse
   asynchronous gesture events.
3. **Body model** — gesture trajectory + speaker vector → continuous
   neural-field area function `A_θ(s, t)` over tract location and
   time. Sampled to TEA's 93 sections at the renderer boundary.
4. **TEA renderer** — physical 1-D transmission-line simulator,
   per-sample MNA solve, fully differentiable in JAX.
5. **Speaker inference** — 30-s reference audio → kilobyte-scale
   speaker object (anatomy bias + source defaults + motor-style
   embedding + optional channel model).

**SPARC and WavLM Large supervise training; neither runs at
inference.** Acoustic loss through TEA is the primary training
objective. Story 1996/2008 anchors are sanity-check held-out gates,
not supervision targets.

> **[Note per `docs/incorrect_assumptions.md` A6]** The budget numbers
> below (≤ 8 GB / ≤ 30 M / ≤ 5× realtime) are unsourced as of this
> audit. They represent aspirational positioning ("consumer hardware",
> "far smaller than Kokoro 82 M") rather than hard product constraints
> with a named source. Treat them as soft targets until a developer
> decision or Task 010 formally sources them.

| Axis | Target |
|---|---|
| Inference VRAM | ≤ 8 GB |
| Total trainable params | ≤ 30 M |
| Per-speaker object | ≤ 2 KB |
| Reference audio for speaker fit | 30 s consented |
| Sample rate | 22.05 kHz (project output target), 16 kHz (floor) — note: TEA renders internally at 44.1 kHz (`TIMESTEP_S = 1/44100`) and decimates 2× at the boundary; see `docs/incorrect_assumptions.md` A1 |
| Inference latency | ≤ 5× realtime on consumer GPU |
| Precision (TEA) | fp32 production, fp64 parity-debug only |

**Compute thesis.** Designed for ≤ 8 GB *inference*, not ≤ 8 GB
naive end-to-end backprop. Training-time gradients through TEA are
short-window, checkpointed (`jax.checkpoint`), and supplemented by
SPARC kinematic supervision so the body model has a non-acoustic
anchor.
(Note: "supplemented by SPARC kinematic supervision" is the planned
training architecture; whether kinematic supervision is structurally
required vs. one useful add-on is under review pending Task 007 —
see `docs/incorrect_assumptions.md` A4.)

**v0 scope.** Constrained articulatory TTS with anatomy + source
fitting from a 30-s reference. Full zero-shot cloning is post-v0
research. v0 ships when 5 unseen Harvard sentences pass Whisper at
WER ≤ 0.5 in the developer's voice, TEA-only path with no
post-filter rescuing.

---

## 1. Vision

### 1.1 What we are building

A TTS system whose **inference-time bottleneck is a differentiable
physical speech-production stack**: a small learned controller emits
sparse asynchronous gesture intentions; a learned but physically
constrained body model expands them into a continuous tract area
field; a 1-D transmission-line acoustic simulator renders that
field as a waveform.

User-facing capability: voice cloning. 30 seconds of consented
reference audio is enough to fit a per-speaker object (anatomy
bias + source defaults + motor-style embedding); arbitrary text
is then rendered in that voice.

### 1.2 Why physics-first

The articulatory bottleneck is interpretable, identifiable, and
generalizable. A neural latent vocoder of the same dimensionality
can sound as good but produces representations that don't transfer
across speakers, languages, or registers without retraining. The
physical stack gives a small, semantically labeled control space —
the same across every speaker — with per-speaker variation
localized in the anatomy and motor-style objects.

Side benefit: the physical state is **auditable**. If the system
produces gibberish, you can plot the area function and the source
flow and see why. Black-box embeddings can't do this.

### 1.3 The body should be a continuous field, not a tube

This is the lesson from the 2026-05-09 pivot. Earlier iterations
parameterized the body as direct per-section areas + lengths (93
free DOF per frame). Two diagnostic findings rejected this
parameterization:

- **`infra/006`** — gradient descent through TEA on free area bins
  could lock one formant per vowel but not all three jointly.
  Consistent with Chen & Sitzmann 2025, who explicitly report
  discrete-bin tube optimization fails for static vowels and
  switch to a continuous neural field to smooth the loss surface.
- **`correctness/004b`** — the smoke pipeline produced empty/garbage
  Whisper transcripts both pre- and post-frication.

The current architecture treats TEA's 93 sections as a **numerical
discretization of a continuous field**, not as the body's ontology.
The body model emits a smooth function `A_θ(s, t)`; the renderer
samples it. Speaker variation is a small bias on a shared field,
not a 93-section snapshot.

### 1.4 What "physics-first" does NOT mean

- Not the entire system is hand-written physics. Three of the five
  modules (planner, controller, body model) are conventional
  neural networks. TEA is hand-written physics. Speaker inference
  is partly NN, partly optimization.
- Not the renderer is the only path to audio. A small constrained
  inference residual / post-filter is permitted in principle
  (Module 6B) but **off by default** in v0. Prosody control lives
  in the controller, not in a post-net; mel-converter rescue
  layers are explicitly rejected.
- Not the body model is hand-coded jaw/tongue kinematics. The body
  model is a learned continuous field with smoothness +
  reachability + speaker priors. Procedural physics is a research
  branch (post-v0).
- Not VTL-with-gradients. VocalTractLab's articulator-space →
  midsagittal-distance → tube-area pipeline was the historical
  starting point but is not the architecture. Maeda 1990 and
  Story 2005 area-mode parameterizations are sanity references,
  not control surfaces. See the course-correction doc for the
  rationale.

### 1.5 Quality and compute targets

These are stretch targets, not commitments. The literature on
small articulatory TTS (SPARC's 12 EMA channels at 50 Hz +
speaker emb, DDSP-articulatory at 0.4 M params, RT-VC's articulatory
codec) supports the small-VRAM ambition; the per-speaker identity
bar is empirical.

> **[Note per `docs/incorrect_assumptions.md` A6]** The ≤ 8 GB and
> ≤ 30 M targets below are aspirational positioning ("Consumer hardware",
> "Far smaller than Kokoro 82 M"), not documented product constraints.
> No specific hardware profile or product decision is on record as
> requiring these exact numbers. Use as planning guides, not hard gates.

| Axis | Target | Reference |
|---|---|---|
| Inference VRAM | ≤ 8 GB | Consumer hardware (aspirational; see A6 note above) |
| Trainable params | ≤ 30 M total (planner ≤ 10 M + controller ~5 M + body ~0.5 M + speaker enc ~5–15 M) | Far smaller than Kokoro 82 M (aspirational; see A6 note above) |
| Per-speaker object | ≤ 2 KB | Anatomy + speaker_vec + source defaults |
| Reference for cloning | 30 s consented | |
| Sample rate | 22.05 kHz project output target, 16 kHz floor — TEA renders internally at 44.1 kHz; see `docs/incorrect_assumptions.md` A1 | |
| Inference latency | ≤ 5× realtime on consumer GPU | Not real-time, not painful |
| Output quality bar | Recognizable speaker identity on held-out phrases via 2/3 listener A/B | Empirical |

---

## 2. The five modules

> Critical separation:
> - **Planner** decides *which words and which prosody.*
> - **Controller** decides *what to do.*
> - **Body model** defines *what the body can do.*
> - **TEA** renders *what the body sounds like.*
> - **Speaker inference** captures *which body and which habits.*

### 2.1 Module 1 — Linguistic / Prosody Planner

**Role.** Maps text (with optional style/intent tags) to a symbolic
timing and prosody plan: phone segments, durations, stress, phrase
boundaries, F0/loudness envelopes.

**Trains on.** Text–audio aligned corpora (LibriTTS-R, VCTK), MFA
phoneme-duration labels, F0/loudness from PyWorld DIO analysis.

**Inference.** Frozen, shared across speakers. The planner does
**not** see speaker_embed beyond F0 *range* (not contour); a shared
planner is the whole point.

**Interface.**

```
in:  tokens:       int[L]              # phones via G2P
     speaking_rate: float
     register:     int
     affect:       float[D_affect]
     emphasis:     bool[L]
     speaker_F0_range: float[2]
out: segments:     (phone_id, dur_s)[N]
     stress:       int[N]
     phrase_break: bool[N]
     f0_env:       float[T_f0]         # log-Hz @ ~100 Hz
     loud_env:     float[T_f0]
     affect_plan:  float[D_affect]
```

Affect is first-class (modern TTS conditions on emotion / intent
tokens; we do the same). Affect propagates via the controller's
gesture-distribution biases (duration, strength, adduction,
breathiness, F0 range usage, articulatory precision), not via the
body model.

**Size.** ≤ 10 M params. v0 may use a rule-based G2P + small
duration/F0 heads or borrowed weights. Defer scaling-up until
after a v0 demo lands.

### 2.2 Module 2 — Neuromotor Controller

**Role.** Converts the planner's segment + prosody plan into a list
of asynchronous overlapping `GestureEvent`s. This is where the
*vocal habit* lives: coarticulation timing, undershoot, consonant
precision, voicing-style preference, microprosody.

The controller does NOT decide where every articulator is at every
10 ms tick. It emits sparse intentions ("close lips here," "open
velum there," "raise tongue tip here," "increase adduction here"),
which the body model's interpolator composes into a continuous
trajectory. This is the Articulatory Phonology framing
(Browman & Goldstein 1992, Saltzman & Munhall 1989).

**Interface.**

```python
@dataclass(frozen=True)
class GestureEvent:
    channel_id:   str
    onset_s:      float
    target_value: float
    duration_s:   float
    release_s:    float | None
    strength:     float  # 0–1
    ramp_shape:   Literal["pchip", "critically_damped", "ballistic", "hold_release"]
    prosody_tags: tuple[str, ...]
    affect_tags:  tuple[str, ...]
```

Density: ~30 events/s averaged across channels — varies with
speaking rate. NOT a fixed update grid.

**Channel inventory (20 channels, pinned in
`tractus_controller/.../body_model/channels.py`).**

- 12 articulator: `jaw_open`, `tongue_body_x`, `tongue_body_y`,
  `tongue_tip_x`, `tongue_tip_y`, `lip_aperture`, `lip_protrusion`,
  `velum_opening`, `larynx_height`, `tongue_tip_side`, `tongue_side`,
  `pharyngeal_width`.
- 8 source: `F0`, `F0_jitter`, `lung_pressure`, `glottal_adduction`,
  `vocal_fold_tension`, `breathiness`, `creak_gate`, `pressed_voice_gate`.

**Trains on.** SPARC-extracted articulatory kinematics from real
audio (training-time supervision); reconstruction-through-TEA on
short windows.

**Size.** ~5 M params. NAT-style or small autoregressive transformer.

### 2.3 Module 3 — Biomechanical Body Model — neural field

**Role.** Maps `(gesture_trajectory, speaker_vec)` → continuous
log-area field `log A_θ(s, t)` over tract location `s ∈ [0, 1]`
(normalized) and time `t ∈ [0, T]`. The renderer samples this
field at TEA's 93 sections per frame.

**Architecture.**

```
gesture_traj[T, 20]   ← controller output, 50 Hz interp
speaker_vec[32]       ← from speaker inference (Module 5)

z_t = encoder(gesture_traj[t])      # small MLP / conv, ~50–80 K params

log A(s, t) = log A_default(s)
            + base_θ(s, speaker_vec)                # speaker-conditioned anatomy bias, slow
            + delta_θ(s, z_t, speaker_vec)          # kinematic + speaker-conditioned dynamics, fast

A(s, t) = exp(log A(s, t)),  hard-clamped [A_min, A_max]
```

Field MLP uses sinusoidal positional encoding over `s`, condition
features (`z_t`, `speaker_vec`) FiLM-injected, ~4 layers @ hidden
128. Total body-model param count target: **0.3 – 0.8 M**. If it
exceeds 1 M params, the architecture is overspecified.

**Speaker conditioning is a small bias, not a free field.** The
per-speaker object (≤ 2 KB) holds `speaker_vec[32]` plus source
defaults plus optional anatomy coefficients. Cloning fits this
vector, not the full field.

**Length.** Mostly fixed; one global tract-length scale + a
small lip-extension scalar are the only length DOF in v0.
Per-section length is not a free variable.

**Auxiliary outputs** (per frame, derived alongside `A`):

```
glottis_state[11]          # source controls — F0, PR, XB, …
nasal_coupling             # scalar, low-rank in v0
subglottal_press           # scalar
frication_gain[93]         # f_phys(constriction, pressure_drop, place, voicing, obstacle_proxy)
frication_color[93]        # small learned residual, energy-capped
```

**Trains on.**
- SPARC EMA labels as kinematic supervision targets (Task 003).
- Acoustic loss (log-mel + MR-STFT) through TEA on short windows.
- Smoothness priors (`d log A / ds`, `d log A / dt`, curvature).
- Reachability barriers from
  `tractus_controller/.../body_model/reachability.py`.
- **Story anchors only as held-out sanity gates** — render `/a, i, u/`
  through the body model + TEA, measure F1/F2/F3, check ±20 %.

**Cheat-channel discipline (load-bearing).**

- Wall mass, resistance, stiffness are **anatomy-owned static**,
  not per-frame body-model outputs. Per-frame walls would let the
  optimizer hide source/shape information in EQ.
- `frication_gain` is **physically derived** from
  `(constriction_area, pressure_drop, place, voicing, obstacle_proxy)`.
  A free `frication_gates` channel is forbidden — it can inject HF
  energy with no corresponding physical condition.
- `frication_color` is the only learned residual on the frication
  path and is energy-capped + ablated in every eval.

### 2.4 Module 4 — TEA Acoustic Renderer

**Role.** Differentiable physical synthesis of the **whole
instrument** — glottal source + branched 1-D transmission-line
tract + nasal coupling + frication + radiation. Takes the body
model's per-frame field samples and renders to waveform.

**Topology.** Branching transmission-line over 93 sections +
trachea + glottis + pharynx + mouth + nose + sinus + fossa branches.
Per-sample MNA solve of a 97×97 banded SPD matrix; `lax.scan` over
audio samples; THETA-trapezoidal time discretization (THETA = 0.505);
soft walls; Bernoulli losses; smooth area clamps. 8th-order
Chebyshev radiation IIR.

**Glottis.** Pure-JAX analytical GeometricGlottis. Static anatomy
(RT, RL, RF0) from speaker inference; dynamic source state (F0,
PR, XB, XT, CA, PL, RA, DP, PS, FL, AS) from the body model.
TwoMassModel and TriangularGlottis are deferred to v1+.

**Frication.** Per-place SOS biquad PSD shaping over the constriction
geometry. MNA injection at the post-solve net-flow term (stable
at 1/5/15 cm³/s). Physical-derivation rule above: gain is
deterministic from constriction + pressure + place + voicing +
obstacle_proxy.

**Trains on.** Nothing — the parameters are physical constants
(THETA, MIN_AREA, Tikhonov ε, Chebyshev coefs). Calibration via
forward parity tests + soft-mode gradient stability + frication
audibility gates.

**Inference.** Default fp32. Per-sample MNA solve runs at
44.1 kHz (`TIMESTEP_S = 1/44100`); output is decimated 2× to
22.05 kHz at the boundary. At 50 Hz frames, that is 882 samples/frame
internally (note: `samples_per_frame=441` in the task spec refers to
the post-decimation count at 22.05 kHz — see `docs/incorrect_assumptions.md`
A1). Backward through TEA at
T=118 b=4 fits 708 MB peak via `jax.checkpoint` + `custom_vjp` on
`step_one` + `custom_linear_solve` on the MNA solve.

### 2.5 Module 5 — Speaker Inference & Adaptation

**Role.** Stores the speaker's parameterization across three
sub-roles:

- **Anatomy + source defaults** for the body model.
- **Motor-style embedding** for the controller.
- **Channel model** for evaluation only — does NOT contaminate the
  cloned voice unless explicitly requested.

**The recording-channel nuisance bucket is non-negotiable.**
Reference audio carries mic, codec, reverb, denoising artifacts.
Without an explicit channel model, the anatomy/source fit absorbs
these and the cloned voice has permanent room coloration baked in.

**Two paths to populate the per-speaker object.**

- **Learned encoder** (fast). 30 s reference → speaker_vec +
  initial anatomy + source defaults + channel model. Trained
  jointly on multi-speaker corpora using SPARC speaker embeddings
  as a teacher signal at training time.
- **Short-window TEA inverse refinement** (slow, accurate).
  Refines anatomy / source / channel by short-window backprop
  through TEA on chunks of the reference. Outer loop on anatomy,
  inner loop on per-window gestures.

Use both. Pure inversion from random init fails (30 s is too little
signal, the inverse is degenerate). The encoder gives a basin of
attraction; the TEA inverse pulls to the local optimum.

**Per-speaker output.**

```
speaker_vec:        float[32]            # body-model conditioning
source_defaults:    float[D_source]      # mean F0, tilt, breathiness baseline
motor_embed:        float[D_motor]       # 64–256 dim, controller conditioning
channel_model:      ChannelParams        # EQ tilt, room, mic proximity (eval-only)
anatomy_extras:     float[D_anatomy]     # optional low-rank tract / glottis anatomy
```

Total ≤ 2 KB. Cloning is fitting this object, not retraining the
controller or body model.

**Anti-cheat.** Some training samples have gesture-unreachable
targets (e.g., formant-shifted self audio). The body model must
**refuse** to find an impossible posture. Refusal rate ≥ 80 % on
unreachable targets is an acceptance gate.

---

## 3. Training story

### 3.1 SPARC as training-time-only teacher

SPARC (Cho et al. 2024, `docs/references/core/2024_sparc_vocal_tract_kinematics.pdf`)
is a pretrained articulatory inversion model: audio → 12 EMA-like
articulator channels @ 50 Hz + F0 + loudness + 64-d speaker
embedding. Underlying feature extractor is WavLM Large
(~316 M params, 1.26 GB).

**Role in this project.** Training-time teacher only. Run offline
on a corpus → freeze labels as `.npz` → JAX-side training loop
consumes the `.npz` files. **WavLM Large is never present in the
inference stack.** This is enforced by env isolation: SPARC
extraction lives in `tools/sparc/.venv/` (PyTorch); the JAX env has
zero PyTorch dependency.

Why SPARC vs Maeda / Story / VTL teachers:
- Real-data-grounded; not derived from MRI snapshots of static
  vowels.
- Produces a 50 Hz kinematic representation that maps cleanly to
  the controller's output space.
- Comes with a published 64-d speaker embedding that doubles as a
  teacher target for our speaker encoder.

### 3.2 Primary losses

| Loss | Where | Notes |
|---|---|---|
| log-mel + MR-STFT | through TEA, short windows | Primary acoustic objective. Existing JAX impl in `tractus_controller.copy_synth.tea_backend`. |
| Kinematic MSE | body model intermediate `z_t` ↔ SPARC EMA labels | Anti-circularity anchor — body model has supervision that doesn't pass through TEA. |
| Smoothness | `d log A / ds`, `d log A / dt`, curvature | Prevents pathological field shapes. |
| Reachability barrier | `body_model/reachability.py` clamps as soft penalties | Field stays in physical range. |
| Refusal loss | unreachable-target training samples | Gesture-unreachable inputs should produce flat / declined fields. |
| Identity loss | speaker encoder vs SPARC speaker_emb (training only) | Speaker encoder learns to predict the same identity SPARC sees. |

**Area-MSE against Story anchors is NOT a primary loss.** Story
anchors are **held-out sanity gates** only.

### 3.3 Training data

| Corpus | Use | Status |
|---|---|---|
| LibriTTS-R | planner duration/F0 + body-model SPARC labels | reference; not yet ingested |
| VCTK | speaker encoder | reference; not yet ingested |
| Developer-recorded | Week-1 phrase, Week-2 fitting, Week-4 demo | in-progress |
| Story 1996/2008 anchors | held-out vowel sanity gate | ingested (`outputs/fixtures/story_1996_anchors.npz`) |
| MFA forced alignments | planner duration supervision | reference |

### 3.4 Voice-cloning ethics

Every training/fitting manifest carries `consent_verified: true`.
CI lint enforces. No public figures. Channel model eval-only by
default.

---

## 4. Inference story

```
text
 └─► Planner             (≤10 M params, frozen)
      └─► segments + prosody
           └─► Controller   (~5 M params)
                └─► GestureEvent[]   (sparse async, 20 channels)
                     └─► gesture_traj[T, 20]   (50 Hz interp)
                          └─► Body model        (~0.5 M params, neural field)
                               └─► A(s, t) sampled to [T, 93]
                                    + glottis state, nasal, frication
                                    └─► TEA              (frozen physics)
                                         └─► waveform
```

**Per-speaker object** loaded once, parameterizes:

- `speaker_vec[32]` → body-model conditioning.
- `motor_embed` → controller conditioning.
- `source_defaults` → glottis state baselines.
- `anatomy_extras` → optional wall / glottis static.

**Inference VRAM budget** — backward not required, only forward.
Per-frame MNA solve dominates. Expect 4–8 GB on a ~3-second utterance.
(Note: TEA renders at 44.1 kHz internally, not 22.05 kHz; the memory
estimate is for the 44.1 kHz engine rate with 2× decimation at the
boundary. See `docs/incorrect_assumptions.md` A1.)
Streaming/chunking can lower this if needed.

**Latency target.** ≤ 5× realtime. Real-time streaming is post-v0.

---

## 5. Validation and acceptance rules

### 5.1 Forward correctness (TEA)

- Forward parity at every milestone (M1 glottis, M2 TDS step,
  M4 output IIR, M5 full pipeline, M7 frication). Already done.
- Soft-mode gradient stability over 10 K seeds (M6). Already done.
- fp32-vs-fp64 self-consistency on synthetic corpus.

### 5.2 Backward stability

- T=118 b=4 backward fits ≤ 2 GB peak (G2 gate). Already passing
  at 708 MB.
- No NaN gradients on any acceptance test.

### 5.3 Cheat-channel ablation

CI fails if:
- `wall_dynamic_residual` magnitudes exceed cap in eval.
- `frication_color` magnitudes exceed cap in eval.
- Ablating either channel changes audio loss by < 5 % (channel was
  doing nothing) or > 20 % (channel was doing too much).

### 5.4 Identifiability

- Anti-cheat refusal rate ≥ 80 % on unreachable-target eval set.
- Source/tract separation: per-axis attribution (F0, tilt,
  frication, nasality, formants) checked with
  `runbooks/source_tract_sep.py` on every pretraining + cloning eval.

### 5.5 v0 demo gates (Week 4)

5 unseen Harvard sentences, TEA-only path, no post-filter rescue:

- Whisper WER ≤ 0.5 averaged over the 5 phrases.
- No phrase WER > 0.7.
- Audible fricatives / stops / vowels in spectrograms.
- Listener identifies developer's voice on at least one of three
  A/B comparisons against a random-other-voice render.
- ≤ 8 GB peak inference VRAM.
- ≤ 5× realtime per phrase wall-clock.
- Per-speaker object ≤ 2 KB.

If any of these fail, exercise the kill criterion in the
course-correction doc Risk C: ship demo on DDSP-articulatory
runtime, document TEA as research-grade.

---

## 6. Roadmap (active)

The 4-week plan lives in
[`workplan/NOW.md`](workplan/NOW.md) and the five active task files
under [`workplan/tasks/00_active/`](workplan/tasks/00_active/).
Summary:

| Week | Deliverable | Kill criterion |
|---|---|---|
| 1 | Neural-field copy-synthesis of one developer-recorded phrase. Tests "can TEA + differentiable optimization recover intelligible speech?" | Whisper gate fails after 2 debug days → TEA may not be the v0 product renderer (Risk C, course-correction doc). |
| 2 | SPARC label extraction + speaker-conditioned body model. Reproduces Week-1 phrase + generalizes to ≥ 7 of 10 held-out short utterances. | Body model can't generalize beyond the bootstrap utterance even after kinematic-loss tuning. |
| 3 | Controller trained on SPARC pseudo-labels. End-to-end text → audio on 5 unseen sentences. | Controller-driven path worse than SPARC-driven path → demo ships on SPARC-direct, not text→controller. |
| 4 | Constrained demo: 30-s reference fitting + 5 Harvard sentences in developer's voice. Honest baseline vs DDSP-articulatory + SPARC vocoder. | TEA-only WER > 0.5 over 5 phrases → ship on DDSP-articulatory runtime. |

---

## 7. Glossary

| Term | Meaning |
|---|---|
| TEA | tractus engine advanced — the JAX 1-D transmission-line acoustic simulator (Module 4). |
| TC | tractus controller — Python package holding planner, controller, body model, speaker inference. |
| SPARC | Cho et al. 2024 articulatory inversion model. WavLM-Large-based; produces 12 EMA + F0 + loudness + 64-d speaker emb at 50 Hz. Training-time teacher only. |
| MNA | Modified Nodal Analysis — the per-sample 97×97 banded SPD solve inside TEA. |
| GestureEvent | Async gesture intention with channel, onset, duration, target value, ramp shape. Output of the controller. |
| `A_θ(s, t)` | Continuous neural field: log-area as a smooth function of normalized tract location `s` and time `t`. Output of the body model. Sampled to `[T, 93]` for TEA. |
| Story anchors | Story 1996/2008 published area functions for 11 vowels. Held-out sanity gate, NOT primary supervision. |
| Speaker object | Per-speaker `.npz`, ≤ 2 KB. Holds `speaker_vec[32]` + source defaults + motor_embed + channel model + optional anatomy_extras. |
| `D_anatomy` | Anatomy vector dimension. v0 target: 32–64 scalars (low-rank). |
| Cheat channel | A renderer-interface field that becomes a hidden neural EQ if left free. Forbidden by design (wall params are anatomy-static; frication_gain is physically derived). |
| Reachability barrier | Soft-clamp penalty keeping the field in physiological range. |

---

## 8. Hardware + precision

- **Environment.** Install pinned dependencies in a project virtual environment.
  GPU work needs compatible CUDA-enabled JAX; lightweight checks can use CPU.
- **Memory.** Profile workload memory on the target platform. Avoid concurrent
  GPU-heavy jobs. The verifier accepts `TEA_VERIFY_BACKEND=cpu`, which skips
  the GPU-specific batch-four memory gate.
- **Precision policy.** `tractus_engine_advanced.precision` owns
  the dtype policy. Default fp32. Set `TEA_PRECISION=fp64` BEFORE
  the first JAX import for parity tests; the policy module calls
  `jax.config.update("jax_enable_x64", True)` exactly once. Do NOT
  call it from outside the policy module.
- **fp64 is parity/debug only.** No fp64 in production paths. CI
  lint catches regressions.

---

## 9. Conventions

- **Commits.** Conventional format (`type(scope): subject`). No
  `Co-Authored-By: Claude` lines.
- **`outputs/` is gitignored.** Reports, fixtures, scratch data
  live there; reproducible verifiers live under `scripts/` or
  `runbooks/`.
- **Numeric prefixes** on phase folders give execution order at a
  glance — but this is now a single-track sprint, not a phase
  graph. Active work is `workplan/tasks/00_active/`.
- **Auto-tests beat ear-tests.** When a cheap automated check
  (Whisper WER on a fricative-rich sentence) does the same job as
  human listening, prefer the auto-test. The runbook stack
  (`runbooks/whisper_gate.sh`, `runbooks/source_tract_sep.py`)
  exists — use it. The `User-input: review` flag in a task file is
  permission for a human ping if needed, not a mandate.
- **Voice-cloning ethics.** Every training/fitting manifest carries
  `consent_verified: true`. CI lint enforces.
- **Whisper sidecars** write to `outputs/whisper/`, not the
  project root.

---

## 10. References

The project's reading list is in `docs/references/`. Tier 0
(load-bearing) for the current architecture:

- **Chen & Sitzmann 2025** — *Differentiable simulator with analytic
  gradients.* Reports discrete tube-section optimization fails for
  static vowels; switches to continuous neural-field area
  representation. Direct precedent for the body-model
  parameterization. `core/2025_chen_sitzmann_diff_vocal_simulator.pdf`.
- **SPARC 2024 (Cho et al.)** — *SPeech Articulatory Coding.*
  WavLM-based articulatory inversion. Training-time teacher.
  `core/2024_sparc_vocal_tract_kinematics.pdf`.
- **DDSP-articulatory 2024** — 0.4 M-param articulatory vocoder.
  Sizing precedent. `baselines/2024_ddsp_articulatory_vocoder.pdf`.
- **Narayanan & Alwan 2000** — distributed aerodynamic frication
  noise. TEA frication module is built on this.
- **Stevens 1998** *Acoustic Phonetics* — formant tables, source
  spectra, radiation models.
- **Fant 1960** — source-filter foundations.
- **Saltzman & Munhall 1989; Browman & Goldstein 1992** — gesture
  task-dynamics + articulatory phonology. Foundation for the
  controller's async gesture-event abstraction.
- **Story 1996/2008** — published vocal-tract area functions. Used
  as held-out sanity gate.

Sanity references (NOT architecture):

- **Maeda 1990** — low-d articulatory PCA. Foundational, but the
  modern differentiable line does not pass through it.
- **VocalTractLab / Birkholz lineage** — historical starting point
  for this codebase. Not the architecture; see course-correction
  doc.

Adjacent / future:

- **rtMRI inversion 2026** — complete tract geometry from real audio.
  Candidate v1+ teacher.
- **MOCHA-TIMIT / mngu0 / USC-TIMIT** — EMA corpora. Validation
  candidates.
- **RT-VC 2025**, **CosyVoice / CosyVoice 2**, **F5-TTS** — neural
  baselines for the v0 demo report.
- **Teaching Machines to Speak 2025** — RL articulatory control,
  syllable-level. Adjacent.

---

**House rule.** If you change the project's architecture or vision,
change this doc in the same commit. If a planning artifact disagrees
with this doc, this doc wins or the disagreement gets reconciled
here. The course-correction doc at
`workplan/handoff/2026-05-09_course_correction.md` is the bridge
record from the previous architecture; read it for the *why*. This
doc is the *what*.
