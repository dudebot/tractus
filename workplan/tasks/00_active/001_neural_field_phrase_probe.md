# Task 001 — Neural-field phrase probe (Week 1)

> **Read first:** `workplan/handoff/2026-05-09_course_correction.md`
> for the why. This task is the load-bearing diagnostic vertical
> slice that determines whether TEA is the v0 product renderer.

## Goal

Produce one intelligible phrase WAV by **gradient-descent
copy-synthesis** of a developer-recorded reference audio, where the
optimization variables are the parameters of a continuous neural
field `A_θ(s, t)` over tract location `s` and time `t`. No learned
controller, no trained body model — just `A_θ` parameters being
optimized to minimize an acoustic loss through TEA.

This tests whether TEA + differentiable optimization + a continuous
parameterization can recover recognizable speech. If yes: the rest
of the 4-week plan stands. If no: TEA isn't the v0 product renderer
and we ship on DDSP-articulatory or SPARC vocoders instead (see Risk
C in the course-correction doc for the exact salvage path).

## Inputs

- `data/probes/juice_lemons_ref.wav` — developer-recorded, 22.05 kHz
  mono, ~2-3 s, of: *"The juice of lemons makes fine punch."*
  (If file missing: ask developer to record before starting. This
  is a hard prerequisite.)
  > **[Correction per `docs/incorrect_assumptions.md` A10]** Despite
  > the "developer-recorded" label here, the actual Week-1 file used
  > is ElevenLabs "Natasha" synthesis, confirmed in the handoff doc
  > (`workplan/handoff/2026-05-13_week1_throughput.md §"Consent note"`).
  > The file is MP3-roundtripped (ElevenLabs export → WAV conversion)
  > and carries lossy-compression artifacts above ~16 kHz. It is not
  > pristine PCM and must not be used as a psychoacoustic reference.
  > Week-4 cloning will use a real developer recording.
- `tractus_controller/outputs/fixtures/default_tube.npz` — neutral
  tract for `A_default(s)`.
- `tractus_controller/outputs/fixtures/story_1996_anchors.npz` —
  reference area functions for `/a, i, u/` sanity gate.

## Deliverable

One file: `scripts/copyfit_neural_field_phrase.py`.

Public function:

```python
def copyfit_neural_field_phrase(
    audio_path: Path,
    transcript: str,
    out_dir: Path,
    *,
    n_iters: int = 250,
    frame_rate_hz: int = 50,
    sample_rate: int = 22050,
    seed: int = 0,
    sparc_npz: Path | None = None,   # unused in Week 1; reserved for Week 2
) -> dict:
    ...
```

CLI:

```bash
python3 scripts/copyfit_neural_field_phrase.py \
  --audio data/probes/juice_lemons_ref.wav \
  --transcript "The juice of lemons makes fine punch." \
  --out outputs/copyfit_neural_field/juice_lemons \
  --iters 250 --seeds 3
```

Outputs to `outputs/copyfit_neural_field/juice_lemons/`:

- `recon.wav` — reconstructed audio at 22.05 kHz.
- `field_snapshot.npz` — `A_θ` evaluated at the optimization grid
  (for visualization + Week 2 reuse).
- `report.json` — losses, gate results, optimizer trace, seeds.
- `optim_trace.png` — loss curve (matplotlib OK).
- `area_function_t0_t1_t2.png` — area function at three time
  snapshots (qualitative plausibility check).

## Implementation recipe

### 1. Parameterization

```python
# Sinusoidal positional encoding over (s, t).
def pe(x, n_freqs=8, max_freq=8.0):
    freqs = 2.0 ** jnp.arange(n_freqs) * jnp.pi / max_freq
    out = jnp.concatenate([jnp.sin(x * freqs), jnp.cos(x * freqs)])
    return out  # shape: 2 * n_freqs

# Field MLP: PE(s) ⊕ PE(t) → log-area delta.
# 4 layers, hidden 128, GELU activations.
# Output: scalar log-area delta in [-δ_max, δ_max] via tanh.
def log_area_delta(s_norm, t_norm, params):
    h = jnp.concatenate([pe(s_norm), pe(t_norm)])
    for W, b in params:
        h = jax.nn.gelu(h @ W + b)
    return jnp.tanh(h[0]) * delta_max  # delta_max ≈ 1.5 (log cm²)

def A(s_norm, t_norm, params):
    A_default_s = sample_default_tube(s_norm)  # smooth interp of default_tube.npz
    return jnp.exp(jnp.log(A_default_s) + log_area_delta(s_norm, t_norm, params))
```

`s_norm` ∈ [0, 1] over 93 tract sections. `t_norm` ∈ [0, 1] over the
phrase duration. Length is **not** a free variable in Week 1 — keep
fixed at default. Lip extension can be a global scalar if needed for
final-vowel matching, but defer until you see whether the field
alone gets in the right ballpark.

### 2. Source extraction (frozen)

Extract from target audio:
- F0 via PyWorld DIO (already a project dep). Smooth with a 5-frame
  median to suppress octave errors.
- Loudness as RMS in a 10 ms window.
- Voicing as PyWorld voiced/unvoiced flag; threshold for frication
  vs voicing.

Pass these directly into the TEA glottis state vector. **Do not let
them be optimized.** The point is to isolate "can the *tract* field
recover the phrase given the right source." If the source is also
free, every attribution becomes ambiguous.

### 3. TEA renderer call

Use existing `tractus_engine_advanced.pipeline.synthesize_dynamic_tube`
with:

- `length_cm[T, 93]` = default (frozen).
- `area_cm2[T, 93]` = sampled from `A(s_norm, t_norm, params)` on
  the (T, 93) grid.
- `glottis_traj[T, 11]` = source schedule from extraction above.
- `nasal_coupling[T]` = small constant for now (defer nasal modeling
  to Week 2).
- `wall_*` static = default-tube values.
- `frication_gain[T, 93]` = derived from constriction geometry per
  the existing `tractus_engine_advanced.frication` module
  (`correctness/004a` plumbing).
- `samples_per_frame = 441` for 50 Hz × 22.05 kHz output (note: TEA
  renders internally at 44.1 kHz so 882 samples/frame at the engine
  level, decimated to 441 at the 22.05 kHz boundary output — see
  `docs/incorrect_assumptions.md` A1).

### 4. Loss

```
L = w_logmel * L_logmel(audio_recon, audio_target)
  + w_stft   * L_mr_stft(audio_recon, audio_target)
  + w_smooth_s * mean((d log A / ds)^2)
  + w_smooth_t * mean((d log A / dt)^2)
  + w_curv     * mean((d² log A / ds²)^2)
  + w_clamp    * clamp_barrier(A, A_min=1e-4, A_max=12.0)
  + w_default  * mean((log A - log A_default)^2)
```

Suggested initial weights (tune in Week 1 itself):
- `w_logmel = 1.0`, `w_stft = 0.5`
- `w_smooth_s = 1e-3`, `w_smooth_t = 1e-3`
- `w_curv = 1e-4`
- `w_clamp = 10.0` (barrier should be steep)
- `w_default = 1e-2` (weak prior; field shouldn't drift far from
  neutral on average)

> **[Note per `docs/incorrect_assumptions.md` A9]** The smoothness/
> curvature weights above are initial-guess defaults that were never
> tuned during Week 1. Codex (commit `c639a73`) recommended 10-30×
> higher smoothness values, which are available via CLI overrides in
> the shipped script but are not the defaults. Do not treat these values
> as validated or tuned — they are a starting point only.

Use the existing `tractus_controller.copy_synth.tea_backend`
log-mel + MR-STFT JAX implementations (already wired through TEA's
custom_vjp).

### 5. Optimizer

Adam, learning rate `3e-3`, 250 iterations. Run **3 seeds**, report
best by composite loss. No L-BFGS — too easy to introduce dependency
issues, and Adam should be sufficient for a single-utterance fit at
this dim count.

Memory: with `frame_chunk_size=1` and `jax.checkpoint` already in
the pipeline, expect 4-12 GB VRAM. If OOM, drop `n_iters` first;
don't disable checkpointing.

## Acceptance gates (all required to pass)

Run from `tractus_controller/.venv/bin/python` (CUDA-enabled).

### Gate 1 — Whisper intelligibility

```bash
bash runbooks/whisper_gate.sh \
  --audio outputs/copyfit_neural_field/juice_lemons/recon.wav \
  --target "The juice of lemons makes fine punch." \
  --model base \
  --threshold-wer 0.75 \
  --threshold-jaccard 0.30
```

**Pass:** transcript non-empty AND WER ≤ 0.75 AND jaccard ≥ 0.30
AND ≥ 3 of `{juice, lemons, makes, fine, punch}` recognized.

### Gate 2 — Loss improvement

MR-STFT loss at iteration 250 ≤ 0.6 × MR-STFT loss at iteration 0,
on the best seed. Measured in `report.json["loss_trace"]`.

### Gate 3 — Field plausibility

In `report.json["plausibility"]`:
- `clamp_occupancy` < 0.05 (fraction of `(T × 93)` cells at the
  area clamp boundary).
- `max(|d log A / ds|)` < 3.0 (no pathological tract shapes).
- `max(|d log A / dt|)` < 5.0 cm²/cm/s scaled.

### Gate 4 — Vowel sanity

Render `/a/`, `/i/`, `/u/` separately by **freezing the field
parameters from the optimized run** and replacing the source
schedule with sustained voicing at F0=120 Hz, then rendering each
vowel's area-function snapshot from the field for the corresponding
mid-phone time (use forced alignment or hand-pick by listening).

For each vowel: F1, F2, F3 within ±20 % of Story 1996 anchor target
windows. Use `scripts/transfer_function_oral_only.py` methodology.

This is a *qualitative* gate — if F1 is at 800 Hz vs target 730 Hz
that's fine; if F1 is at 180 Hz, the field collapsed.

### Gate 5 — Human five-choice listening

Show the developer four other Harvard sentences alongside the target.
Play the reconstructed WAV once on headphones. Developer picks one
of five.

**Pass:** correct pick on first listen.

## Kill criterion (when to stop, not when to retry)

If after **Week 1 + 2 focused debug days** the gates still fail:

1. Stop tuning.
2. Run `scripts/inverse_fit_demo.py` (the existing
   discrete-bin baseline) on the *same* phrase, source, and gates,
   for cross-comparison. Note the result.
3. Render `/a/, /i/, /u/` from the **frozen R1 fixture corpus**
   tubes (sustained, voiced, F0=120). If those produce intelligible
   vowels, the field optimization or source extraction is the
   problem. If those *also* produce garbage, TEA itself has a
   correctness gap upstream of any body model.
4. Surface the result in a new `workplan/handoff/` doc and stop.
   The next move is in the course-correction document Risk C.

Do **not** spend more than 24 focused hours on Week 1 tuning. The
kill criterion exists to prevent sunk-cost spending; respect it.

## Anti-deliverables

Things this task should **not** produce:

- A trained body-model checkpoint. The body model arrives in Week 2.
- SPARC integration. Week 1 is unconditioned; SPARC is Week 2.
- Multi-phrase generalization. One phrase is enough.
- Speaker conditioning. Week 1 fits one speaker (the developer).
- Any modification to TEA pipeline internals. If it doesn't work,
  fix the field side or surface the TEA bug — don't patch through.
- New runbooks. `runbooks/whisper_gate.sh` is the gate; don't add
  parallel runbooks.
- Updates to `ARCHITECTURE.md` or any of the prepivot task files.
  This is implementation; documentation lives in the
  course-correction handoff.

## Reporting

When done, append to `workplan/LOG.md`:

```
## YYYY-MM-DD — Week 1 result

- Gate 1 (Whisper): pass | fail (WER=__, jaccard=__, words=__)
- Gate 2 (loss):    pass | fail (init=__, final=__)
- Gate 3 (field):   pass | fail (clamp_occ=__, max_grad_s=__, max_grad_t=__)
- Gate 4 (vowels):  pass | fail (/a/ F1F2F3=__, /i/=__, /u/=__)
- Gate 5 (human):   pass | fail
- Verdict:          GREEN (Week 2 unblocked) | RED (apply kill criterion)
- Notes:            __
```

If GREEN, dispatch task `002_sparc_extraction_pipeline.md`.

If RED, write up the salvage analysis as a new handoff doc and stop
all body-model work pending developer review.

## References

- Course-correction context: `workplan/handoff/2026-05-09_course_correction.md`.
- Chen & Sitzmann 2025 differentiable simulator with neural-field
  area: `docs/references/core/2025_chen_sitzmann_diff_vocal_simulator.pdf`.
- Existing discrete-bin baseline (failed): `scripts/inverse_fit_demo.py`.
- TEA pipeline entry: `tractus_engine_advanced/.../pipeline.py:synthesize_dynamic_tube`.
- Codex thread: `[session identifier omitted]` — continue with
  `codex-reply` if stuck on parameterization or loss recipe.
