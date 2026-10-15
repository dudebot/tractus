# Task 003 — Speaker-conditioned body model (Week 2)

> **Blocked by:** Task 002 GREEN. **Read first:**
> `workplan/handoff/2026-05-09_course_correction.md`.

## Goal

Replace the per-utterance neural-field copy-fit (Task 001) with a
**trained body model** that predicts the field from kinematic
features + a small speaker vector. The body model must:

1. Reproduce Task 001's phrase when conditioned on the SPARC labels
   for that phrase.
2. Generalize to ≥ 7 of 10 held-out short utterances passing the
   Whisper non-empty gate.
3. Render `/a, i, u/` jointly within ±20 % of Story formant targets.

This is the move from "copy-synthesis tool" to "body model that
speaks new utterances."

## Architecture

```python
# tractus_controller/tractus_controller/body_model/neural_field.py

@dataclass(frozen=True)
class NeuralFieldConfig:
    n_freqs_s: int = 8
    n_freqs_t: int = 8         # only used if explicit-t variant
    hidden_dim: int = 128
    n_layers: int = 4
    delta_max: float = 1.5     # max log-area excursion
    speaker_dim: int = 32      # tight per-speaker bottleneck
    kinematic_dim: int = 14    # 12 EMA + F0 + loudness from SPARC

# Field: A_θ(s | z_t, speaker)
#
# At inference, t is implicit — for each frame t we have a kinematic
# vector z_t and we sample the field over s.
#
# log A(s, t) = log A_default(s) + base_θ(s, speaker)
#                                + delta_θ(s, z_t, speaker)
#
# base_θ:   speaker-conditioned neutral tract bias (slow, anatomy)
# delta_θ:  kinematic + speaker-conditioned dynamic field (fast)
```

Param budget: **0.3 - 0.8 M total.** Codex's recommended split:

- gesture/SPARC encoder (z_t producer): 20-80 K
- field MLP (delta_θ): 100-300 K
- base + speaker conditioning: 50-200 K

If you blow past 1 M params, stop and reconsider — the architecture
is overspecified.

## Training pipeline

### Data

Combine:
- Task 001 reference utterance + Week-1 optimized field as a
  **bootstrap target**.
- 50-100 SPARC-labeled utterances from Task 002.

For each utterance, supervision is:
- **Acoustic loss:** log-mel + MR-STFT through TEA (short windows
  only — `frame_chunk_size=1` with `jax.checkpoint`).
- **Kinematic loss:** the body model exposes an intermediate
  `z_t = encoder(gestures or sparc_features)` — train this to match
  SPARC's EMA + F0 + loudness directly.
- **Smoothness priors:** spatial and temporal `d log A` regularizers
  from Task 001.
- **Reachability barriers:** `reachability.py` clamps as soft
  penalties.
- **Story sanity** as a separate held-out validation gate, NOT in
  the training loss.

### Optimizer

Adam, lr 1e-3, 50 K-100 K steps. Batch size 1-2 utterances (memory-
limited by TEA backward). Use `bfloat16` for the body-model neural
nets if it makes throughput tolerable; **never** for the TEA solver
state.

### Speaker fitting

For Week 2's developer voice: fit `speaker_vec` (32-d) on 5-10
developer-recorded utterances with the rest of the body model
**frozen**. ~1 K optimization steps. Output: a 128-byte `.npz` per
speaker.

This is the cloning surface. Confirm it stays small.

## Acceptance gates

### Gate 1 — Reproduce Task 001

When given the SPARC labels for the Week-1 phrase + the developer's
fitted `speaker_vec`, the body model produces an `A(s, t)` whose
TEA-rendered audio passes the same Whisper gate as Task 001.

WER ≤ 0.75, jaccard ≥ 0.30, ≥ 3 content words. Loss can be slightly
worse than the per-utterance fit; that's fine — generalization
costs.

### Gate 2 — Held-out generalization

10 developer-recorded short utterances (3-5 words each, varied
phonetic content). Run through SPARC → body model → TEA → Whisper.

**Pass:** ≥ 7 of 10 produce non-empty Whisper transcripts. WER not
required to be tight; we're checking "does it speak at all."

### Gate 3 — Vowel sanity (joint formants)

Render `/a, i, u/` sustained for 0.4 s each via:
- A held-tongue gesture trajectory at the Story-target articulation.
- Frozen body model + developer speaker_vec.
- Voiced source at F0=120 Hz.

Measure F1/F2/F3 via `scripts/transfer_function_oral_only.py`.

**Pass:** all three formants within ±20 % of Story target windows
for all three vowels. This is the gate `infra/006` failed under
discrete-bin parameterization; passing here is direct evidence the
neural-field pivot worked.

### Gate 4 — Field plausibility

Same as Task 001 Gate 3: clamp_occupancy < 5 %, gradient bounds
respected, no pathological tract shapes on any of the 10 held-out
utterances.

### Gate 5 — Speaker object size

`outputs/speakers/<dev>.npz` ≤ 2 KB. (Anatomy + speaker_vec +
source defaults; no full-tract snapshots.)

## Kill criterion

If the trained body model cannot reproduce Task 001 OR generalize to
5+ held-out utterances after one full training round + 1 debug day,
diagnose between:

- (a) **SPARC labels too noisy.** Check Task 002 Gate 2 outputs;
  inspect EMA channels visually.
- (b) **Field capacity too low.** Bump hidden_dim 128 → 192 or
  n_layers 4 → 5. Stop if it goes past 1 M params.
- (c) **Acoustic loss too dominant.** Increase kinematic-loss weight
  by 10×; if that helps, the body was bypassing the kinematic
  bottleneck.

Spend ≤ 2 days on these probes. If still failing, surface to the
developer; the next move is the Week-3 controller work or a
broader rethink.

## Anti-deliverables

- No re-fitting per utterance at inference. The body model **predicts**;
  it does not optimize.
- No reintroducing the conv-MLP `model.py`. That's archived.
- No Story-MSE in the training loss. Story is a sanity gate.
- No SPARC weights at inference. SPARC labels were extracted in
  Task 002 and frozen.
- No multi-speaker corpus run yet. Week 2 is one speaker
  (developer). Multi-speaker is Week 3+.

## Reporting

`workplan/LOG.md`:

```
## YYYY-MM-DD — Week 2 body model

- Param count:      __ M
- Gate 1 (reproduce):     pass | fail
- Gate 2 (held-out 10):   __ / 10 non-empty
- Gate 3 (vowel formants):pass | fail
- Gate 4 (plausibility):  pass | fail
- Gate 5 (speaker size):  __ KB
- Verdict:                GREEN | RED
```

If GREEN: Task 004 unblocked.
If RED: surface diagnosis from kill criterion to developer; do not
auto-promote.

## References

- Course correction: `workplan/handoff/2026-05-09_course_correction.md`
- Codex thread: `[session identifier omitted]`
- Existing reachability code: `…/body_model/reachability.py`
- Task 001 deliverables (reference): `outputs/copyfit_neural_field/juice_lemons/`
