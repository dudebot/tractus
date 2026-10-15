# Task 007 — Frication-enabled copyfit retest

> **Blocker:** none. **Read first:** `docs/week1/findings.md`
> §"Open issues" item-1 (frication wiring), and the loss-curve note
> below — the Week-1 baseline did NOT plateau, which changes how we
> interpret the comparison.

## Why this exists

Two things showed up when reviewing Week-1 closure:

1. **Frication was disabled in copyfit.** `synthesize_dynamic_tube`
   is called without a `frication=` kwarg, so the default
   `Frication.disabled()` fires and the optimizer cannot produce
   turbulence noise. The reference phrase ("The juice of lemons makes
   fine punch") has ~9 fricatives + affricates; without frication
   the optimizer must substitute voiced CV gestures, and Whisper
   transcribed the result as a single "I". H7 in Codex's review
   flagged this as a likely contributor to the gibberish.

2. **The 100-iter run did not plateau.** From `juice_lemons_run100/
   report.json`, the acoustic loss trace's last 4 logged points are
   `3.404 → 3.374 → 3.366 → 3.357` — still descending ~0.01 per
   logged step (which is every ~5 iters), no flatlining. Total
   improvement init→final is ~17% (4.058 → 3.357). At the observed
   rate, another 100 iters would plausibly bring acoustic loss to
   ~3.0-3.1. The "phonetic wall" finding in Week-1 closure was
   declared on a non-converged optimizer.

Both are confounders for the Week-1 conclusion. Before SPARC arrives
in Week 2, we want the cleanest possible negative-result statement:
"copyfit with frication enabled and the optimizer actually converged
still cannot recover phonemes." Or — and this is the interesting
case — copyfit with frication produces meaningfully better fits and
the SPARC roadmap needs a softer framing.

## Goal

Rerun the `juice_lemons` copyfit with frication wired in **and** with
the iteration budget extended past where the baseline stopped. Compare
to `outputs/copyfit_neural_field/juice_lemons_run100/` head-to-head.

## Inputs

- `data/probes/juice_lemons_ref.wav` — same reference as baseline
- `scripts/copyfit_neural_field_phrase.py` — script to extend
- `tractus_engine_advanced.frication.Frication` — physically-derived
  frication module; defaults are calibrated for sibilant lift in
  4-8 kHz band

## Implementation

### Frication wiring

`synthesize_dynamic_tube` accepts:
- `frication: Frication | None`
- `place_index[T, 93]` — int32, PLACE_* code per section per frame
- `obstacle_proxy[T, 93]` — float in [0,1], dental/teeth presence
- `voicing[T]` — float in [0,1], voicing scalar per frame
- `frication_color[T, 93]` — float, multiplicative color residual
- `noise_key` — JAX PRNG key for deterministic noise

This is unsupervised inverse-fit; we do NOT have phoneme labels for
the reference. So we use a physics-bound default: the frication
module's `physical_gain` already gates on `Re² · area` (turbulence
threshold) and pressure drop. Tag every supraglottal section as a
generic sibilant site and let the physics decide where noise fires.

Concrete first probe:
- `frication = Frication.default()` (enabled, all defaults)
- `place_index = PLACE_ALVEOLAR` (code 3, gain 2.2 — sibilant) on
  the supraglottal range `FREE_LO:FREE_HI`, `PLACE_NONE` (0)
  elsewhere. Constant across time.
- `obstacle_proxy = zeros` (no learned obstacle modeling here)
- `voicing[t]` = derived from CREPE periodicity of the reference
  audio at 50 Hz, OR if that's a bigger lift, set `voicing = 0`
  uniformly (full unvoiced gain — physics gate filters)
- `frication_color = zeros`
- `noise_key = jax.random.key(0)`

If the simple uniform-PLACE_ALVEOLAR setup produces no observable
change, swap to a per-frame schedule (PLACE_ALVEOLAR on alveolar
sections, PLACE_GLOTTAL on glottis sections, etc.) — but try the
simple case first.

### Iteration budget

Run 200 iters minimum (vs baseline's 100). Same seed (0) to make the
comparison clean. Same recipe everywhere else — DO NOT tune
smoothness/curvature weights, init, or LR. The only experimental
variable is frication on/off + extended iters.

Output to `outputs/copyfit_neural_field/juice_lemons_frication200/`.

### Side-comparison run

Also run a 200-iter no-frication baseline to isolate the frication
effect from the iteration-budget effect. Output to
`outputs/copyfit_neural_field/juice_lemons_nofric200/`.

So three runs to compare:
- `juice_lemons_run100/` — existing baseline (no frication, 100 iters)
- `juice_lemons_nofric200/` — no frication, 200 iters (controls for
  the iter-budget confounder)
- `juice_lemons_frication200/` — frication on, 200 iters (the actual
  experiment)

If GPU time is constrained, the priority order is: frication200 first,
then nofric200, then keep the baseline as-is.

## Acceptance gates

### Gate 1 — Loss-trajectory comparison

Plot all three loss curves on the same axes. Specifically check:
- Does the no-fric 200-iter run continue to descend past iter 100,
  or does it plateau?
- Does frication200 reach a lower acoustic loss than nofric200 at
  matched iter count?
- If frication200's acoustic loss is HIGHER than baseline at iter 100,
  flag — frication may be acting as noise injection that the optimizer
  has to fight.

Save plot to `outputs/copyfit_neural_field/comparison/loss_curves.png`.

### Gate 2 — Whisper transcription comparison

Run `runbooks/whisper_gate.sh` (or the Python impl directly) on each
of the three `recon.wav` files. Record transcription verbatim. The
reference is "The juice of lemons makes fine punch."

Pass criteria: transcription comparison is informative regardless of
result. We're not gating on WER ≤ X; we're gating on having the
data. **A negative result here is still a Gate-1 pass** as long as
the comparison is documented.

### Gate 3 — Fricative-band energy

For each recon, compute 4-8 kHz band energy and 8-11 kHz band energy
using the same machinery as `runbooks/source_tract_sep.py` /
`scripts/m7_frication_band.py`. Report:
- band energy of reference vs each recon (dB)
- delta of frication200 vs nofric200 in 4-8 kHz (positive = frication
  is doing something)

The frication module's own gate (`m7_frication_band.py` test) requires
≥ 1 dB lift, mean post/pre ratio ∈ [1.5x, 30x]. Use this as the
sanity check that the wiring is functional.

### Gate 4 — Vowel formant comparison

Same `vowels` block as in baseline `report.json` — measure F1/F2/F3
at the /u/, /i/, /a/ frame indices. Report side-by-side with
baseline. Frication shouldn't meaningfully change voiced vowel
formants; if it does, something's wrong with the wiring.

## Reporting

Append to LOG.md:

```
## YYYY-MM-DD — Frication copyfit retest

- juice_lemons_run100 (baseline):    acoustic=__, Whisper="__"
- juice_lemons_nofric200 (control):  acoustic=__, Whisper="__"
- juice_lemons_frication200 (test):  acoustic=__, Whisper="__"
- 4-8 kHz delta (frication vs control): __ dB
- Plateau test (control 100→200): converged | still descending
- Verdict: GREEN (frication helped meaningfully)
        | NEUTRAL (no significant change)
        | RED (frication made it worse — wiring bug or physics misuse)
```

Also write a 1-page comparison note to
`docs/week1/frication_retest.md` with the loss curves, transcriptions,
and one paragraph of interpretation. This is a sister doc to
`docs/week1/findings.md`, not a replacement.

## Anti-deliverables

- **Don't tune smoothness/curvature/LR/init** to make frication look
  better. The point is a clean comparison; introducing other variables
  defeats the experiment.
- **Don't try to learn `place_index`** — that's a body-model concern,
  not an inverse-fit one. If physics-gated frication at uniform
  PLACE_ALVEOLAR produces no change, the result is informative as-is.
- **Don't add new loss terms** for fricative-band energy. The acoustic
  log-mel + MR-STFT loss already weights high-frequency content; let
  it do its job.
- **Don't rewrite the copyfit script** — wire frication in with the
  smallest possible diff.

## References

- `docs/week1/findings.md` §"Open issues" item-1
- `outputs/copyfit_neural_field/juice_lemons_run100/report.json` —
  baseline trace
- `tractus_engine_advanced/src/tractus_engine_advanced/frication.py` —
  module + PLACE_* constants + defaults
- `tractus_engine_advanced/src/tractus_engine_advanced/pipeline.py:200` —
  `synthesize_dynamic_tube` signature
- `scripts/m7_frication_band.py` — band-energy gate
- Codex thread `[session identifier omitted]` — H7 frication finding
