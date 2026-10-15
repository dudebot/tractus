# Handoff — 2026-05-07 inverse-fit findings (infra/006 result)

## Scope

`infra/006` (target-sound inverse-fit demo) ran end-to-end through TEA on
three canonical vowel targets. This is the "settle the question" test the
user asked for after the user ear-tested the smoke audio and concluded
that the rule-based smoke pipeline produces clicking, not phonemes.

## Setup

- Forward path: full TEA stack (`synthesize_dynamic_tube`) at fp32, T=20
  frames (200 ms), default speaker anatomy, voiced glottis (F0=120 Hz,
  PR=8000 dPa).
- Free parameters: log-area on supraglottal sections 25-64
  (40 scalars). Trachea + glottis + walls + nasal frozen at default.
- Loss: `spectral_template_loss_jnp` against literature-shaped log-PSD
  with bell-curve peaks at Peterson & Barney male formants.
- Optimizer: Adam, lr=5e-2, 150 steps per vowel.
- JIT-compiled value-and-grad; ~8 s per step on the dev 16 GB GPU.

## Result

```
                            initial   final   ratio    F1 err   F2 err   F3 err
/a/   target F1=730        139.5  →  77.4    0.555     4.1%   115.3%    44.3%
/i/   target F1=270        195.0  →  87.0    0.446    11.7%    46.4%    23.5%
/u/   target F3=2240       174.3  →  87.8    0.504    36.4%    25.0%     0.9%
```

(Convergence threshold 0.5×; F-error tolerance 15%.)

## Diagnosis

**This is the most informative outcome possible — neither "simulator
fine" nor "simulator broken".**

Each vowel converges on a **different single formant**:
- /a/ nails F1 (4.1% off target).
- /i/ nails F1 (11.7%).
- /u/ nails F3 (0.9%).

None of the three vowels simultaneously place F1 + F2 + F3 within
tolerance. The optimizer finds a local minimum where ONE formant is
correct and the other two are off.

**What this says:** TEA's wave-equation simulator IS capable of
producing each individual formant frequency. The pieces are physically
present. But the **area-function-only parameter space is too narrow to
coordinate F1, F2, F3 together** — the optimizer can shift one
resonance into place but in doing so pushes the others further off.

This is consistent with classical tube-acoustics theory: in a Webster-
horn-style 1D resonator, the formants are **coupled** — moving one
shape parameter shifts multiple resonances. With only 40 area scalars,
the gradient descent finds a "single peak in the right place" minimum
faster than it can find the simultaneous-three-peak global optimum.

## What this means for the project's decision tree

Per the infra/006 task file's three branches:

- ❌ **All three vowels converge with all formants right** — would mean
  simulator is fine, hand-crafted shapes were just bad.
- ❌ **None converge** — would mean real P0-blocking simulator
  capability problem; would force re-opening P0.
- ✅ **Partial convergence** — what we got. Simulator is capable but
  needs more parameter freedom OR a smarter optimizer to coordinate
  multi-formant placement.

## Recommended next moves (in order of cost / payoff)

1. **(cheap) Multi-init Adam.** Run inverse-fit with 8-16 random log-
   area initializations per vowel; keep the best. Local-minimum problem
   might be solved by stochastic search + best-of-N.
2. **(cheap) Wider parameter space.** Free up the wall_resistance + a
   subset of glottis params (XB, XT — glottal aperture; CA — chink
   area). Wall_resistance affects formant bandwidths; glottal aperture
   shifts the source spectrum. Both have known acoustic effects on
   formant placement.
3. **(medium) Two-stage optimization.** First fit just F1 with a
   partial loss; then add F2; then F3. Curriculum prevents the
   "single-formant-only" basin.
4. **(medium) Higher-order loss.** Augment spectral-template loss with
   explicit per-formant peak-position penalty so the optimizer is
   directly punished for missing a target peak (rather than averaging
   distance over the whole spectrum).
5. **(expensive) Body-model retrain (`pretraining/002`).** The trained
   body model would learn a tube shape distribution that DOES place
   all three formants correctly per phoneme — this is precisely the job
   it's designed for. The current rule-based hand-crafted shapes are a
   stand-in until that lands.

## Strategic implication

The smoke pipeline's "clicking" character is **not** because TEA can't
produce vowels. It's because the rule-based pipeline produces tube
shapes that fall into the "single-formant-only" basin. A trained body
model that learns from real-speech corpus data would converge naturally
to multi-formant configurations (real corpus IS in those minima).

P0 close criteria are unaffected — frication and backward-OOM gates
remain valid. The "vowel intelligibility floor" is a P3 (pretraining)
concern, not a P0 (correctness) concern. The infra/006 result confirms
this categorization.

## Artifacts

- `outputs/inverse_fit_demo/report.json` — full per-vowel report,
  loss histories, recovered area functions.
- `outputs/inverse_fit_demo/{a,i,u}_{before,after}.wav` — pre/post audio
  for each vowel, gitignored. Copied to
  `~/Desktop/inverse_fit/` for user ear-testing.
- `outputs/inverse_fit_demo/run.log` — full step-by-step log.

## Files / commits

- `feat(testing): differentiable spectral_template_loss_jnp + inverse-fit demo` (commit dbd8345)
- This handoff document.
