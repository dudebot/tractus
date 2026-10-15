# infra/006 — Target-sound inverse-fit demo + acceptance gate

**Phase:** infra    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** —
**Blocked by:** —
**How to dispatch:** fresh Claude Code session

---

## Context

The user's framing during the testing-apparatus overhaul (commit
`4763881`):

> "The shape of these sounds shouldn't be alien to time series
> analysis and Fourier analysis. This should be something that in my
> opinion should be something that should be framed as a very simple
> gradient descent for target sounds."

The acoustic-gates module
(`tractus_engine_advanced.acoustic_gates`) ships
`spectral_template_loss_np` — the numpy version of the L2 distance
between a rendered audio's log-PSD envelope and a target template.
This task lands the **JAX-differentiable counterpart**
(`spectral_template_loss_jnp`) AND a working inverse-fit demo that
gradient-descends through TEA on tube + glottis parameters until
the rendered spectrum matches a target.

The test this catches: when manual phoneme-test renders show
TEA-produced vowel formants **placed lower than literature targets**
(/a/ envelope peaks at ~400 Hz instead of 730 Hz; /i/ and /u/ both
peak at 200 Hz with no distinct F2), it's unclear whether the
problem is (a) the simulator genuinely can't produce those formant
shapes, or (b) the hand-crafted tube shapes are wrong. **The
inverse-fit answer to (a) is "yes, TEA can produce them" exactly
when gradient descent converges to a low loss; the answer is "no"
when it stalls at high loss.** This converts an open-ended
diagnostic question into a closed automated test.

## Deliverable

1. **`spectral_template_loss_jnp`** in `acoustic_gates.py`. Same
   structure as the numpy version: pre-emphasis (optional) →
   Hann-window → FFT → log → cepstral lifter (optional) → restrict
   to `[f_min, f_max]` band → L2 against target. Pure jnp, no
   numpy ops. Works through `jax.grad` end-to-end.
2. **`scripts/inverse_fit_demo.py`**:
   - Target: a literature-shaped /a/ spectrum with bell-curve
     peaks at F1=730, F2=1090, F3=2440 (Peterson & Barney male).
   - Variables: log-area-cm2 per supraglottal section (sections
     25-64, the pharynx + mouth — 40 free parameters total).
     Trachea + glottis + walls held at default-speaker baseline. Glottis
     trajectory held at default voiced (F0=120, PR=8000).
   - Loss: spectral_template_loss_jnp against the target.
   - Optimizer: optax adam, ~100-300 steps, lr~1e-2.
   - Render T=20 frames (~ 200ms) so backward fits comfortably in
     the post-correctness/008 ≤2 GB budget at b=1.
   - Output: pre/post WAVs to `outputs/inverse_fit_demo/`,
     before/after FFT plot (matplotlib), JSON report with final
     loss + recovered area function.
3. **Acceptance gate** (automated):
   - Final loss ≤ 0.5× initial loss. (Genuine convergence.)
   - Final rendered F1, F2 within 20% of target. (Real formant
     match, not just spectrum-shape match.)
   - Recovered area function is physically plausible (areas > 0,
     no extreme spikes >10× the default-speaker baseline).
4. **Per-vowel coverage**: repeat for /i/ and /u/ targets. Three
   total demos, one per vowel.

## Acceptance gates

- ✅ All three vowel demos converge: final loss ≤ 0.5× initial.
- ✅ Recovered F1/F2 within 20% of literature target on at least
  2/3 vowels.
- ✅ JSON report at `outputs/inverse_fit_demo/report.json`.
- ✅ Before/after WAV pairs on disk for ear-test (acceptance is
  automated; ear-test is qualitative confirmation).

## Anti-deliverable / scope guard

- Do NOT optimize over wall_mass / wall_resistance / wall_stiffness
  — those are anatomy-static per ARCH §510-515. Only area function
  is variable.
- Do NOT increase render duration past T=20 unless backward
  memory has been re-verified at the chosen T. correctness/008's
  ≤2 GB gate is for T=118 b=4 production; this demo is short-window
  inversion at T=20 which is well within budget.
- Do NOT use this loss as a training objective in the body-model
  pretraining path. ARCH §923-927 mandates short-window TEA
  anchors only; full-utterance gradient descent through TEA is the
  G4 escape hatch the architecture explicitly forbids.
- Do NOT skip the per-vowel /i/ and /u/ runs — single-vowel
  convergence on /a/ alone is consistent with TEA having only ONE
  formant pattern it can produce. The /i/ + /u/ test is what
  proves the simulator has a meaningful tube→spectrum
  differentiation surface.

## Outcomes by branch

- **All three vowels converge → simulator is fine, hand-crafted
  shapes were bad.** Land the recovered area functions as
  canonical phoneme tube fixtures (`outputs/fixtures/canonical_<phone>.npz`)
  and use them in the smoke pipeline going forward.
- **One or two converge → partial capability.** File specific
  follow-ups for the failing vowels; investigate wall losses,
  radiation impedance, or tube-section topology for that case.
- **None converge → simulator has a structural problem.** Real
  P0 issue. Likely candidates: wall losses too high (over-
  damping), radiation impedance wrong (high-freq tilt), or
  body model + frication smearing the formant peaks. File
  `correctness/009_simulator_formant_capability.md` and treat
  P0 as not-actually-closed.

## Notes

This task is the practical version of the user's "gradient descent
for target sounds" framing. The acoustic-gates module already has
the perceptual-template scaffolding; this task is what makes those
gates *actionable* — when one fails, the worker can re-run the
inverse-fit to see whether the simulator can produce the target at
all, or whether the upstream layer needs the fix.
