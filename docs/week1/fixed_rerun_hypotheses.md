# Fixed copyfit rerun — explicit hypotheses

> Pre-registration of what we expect from each experimental run, so
> when results land we can quickly judge "this went better than
> expected, ship the next thing" vs "this is shit, step back."
>
> Written before any results from `juice_lemons_fixed_nofric200` are
> in. Updated post-hoc with verdicts when results land.
>
> Sister docs: `docs/incorrect_assumptions.md` (the bug list),
> `docs/assumptions_audit.md` (the audit synthesis).

## What's actually being varied

| Run | A1 fix | Frication | Iters | Expected wall time |
|---|---|---|---|---|
| `juice_lemons_run100` (existing baseline) | ✗ | off | 100 | 73 min (already done, broken) |
| `juice_lemons_fixed_nofric200` | ✓ | off | 200 | ~5 h |
| `juice_lemons_fixed_fric200` | ✓ | on | 200 | ~5 h |

Single seed (0) for all three. Same MLP architecture, same loss
weights (initial defaults — A9 unverified), same LR (3e-3).

## Run 2 (`fixed_nofric200`) — what each outcome means

The point is to isolate the A1 fix's effect alone. Frication is
still off, so consonant-poor recovery is expected on principle.

### Outcome 2A — "fix unblocked optimization"
**Signal:** acoustic loss drops below baseline's final 3.357 within
the first 50-100 iters (rough proxy: 2.5 or below). Loss curve
shows clear plateau by iter 200. Whisper transcribes meaningful
content (e.g., "the use of lemons", "i hear lemons", at least 2
recognizable content words). Vowel formants now within 5% of target
(not 12-19%).

**Interpretation:** A1 was the dominant problem. The phonetic-wall
verdict was an artifact of the broken contract. Acoustic supervision
*can* drive articulator recovery on real audio with the bug fixed.

**Action:** sharply downgrades urgency of SPARC. Dispatch Task 009
(TEA-to-TEA discriminating experiment) to confirm in-model. If 009
also passes, **the entire Week-2 SPARC pipeline becomes optional
auxiliary, not critical-path.** Re-evaluate the 4-week plan.

### Outcome 2B — "fix helped but not enough"
**Signal:** acoustic loss reaches ~2.5-3.0 (better than baseline's
3.357 but not dramatically). Loss continues descending at iter 200
without plateau. Whisper transcribes 1-2 partial words, mostly
vowel-leaning. Vowel formants improved (5-10% error) but not great.

**Interpretation:** A1 was a major confounder but not the only one.
Frication, longer optimization, or actual non-uniqueness all still
in play.

**Action:** run frication run (3) immediately. The next bit of signal
should come from comparing 2 vs 3.

### Outcome 2C — "fix changed nothing"
**Signal:** acoustic loss trajectory looks similar to broken baseline
(final ~3.3-3.5). Whisper transcription still nonsense. Vowel formant
errors still 12-19%.

**Interpretation:** Either we missed another bug as load-bearing as
A1 (most likely), or the broken pipeline was coincidentally
producing similar quality of output (less likely — the math says
octave shift should have severely degraded everything).

**Action:** STOP. Audit what's left. Things to check:
- Is the resampling actually doing what we think? Plot
  `target_for_feats` spectrogram vs original Natasha
- Is the glottis schedule still being interpreted correctly at the
  new frame timing?
- Is the FeatureConfig change to `(1024, 2048, 4096)` actually
  taking effect?
- Did we somehow break the gradient signal entirely?

### Outcome 2D — "fix made things worse"
**Signal:** acoustic loss higher than baseline at matched iters.
Whisper output noisier than baseline. Vowel formant errors > 20%.

**Interpretation:** Most likely a wiring bug introduced by the fix.
Possible culprits:
- DIO/glottis schedule misaligned with new frame timing
- Resampling artifact getting absorbed into loss
- MR-STFT FFT-size change (1024, 2048, 4096) shifted what the loss
  attends to in a way that derails optimization

**Action:** Revert the FFT-size override (try the run with default
(512, 1024, 2048) at 44.1 kHz) — that's the most-suspect change.
If still bad, deeper inspection.

## Run 3 (`fixed_fric200`) — what each outcome means

Compares to Run 2. Frication is the only variable.

### Outcome 3A — "frication clearly helps"
**Signal:** acoustic loss reaches ~10-20% lower than Run 2 at
matched iters. Whisper transcription noticeably better, especially
on consonants. Fricative-band energy (4-8 kHz) ≥ 3 dB higher than
Run 2 over the fricative-bearing windows ("juice", "lemons", "fine",
"punch"). m7_frication_band gate (1 dB lift, [1.5x, 30x] ratio)
passes.

**Interpretation:** Frication is doing useful work. The
uniform-PLACE_ALVEOLAR + physics-gate strategy is enough to recover
fricatives without place-index supervision.

**Action:** Make `--frication` the default. Update task 001 spec.
SPARC's role narrows further.

### Outcome 3B — "frication adds bandwidth, not phonemes"
**Signal:** fricative-band energy IS higher than Run 2 (m7 gate
passes), but Whisper transcription doesn't improve (or improves
trivially). Loss similar to Run 2.

**Interpretation:** Optimizer can produce noise, but can't route it
to phoneme-meaningful events without place_index supervision.
This is the "frication needs phone labels to be useful" case.

**Action:** SPARC place_index extraction becomes load-bearing for
fricative recovery. Update Task 002 to extract per-frame place
codes from SPARC's output (if SPARC produces them) or derive via a
forced-alignment step.

### Outcome 3C — "frication destabilizes optimization"
**Signal:** loss higher than Run 2. Whisper output noisier (literal
noise). Optimizer trace shows oscillation or divergence.

**Interpretation:** Uniform PLACE_ALVEOLAR everywhere is too
aggressive — frication noise injected at every supraglottal section
adds unwanted energy that the optimizer fights. The physics gate
isn't enough to localize.

**Action:** Try a more restrictive place_index — e.g., only the most
constricted N sections per frame get PLACE_ALVEOLAR; rest stay
PLACE_NONE. Or schedule frication on/off per frame from a coarse
voicing-vs-unvoiced split derived from the reference audio.

### Outcome 3D — "no measurable difference"
**Signal:** loss, Whisper output, and fricative-band energy all
within noise of Run 2.

**Interpretation:** The physics gate (Re² · area + Δp threshold)
is filtering frication noise to ~zero because the optimizer hasn't
created any sufficiently constricted geometries. Or `noise_amp_scale=8.0`
is just too low to be perceptible at our scale.

**Action:** Sanity-check by looking at the trained area function:
are there any sections with area < 0.1 cm² for sustained windows?
If not, the optimizer simply hasn't created fricatives. If yes, the
gain is too low.

## Cross-run hypotheses (interactions between 2 and 3)

If 2A AND 3A: best case. Acoustic supervision works, frication
helps, SPARC-as-required is firmly disproved.

If 2A AND 3D: surprising. Means the fix alone is enough to recover
phonemes including fricatives somehow (maybe via the body model
finding alternative articulations that produce fricative-like
spectra without using the frication module).

If 2B AND 3A: frication is doing the heavy lifting. The acoustic
loss + frication recipe is the recovery path. Strong argument for
making frication default.

If 2C AND 3C: deep problem. Either we have multiple bugs or the
inverse problem really is hard. Time for the discriminating
experiment (Task 009) to be the arbiter.

## Things this set of runs CANNOT tell us

These runs do NOT establish:

1. **"Kinematic supervision is structurally required."** That
   requires the in-model TEA-to-TEA experiment (Task 009). Real-audio
   results have too many out-of-model confounders (ElevenLabs vocoder
   character, MP3 roundtrip, microphone, room) that TEA can't
   reproduce regardless.
2. **Whether SPARC labels would be useful as auxiliary supervision.**
   Even if 2A + 3A, SPARC may still help (e.g., constraining to
   speaker-realistic articulator trajectories). That's a separate
   research question.
3. **Whether the body model trained on a corpus would generalize.**
   Single-utterance copyfit doesn't tell us about cross-utterance
   transfer.

## Verdict template (fill in post-hoc)

```
## YYYY-MM-DD — Fixed copyfit rerun results

### Run 2 (fixed_nofric200)
- Final acoustic loss:    ___
- Loss plateau by iter:   ___ (or "did not plateau")
- Whisper:                "___"
- Vowel /u/ formant err:  __% / __% / __%
- Vowel /i/ formant err:  __% / __% / __%
- Vowel /a/ formant err:  __% / __% / __%
- Outcome:                2A | 2B | 2C | 2D

### Run 3 (fixed_fric200)
- Final acoustic loss:    ___
- Loss plateau by iter:   ___
- Whisper:                "___"
- 4-8 kHz energy delta vs Run 2: __ dB
- m7 gate:                pass | fail
- Outcome:                3A | 3B | 3C | 3D

### Synthesis
- Cross-outcome:          (e.g., 2A+3A — best case)
- Verdict on A4 ("kinematic supervision required"):  REINFORCED | WEAKENED | UNCHANGED
- Next experiment:        Task 009 | re-tune | step back
```
