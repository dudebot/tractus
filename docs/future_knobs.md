# Future configuration knobs

> Architectural decisions we've identified as load-bearing for v0
> quality but deferred from current work. Each entry has a sentence
> on the motivation, current behavior, what the knob would change,
> and rough cost.
>
> **Bar for adding here:** something that would otherwise become a
> code refactor instead of a config flag. The user explicitly asked
> for these to be configuration items going forward, not refactors.
>
> When a knob graduates to implemented, mark `**IMPLEMENTED in <commit>**`
> at the top of the entry. Don't delete entries — they're the durable
> record of "here's what we knew was a knob and chose to defer."

---

## K1 — Frame-rate / temporal interpolation

**Motivation:** Current pipeline operates at 50 Hz frame rate (20 ms
per frame). The body model emits one area-function vector per frame;
TEA holds it constant for `samples_per_frame` audio samples then
steps to the next frame's value. Step changes in area function
produce frame-rate harmonics in the output spectrum (50 Hz +
harmonics → audible buzz).

The TEA glottis ALREADY supports `interpolate_glottis_within_frame=True`
(default), which linearly interpolates per-sample glottis params. The
area function does not have an analogous knob.

Phonologically relevant time scales:

| Event | Time scale |
|---|---|
| Articulator movement | 50-200 ms (50 Hz adequate) |
| Vowel formant transitions | 30-100 ms (50 Hz adequate) |
| Voice onset time | 5-100 ms (100 Hz preferred) |
| Frication onset | 5-20 ms (100 Hz preferred) |
| Stop bursts | 1-5 ms (200 Hz needed) |

**Current behavior:** 50 Hz step area, 50 Hz interpolated glottis.
Stop bursts are smeared. Frame-rate harmonics in output.

**Proposed knob:** `interpolate_area_within_frame: bool = False`
kwarg in `synthesize_dynamic_tube`. Linear is the default
implementation; cubic Hermite or B-spline as future refinements.
Same machinery as the existing glottis interpolation.

**Cost:** Modest — TEA already does per-sample glottis math, adding
per-sample area interpolation is similar complexity. May invalidate
existing fixture comparisons (correctness/001 / 002) since the
forward render bit-equality changes when the flag flips.

**When to implement:** After v0 base demo works. The audible buzz
from step area is a quality issue, not a correctness issue —
intelligibility tests will pass without this. Reach for it when
"the demo sounds robotic" becomes the dominant complaint.

---

## K2 — Body model output frame rate (independent of TEA frame rate)

**Motivation:** The body model currently outputs one area-function
per 20ms window (50 Hz). SPARC also extracts kinematic features at
50 Hz (locked to WavLM transformer stride). For high-quality stop
bursts we'd want 100-200 Hz body-model output.

**Current behavior:** 50 Hz throughout. Body model and TEA frame
rates are coupled.

**Proposed knob:** Decouple. Body model emits at a configurable
rate (default 50 Hz, optionally 100 Hz / 200 Hz). Interpolate up to
TEA's per-sample rate via K1's mechanism.

**Cost:** Larger. Body-model architecture change (more parameters,
2× or 4× output channels), retraining, possibly affects SPARC
supervision alignment.

**When to implement:** Post-v0. Joint with K1.

---

## K3 — Sample rate per pipeline component

**Motivation:** Currently the project commits to 22.05 kHz user-facing
output, TEA renders at hard-coded 44.1 kHz, SPARC consumes at hard-
coded 16 kHz. Each constraint is its own story; no single project-
wide rate. We've documented this; future-us should NOT try to
"unify" by forcing one rate everywhere — the constraints are real.

**Current behavior:** TEA 44.1 kHz internal, decimate at boundary
to 22.05 kHz user output (post-A1 fix); SPARC ingestion at 16 kHz
via internal resample.

**Proposed knob:** None at the project level. Per-component config:
- TEA: `--target-output-rate` post-decimation (default 22050)
- Final TTS render: `--output-rate` (default 22050; could go 24 kHz
  or 16 kHz if user demands)

**Cost:** Trivial.

**When to implement:** Whenever the v0 demo grows a real CLI.

---

## K4 — Loss-side feature configuration (loss-vs-output rate)

**Motivation:** The loss currently computes mel + MR-STFT at the
same rate as the rendered audio. This couples loss bandwidth to
output bandwidth. There's no fundamental reason they need to match
— for Week-1's experiment we made them match because the broken
contract obscured the question. Going forward, supervising at 44.1
kHz while outputting at 22.05 kHz might be useful (loss sees full
fricative spectrum, output is bandlimited).

**Current behavior:** Loss at 44.1 kHz (post-A1 fix), output at
44.1 kHz (still pre-decimation in copyfit).

**Proposed knob:** `loss_sample_rate` separate from `output_sample_rate`
in copyfit and any future training script.

**Cost:** Small.

**When to implement:** Only if we discover the loss-rate matters
empirically (e.g., 16 kHz loss on 22.05 kHz output).

---

## K5 — Frication place_index supervision strategy

**Motivation:** Currently copyfit's frication wiring tags every
supraglottal section as `PLACE_ALVEOLAR` and lets physics gates
filter. This is a coarse proxy for not knowing the actual
phonetic place at each frame. With SPARC labels, we could derive
more accurate place_index from EMA articulator positions.

**Current behavior:** uniform PLACE_ALVEOLAR across supraglottal
sections, all frames.

**Proposed knob:** `place_index_strategy` enum:
- `"uniform_alveolar"` (current)
- `"physics_only"` (set place_index=PLACE_NONE everywhere — frication
  is gated entirely by physics, no place-bias)
- `"sparc_derived"` (use SPARC EMA + a simple constriction-detector
  to assign per-frame place codes)
- `"forced_alignment"` (use phoneme-level alignment from a CTC
  recognizer to assign codes per-phone)

**Cost:** Small wrapper code; the alternatives mostly already exist
or are easy to derive. The SPARC-derived option requires SPARC
labels (Task 002).

**When to implement:** After Run 3 reveals which strategy actually
matters. If `uniform_alveolar` works fine → skip. If frication needs
phone supervision → implement `sparc_derived` or `forced_alignment`.

---

## K6 — Smoothness / curvature / barrier weight defaults

**Motivation:** Per `docs/incorrect_assumptions.md` A9, the loss
weight defaults are unverified initial values from the Week-1 task
spec and were never tuned. Codex recommended 10-30× higher
smoothness in commit `c639a73` but only as CLI overrides; the code
defaults still hold the original guesses.

**Current behavior:** `W_SMOOTH_S=1e-3, W_SMOOTH_T=1e-3, W_CURV=1e-4`
defaults; CLI flags `--w-smooth-s, --w-smooth-t, --w-curv` override.

**Proposed knob:** Either (a) update defaults to Codex's recommended
values after empirical confirmation they work better, or (b) make
the recipe pre-set: `--recipe codex` flag picks Codex's values,
`--recipe legacy` keeps current.

**Cost:** Trivial.

**When to implement:** After comparison runs show which weights
actually behave better. Defer until we have Week-1-fixed-rerun data.

---

## Notes on bar / rationale

The user explicitly asked: "i would just hope that whatever we
choose in the future would be a configuration item, not a code
refactor item."

This file's purpose is to NAME the config items we know we'll want
later, so when we add them they slot into existing knob slots
rather than requiring refactors. Adding a new entry here is cheap;
adding it later as a refactor is expensive. Bias toward over-
documenting the knob surface area.

If a knob graduates to "actually we want this now" status, raise a
task file (`workplan/tasks/00_active/NNN_<slug>.md`) referencing
the knob's Kx ID. Don't delete the entry here — annotate
`**IMPLEMENTED in <commit>**` at the top.
