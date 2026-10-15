# Assumptions audit — TEA simulator + Loss/optimization

Scope: Categories 2 (TEA simulator) and 4 (loss/optimization) of
`workplan/tasks/00_active/008_assumptions_audit.md`. Categories 1
(SPARC) and 3 (workplan budgets) are owned elsewhere and not touched
here.

Method: for each load-bearing assumption, locate the source line(s),
trace the introducing commit via `git blame`/`git log -S`, read the
introducing commit message and surrounding docstrings for the
original rationale, then assign one of four verdicts:
`verified`, `inherited unverified`, `cargo-cult`, `malformed`.
Codex was used adversarially on the most consequential finding
(thread `[session identifier omitted]`); its disagreement is
recorded inline.

**No code changes were made.** This is documentation only.

---

## Top-3 concerns (read this first)

1. **Sample-rate contract is silently broken in the copyfit script.**
   `scripts/copyfit_neural_field_phrase.py` declares `SAMPLE_RATE =
   22050` and `SAMPLES_PER_FRAME = 441`. It loads target audio at
   22050 Hz, passes `samples_per_frame=441` to TEA without
   overriding `dt`, and TEA's `dt` is hard-coded to `1/44100`.
   Both the rendered (44.1 kHz time-base) and target (22.05 kHz
   time-base) audio arrays then feed the same `log_mel(...)` with
   `FeatureConfig(sample_rate=22050)`. Codex (thread `[session identifier omitted]`)
   confirms: the rendered acoustic content appears **half-speed /
   one octave down** to the loss because every TEA-internal physical
   frequency is divided by two when the same samples are interpreted
   as 22.05 kHz instead of 44.1 kHz. **All Week-1 copyfit results
   were trained against an octave-mismatched target.** This alone
   makes the "phonetic wall / non-uniqueness" Week-1 narrative
   un-evaluable — the optimizer was being asked to match a target
   the geometry physically cannot reach. `scripts/render_m5_oracle.py`
   lines 17-30 already documented this exact engine constraint
   ("TEA's TIMESTEP_S is hard-coded to 1/44100... the spec'd
   samples_per_frame=50 / sample_rate=22050 ... is aspirational");
   the copyfit script ignored the documented constraint.

2. **ARCHITECTURE.md §5.2 carries the same 22.05 kHz claim that the
   engine cannot satisfy.** Line 347 reads "Per-sample MNA solve
   runs at 22.05 kHz (50 Hz frames × 441 samples)." This is wrong:
   the engine runs at 44.1 kHz, full stop. The doc was written as
   if the aspirational rate were achieved. Anyone reading
   ARCHITECTURE.md as ground truth (a future worker, a Codex
   session, a fresh Claude) will inherit the same wrong assumption
   that bit copyfit. This is a **doc bug that propagates the bug**.

3. **The copyfit loss-weight recipe is "Suggested initial ... tune
   in Week 1 itself" verbatim from the task spec, never tuned.**
   `W_SMOOTH_S = 1e-3`, `W_SMOOTH_T = 1e-3`, `W_CURV = 1e-4`,
   `W_CLAMP = 10.0`, `W_DEFAULT = 1e-2` were all copied straight
   from `workplan/tasks/00_active/001_neural_field_phrase_probe.md`
   §4 ("Suggested initial weights — tune in Week 1 itself"). The
   script later got CLI overrides (commit `c639a73`) only because
   Codex suggested raising smoothness 10×–30× to combat jaggedness;
   no other ablation occurred. The "tune in Week 1" instruction
   was never executed; the values became architectural after
   first use.

---

## Category 2 — TEA simulator assumptions

### 2.1 Precision policy: fp32 default, fp64 opt-in via `TEA_PRECISION`

**Source.**
`tractus_engine_advanced/src/tractus_engine_advanced/precision.py:36-46`.
Module-level docstring cites ARCHITECTURE.md §7.

**Verdict.** `verified`.

The fp32 default is justified by two stated reasons:
(i) `glottis_geometric.py` lines 170-176 document that the 25 Hz
Chebyshev filter is fp32-unstable in direct-form-I and required a
DF2T/SOS rewrite (correctness/006) — i.e., fp32 is the chosen
precision and the code has been adapted to make it stable, rather
than fp32 being a guess. (ii) The fp64 path is preserved as a parity
oracle (subprocess-isolated test markers) — it's not gone, just
opt-in. The mechanism (`jax.config.update` at module load + import
ordering invariant in `__init__.py`) is correct; CLAUDE.md flags the
landmine that bit `tea_backend.py` previously.

**Failure mode if wrong.** Production fp32 forward already passes
parity vs upstream (M2/M4/M5) and against the fp64 oracle within
documented tolerances. If we *needed* fp64 in production, we'd see
parity drift — we don't. Memory cost of fp64 default would 2× VRAM
(directly bites the 16 GB Blackwell budget); compute throughput
roughly halves. Both are immediate and loud.

**Suggested verification.** Re-run `tests/test_tds_parity.py` with
the `parity` and `fp64_oracle` markers enabled; if both pass, the
fp32 production default is justified to within current tolerance.
None needed if no recent change to MNA / glottis fp-arithmetic.

**What would change if this is wrong.** Memory budgets in
ARCHITECTURE.md §5.2 G2 (T=118 b=4 = 708 MB peak) would roughly
double. backward/008 custom_vjp work would need to be re-verified at
fp64.

---

### 2.2 `frame_chunk_size = 1` default in `pipeline.py:220`

**Source.** `tractus_engine_advanced/src/tractus_engine_advanced/pipeline.py:220`.
Introduced in commit `2580214` (`tea(P0): correctness/007 partial —
chunked outer-scan kwarg landed`).

**Verdict.** `verified` for the production T=118 b=4 backward case;
`inherited unverified` for the typical inverse-fit T=120 b=1 case.

The introducing commit message explicitly states K=1 is the
strongest backward-memory bound and is what brings T=118 b=4 under
the 2 GB G2 budget (now actually 708 MB after correctness/008's
custom_vjp). For the production target it is verified. **However**,
the copyfit script overrides this to K=10 (line 461), citing Codex
thread `[session identifier omitted]` recommending K=4 as a throughput experiment and
choosing K=10 as "a more aggressive trade." K=10 at b=1 was deemed
to fit ~156 MB which is well inside headroom. This is a forward-pass
estimate; backward at K=10 wasn't OOM-verified empirically — the
script just runs and either fits or doesn't.

**Failure mode if wrong.** K too large → backward OOM (loud, kills
training). K=1 too pessimistic for b=1 → wasted compile + iter
time (quiet, but observable as iter rate). Currently the script
runs fine at K=10 for copyfit's actual T (~120) so the practical
risk is low.

**Suggested verification.** Run `scripts/verify_backward_oom.py`
with K=10 at copyfit's actual `(T, b)` once GPU is free. If it
passes, copyfit's K=10 is verified for its operating regime.

**What would change if this is wrong.** Just the per-script
default. Not architectural.

---

### 2.3 T=118 batch=4 as the production memory target

**Source.** `ARCHITECTURE.md:347-349` and `:500`. Quoted from the
former: "Per-sample MNA solve runs at 22.05 kHz (50 Hz frames × 441
samples). Backward through TEA at T=118 b=4 fits 708 MB peak via
`jax.checkpoint` + `custom_vjp` on `step_one` + `custom_linear_solve`
on the MNA solve." Origin in archived correctness/007 + 008 task
files.

**Verdict.** `inherited unverified` (where 118 came from) +
`verified` (that the budget passes).

The "T=118 b=4 at ≤ 2 GB" gate has been empirically demonstrated
(now 708 MB) — that part is verified. What is **not** documented
anywhere I could find: where "118 frames" came from. It is not a
corpus statistic, not a planner-output bound, not derived from a
phrase length distribution. It appears in the archived
correctness/007 task file as if pre-given. Likely a Codex
deliberation artifact ("typical short utterance ≈ 2.4 sec at 50 Hz
= 120 frames; round to 118 because it's not divisible by anything
useful and exposes padding bugs") but no commit message or doc
captures the rationale.

The "b=4" choice is similarly under-justified. Plausible reasons:
(a) typical mini-batch for inverse fit, (b) a Codex
test-the-VRAM-headroom value, (c) inherited from an earlier
training-time-only plan that no longer exists.

**Failure mode if wrong.** If true production batch is 8 (or 16),
the budget is loose; if it's 1, we over-engineered the entire
correctness/005-008 stack. Not catastrophic — just wasted effort.
The structural fixes (custom_vjp / custom_linear_solve) help for
any backward through TEA, regardless of batch. No active failure
right now.

**Suggested verification.** Decide at the workplan level whether
T=118 b=4 is still the right operating point in the post-pivot
plan (Cat 3, not my scope). If T is ever 240+ or b is ever 16+,
re-verify the gate.

**What would change if this is wrong.** Budget claims in
ARCHITECTURE.md §5.2 would need updating. The structural fixes
remain useful regardless.

---

### 2.4 Per-sample MNA solve

**Source.** TEA design from m4/m5; the per-sample step body sits
in `tractus_engine_advanced/src/tractus_engine_advanced/tds/step.py`,
specifically `_step_one_impl` and `_mna_solve` at lines 63-93.

**Verdict.** `verified`.

The per-sample structure is inherited from the upstream
VocalTractLab `Synthesizer.cpp:200-242` (cited in `pipeline.py`
docstring). It is what gives the soft-walls + turbulence-loss
algebraic loop a one-sample break (SCOPE.md §3.C row N) and
matches the parity oracle that both M4 and M5 are validated
against. Trying to vectorize across multiple samples would either
(a) break the implicit-state recurrence assumption inside MNA, or
(b) require flattening the state-update equations across N samples
which is the same cost as N per-sample solves.

The autograd-side complexity (custom_linear_solve in
correctness/007 iter 2, custom_vjp on step_one in correctness/008)
is the cost of keeping the per-sample structure under autodiff —
not the cost of the per-sample solve itself.

**Failure mode if wrong.** Fp32 forward parity vs the C++ oracle
would break (it doesn't — M5 corpus regen passes). Algebraic-loop
artifacts would appear in pressure traces (they don't).

**Suggested verification.** None needed unless someone proposes a
per-frame-block solver as an optimization; that proposal would need
to demonstrate forward-parity to the existing oracle.

**What would change if this is wrong.** Wholesale rewrite of `step.py`
+ `mna.py` + the entire correctness/005-008 sequence. Very expensive
to revisit.

---

### 2.5 `samples_per_frame` calculation in copyfit (the load-bearing bug)

**Source.** `scripts/copyfit_neural_field_phrase.py:79-81`:
```
SAMPLE_RATE = 22050
FRAME_RATE_HZ = 50
SAMPLES_PER_FRAME = SAMPLE_RATE // FRAME_RATE_HZ  # 441
```
`git blame` → commit `fad6fff` (`feat(week1): neural-field copy-synth
phrase probe — initial implementation`). `git log -S "SAMPLE_RATE = 22050"`
confirms initial-add commit.

**Verdict.** `malformed`. (Disagreement with the spec it inherited
from is captured below.)

The script enforces `if sr != 22050: raise ValueError` on the
target WAV, then passes `samples_per_frame=441` directly to
`synthesize_dynamic_tube` without overriding `dt`. TEA defaults
`dt=TIMESTEP_S = 1/44100`. The rendered audio array therefore
contains 441 samples per frame at a **44.1 kHz time-base** =
10 ms of TEA-physical audio per frame. The target audio is at a
**22.05 kHz time-base** = 20 ms per frame. Both arrays have
identical length but encode different real-time durations.

The `log_mel` + MR-STFT loss is computed with
`FeatureConfig(sample_rate=22050)`, which means the rendered
audio is interpreted as 22.05 kHz when it is actually 44.1 kHz.
Per Codex thread `[session identifier omitted]`:
`f_apparent = f_physical × 22050/44100 = f_physical / 2`. Every
TEA-internal physical frequency appears **half-speed / one octave
down** to the loss.

I initially suspected the inverse direction (rendered F0 doubled),
Codex caught the sign error: the same samples reinterpreted at
half the rate stretch in time, lowering pitches. Recorded as
disagreement-then-correction.

**Cross-reference.** `scripts/render_m5_oracle.py:17-30` already
documented this exact constraint:

> TEA's TIMESTEP_S is hard-coded to 1/44100 [...] The audio
> synthesize_dynamic_tube produces is therefore at 44100 Hz [...]
> The spec'd samples_per_frame=50 / sample_rate=22050
> (ARCHITECTURE.md §8.1) is aspirational — it would require either
> an engine-level re-derivation of TIMESTEP_S or a post-render
> decimation. Both are out-of-scope for infra/005

So one author already knew. The copyfit author either didn't
read this comment or assumed a fix had landed. The
`scripts/inverse_fit_demo.py:61-62` script gets it right
(`SAMPLE_RATE = 44100; SAMPLES_PER_FRAME = 441   # 10 ms / frame`),
which is internally consistent.

**Failure mode.** Already realized. The Week-1 "phonetic wall"
finding and "loss plateau" interpretation are downstream of an
optimizer being asked to match a target whose physical content
the geometry cannot reproduce (target has formants/F0 at locations
the rendered audio reaches one octave below). The audio you'd
*hear* from `recon.wav` (saved at `sample_rate=22050` per
copyfit:1022) is a half-speed/octave-low rendering of what TEA
actually computed. This explains both the "bo bo ne gardeh"
percept (slowed-down speech sounds slurred and pitched-down) and
the apparent inability to reach phonetic targets.

**Suggested verification.** Two cheap actions:
(a) Read off the F0 trajectory passed in (`extract_source` returns
F0 in physical Hz, e.g., 120.0 = 120 Hz). Render audio. Whisper or
CREPE-pitch-track the output; expected ≈ 60 Hz if my read is
right, ≈ 120 Hz if the engine is actually compensating somewhere.
(b) Render a 1-second sustained vowel at known F0=200 Hz, save
WAV labeled at 44100, and at 22050. Compare spectrograms.

**What would change if this is wrong.**
- Fix is one of: (i) change copyfit's `SAMPLE_RATE` to 44100 +
  resample target audio, (ii) post-render decimate TEA output
  before loss, (iii) rebuild engine `TIMESTEP_S` to actually be
  1/22050 (much larger blast radius).
- Week-1 closure narrative (`docs/week1/findings.md`) needs
  re-evaluation: "kinematic supervision is structurally required"
  is partly grounded on a non-converged optimizer and partly on
  this sample-rate bug. Cat-3 owner should be alerted.
- ARCHITECTURE.md §5.2 (line 347 + §8.1) needs the doc-side fix to
  match engine reality.

---

## Category 4 — Loss / optimization assumptions

### 4.1 log-mel + MR-STFT as the acoustic loss

**Source.** `scripts/copyfit_neural_field_phrase.py:478-485`
(`acoustic_loss(...)`); recipe from
`workplan/tasks/00_active/001_neural_field_phrase_probe.md:139-140`
and ARCHITECTURE.md:430. `FeatureConfig` defaults in
`tractus_controller/tractus_controller/copy_synth/features_jax.py:35-44`.

**Verdict.** `inherited unverified`.

The combination is standard in differentiable-DSP literature
(DDSP, RAVE, Neural Source-Filter, MelGAN-style discriminators)
and the rationale "log-mel for perceptual scaling + MR-STFT for
phase/transient detail" is widely cited. But neither
ARCHITECTURE.md nor the task spec records an ablation in this
codebase. We don't know whether MR-STFT alone, log-mel alone, or
some weighted combo would behave better on the actual TEA forward.
The weights `W_LOGMEL=1.0`, `W_STFT=0.5` come from the task spec's
"Suggested initial weights" line and were never tuned.

**Failure mode if wrong.** Optimizer plateaus at a local minimum
that the loss can't see past, manifesting as "loss flat from iter
N." Plausibly contributed to the Week-1 stall in addition to (and
correlated with) finding 2.5. Loud-ish — visible in loss traces.

**Suggested verification.** Cheap ablation post-Week-1:
log-mel only, MR-STFT only, change weight ratio. Three runs at
T=120 b=1, ~50 iters each, compare final loss + spectrogram
similarity. None of these conclusions are valid until 2.5 is fixed
first.

**What would change if this is wrong.** Just the loss recipe.
Not architectural.

---

### 4.2 Smoothness / curvature / barrier weight defaults

**Source.** `scripts/copyfit_neural_field_phrase.py:122-129`:
```
W_SMOOTH_S = 1e-3
W_SMOOTH_T = 1e-3
W_CURV = 1e-4
W_CLAMP = 10.0
W_DEFAULT = 1e-2
```
`git blame` → commit `fad6fff` (initial implementation), copied
verbatim from `workplan/tasks/00_active/001_neural_field_phrase_probe.md:148-154`
which itself labels them "Suggested initial weights (tune in Week 1
itself)". Commit `c639a73` later added CLI overrides because Codex
suggested 10×–30× higher smoothness to combat jaggedness — but
the script *defaults* still hold the unverified spec values.

**Verdict.** `cargo-cult` (defaults remain the spec's "starting
point" line, never tuned).

**Failure mode if wrong.** Wrong weight balance → either the prior
dominates (field stays near default tube → can't fit) or the
acoustic loss dominates (field thrashes → audible "bo bo ne
gardeh" jaggedness, which Codex actually diagnosed and led to the
CLI override). Loud — visible in `delta_sat_frac`, smoothness
trace, and ear test.

**Suggested verification.** The CLI override path makes this
easy: launch a sweep over (W_SMOOTH_S, W_SMOOTH_T, W_CURV) with
log-spaced values around the current defaults at fixed seed.
Re-evaluate with audio + loss decomposition once 2.5 is fixed.

**What would change if this is wrong.** Recipe tuning, not
architecture. But because these gate the perceived quality of
copyfit output, they affect downstream Week-2 / Week-N decisions
on whether copyfit-without-supervision is viable.

---

### 4.3 `DELTA_MAX = 3.0`

**Source.** `scripts/copyfit_neural_field_phrase.py:99-102`. `git
log -S` → introduced at value `1.5` in `fad6fff`, bumped to `3.0`
in commit `9d96c30` (`feat(copyfit): DELTA_MAX=3.0 + Adam warmup +
update-norm cap`).

**Verdict.** `verified` (with a load-bearing caveat).

Commit `9d96c30` records the rationale in the body:
> Default-tube free range is uniform 4.0 cm²; Story vowel anchors
> need log-space excursions up to ~3.0 to be reached. 1.5 was too
> tight; 3.0 gives physiological headroom (area excursion factor
> ~20×, from 0.2 to 80 cm²).

So 3.0 is grounded in (a) the physical area range required to
reach Story 1996 vowel shapes from the uniform-4.0-cm² default
tube, (b) measured saturation (`delta_sat_frac`) at the prior
value of 1.5. This is a verified empirical choice, not cargo-cult.

The caveat: 3.0 in log-cm² lets area go down to e^(log(4) - 3.0)
≈ 0.2 cm² and up to ≈ 80 cm². The 80 cm² ceiling exceeds the
softplus barrier `AREA_MAX = 12.0 cm²`, so the upper barrier
fires before DELTA_MAX gates. The 0.2 cm² floor is comfortably
above `AREA_MIN = 1e-4 cm²`. So DELTA_MAX is mostly a numerical
saturation guard for the tanh; the real area limits are the
softplus barrier. That's fine but worth flagging — DELTA_MAX
mostly controls *gradient flow shape* near the tanh asymptotes,
not actual physical bounds.

**Failure mode if wrong.** Too small → tanh saturation, gradient
dies (the symptom that motivated the bump). Too large → numerical
instability in the tanh (very unlikely; tanh is well-conditioned).

**Suggested verification.** None — the bump-from-1.5 commit is
the verification.

**What would change if this is wrong.** Local optimizer recipe;
not architectural.

---

### 4.4 Adam warmup + update-norm cap

**Source.** Commits `e17e7b8` (mask + barrier + telemetry) and
`9d96c30` (DELTA_MAX + warmup + cap).

**Verdict.** `verified`.

The introducing commit message records the empirical motivation:
> Adam's first-step problem became the next blocker — `upd ≈ lr
> × sign(grad)` at iter 1 drove the field into tanh saturation by
> iter 4.

Three layered observations:
1. 70-iter run: loss flat from iter 10 (saturation, buggy
   barrier).
2. 25-iter run with mask + LR=1e-2: same saturation, faster
   (`sat=1.0` by iter 10).
3. 5-iter run with margin barrier + LR=1e-2: acoustic loss
   finally moves but `sat=0.42` by iter 5 from Adam's aggressive
   first steps.

That's three failed runs converging on the same diagnosis (Adam
first-step blowup with cold m,v). The warmup + cap fixes are the
standard remedy. Verified by symptom-disappearance in subsequent
runs.

**Failure mode if wrong.** Warmup too short → first-step blowup
returns. Cap too tight → optimizer can't make progress. Both are
loud (visible in `update_l2`, `delta_sat_frac`).

**Suggested verification.** None — the commits document the
verification.

**What would change if this is wrong.** Local optimizer recipe.

---

### 4.5 25 Hz Chebyshev filter on supraglottal pressure

**Source.**
`tractus_engine_advanced/src/tractus_engine_advanced/glottis_geometric.py:50-56,
170-176, 224`. `git log -S "supra_filter_cutoff_hz"` → first
introduced in commit `6fb36df` (`initial vibe`), as part of the
upstream port. Later refactored to SOS form in correctness/006
(commit `d0e4fe6`).

**Verdict.** `verified`.

The 25 Hz Chebyshev is a **direct port from upstream
VocalTractLab `IirFilter`** (cited in the docstring at line 50).
It models the slow trans-pressure (subglottal − filtered
supraglottal) feedback used in the glottis amplitude calculation
at line 386:
```
trans_pressure = soft_relu(cp[PR] - state.supra_filtered_dPa, tau_relu)
amplitude = 0.0011 * jnp.sqrt(trans_pressure + 1e-12)
```
The 25 Hz cutoff isolates the slowly-varying mean supraglottal
pressure (filtering out the kHz-scale acoustic resonances), which
is what the upstream Birkholz / Story formulation uses to drive
the source amplitude. This is physiology, not a tuning knob —
it's modeling the acoustic-mechanical feedback loop that opens
the glottis based on the mean pressure differential rather than
the instantaneous waveform.

The DF1→SOS refactor is a numerical-stability fix (correctness/006)
not a redesign of the cutoff itself; the cutoff value 25 Hz is
preserved bit-for-bit.

**Failure mode if wrong.** If 25 Hz were wrong, glottis amplitude
would be either over-modulated by acoustic resonances (cutoff too
high) or unresponsive to phrase-level pressure changes (cutoff too
low). Forward parity vs upstream would fail. It doesn't.

**Suggested verification.** None needed — upstream port + parity
test cover this.

**What would change if this is wrong.** This would require
revisiting the entire glottis-source model parity assumption,
which would invalidate M5/G3a + correctness/006. Very high blast
radius; unlikely worth questioning.

---

## Cross-cutting observation

Of the eleven items audited, exactly one is a real bug (2.5 / the
sample-rate mismatch). The rest are split between "verified
empirically" (precision policy, DELTA_MAX bump, Adam recipe,
25 Hz filter, per-sample MNA, T=118 b=4 budget achievement) and
"inherited from spec, never tuned" (loss-weight defaults,
log-mel + MR-STFT combo).

The pattern that the task brief warned about — "load-bearing
constants get inherited from prior sessions / Codex
deliberations / paper-section-X without anyone verifying they're
still load-bearing" — is **most strongly visible in the
sample-rate finding**, where a documented engine constraint
(render_m5_oracle.py:17-30) was contradicted by a later script
(copyfit) without the contradiction being caught. The doc bug in
ARCHITECTURE.md §5.2 (line 347, "22.05 kHz") propagated the same
wrong assumption forward.

The Week-1 closure verdict ("loss plateaued / phonetic wall /
kinematic supervision is structurally required") is downstream of
finding 2.5. Whether that verdict is recoverable after the
sample-rate fix is a Cat-3 question for the workplan/findings
owner, but the audit's responsibility is to surface that the
basis for the Week-1 conclusion is more fragile than it looked.

---

## References

- Codex thread `[session identifier omitted]` — adversarial
  review of finding 2.5 (sample-rate). Recorded sign correction.
- Codex thread `[session identifier omitted]` — Week-1 stall + DELTA_MAX bump rationale,
  cited by commits `e17e7b8` and `9d96c30`.
- Codex thread `[session identifier omitted]` — frame_chunk_size throughput
  recommendation, cited by copyfit script docstring.
- `scripts/render_m5_oracle.py:17-30` — pre-existing documentation of
  the engine 44.1 kHz constraint that the copyfit script later
  violated.
- `workplan/tasks/00_active/001_neural_field_phrase_probe.md` — origin
  of the loss-weight recipe and the "samples_per_frame=441 for 50 Hz
  × 22.05 kHz" claim that became copyfit's bug.
- `ARCHITECTURE.md:347` — propagates the "22.05 kHz" claim that the
  engine cannot satisfy.
