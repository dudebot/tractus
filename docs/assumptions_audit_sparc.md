# Assumptions audit — SPARC stack (Category 1)

**Scope:** Category 1 only of `workplan/tasks/00_active/008_assumptions_audit.md`
— the SPARC integration assumptions currently shaping Week 2. Categories
2/3/4 are owned elsewhere.

**Sources:** SPARC paper (`docs/references/core/2024_sparc_vocal_tract_kinematics.pdf`),
upstream source at `github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding`
(commit at `main` HEAD on 2026-05-15: files `sparc/sparc.py`, `inversion.py`,
`src_extractor.py`, `speech.py`, `spk_encoder.py`), HF model card
`microsoft/wavlm-large`, plus one Codex cross-check (thread
`[session identifier omitted]`, fresh / cold review).

## Top-line summary

Three findings worth surfacing before the per-item detail:

1. **Periodicity threshold is NOT applied by SPARC out of the box.** Paper §B.1
   says "we threshold the periodicity bigger than 0.4." Source code defaults
   `periodicity_threshold=0.0` in both `SPARC.__init__` and
   `SourceExtractor.__init__`, and the public `load_model()` loader does not
   override it. Task 002 already plans to mask `f0` where `periodicity < 0.4`
   downstream — that is the right thing to do, but **not** because SPARC did it
   for us. (See entry 5.) Worse, SPARC's internal `_threshold_periodicity()`
   only zeros the *periodicity* trace, never the pitch — so anybody reading
   `pitch` and trusting it without looking at `periodicity` is reading raw
   torchcrepe output everywhere.

2. **EMA "z-scored within utterance" is a *training-time* preprocessing step,
   not an inference-time output guarantee.** The paper sentence in §III.A.1
   describes how MNGU0 EMA was prepared before fitting OLS; the inverter at
   inference time is just `WavLM features -> Butterworth lowpass -> Linear`
   with no normalization step. The output is in "MNGU0 z-score space" —
   dimensionless, with the per-channel mean/std baked into the OLS bias and
   coefficients. Because the training z-score was *within-utterance*, there is
   no single global (mean, std) pair that would invert it back to mm. The
   schema in Task 002 currently lists `"z_score_within_utterance"` as a
   possible value of `ema_normalization` — that label would be misleading;
   "MNGU0-OLS-output-space" is more honest. (See entry 6.)

3. **WavLM at non-16-kHz input is "inherited unverified", not "verified
   refusal".** The HF model card says "make sure that your speech input is
   also sampled at 16kHz" — that is normative, not a runtime assertion. The
   model has a fixed 16 kHz convolutional front-end (320-sample stride
   producing the 50 Hz output rate), so feeding 24 kHz audio would just be
   interpreted as 24/16 = 1.5x time-stretched signal — frequencies would shift
   downward by 33%. SPARC's loader resamples for you (`speech.py::_load_wav`
   calls `librosa.resample`), so within SPARC the rule is enforced. The risk
   is in any downstream code that bypasses `Coder.encode()` and feeds WavLM
   directly. (See entry 1.)

The Week-2 plan is mostly safe — Task 002 already prescribes resampling to
16 kHz, masking `f0` by periodicity, and saving as `sparc_api_native`. The
audit's actionable surface is: (a) update the `ema_normalization` schema
language, (b) confirm the periodicity threshold is documented as
*downstream-applied*, not *SPARC-applied*, and (c) decide whether the
"physical mm" recovery question matters for the body model in Task 003.

---

## Per-assumption entries

### 1. 16 kHz audio input requirement

- **Source citation.**
  - WavLM model card (HF `microsoft/wavlm-large`): "The large model
    pretrained on 16kHz sampled speech audio. When using the model, make sure
    that your speech input is also sampled at 16kHz." (Only sentence on the
    topic; no quantitative degradation curve given.)
  - SPARC source: `sparc/sparc.py:82` defaults `sr=16000`; `speech.py:80–81`
    auto-resamples any other input rate via
    `librosa.resample(wav, orig_sr=sr, target_sr=self.sr)`.
- **Verdict.** `inherited unverified` (for the *runtime behavior at other
  rates*) / `verified` (for the *resampling-required-and-handled-by-SPARC*
  rule). The 16 kHz rule is real and load-bearing — WavLM's CNN frontend has
  a fixed 320-sample stride that produces the 50 Hz feature rate only when
  the input is at 16 kHz — but the *failure mode* at e.g. 24 kHz input is not
  a refusal; it is silent semantic distortion (frequencies shifted, feature
  rate changed to 75 Hz).
- **Failure mode if wrong.** Loud and immediate if SPARC's own loader is
  bypassed: features would be extracted at the wrong frame rate (e.g.
  24 kHz / 320 = 75 Hz instead of 50 Hz), AAI EMA would be temporally
  misaligned with downstream 50 Hz consumers, and the linear MNGU0 head
  would receive WavLM features computed on a frequency-shifted spectrum →
  garbage EMA. Quiet if SPARC's loader is used (it resamples for you).
- **Suggested verification.** None needed for the v0 pipeline since Task 002
  resamples explicitly. If/when anyone adds a "skip SPARC, use WavLM
  directly" path, run a one-line probe:
  `librosa.resample(audio, 24000, 16000)` *before* WavLM, never after.

### 2. 50 Hz frame rate output

- **Source citation.** Paper §III.A.1: "The original 200 Hz EMA data is
  downsampled to 50 Hz to match the sampling rate of the SSL features." Code
  `sparc/sparc.py:82` defaults `ft_sr=50`; `src_extractor.py:51`
  `pitch_hop_length = int(self.sr/(self.ft_sr*self.q))` = 320 at sr=16000.
- **Verdict.** `verified`. The 50 Hz rate is forced by WavLM's transformer
  stride (320 samples / 16000 Hz = 50 Hz). SPARC matches CREPE and loudness
  to that grid by setting their hop lengths to 320 (with a paper-noted hack:
  CREPE actually runs at 200 Hz with hop=80, then downsamples by 4 — see
  §B.1 — to avoid octave errors on high-pitched female speakers).
- **Failure mode if wrong.** N/A — this is fixed by the SSL backbone. To get
  100 Hz EMA you would need to either (a) use a different SSL model with
  finer stride, (b) interpolate post-hoc, or (c) run two staggered passes —
  none free.
- **Suggested verification.** None. If a future task wants higher temporal
  resolution, the question becomes "which SSL model?", not "can SPARC do
  100 Hz?" (it can't).

### 3. EMA channel order = TD, TB, TT, LI, UL, LL (XY pairs)

- **Source citation.** SPARC repo README states the order as
  `'TDX','TDY','TBX','TBY','TTX','TTY','LIX','LIY','ULX','ULY','LLX','LLY'`.
  `inversion.py:40–44` loads `linear_model.coef_` and instantiates
  `nn.Linear(input_dim, output_dim)` blindly from the pickle — channel order
  is whatever the published `wavlm_large-9_cut-10_mngu_linear.pkl` was
  trained in.
- **Verdict.** `verified` for the published checkpoint; `inherited unverified`
  for "is this stable across SPARC versions or model variants." The order is
  not self-describing in the output dict — you trust the README + the
  pickled OLS row order.
- **Failure mode if wrong.** Silent. If two channels swap (e.g. TT/TB), every
  downstream consumer learns the wrong articulator-to-feature binding;
  shows up as "the body model converges but the synthesized voice has tongue
  positions that look right on a plot but sound wrong." Hard to debug
  because nothing crashes.
- **Suggested verification.** **One-shot at extraction time:** the moment
  the SPARC venv is stood up in Task 002, run `Coder.encode()` on a vowel
  with known articulator targets (e.g. the high-front vowel /i/ → expect
  TT/TB high-Y), inspect the per-channel time series, and freeze a
  `ema_channel_order` constant. This is what Task 002 already does in
  Gate 2 — the gate is correct, just make sure it actually verifies the
  assignment, not just that 12 channels exist.

### 4. MNGU0 as canonical articulator space

- **Source citation.** Paper §III.A.1: "We selected one of the largest
  single-speaker EMA datasets, MNGU0, that includes 75 minutes of EMA
  collected while reading newspapers aloud." Justification: §IV-A and
  Appendix A.4 — affine transformations between speakers' EMA spaces are
  near-isometric (Cho et al. [10]).
- **Verdict.** `verified` (for SPARC's stated use case: cross-speaker AAI).
  `inherited unverified` for *our* use case, which is not cross-speaker AAI
  but a kinematic teacher for one speaker-conditioned body model. We don't
  care that MNGU0 generalizes to other speakers — we care whether MNGU0's
  articulator coordinate system is a useful *target* for our body model to
  learn against.
- **Failure mode if wrong.** Soft. If MNGU0's articulator placements are
  e.g. systematically biased relative to "average vocal tract anatomy",
  the body model trained against SPARC labels would inherit that bias —
  but since the body model is *learned* (not fitted to a specific physical
  reference), it can absorb arbitrary linear offsets. The risk is more
  about *interpretability* of the resulting EMA-like representation than
  about correctness of the synthesized audio.
- **Suggested verification.** Skip for now. Re-raise if/when we want to use
  the body model's predicted EMA for human-readable visualization
  (e.g. an interactive vocal-tract animation), at which point the
  MNGU0-vs-our-anatomy alignment becomes a UX question.

### 5. Periodicity threshold = 0.4

- **Source citation.** Paper §B.1: "we threshold the periodicity bigger than
  0.4. The periodicity thresholding is only applied for the inference time."
  Source code: `sparc/sparc.py:83` defaults `periodicity_threshold=0.0`;
  `src_extractor.py:43` same default; `_threshold_periodicity()` at
  `src_extractor.py:145–152` only zeros *periodicity*, never *pitch*; the
  public `load_model()` does NOT override the default. **The published
  checkpoint as released does not apply the 0.4 threshold the paper
  describes.** Codex (`[session identifier omitted]`) confirmed this reading.
- **Verdict.** `malformed` (paper says one thing, source does another) — but
  the *downstream effect* on Task 002 is benign because Task 002 explicitly
  applies `f0[periodicity < 0.4] = NaN` itself.
- **Failure mode if wrong.** Two layers:
  - If you trust the paper and assume SPARC pre-thresholds, you get raw
    CREPE pitch contours including unvoiced frames where CREPE's f0 estimate
    is meaningless. Loud as soon as anyone plots f0 — the contours go
    haywire in unvoiced regions. Task 002 dodges this.
  - The 0.4 value itself is paper-tuned for SPARC's *resynthesis* pipeline
    (where over-aggressive masking removes useful pitch context for the
    HiFi-GAN). For our use case (kinematic teacher, training-time only), we
    might want a *higher* threshold (cleaner voiced/unvoiced labels) or to
    keep the raw periodicity and let the body model decide. Task 002 already
    saves `f0_periodicity` raw alongside the masked f0 — that is the right
    call.
- **Suggested verification.** None for the value of 0.4 specifically — it is
  a downstream tunable for us, not a SPARC requirement. **Do** add a
  one-line comment in `extract_sparc_labels.py` clarifying that SPARC's
  default is 0.0 and the 0.4 threshold is being applied by us, not by SPARC,
  to avoid future confusion.

### 6. EMA z-scored within utterance

- **Source citation.** Paper §III.A.1: "each channel is z-scored within
  utterances" — but in the context of *MNGU0 training data preparation*,
  before fitting OLS. Source: `inversion.py::_extract_ema` (the inference
  path) does NOT apply any normalization to the EMA output; it is just
  `WavLM features -> Butterworth lowpass -> nn.Linear -> reshape`. There is
  no de-normalization function in the SPARC API. Codex confirmed
  (`[session identifier omitted]`).
- **Verdict.** `malformed` (the assumption as stated mixes training-time and
  inference-time semantics).
- **Failure mode if wrong.** The Task 002 schema offers
  `ema_normalization ∈ {sparc_api_native, z_score_within_utterance,
  physical_mm}` and asks the worker to pick one. **None of the three is
  fully correct**:
  - `sparc_api_native` is the most honest label (just record what the API
    returned).
  - `z_score_within_utterance` is *false* — SPARC's output is not z-scored
    on the *current* utterance; it is in MNGU0-OLS-output-space. The raw
    output statistics will *approximate* zero-mean unit-variance only if
    the test utterance happens to match MNGU0's training distribution.
  - `physical_mm` is unrecoverable — the per-utterance z-score destroys the
    global scale, so even with the saved MNGU0 channel means/stds (which
    SPARC does not expose) you couldn't invert it.
- **Suggested verification.** Update the schema in
  `workplan/tasks/00_active/002_sparc_extraction_pipeline.md` to drop
  `physical_mm` as an option and re-word `z_score_within_utterance` →
  `mngu0_ols_output_space` (or just collapse to `sparc_api_native` always).
  Add the empirical per-channel `[min, max, mean, std]` sidecar (already
  planned) so that Task 003 can do its own normalization at training time
  if it wants.

### 7. Speaker-embedding FFN inside vocoder checkpoint

- **Source citation.** Paper §III.A.3: "the encoder consists of the frozen
  CNN extractor from WavLM Large followed by a weighted pooling layer and a
  learnable feedforward network (FFN)." Code: `sparc/sparc.py:54` —
  `config['spk_ft_ckpt'] = ckpt['state_dict']['spk_ft']`. The `spk_ft` (FFN)
  weights are bundled into the same `model_english_1500k.ckpt` file as the
  HiFi-GAN generator weights. The `feature_extraction` model variant
  (`sparc/sparc.py:29–32`) only loads the inversion linear pickle and does
  NOT include `spk_ft` — so without the full checkpoint, you do not get the
  64-d speaker embedding. Codex confirmed (`[session identifier omitted]`).
- **Verdict.** `verified`.
- **Failure mode if wrong.** Not a deal-breaker — the full
  `model_english_1500k.ckpt` is what Task 002 plans to download anyway, so
  the FFN comes "for free." The risk is purely about the *license-and-disk-
  space* hard gate in Task 002:
  - **License:** the FFN ships as part of the same checkpoint as the
    HiFi-GAN — same license. No ability to "use only the analysis FFN
    under a more permissive sub-license."
  - **Disk / VRAM:** ~1.26 GB checkpoint download is mandatory. No
    "encoder-only lite" build available without retraining. Task 002 already
    accounts for this.
- **Suggested verification.** None — the constraint is well-understood and
  Task 002 handles it. Confirm during Task 002's license-check gate that
  the checkpoint being downloaded is `model_english_1500k.ckpt` (full
  bundle), not `feature_extraction` (linear-only, no speaker embedding).

---

## What would change if these are wrong

Mapping each entry to the architecture decisions it underwrites:

- **Entries 1, 2.** If the 16 kHz / 50 Hz assumptions broke, the entire
  Task 002 sample-rate handling block (resample to 16 kHz, save
  `sparc_input_sample_rate=16000`) would need to change, AND every
  downstream consumer that aligns SPARC labels with TEA's internal sample
  rate would need a new alignment scheme. Currently safe.
- **Entry 3.** If the channel order assumption broke, the body-model
  (Task 003) would learn a permuted EMA target — silent failure mode, very
  hard to diagnose. The fix-cost is low (re-train), the detection-cost
  is high. **The Gate-2 sanity check in Task 002 must verify the
  assignment per-channel, not just shape.**
- **Entry 4.** If MNGU0's articulator space is not a good target, the body
  model still trains, but the resulting EMA-like representation is not
  visually interpretable as "tongue tip / tongue blade / etc." This only
  matters for downstream UX/interpretability, not for synthesis quality.
- **Entry 5.** If the periodicity-threshold story is wrong, Task 003's
  voiced/unvoiced supervision is degraded but recoverable — Task 002 saves
  raw periodicity alongside masked f0. The risk is purely "did anyone read
  the paper and assume SPARC applied the 0.4 threshold?" — which would
  cause double-masking. Add a script comment to prevent this.
- **Entry 6.** If the z-score interpretation is wrong, the *current*
  Task 002 schema offers a misleading label set. Fix is documentation-only:
  drop `physical_mm`, re-word `z_score_within_utterance`. The body model in
  Task 003 already needs to do its own normalization at training time
  (otherwise different speakers' EMA traces would have different scales),
  so the *correctness* impact is zero — only the schema honesty matters.
- **Entry 7.** If the FFN-in-vocoder-checkpoint assumption broke (i.e. the
  upstream maintainers split the FFN out into a separate file), the only
  impact is on Task 002's download script and the license capture step.
  Trivially recoverable.

## Codex disagreements

None on substance. Codex (thread `[session identifier omitted]`) added one nuance on entry 6
that is worth keeping: even *with* the saved MNGU0 channel means/stds, the
within-utterance z-score on training data means there is no single global
(mean, std) pair to invert. So `physical_mm` is not just "API doesn't
expose it" — it is structurally unrecoverable from the published checkpoint
alone. That is now reflected in the entry.
