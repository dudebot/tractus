# infra/003 — Runbook scaffolding

**Phase:** infra    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** correctness/{002,003,004,005,006} acceptance, smoke/002, inversion/*, cloning/*
**Blocked by:** —
**How to dispatch:** fresh Claude Code session

---

## Context

Acceptance gates throughout the workplan invoke standalone scripts
(no LLM in the loop) that produce JSON reports + exit codes. This
task lands those scripts as production-ready CLI tools.

`runbooks/` already contains skeleton stubs (created with
the workplan); this task fleshes each into a working tool.

## Runbooks to land

For each, the contract is: read inputs from CLI args, produce a JSON
report on stdout + a one-line summary on stderr, exit 0 on pass /
non-zero on fail.

### 1. `whisper_gate.sh`

ASR transcription + WER + jaccard vs target text. Whisper-jaccard
alone is a noisy stop-loss (ARCH §1147-1150); the gate must be paired
with the plausibility panel (#3) and a template-only baseline before
declaring intelligibility.
- Input: `--audio <path> --target "transcript text" [--threshold-wer 0.25 --threshold-jaccard 0.30]`
- Uses `whisperx` or `faster-whisper` (pick the lighter wheel).
- Output: `{"wer": float, "jaccard": float, "asr_text": str, "pass": bool}`. WER is **word-level** (Whisper is word-level; do not advertise phone-WER without an MFA aligner).
- Exit: 0 if WER < threshold AND jaccard > threshold; else 1.
- Implementation: standalone WER/jaccard utility under
  `runbooks/`; do not import from
  `tractus_controller.copy_synth.gates` (that module is not part of
  the runbook contract).

### 2. `source_tract_sep.py`

Per-axis decomposition: F0 (cents), spectral tilt (dB/octave),
frication band energy (dB), nasalization metric, formants F1–F3.
- Input: `--audio <path> --reference <path> [--ref-anatomy <path>]`
- Outputs per-axis distance metrics in JSON. Pass requires:
  F0 RMSE < 50 cents, tilt within ±2 dB/octave, formant within
  5%, frication band energy within 3 dB.
- Use PYIN+Crepe agreement for F0; Praat or `librosa.lpc` for
  formants; `librosa.feature.spectral_*` for tilt.

### 3. `plausibility_panel.py`

Continuous-time velocity / acceleration / jerk caps + drift bounds
+ phone-boundary alignment (MFA-anchored).
- Input trajectory NPZ schema (required keys):
  `trajectory: float32[T, D]`,
  `channel_names: str[D]` (e.g., `["jaw_open_cm", "tongue_body_x_cm", ...]`),
  `units: str[D]` (per-channel; cm / cm-per-s / dPa / Hz / dimensionless),
  `frame_rate_hz: float`,
  `training_std: float32[D]` (channel-wise std from the training
  distribution; the runbook does NOT estimate this on the fly),
  optional `caps_config: str` (path to per-channel velocity / accel /
  jerk caps JSON).
- CLI: `--trajectory <path-to-(T,D)-npz> [--mfa <textgrid> --caps-config <path>]`
  (frame rate, channel names, units, training-std come from the NPZ).
- Caps in continuous-time units (cm/s, cm/s², cm/s³), per channel.
- Default caps: jaw 50 cm/s peak, tongue body 80 cm/s, lip 60 cm/s,
  glottis F0 jump 200 Hz/100ms. Configurable via JSON.
- Drift: rolling mean per channel over T should not exceed 2× the
  channel's training-distribution std (read from NPZ).
- Phone-boundary alignment: trajectory inflections within 30 ms of
  MFA-anchored phone boundaries when available.
- Output: per-channel pass/fail + worst offenders.

### 4. `anticheat_unreachable.py`

Generate transcript-preserving but gesture-unreachable target audio
(formant shift, EQ tilt, pitch shift, reverb), then run a fitter and
gate on **refusal**.
- Input: `--source-audio <path> --transcript "..." --perturbation
  {formant|eq|pitch|reverb|all}`
- Generates perturbed target; runs a fitter (passed in via
  `--fitter-cmd`); checks the fitter's loss-improvement on the
  unreachable target.
- Pass criterion: optimizer **refused** to find spurious
  improvement (loss change < 5% from initialization, OR
  optimizer reported "refused" explicitly).

### 5. `dimensionality_report.py`

Two analyses + a sweep summary.
- Input: `--body-trajectories <path-to-trajectories.npz>
  --renderer <module-spec> [--sweep 24,32,48,64,96,132]`. The default
  sweep set is `{24,32,48,64,96,132}` per ARCH §1206-1211; do not add
  D=16 or other values without an architecture update.
- Analysis A: PCA / SVD on body trajectories. Cumulative variance
  table at standard D values.
- Analysis B: Acoustic-Jacobian active-subspace SVD. Compute
  J = d(audio_features)/d(body_state) per frame; SVD; rank by
  singular value. Features: F0, F1–F3, spectral tilt,
  frication-band energy, nasalization, log-mel bands.
- Analysis C: per-D held-out gate metrics. For each D in the sweep,
  report intelligibility (word-WER + jaccard via #1), identity
  (ECAPA-TDNN cosine), anti-cheat refusal (#4), and plausibility (#3)
  on the same held-out set. inversion/006 reads these to pick
  `D_body_effective`.
- Dynamic-source parity check on renderer-backed analyses: the
  Jacobian must use the dynamic-source path (per ARCH §1185-1191),
  not a frozen-source approximation.
- Output: combined report `{"pca": {...}, "jacobian_svd": {...},
  "per_d_gates": {...}, "candidate_D_body": [int, ...],
  "recommended_min": int}`.
- Used by inversion/006 (the dimensionality-discovery decision
  task).

### 6. `matched_reference_eval.sh`

Commercial-baseline comparison at matched reference length.
- Input: `--our-audio <path> --reference-30s <path>
  --commercial-clone-cmd <cmd-template-with-{ref}-and-{text}>
  --eval-text "..." [--ivc-mode | --pvc-mode]`
- Invokes the commercial cloning at the same reference length we
  use, with their cloning mode (Instant vs Professional)
  explicitly labeled.
- Computes WER + jaccard + ECAPA-TDNN identity cosine vs the
  reference speaker for both our system and the commercial system.
- Output: matched-reference comparison table per-utterance.

## Acceptance gates (automated)

- ✅ Each of the 6 runbooks has a `--help` output, an executable
  bit, and BOTH a passing smoke fixture AND a deliberately-failing
  fixture under `runbooks/test_fixtures/`.
- ✅ `bash runbooks/<each>.sh --help` (or `python -m
  runbooks.<each> --help`) returns 0.
- ✅ Smoke tests assert: pass-fixture exits 0 with valid JSON;
  fail-fixture exits non-zero with a one-line stderr summary that
  names the failed gate; output JSON validates against the runbook's
  schema (ship the JSON-schema files under
  `runbooks/schemas/`).
- ✅ Each runbook has a 1-line description in `INDEX.md` runbook
  inventory matching its actual behavior.

## Anti-deliverable / scope guard

- Do NOT make these scripts depend on the rest of the project
  beyond stable interfaces (audio in, JSON report out). They must
  be runnable from a fresh checkout with only their own
  dependencies installed.
- Do NOT vendor large model weights (Whisper, ECAPA-TDNN). Scripts
  download on first run via cached HuggingFace pulls.
- Do NOT introduce LLM calls. These are pure-data verification
  scripts.

## Dispatch notes

Implementation can be parallelized: each runbook is a small
self-contained tool. Worker should pick a sensible Python
environment manager (uv pyproject is what the project uses) and
keep dependencies minimal.

This task is overscoped at M for one fresh session. If the worker
finds the six production CLIs cannot land cleanly in 1–2 sessions,
split per runbook (`infra/003a` whisper_gate + source_tract_sep,
`infra/003b` plausibility + anticheat, `infra/003c` dimensionality +
matched_reference) and file follow-ups; the acceptance gates above
are the contract regardless of split.
