# Task 002 — SPARC extraction pipeline (Week 2 prep)

> **Blocked by:** Task 001 GREEN result. **Read first:**
> `workplan/handoff/2026-05-09_course_correction.md` for the why.

## Goal

Stand up a **training-time-only** SPARC inversion pipeline that maps
audio → kinematic labels and freezes them as `.npz` artifacts, with
**zero runtime impact on the JAX inference stack**. SPARC labels
become the supervision target for the speaker-conditioned body model
in Task 003.

SPARC is **not in the inference stack**. Period. WavLM Large is
1.26 GB / ~316 M params; that blows the project's ≤8 GB / ≤30 M
inference budget on its own. Treat SPARC like rtMRI corpora — labels
once, frozen forever.

## Why this comes before the body-model training

Task 003 (speaker-conditioned body model) needs supervision data
beyond the single Week-1 phrase. SPARC provides:

- 12 EMA-like articulator channels @ 50 Hz (kinematic teacher).
- F0 + loudness (source teacher).
- 64-d speaker embedding (anatomy/identity teacher).

Without it, the body model has only one labeled utterance to
generalize from, and Task 003 will fail.

## Inputs

- **Primary corpus**: LibriTTS-R `train-clean-100`, 50-100 utterances.
  In-distribution for the SPARC `en` checkpoint (SPARC was trained on
  LibriTTS-R), so labels should be most reliable here. Hard cap ≤ 500
  utterances for the v0 round; this isn't a corpus build.
- **Optional OOD stress test**: 10 VCTK utterances. Useful for the
  speaker-embedding diversity check; NOT required for Gate 3 pass.
- 10 developer-recorded utterances for held-out validation +
  speaker-fitting.
- The Week-1 reference (`juice_lemons_ref.wav`) — sanity check
  that SPARC produces reasonable labels on it.

**Sample-rate handling.** Source corpora arrive at varying rates
(LibriTTS-R is 24 kHz, VCTK is 48 kHz, dev recordings vary). The
script MUST resample to **16 kHz mono** before SPARC/WavLM ingestion;
WavLM Large requires 16 kHz input. Record `source_sample_rate`,
`sparc_input_sample_rate=16000`, source duration, and resampler
library+version (`librosa.resample` with `res_type="soxr_hq"`
recommended) in each `.npz`.

## Deliverable

`scripts/extract_sparc_labels.py` — a script that:

1. Lives in a separate Python env with PyTorch + WavLM +
   SPARC weights. **Do not merge into the JAX env.** Suggested:
   `tools/sparc/.venv/`. (uv supports this.)
2. Reads a manifest of audio paths.
3. Outputs one `.npz` per utterance under
   `outputs/sparc_labels/<corpus>/<utt_id>.npz` with the following
   keys:

   ```
   ema[T, 12]                float32, 50 Hz, channel order = SPARC API native:
                             [TDx, TDy, TBx, TBy, TTx, TTy, LIx, LIy, ULx, ULy, LLx, LLy]
                             (verified against SPARC Coder.encode() output; do NOT reorder)
   ema_channel_order         np.array of 12 strings, the names above in order
   ema_normalization         str — exactly one of:
                               "sparc_api_native"          (default — saved as returned)
                               "z_score_within_utterance"  (only if SPARC code/docs confirm
                                                            that de-normalization is applied
                                                            at inference; per audit A5 below,
                                                            this is likely structurally
                                                            unrecoverable — training z-scoring
                                                            discards per-utterance scale)
                               "physical_mm"               (only if SPARC code/docs confirm;
                                                            per audit A5 below, this is
                                                            almost certainly incorrect)
                             Worker MUST inspect SPARC source for the truth; if neither
                             can be confirmed from source, save as "sparc_api_native"
                             and record empirical per-channel [min, max, mean, std] in a
                             sidecar `ema_stats.json` for the Week-1 probe.
                             Do NOT guess "physical_mm".
                             > **[Note per `docs/incorrect_assumptions.md` A5]** SPARC's
                             > published checkpoint outputs EMA in MNGU0-OLS-output-space.
                             > Training z-scoring was within-utterance, so per-utterance
                             > scale was discarded during training and is structurally
                             > unrecoverable at inference. "sparc_api_native" is almost
                             > certainly the correct label. Body-model code (Task 003)
                             > must learn the projection rather than assume a known
                             > coordinate system.
   f0[T]                     float32, Hz; SPARC API output as-is, with NaN where
                             f0_periodicity < 0.4 (this is a tractus downstream
                             convention, NOT SPARC's default; SPARC source code
                             defaults periodicity_threshold=0.0 — see
                             `docs/incorrect_assumptions.md` A8; the 0.4 figure
                             is from the paper's training-time config but is not
                             enforced in the published checkpoint pipeline)
   f0_periodicity[T]         float32, raw periodicity from SPARC pitch tracker
                             (kept so downstream can re-mask)
   loudness[T]               float32, SPARC API loudness output AS-IS — do NOT re-derive
   loudness_units            str — "sparc_api_native"; one-line comment in the script
                             notes what SPARC docs say the unit is. Trust the API output
                             over the paper text if they conflict.
   speaker_emb[64]           float32
   frame_rate_hz             50
   source_sample_rate        int
   sparc_input_sample_rate   16000
   source_duration_s         float
   resampler                 str (e.g. "librosa.resample/soxr_hq")
   sparc_version             str (PyPI version OR git SHA of cloned repo)
   sparc_checkpoint_sha256   str (computed once at script start, copied into each .npz)
   wavlm_source              str (e.g. "microsoft/wavlm-large @ HF revision <hash>")
   wavlm_checkpoint_sha256   str (computed once at script start)
   torch_determinism         str (concise tag, e.g. "fp32_tf32off_cudnnbench_off_eval_nograd")
   extraction_timestamp      str (ISO 8601 UTC)
   audio_path_hash           str (SHA-256 hex of source audio bytes)
   ```

4. Writes a manifest `outputs/sparc_labels/<corpus>/manifest.json`
   listing all extractions + their hashes.

A single **JAX-side** loader at
`tractus_controller/tractus_controller/data/sparc_labels.py`
that consumes the `.npz` files. The loader has zero PyTorch
dependency. String metadata is saved as fixed Unicode arrays
(`np.array(s)`) — NOT `dtype=object` — so loading does not need
`allow_pickle=True`. Decode pattern in the loader:
`str(arr.item())` for scalars, `arr.tolist()` for 1-D string
arrays.

## Implementation notes

### Env isolation

```bash
mkdir -p tools/sparc
cd tools/sparc
uv venv .venv
.venv/bin/uv pip install torch transformers
# Then SPARC weights — see SPARC paper / repo for canonical install.
```

The JAX-side env never sees this. The handoff between envs is
`.npz` files.

### License + weights check (hard gate)

Capture and pin the licenses of FOUR artifacts in
`outputs/sparc_labels/LICENSE.md` before any bulk extraction:

1. **WavLM Large checkpoint** — record the actual download source and
   the license shown at THAT source. The HF model card
   `microsoft/wavlm-large` is the recommended source; copy the license
   from there verbatim, NOT from the `microsoft/unilm` repo root, as
   they may differ.
2. **SPARC package + checkpoint** — record the SPARC repo/PyPI
   license and any separate license attached to the published vocoder
   checkpoint (the speaker-embedding FFN lives inside it).
3. **CREPE / torchcrepe** — record license.
4. **Source corpus** — LibriTTS-R (CC-BY-4.0) and any per-corpus
   terms.

Bar for "pass": all four permit non-commercial research use and
internal label generation. The worker is NOT required to audit
transitive pip dependencies. If any of the four prohibits this use:
**stop and surface to developer** — the fallback per Risk B in the
course-correction doc is HuBERT-base or WavLM-base as a weaker
teacher, and is a developer decision.

### Speaker-embedding checkpoint dependency

The 64-d speaker embedding requires the FFN weights from the SPARC
HiFi-GAN vocoder checkpoint, NOT WavLM alone. The published SPARC
`Coder.encode()` API returns the embedding directly — use it. Do not
try to reconstruct just the FFN. If the vocoder checkpoint is
unreachable, this is an immediate STOP-and-surface (matches the "no
audio-only hacks" guard).

### Determinism controls

The extraction script MUST set, before any model load:

```python
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
```

All models in `model.eval()`, all forward passes in `torch.no_grad()`,
fp32 throughout (no autocast). Record the resulting
`torch_determinism` tag in each `.npz`. With this profile, GPU runs
SHOULD be near-bit-equal across re-runs on the same machine; Gate 4
gates on cosine similarity (not strict bit-equality) to allow for
benign floating-point reduction-order differences.

### Checkpoint hashing

Hash once per script invocation, before the extraction loop:
- Resolve actual local file paths for both checkpoints (HF cache for
  WavLM, SPARC's checkpoint download path for the SPARC vocoder).
- SHA-256 each file. Log `relative_path=sha256` to stderr at startup.
- Cache the hash strings in memory; copy into every `.npz` written
  in this run. Do NOT re-hash per utterance.
- If a checkpoint resolves to multiple files, hash each and join as
  `path1=hash1:path2=hash2:...` in declared order.

### Sanity check before bulk extraction

Run on `juice_lemons_ref.wav` first. Inspect:
- Are the EMA channels in physically reasonable ranges (no NaN,
  bounded magnitudes)?
- Does F0 match what PyWorld DIO extracted for Task 001?
- Is the speaker embedding deterministic across two runs?

If any of these are off, fix before running on the full corpus.

### Provenance discipline

Every `.npz` carries:
- `sparc_version` — git SHA or pip version
- `wavlm_version` — checkpoint hash
- `extraction_timestamp` — ISO 8601
- `audio_path_hash` — SHA-256 of source audio for traceability

This matches the project's existing fixture-provenance conventions.

## Acceptance gates

### Gate 1 — Env isolation verified

Two checks, both must pass:

1. `grep -rE "import torch|from torch" tractus_controller/ tractus_engine_advanced/`
   returns only test files (if any) and never the body-model or
   pipeline code.
2. From the JAX env, both of these succeed:
   ```bash
   tractus_controller/.venv/bin/python -c "from tractus_controller.data.sparc_labels import load_sparc_labels; d = load_sparc_labels('outputs/sparc_labels/probe/juice_lemons_ref.npz'); print(sorted(d.keys()))"
   tractus_controller/.venv/bin/python -c "import sys; import tractus_controller.data.sparc_labels; assert 'torch' not in sys.modules and 'transformers' not in sys.modules, sorted(k for k in sys.modules if k.startswith(('torch','transformers')))"
   ```

### Gate 2 — Round-trip on Week-1 phrase

Load `juice_lemons_ref.npz` from the JAX-side loader and run:

1. **EMA sanity:** all 12 channels finite, no NaN/Inf. Plot the 12
   channels over time, save to `outputs/sparc_labels/probe/ema.png`.
   Print per-channel value range. If
   `ema_normalization == "z_score_within_utterance"`, expect roughly
   N(0, 1) over the utterance. (Note per `docs/incorrect_assumptions.md`
   A5: this value is almost certainly structurally unavailable — SPARC
   discarded per-utterance scale during training; the check is left
   for completeness but expect `"sparc_api_native"` in practice.) If
   `"sparc_api_native"` and the distribution is non-obvious, write
   empirical stats to `ema_stats.json`.

2. **F0 comparison vs PyWorld DIO:**
   - Extract DIO F0 at 5 ms frame period (200 Hz), then average-pool
     by 4 to 50 Hz to align with the SPARC grid.
   - tractus unvoiced mask convention: `f0_periodicity < 0.4` (our
     downstream convention, not SPARC's API default — see
     `docs/incorrect_assumptions.md` A8; SPARC source defaults to 0.0).
   - DIO unvoiced mask: `dio_f0 <= 0` (DIO returns 0 for unvoiced —
     not a separate flag).
   - Compute median relative error
     `|f0_sparc - f0_dio| / f0_dio` on the **voiced-overlap frames
     only** (both masks unvoiced=False).
   - **Pass criteria:**
     - Median relative error ≤ 10% on voiced-overlap.
     - Voiced-overlap ≥ 50% of total frames for clearly-voiced clips
       (Week-1 probe is voiced-heavy → hard gate). For
       unvoiced-heavy clips, "inspect and justify" rather than
       hard-fail.
   - Save plot overlaying both contours to
     `outputs/sparc_labels/probe/f0_compare.png`.

3. **Loudness sanity:** monotonic with audio RMS energy on a
   held-out 200 ms window. Plot to
   `outputs/sparc_labels/probe/loudness.png`.

### Gate 3 — Bulk extraction completes

50-100 utterances extracted with no failures. `manifest.json`
listing matches actual file set. Per-utterance manifest entry
records original SR, output T count, and (on failure) reason +
duration so a debug pass doesn't need to re-derive context.

### Gate 4 — Speaker embedding stability

Run extraction on `juice_lemons_ref.wav` twice within the same
process and once in a fresh subprocess. For each pair, compute
cosine similarity AND `np.array_equal` (informational).

- Same-process: cosine ≥ 0.99999 (gate). `array_equal` reported
  informationally — bit-equality is nice-to-have, not required.
- Fresh-subprocess: cosine ≥ 0.999 (gate). Allows for nondeterministic
  checkpoint load order.

If same-process cosine drops below 0.99999, the determinism profile
(see "Determinism controls") failed — investigate before bulk
extraction.

## Anti-deliverables

- **No SPARC EMA → TEA cross-section mapping in this task.** That
  is the body model, owned by Task 003. Task 002 freezes labels in
  SPARC's native coordinate system; downstream consumers handle
  reprojection. Do not pre-bake any tube-area mapping here even as
  scaffolding. (Note: TEA itself is a Maeda-style 1-D simulator,
  so the previous "no Maeda bridge" wording was misleading — the
  prohibition is specifically against bridging *here*, not against
  Maeda-family geometry generally.)
- No PyTorch in the JAX env. Period.
- No re-deriving SPARC from scratch. Use the published model.
- No per-frame "physical interpretation" of EMA channels. They're
  features; the body model in Task 003 learns the mapping.
- No re-deriving SPARC's loudness or F0 — save the API outputs as
  returned, even if they disagree with the paper text.

## Reporting

Append to `workplan/LOG.md`:

```
## YYYY-MM-DD — Week 2 SPARC extraction

- Env isolation:    pass | fail
- Round-trip:       pass | fail
- Bulk extraction:  __ / __ utterances
- Speaker stability: cosine = __
- License check:    pass | fail
- Verdict:          GREEN | RED | BLOCKED-ON-LICENSE
```

If GREEN: Task 003 unblocked. If license-blocked: surface to
developer for fallback teacher decision.

## References

- SPARC paper: `docs/references/core/2024_sparc_vocal_tract_kinematics.pdf`
- Course correction: `workplan/handoff/2026-05-09_course_correction.md`
- Codex thread: `[session identifier omitted]`
