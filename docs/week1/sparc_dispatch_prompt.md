# Week 2 — SPARC supervision dispatch prompt

> Copy-pasteable opening message for a fresh Claude Code session (or
> `general-purpose` subagent with `isolation: "worktree"`). Frames
> Week 2's first concrete task. Worker reads the spec, builds the
> extraction pipeline, gates against acceptance, reports verdict.

---

```
You are picking up tractus Week 2. The project is a physics-first
TTS / voice-cloning system built around TEA, a differentiable
1-D transmission-line vocal-tract simulator written in JAX. Week 1
just closed; read `docs/week1/findings.md` first. The TL;DR for
your context:

Week 1 proved that gradient descent through TEA can recover
**speech-like audio** from a reference clip via log-mel + MR-STFT
loss, but **cannot recover the correct articulation** even when given
a TEA-renderable synthetic target. The mechanism is
articulatory-acoustic non-uniqueness: many tract shapes produce
similar acoustic features, and acoustic loss has no preference
between them. On the synth-/A/ probe, the recovered tube had a
narrow front-of-mouth where ground truth was wide (log-MAE 0.87
vs truth, Whisper transcribed as "Uh" — close, not /A/).

Conclusion: kinematic supervision is structurally required. Week 2
is about pre-extracting SPARC articulatory + speaker labels on a
small corpus, frozen as .npz artifacts, that the JAX training
loop will consume in Week 3.

> **[Note per `docs/incorrect_assumptions.md` A4]** "Structurally
> required" is the Week-1 closure claim, but Codex adversarial review
> rated it UNDETERMINED pending a converged, frication-capable negative
> result. Task 007 is the discriminating experiment. SPARC is the planned
> Week-2 path regardless, but treat this claim as strong-prior, not
> proved fact.

Your task is `workplan/tasks/00_active/002_sparc_extraction_pipeline.md`.
Read it end-to-end before writing any code. The acceptance gates
are in §"Acceptance gates"; gate yourself against them before
reporting done.

Critical SPARC-side facts the task spec calls out — surface here so
you don't get caught:

- **Sample rate is 16 kHz mono.** WavLM Large expects 16 kHz input.
  Non-16-kHz input does NOT cause a hard error — WavLM silently
  processes it, producing frequency-shifted and frame-rate-shifted
  features that look plausible but are wrong (see
  `docs/incorrect_assumptions.md` A7). SPARC's `Coder.encode()` API
  resamples internally, so the v0 SPARC pipeline is safe; the risk
  is any future code path that bypasses `Coder.encode()` and feeds
  WavLM directly. Source corpora (LibriTTS-R 24 kHz, VCTK 48 kHz,
  dev recordings varied) MUST be resampled before any direct WavLM
  ingestion.
- **EMA channel order is SPARC API native:**
  `[TDx, TDy, TBx, TBy, TTx, TTy, LIx, LIy, ULx, ULy, LLx, LLy]`.
  Do NOT reorder. Saving them in `[UL, LL, LI, TT, TB, TD]` order
  would silently mis-train Task 003.
- **EMA scale is unknown until verified.** SPARC was trained with
  z-scored EMA but the API may or may not de-normalize. Inspect
  SPARC source; if you can't confirm, save as `"sparc_api_native"`
  with empirical stats. Do NOT label as `"physical_mm"` on a guess.
- **Speaker embedding requires the SPARC vocoder checkpoint.** The
  64-d FFN that produces the embedding lives inside the HiFi-GAN
  vocoder, jointly trained. Use `Coder.encode()`; don't try to
  reconstruct the FFN from WavLM alone.
- **Loudness + F0 from SPARC API as-is.** Don't re-derive from the
  paper text — trust the API output if they conflict.
- **F0 vs DIO comparison needs voiced-only masking** (DIO returns 0
  for unvoiced, not a flag) and 200→50 Hz alignment. Median relative
  error ≤ 10% on voiced-overlap is the gate, not raw all-frame ±10%.
- **Determinism profile is mandatory** (TF32 off, cuDNN benchmark off,
  `eval()`, `no_grad()`, fp32). See task-spec §"Determinism controls".
  Without it, Gate 4 is not testable.

Scope guards (anti-deliverables from the task spec, plus fresh ones
learned during Week 1):

- **SPARC runs TRAINING-TIME ONLY.** Never imported in the JAX
  inference stack. WavLM Large is 1.26 GB / 316 M params; it blows
  the project's ≤8 GB inference budget on its own. Build SPARC
  extraction in a SEPARATE PyTorch venv at `tools/sparc/.venv/`.
  The JAX side consumes only the frozen .npz outputs. **No PyTorch
  imports in `tractus_controller/` or `tractus_engine_advanced/`,
  period.**
- **License + weights check is a hard gate.** Capture FOUR licenses
  in `outputs/sparc_labels/LICENSE.md`: (1) WavLM Large checkpoint —
  copy from the actual download source (HF model card, not unilm
  repo root; they may differ), (2) SPARC package + checkpoint, (3)
  CREPE/torchcrepe, (4) source corpus. Bar: all four permit
  non-commercial research use. If any prohibits, STOP and surface —
  fallback teacher (HuBERT-base / WavLM-base per Risk B in the
  course-correction doc) is a developer decision.
- **No re-deriving SPARC from scratch.** Use the published model
  weights from the SPARC repo. If those weights are pulled or
  inaccessible, surface.
- **No "lets just use audio" hacks** to bypass SPARC. The whole
  point of Week 2 is to add a non-acoustic supervision signal. If
  SPARC blocks for any reason, surface to the developer; do not
  silently substitute mel features or vocoder outputs.
- **Sanity check first, bulk extraction second.** Run on the
  Week-1 probe (`data/probes/juice_lemons_ref.wav`) and one
  developer recording before processing 50-100 LibriTTS-R
  utterances. Inspect the EMA channels visually, compare F0 vs
  PyWorld DIO baseline.
- **Provenance discipline.** Every .npz carries `sparc_version`,
  `wavlm_version`, `extraction_timestamp`, `audio_path_hash`.
  Matches existing fixture conventions.

GPU / hardware notes that bit Week 1:

- CUDA-enabled JAX is in `tractus_controller/.venv/`, NOT the
  engine venv. Don't run two GPU processes concurrently — caused a
  6.5-hour hang in Week 1.
- For SPARC: WavLM Large needs roughly 3-4 GB VRAM in fp16 inference
  mode. Extraction is one-shot offline, so contention isn't an issue
  during extraction itself; just don't run alongside copyfit later.

Reference material:

- Course-correction doc:
  `workplan/handoff/2026-05-09_course_correction.md`
- Architecture: `ARCHITECTURE.md` §2.5 (Speaker Inference) and §3.1
  (SPARC as training-time teacher).
- SPARC paper: `docs/references/core/2024_sparc_vocal_tract_kinematics.pdf`.
- Week-1 findings + open issues: `docs/week1/findings.md`.
- Existing Codex thread to continue if useful:
  `[session identifier omitted]` (has Week-1 stall +
  non-uniqueness data in context; recommended for Week-2 design
  questions).

Implementation order (suggested):

1. Read `002_sparc_extraction_pipeline.md` in full.
2. Set up `tools/sparc/.venv/` with PyTorch + transformers + SPARC
   weights. Document the dependency tree in `tools/sparc/README.md`.
3. Run the license + weight verification gate. Halt if any concern.
4. Write `scripts/extract_sparc_labels.py` — single-utterance smoke
   test on `data/probes/juice_lemons_ref.wav` first, then accept a
   manifest.
5. Write the JAX-side loader at
   `tractus_controller/tractus_controller/data/sparc_labels.py`.
   Zero PyTorch deps; pure NumPy load.
6. Gate-2 sanity round-trip on the Week-1 probe.
7. Bulk extraction on a small LibriTTS-R subset (50-100 utterances)
   + 10 developer-recorded clips if available.
8. Acceptance gates 1-4 per the task spec.
9. Report verdict by appending to `workplan/LOG.md` with the
   template format (Week 2 task 002, gates pass/fail, env isolation
   verdict, license-check verdict, total utterances extracted).

If you hit a wall and your hypothesis stack doesn't unstick it,
spawn a fresh Codex thread (NOT a continuation of the Week-1 thread
unless your question is directly about non-uniqueness data). Use
the verification-mode framing: "here's my plan/code, poke holes;
KEEP/DROP per gate."

Begin by reading `docs/week1/findings.md` and
`workplan/tasks/00_active/002_sparc_extraction_pipeline.md`. Then
report back what you've understood about the scope and what you
intend to build first.
```
