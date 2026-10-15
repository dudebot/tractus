# pretraining/006 — SPARC bootstrap as body-model teacher (decision deferred)

**Phase:** pretraining    **Status:** pending    **Effort:** M
**User-input:** **decision** (yes/no on adopting SPARC as body-model teacher)
**Blocked by:** pretraining/002b (R1-distill baseline lands so we have a
baseline to compare against)
**Blocks:** pretraining/002 v2 (if SPARC adopted)
**How to dispatch:** fresh Claude Code session for analysis;
user-decision step at end

**Re-prioritization + framing note (2026-05-08, updated 2026-05-09):**
Codex review (thread `[session identifier omitted]`) on the
abstract recommendation: move SPARC analysis EARLIER in the dependency
order — before pretraining/002 v1, not just before v2. Treat the
previous "pretraining/002 first iteration" prerequisite as satisfied by
pretraining/002b once it lands.

Codex's deeper, fact-grounded follow-up corrected the framing:
**SPARC outputs articulatory KINEMATICS (12-d EMA + pitch + loudness
+ 64-d speaker emb), not TEA renderer-interface targets.** It cannot
substitute as the primary teacher for `target_renderer_interface`
fields (tube area/length, glottis state, frication, walls). What it
CAN do is supervise an auxiliary head on a body-state latent + serve
as a real-audio gesture-plausibility signal at scale.

Adoption framing per the corrected analysis:
- Story 1996 + R1 fixtures + (eventual) inversion-fits → **primary**
  renderer-interface supervision.
- SPARC → **auxiliary** supervision on a body-state latent +
  gesture-plausibility loss.

Adoption gate: after a 10K SPARC-aux corpus slice, the body model
must show improved TEA-facing validation (anchor formant panel /
WER on a held-out renderer-target corpus), NOT just lower SPARC-feature
MSE.

---

## Context

`../../../ARCHITECTURE.md` §2.3 teacher strategy + §13 open question
(f). SPARC (`Coding Speech through Vocal Tract Kinematics`,
arXiv:2406.12998) provides articulatory + source features from real
audio that aren't TEA-circular. Could bootstrap body-model coverage
across many speakers without heavy manual data work.

External-dependency adoption decision — affects whether v1 body
model uses SPARC features as inductive bias / additional supervision.

## Deliverable (analysis phase)

Per Codex 2026-05-09 (verified facts to skip re-deriving):
- Code license: **MIT** (compatible with project's GPL-3.0).
- Repo: `Berkeley-Speech-Group/Speech-Articulatory-Coding`. PyPI:
  `speech-articulatory-coding 0.1.0` (released 2025-01-29).
- Output schema: `ema (L,12)` at 50 Hz with channels
  `[TDX, TDY, TBX, TBY, TTX, TTY, LIX, LIY, ULX, ULY, LLX, LLY]`,
  plus `loudness (L,1)`, `pitch (L,1)`, `periodicity (L,1)`,
  `pitch_stats`, `spk_emb (64,)`, `ft_len`.
- Heavy deps: torch, torchaudio, transformers, **WavLM Large
  (1.26 GB checkpoint, 16 kHz input)**, torchcrepe, penn, librosa,
  huggingface-hub.
- Anatomy normalization: SPARC traces are z-scored within utterance
  (anatomy-normalized MNGU0 template space). Translation to
  physical cm-scale gestures is required and lossy.

Remaining open items the analysis must resolve:

1. **Model-weights license check** (HARD GATE — Codex flagged this
   as not-yet-verified). PyPI / repo declare MIT for code; the
   downloaded model checkpoint files may carry separate
   acceptable-use restrictions. Confirm explicitly before any
   derived training artifact is committed or shipped. Record the
   exact license name + URL.
2. Run SPARC inference on a representative subset (20 utterances
   from VCTK + LibriTTS-R). Record install footprint, runtime
   per utterance, GPU memory.
3. Map SPARC output channels onto the body-model gesture vocabulary
   (`tractus_controller/tractus_controller/body_model/channels.py`):
   - tongue tip / body / lips / jaw → mostly direct (10 of 12 EMA)
   - velum, larynx, pharynx, glottal adduction, voicing-quality
     → NOT covered by SPARC; must come from other sources
   Document the per-channel coverage map.
3a. Compare SPARC features to Story 1996 anchor renders (from
    pretraining/002a) and / or R1 fixtures on the same audio. Are
    they aligned? Are they different in ways that suggest SPARC adds
    signal?
4. Estimate effort to use SPARC features as auxiliary supervision
   in pretraining/002:
   - Add a SPARC-feature-prediction head on the body-model latent.
   - Add SPARC-prediction loss term.
   - Holdout evaluation on speakers SPARC didn't see in training.
5. **Decision report**: pros/cons + estimated cost/benefit. Codex
   estimates: 2 sessions for analysis only, 3-5 for bridge + 10K
   corpus slice, 6-8 for full adoption. Adoption gate: SPARC
   auxiliary loss must improve TEA-facing validation (anchor
   formant panel / WER), NOT just lower SPARC-feature MSE.

## Deliverable (if user decision is YES)

6. Implement SPARC-feature auxiliary loss in pretraining/002 v2.
7. Re-train body model with SPARC supervision.
8. Re-run inversion/004's holdout test to verify the gain.

## Acceptance gates (automated for analysis)

- ✅ Model-weights license confirmed in writing in the report. If
  the weights carry restrictions incompatible with derived-artifact
  publication, the task halts here regardless of feature quality.
- ✅ SPARC inference works on the project's audio format (16 kHz
  resample tested; 22.05 / 24 kHz upstream is the source).
- ✅ Channel-coverage map: which body-model gesture channels SPARC
  covers, which it doesn't.
- ✅ Comparison report produced: SPARC features vs Story-1996 anchor
  renders (from 002a) on matching utterances, with per-feature
  alignment metrics.
- ✅ Cost/benefit estimate concrete: dependency footprint, training
  time delta, expected held-out gain on TEA-facing metrics.
- ⏸️ **User decision**: adopt as auxiliary supervision or defer.

## Acceptance gates (if YES on adoption)

- ✅ pretraining/002 v2 retrained with SPARC supervision.
- ✅ TEA-facing validation improves vs without SPARC: anchor-formant
  panel (≥ 9/12 vowels within 15% on F1+F2) AND held-out log-mel
  / WER on a renderer-target corpus. SPARC-feature MSE alone does
  NOT count — the gate is whether the auxiliary signal flows
  through to better physically-rendered audio.
- ✅ inversion/004's holdout test improves by ≥ 1 dB log-mel
  vs without SPARC supervision (secondary gate).
- ✅ Cross-speaker generalization: 5 speakers held out from SPARC's
  own training corpus pass inversion/004.

## Anti-deliverable / scope guard

- Do NOT replace TEA with SPARC's synthesizer. SPARC provides
  features; TEA provides physics. We're using SPARC as a
  ground-truth-ish teacher signal, not as the renderer.
- Do NOT couple the architecture to SPARC. The interface is
  "auxiliary loss," not "required feature."
- Do NOT sneak SPARC into the inference path. Encoder uses SPARC
  during training only; SPARC features appear in cloned-voice
  provenance metadata, NOT as a runtime dependency. CI guardrail
  (infra/002 doc-coherence + non-circularity) enforces.

## Notes

This is a decision-point task by design. The validator session
shepherds the decision: review the analysis, present cost/benefit,
get user choice, optionally proceed to implementation. The
implementation half is only triggered by YES.
