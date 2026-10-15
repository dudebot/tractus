# Task 005 — Constrained articulatory-TTS demo (Week 4)

> **Blocked by:** Task 004 GREEN (or RED-DEMO-DEGRADED with explicit
> developer sign-off). **Read first:**
> `workplan/handoff/2026-05-09_course_correction.md`.

## Goal

Ship the v0 demo: text → 5 unseen Harvard sentences rendered in the
developer's voice, with anatomy + source + speaker_vec fitted from
30 s of reference audio. Compare honestly against
DDSP-articulatory and SPARC-vocoder baselines at the same
constraint level.

This is **not** zero-shot 30-s cloning — that's a research push
post-v0. v0 ships at "anatomy + source fitting from 30-s reference
audio of one consented speaker."

## Inputs

- 30 s of developer-recorded reference audio: read passage,
  multiple phonetic environments, mono 22.05 kHz. Save as
  `data/probes/dev_reference_30s.wav`.
- 5 Harvard sentences for demo (NOT recorded by developer — these
  are pure synthesis):
  1. "The juice of lemons makes fine punch."
  2. "Four hours of steady work faced us."
  3. "A large size in stockings is hard to sell."
  4. "The boy was there when the sun rose."
  5. "Open the crate but don't break the glass."

## Pipeline

```
30s reference → speaker fit:
  - SPARC speaker_emb extracted (from Task 002 pipeline)
  - Body-model speaker_vec optimized (~1K steps)
  - Source defaults extracted (mean F0, tilt, breathiness)
  - Channel model optional, off by default

5 Harvard sentences:
  text → planner → controller → body model → TEA → WAV
```

## Baselines for comparison

Run the same 5 sentences through:

1. **TEA-only (this project)** — the deliverable above.
2. **DDSP-articulatory** — published 0.4 M-param model. Run
   inference via the published checkpoint if available; if
   unobtainable, document and skip.
3. **SPARC vocoder** — SPARC's published synthesis pipeline.
4. **TEA + small post-filter (ablation)** — only if Module 6B
   residual is wired up; otherwise skip.

For each: record audio, transcript, Whisper WER, listener A/B vs
the developer's reference voice.

## Acceptance gates

### Gate 1 — Intelligibility

All 5 phrases pass Whisper at WER ≤ 0.5 averaged over the set.
Single-phrase WER ≤ 0.7 (no individual outlier worse than 70 %).

### Gate 2 — Audible articulation

For each phrase:
- Audible fricatives where the transcript has them (`/s/, /ʃ/, /f/`).
- Audible stops with discernible burst/release.
- Vowel quality not collapsed (varies across phonemes).

This is a listening gate; document with spectrogram plots.

### Gate 3 — Speaker similarity

2/3 listener A/B test on at least one phrase: TEA-only render vs a
random-other-voice render. Listener picks "which sounds more like
the reference voice." Pass on at least one.

(Don't expect full-blown cloning. v0 ships if listeners can
identify the voice as the developer's at all.)

### Gate 4 — TEA-only (no post-filter)

The TEA-only path passes Gate 1 without any residual / post-filter.
If a post-filter is wired up, ablation reports both.

### Gate 5 — Compute budget

End-to-end inference of one phrase fits in 8 GB VRAM. Wall-clock
per phrase ≤ 5× realtime. (We're not real-time yet; we're not
painful either.) (Note per `docs/incorrect_assumptions.md` A6: the
8 GB figure is an aspirational positioning target without a formally
sourced hardware or product constraint; treat as a reasonable
planning gate, not a hard requirement with a known source.)

### Gate 6 — Speaker object size

`outputs/speakers/dev.npz` ≤ 2 KB.

## Demo report

Land `workplan/handoff/2026-MM-DD_v0_demo.md` with:

- All 5 phrases as WAVs in `outputs/v0_demo/`.
- Spectrograms for each phrase, all four pipelines side-by-side.
- Whisper transcripts for each.
- Listener A/B raw notes.
- Compute profile per phrase.
- Honest comparison: where does TEA win, where does it lose vs
  SPARC / DDSP-articulatory?
- Forward-look: what would a v1 push add?

## Kill criterion

If Gate 1 fails (TEA-only WER > 0.5 average over 5 phrases) and
DDSP-articulatory baseline passes:

- v0 demo ships **with DDSP-articulatory as the runtime path** and
  TEA documented as a research-grade differentiable simulator
  used during training only.
- The "physics-first" framing becomes "physics-anchored
  articulatory TTS with neural production rendering."

This is the exit ramp. Document it in the demo handoff if invoked.

## Anti-deliverables

- No new corpus, language, or speaker beyond the developer.
- No real-time streaming. v0 is offline.
- No GUI or app shell. Command-line + WAV files.
- No marketing copy. Demo handoff is technical.
- No "almost works → ship anyway." If gates fail, exercise the
  kill criterion and ship the honest version.

## Reporting

`workplan/LOG.md`:

```
## YYYY-MM-DD — v0 demo

- Gate 1 (intelligibility): __ / 5 pass, avg WER __
- Gate 2 (articulation):    pass | fail
- Gate 3 (speaker A/B):     pass on __ / 5 phrases
- Gate 4 (no post-filter):  pass | fail
- Gate 5 (compute):         __ GB peak, __× realtime
- Gate 6 (speaker size):    __ KB
- Path:                     TEA-only | DDSP-fallback
- v0 verdict:               SHIP | KILL-AND-PIVOT
```

## References

- Course correction: `workplan/handoff/2026-05-09_course_correction.md`
- DDSP-articulatory paper: `docs/references/baselines/2024_ddsp_articulatory_vocoder.pdf`
- SPARC paper: `docs/references/core/2024_sparc_vocal_tract_kinematics.pdf`
- RT-VC: `docs/references/baselines/2025_rt_vc_articulatory_voice_conversion.pdf`
