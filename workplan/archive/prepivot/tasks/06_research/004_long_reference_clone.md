# research/004 — Long-reference cloning vs ElevenLabs IVC/PVC

**Phase:** research    **Status:** pending    **Effort:** M
**User-input:** review (listener panel for PVC comparison)
**Blocked by:** P4 closed; cloning/005 results determine whether this
makes sense to pursue
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §1.4. v0 target is 30s reference; v2
target is 1–3 min (more robust IVC) and 30+ min (PVC comparator).
This task pushes tractus beyond 30s.

ElevenLabs guidance: IVC works ~30s–2min reference; PVC requires
30–180 min. Matching their PVC tier is a different research
question — fine-tuning a dedicated model on the speaker — and is
NOT what tractus does. Tractus is closer to IVC architecturally.
A fair PVC comparison would need tractus to also fine-tune per
speaker, which is its own multi-month investment.

## Deliverable

1. Re-run cloning/002's pipeline at 1 min, 3 min reference for the
   same 5 speakers.
2. Compare:
   - tractus at 30s vs 1 min vs 3 min.
   - tractus at 1 min vs ElevenLabs IVC at 1 min.
   - tractus at 3 min vs ElevenLabs IVC at 3 min.
   - (Stretch) tractus at 30 min (with per-speaker fine-tuning of
     anatomy + motor adapter, if pretraining/* supports it) vs
     ElevenLabs PVC at 30 min — clearly labeled as architectural
     comparison NOT v0 target.
3. Report: per-reference-length identity gain table; identifies
   the reference-length sweet spot.

## Acceptance gates (automated for objective; user review for listener)

- ✅ tractus at 1 min outperforms tractus at 30s on identity
  cosine by ≥ 0.05 absolute (more reference = more anatomy signal).
- ✅ Anti-cheat refusal stays ≥ 80% at all reference lengths.
- ✅ Report identifies the reference-length sweet spot (where
  identity cosine plateaus relative to additional reference).
- ⏸️ **User review (listener panel)**: at 1 min reference, tractus
  selected as "more like reference" by ≥ 35% of listeners vs
  ElevenLabs IVC at same length.
- ⏸️ Optional PVC stretch goal: documented architectural delta
  (we don't fine-tune dedicated models per speaker; PVC does);
  results presented with that caveat front-and-center. Per-speaker
  fine-tuning would require new training infrastructure beyond
  pretraining/* — file as a separate research item if pursued.

## Anti-deliverable / scope guard

- Do NOT misrepresent tractus as a PVC-tier product. Architecture
  is closer to IVC.
- Do NOT cherry-pick reference lengths that flatter tractus.
- Do NOT ship a PVC fine-tuning pipeline as part of this task —
  that's a separate multi-month research item.

## Notes

This task closes the headline-quality story for the project's
public framing. Outcomes here directly drive what we say about
tractus vs commercial offerings.
