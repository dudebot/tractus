# cloning/005 — Matched-reference evaluation vs commercial baselines

**Phase:** cloning    **Status:** pending    **Effort:** M
**User-input:** review (test-set design + listener panel)
**Blocked by:** cloning/002, cloning/003, cloning/004
**Blocks:** P4 close
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §10 lesson 13 + §15.5. **Don't compare
30s tractus vs 30+ min ElevenLabs Professional**. Match the
reference length and label commercial cloning mode (Instant Voice
Cloning vs Professional Voice Cloning).

`runbooks/matched_reference_eval.sh` is the harness from infra/003.

## Deliverable

1. Define test set:
   - 5 speakers from cloning/002 (consented, used for tractus fit).
   - For each, 5 held-out phrases not seen in any fit.
   - Total: 25 (speaker, phrase) pairs.
2. For each pair, render via:
   - **tractus** at 30s reference (TEA-only + TEA+residual, both).
   - **Commercial baseline at matched 30s reference** (e.g.
     ElevenLabs Instant Voice Cloning at 30s prompt).
   - Optional: commercial Professional-tier at long reference for
     ceiling reference (clearly labeled "longer-reference, do not
     use as architecture comparison").
3. Listener A/B panel:
   - Blind paired comparison.
   - ≥ 5 listeners.
   - Question: "which sounds more like the reference speaker?"
   - Tertiary question: "which sounds more natural overall?"
4. Objective metrics on every render:
   - Identity cosine (ECAPA-TDNN) vs reference.
   - WER vs ground-truth held-out transcript.
   - Per-axis source/tract separation.
   - Spectrogram inspection on sibilants + nasals (visual; flagged
     for validator review).
5. Report `outputs/matched_reference/report.md` with per-pair
   results, aggregate listener preference, fairness analysis
   (matched-reference enforced; no apples-to-oranges).

## Acceptance gates (automated for objective; ≥-prefer for listener)

- ✅ All 25 pairs rendered via both tractus and commercial-IVC,
  with the matched-reference enforcement check (commercial baseline
  used exactly the 30s reference) recorded as a hard pass/fail per
  pair.
- ✅ Identity cosine: tractus ≥ 0.9× commercial-IVC at the same
  reference length, averaged across speakers (one-sided; tractus
  over-performing is not penalized).
- ✅ WER: tractus within ±5 pp of commercial-IVC.
- ⏸️ **User review (listener panel)**: total panel size pinned at
  ≥ 5 listeners (deliverable 3); tractus selected as "more like
  the reference" by ≥ 2 of 5 listeners (≥ 40%). We're not expecting
  to dominate IVC at 30s, but matching this bar means we're in the
  same competitive territory.
- ⏸️ Optional ceiling: commercial Professional-tier at long
  reference is allowed to dominate; documented but not used as
  architecture comparison.

## Anti-deliverable / scope guard

- **Do NOT compare 30s tractus vs 30+ min commercial Professional
  and call it an architecture comparison.** That's the literal
  failure mode this task exists to prevent.
- Do NOT cherry-pick speakers or phrases. Use the same 5 speakers
  from cloning/002, same held-out phrases.
- Do NOT show commercial-Professional results without clearly
  labeling reference length.

## Notes

This task closes Phase 4 and produces the headline result. Outcome
informs research/* priorities: if tractus is competitive at IVC
tier, the next push is research/004 (long-reference Professional
comparison); if not, research/001 (procedural body model) or
research/002 (anatomy beyond 60) become priorities.
