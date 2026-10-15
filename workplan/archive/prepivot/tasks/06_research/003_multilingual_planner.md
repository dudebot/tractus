# research/003 — Multilingual planner

**Phase:** research    **Status:** pending    **Effort:** L
**User-input:** none
**Blocked by:** P4 closed
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §13 open question. v0 planner is English
(LibriTTS-R / VCTK / etc.). Multilingual extension affects Module 1
only; downstream modules (controller, body model, TEA) are
language-agnostic by design (they consume phones + prosody, not
text).

This is research-tier because:
- Multilingual G2P + duration + F0 corpora are heavier to curate.
- Not all phones in the world's languages map cleanly to the
  current ARPAbet inventory; some languages need additional
  named shapes or finer body-state coverage.

## Deliverable (high-level scope)

1. Survey existing multilingual TTS corpora (Common Voice, M-AILABS,
   Hi-Fi TTS, etc.) for license + alignment availability.
2. Pick 3 additional languages to start (e.g. Spanish, Mandarin,
   Hindi — diverse phonological inventories).
3. Extend G2P to those languages. Espeak handles many; gaps
   identified.
4. Extend phone-to-gesture template inventory: which languages
   require new named shapes the default speaker doesn't have? File against
   the body-model / anatomy basis if so.
5. Train multilingual planner head; evaluate per-language
   intelligibility.

## Acceptance gates (automated)

- ✅ 3 languages render via the rule-based smoke path with
  language-conditioning.
- ✅ Whisper-multilingual gating shows ≥ 60% **word-WER** on each
  language (Whisper is word-level; phone-WER would require
  language-specific MFA aligners — note availability per language
  in the report). Quality is lower than English; that's expected at
  v1.
- ✅ Language-coverage report identifies any phone classes the
  current body-model + anatomy basis can't reach, with severity
  (e.g., does Mandarin require novel named shapes vs only timing
  changes?).

## Anti-deliverable / scope guard

- Do NOT block on perfect quality. v1 multilingual is "intelligible
  in all 3 languages," not "11labs-tier."
- Do NOT extend the body model in this task. Identified gaps are
  filed as separate body-model coverage tasks.

## Notes

Research-tier; product priorities will determine when this lands.
