# smoke/001 — Rule-based G2P + template gestures + TEA, regression smoke

**Phase:** smoke    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** smoke/002
**Blocked by:** P0 closed (all correctness/* + infra/001 + infra/003)
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §10 P1 (TEA-only speech smoke), §1.2.1
(non-speech allowed but not gated). End-to-end text → audio path is
broken since the C++ front-end was deleted. This task restores a
**rule-based** smoke path so we can hear something the simulator
produces before any learned modules exist.

Pipeline:
```
text → G2P (rule-based or borrow Misaki/espeak)
    → ARPAbet phones + durations (rule-based)
    → template GestureEvents (one per phone, no overlap, simple
      onset/duration/release/target from the default named-shape inventory)
    → body model R1 NN (the saved checkpoint, body-model baseline)
    → TEA at fp32 (post-correctness/004)
    → audio
```

This is the "Daisy, Daisy" regression anchor — no learned modules,
no anatomy fitting, no surrogate. Just text → physical state →
sound.

## Deliverable

1. CLI smoke command: `tractus-controller smoke --text "..." --out
   <path>` (or equivalent). Sample rate 22.05 kHz default. Add a
   pointer to the new `smoke` verb in ARCH §8.2 in the same commit.
2. G2P: pure-Python ARPAbet via existing
   `tractus_controller/phonemes/` + espeak fallback. NO
   NotImplementedError stubs.
3. Duration: rule-based per-phone, first-pass values from common
   L1-EN ranges (vowels ~120ms, stops ~80ms, fricatives ~150ms, etc.).
   Configurable via JSON; refine when a corpus is in hand.
4. Template gestures: each phone maps to a small list of
   `GestureEvent`s drawn from the default named-shape inventory
   (the default named-shape inventory). E.g.
   /a/ → "jaw open + tongue body low + lips relaxed" gestures.
5. Body model: use the existing `adapters/r1_nn.py` checkpoint as-is.
   Per ARCH §1049, §1126-1128 it's undertrained at 26K rows and
   needs 1M+ for the 1e-3 audio parity gate; failures here may
   indicate the body-model floor, not a pipeline bug. Flag floor
   issues for re-baseline (regenerate corpus from TEA forward) rather
   than retuning the smoke.
6. Render via `synthesize_dynamic_tube` at fp32. WAV output.
7. Sidecar JSON per WAV with this schema:
   `{"git_hash": str, "config": {...}, "gesture_list": [...],
   "anatomy": "default" | str, "input_text": str,
   "consent_verified": bool, "deterministic_seed": int}`. Per
   ARCH consent/provenance rule.
8. Regression set: 20 fixed utterances including "Daisy, Daisy",
   pangrams, fricative-heavy phrases (covering /s/ /ʃ/ /f/ /θ/ /z/
   /v/ + aspiration + bursts so correctness/004 can reuse them), and
   the phonetic-coverage prompt set from `ARCHITECTURE.md` §4.

## Acceptance gates (automated)

- ✅ `tractus-controller smoke --text "Daisy, Daisy, give me your
  answer do" --out /tmp/daisy.wav` produces a WAV without
  exception.
- ✅ All 20 regression utterances render. WAV files produced;
  no NaN, no Inf, RMS in [0.01, 0.5].
- ✅ Sidecar JSON per WAV is well-formed and contains required
  fields.
- ✅ `runbooks/whisper_gate.sh` against the 20 utterances reports
  per-utterance WER. (Pass criterion deferred to smoke/002.)
- ✅ Wall-clock: end-to-end text → 5s audio in ≤ 30s on CPU.
- ✅ Deterministic-seed run produces bit-equal audio across two
  invocations (regression-prevention smoke).

## Anti-deliverable / scope guard

- Do NOT train any model in this task. Body model is the existing
  R1 NN; controller is rule-based templates.
- Do NOT implement R2/R3 from the legacy plan. The new architecture
  bypasses that contract via the GestureEvent representation.
- Do NOT optimize quality. Smoke = "speaks without crashing"; the
  intelligibility/identity bar is smoke/002 + later phases.

## Notes

The interesting design call here: how rich are the rule-based phone
templates? Recommendation: keep them minimal (one anchor gesture
per phone, no coarticulation). Coarticulation emerges later from
overlap in pretraining/003. Worse smoke audio now is better than
fragile templates that hide controller-design issues later.
