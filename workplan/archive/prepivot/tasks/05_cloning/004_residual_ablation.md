# cloning/004 — TEA-only AND TEA+residual ablation reported every eval

**Phase:** cloning    **Status:** pending    **Effort:** S
**User-input:** none
**Blocked by:** cloning/002
**Blocks:** cloning/005
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §2.6 5B (inference residual). The
inference residual / post-filter compensates for known 1-D
acoustic-propagation limits above ~4–5 kHz (sibilants, dental,
radiation). Tightly scoped: input includes TEA state features;
output is energy-capped residual; forbidden from changing F0
contour or erasing phone identity.

Every cloning eval must report **TEA-only AND TEA+residual** so we
can verify (a) TEA-only remains intelligible without help, and (b)
residual adds value on the 1-D-unreliable upper band.

## Deliverable

1. Wire the residual module into the inference path with an
   on/off flag.
2. Modify cloning/001's and cloning/002's eval pipeline to render
   each held-out utterance twice (TEA-only + TEA+residual) and
   produce metrics for both.
3. Comparison report:
   - Per-utterance: log-mel L1 to reference, identity cosine, WER,
     per-axis source/tract separation, frication-band specific
     comparison.
   - Aggregate: residual contribution magnitude (RMS), upper-band
     specific gain (e.g. 4–11 kHz log-mel L1 with vs without).

## Acceptance gates (automated)

- ✅ TEA-only remains intelligible on all cloning/002 held-out
  phrases (WER ≤ cloning/002 threshold without residual).
- ✅ TEA-only identity cosine ≥ 0.65; TEA+residual ≥ 0.70. The
  0.05 gap is the implied "residual carries upper-band identity"
  (per ARCH §2.6 5B); both gates must pass simultaneously.
- ✅ Residual energy contribution band-limited to 4–11 kHz: ≤ 20%
  of total in-band signal energy (the cap is band-specific because
  that's where the residual is allowed to compensate; full-spectrum
  RMS is too lax).
- ✅ Residual upper-band gain ≥ 0.5 dB log-mel L1 vs TEA-only on
  the 4–11 kHz band (it's adding useful signal in the
  1-D-unreliable region).
- ✅ F0 contour drift between TEA-only and TEA+residual < 10 cents
  RMS on voiced segments (residual is not changing pitch).
- ✅ Phone-identity preserved: TEA+residual WER not worse than
  TEA-only WER by more than 5 percentage points.

## Anti-deliverable / scope guard

- Do NOT let the residual carry > 20% of signal energy. Cheat-
  channel CI gate (infra/002) catches this.
- Do NOT report only TEA+residual numbers. The TEA-only ablation
  is required by `ARCHITECTURE.md` §1.3.
- Do NOT enable the residual at training time for any module other
  than the residual itself. Modules pretraining/* train against TEA
  + surrogate, not residual-augmented audio.

## Notes

If TEA-only is unintelligible, the simulator failed and we're in
trouble — the residual would be doing real synthesis, not
compensation. That's a Phase 0 regression. Escalate.
