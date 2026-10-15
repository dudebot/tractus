# correctness/004b — Audible-frication: PSD shaping + injection-point stability

**Phase:** correctness    **Status:** pending    **Effort:** M
**User-input:** **escalation-only** — same Whisper-gate-first rule as 004a;
ping the user only if Whisper produces a genuinely ambiguous result.
**Blocks:** P0 close
**Blocked by:** correctness/004a (commit `96dee93` — physical module + 12
unit-test gates landed; module is correct, output is just inaudible)
**How to dispatch:** fresh Claude Code session

---

## Context

`correctness/004a` shipped the physical-frication module
(`tractus_engine_advanced.frication`) with all 12 unit-test gates
green: zero-constriction → zero gain, monotonicity in constriction,
voicing modulation, color cap, gradient FD agreement. The module
correctly derives `frication_gain[T, 93]` from the
`(constriction_area, pressure_drop, place, voicing,
dental_obstacle_proxy)` contract per ARCH §496-§504.

But the **audible-frication acceptance gates failed**:

1. **Whisper substitution-rate gate** (target ≥ 50% improvement):
   `faster-whisper base.en` produced empty transcripts on BOTH the
   pre-frication and post-frication smoke audio. This isn't a
   frication-module problem — it's the pre-existing smoke/002
   intelligibility floor (smoke audio peak ~0.5 / RMS ~0.03,
   transient-dominated, below Whisper's speech-detection threshold
   per ARCH §1049). Whisper gate has no measurement headroom until
   the smoke pipeline produces audio loud enough to transcribe.
2. **Spectral-band-energy gate** (target ≥ 1 dB lift in 4-8 kHz on
   80% of utterances): the diff signal IS broadband and IS phone-
   aligned (visible per-frame RMS bumps at /s/ and /ʃ/), but at
   numerically-stable MNA injection amplitudes (`noise_amp_scale ≤
   0.5 cm³/s`) the 4-8 kHz contribution is **−46 dB**, well below
   the existing signal's transient floor in the same band. Above
   0.5 cm³/s the per-sample MNA solve NaNs.

So the module is mathematically correct but produces no audible
frication because (a) the noise PSD is broadband-flat (`tanh(z)·k`)
when sibilants need 6-8 kHz peaks, and (b) the injection slot in
`prepare_time_step` (`D += d · source_amp`) is unstable above 0.5
cm³/s — way below physiological turbulence intensities.

This task closes both gaps.

## Deliverable

1. **Sibilant-shaping noise PSD.** Replace the broadband-flat
   `tanh(z) · k` source with a place-conditional spectral envelope:
   - /s/, /z/: peak at ~7 kHz, slope ~−6 dB/oct above peak.
   - /ʃ/, /ʒ/: peak at ~3 kHz.
   - /f/, /v/, /θ/, /ð/: roughly flat through ~5 kHz.
   - /h/-aspiration: low-passed white through tract resonances.
   - Implementation: per-section IIR shaping filter, coefficients
     derived from `place_index[T, 93]`-conditioned interpolation over
     a small set of canonical PSDs (one per place class). Reuse the
     SOS-biquad utility from `glottis/geometric.py` (correctness/006)
     so the filter is fp32-stable.
   - Reparameterized noise (`noise_key`-deterministic) preserved.

2. **Re-stable injection point.** The current `D += d · source_amp`
   slot in `prepare_time_step` (`tds/mna.py`) goes NaN above
   `noise_amp_scale = 0.5 cm³/s`. Two viable replacements:
   - **(a)** Move injection into the per-section branch RHS as a
     forced-flow term — bypasses the `D` matrix conditioning that
     drives the NaN at high amplitude. Read 004a's worker notes in
     the commit body of `96dee93` for context.
   - **(b)** Tikhonov-regularize the MNA matrix proportionally to
     the source amplitude so the solve stays well-conditioned at
     full physiological drive (~5-15 cm³/s for typical sibilant
     turbulence per Stevens 1971).
   - Pick whichever passes the bit-equality regression for the
     no-noise case (`source_amp = 0` → identical audio to the
     pre-004a path).
   - Validate at 5 cm³/s and 15 cm³/s; verifier `scripts/m7_frication_band.py`
     should report 4-8 kHz lift ≥ 1 dB on the fricative-rich corpus.

3. **Smoke audio level fix (orthogonal but bundled here).** The
   pre-existing smoke/002 intelligibility floor is what made the
   Whisper gate uninformative. Investigate whether the rule-based
   smoke pipeline can be peak-normalized OR the gesture-templates
   amplified so the rendered audio is at least loud enough for
   Whisper to transcribe. This is NOT the same as fixing smoke/002's
   intelligibility (that's owned by the body-model retrain in
   pretraining/002), it's just "make the audio audible enough to
   measure". A 6 dB peak-normalize at output stage would probably
   suffice; document the choice.

## Acceptance gates (automated)

- ✅ Forward bit-equality preserved at `source_amp = 0` (max abs diff
  = 0). The whole point of 004a's contract.
- ✅ All 12 of 004a's unit-test gates still pass.
- ✅ MNA solve stable at `noise_amp_scale ∈ {1, 5, 15} cm³/s` —
  no NaN, finite gradients.
- ✅ `scripts/m7_frication_band.py` reports 4-8 kHz lift ≥ 1 dB on
  ≥ 8/10 fricative-rich utterances.
- ✅ Whisper gate (post-smoke-audio-level fix): fricative-character
  substitution rate ≥ 50% lower post-frication-on than post-frication-off
  on the same 10-utterance fricative-rich set. `runbooks/whisper_gate.sh`.
- ✅ JSON report at `outputs/m7_frication/audible_report.json`.

## Anti-deliverable / scope guard

- Do NOT change the `f_phys` formulation 004a established. The PSD
  shaping is a separate concern from the gain derivation.
- Do NOT add a free per-frame frication channel. ARCH §3 still
  forbids it; the only learned residual is the energy-capped
  `frication_color` already in tree.
- Do NOT bypass the MNA-stability issue with a stop-gradient or
  hand-rolled VJP — fix the conditioning, not the symptom.
- Do NOT use the smoke-audio-level fix to pass intelligibility
  gates that aren't really earned by frication. The 6 dB peak-
  normalize is a measurement-headroom fix, NOT a quality fix.

## Notes

The worker for 004a explicitly flagged both failure modes and chose
not to escalate to the user — Whisper-can't-hear-it is the smoke/002
floor (separate task) and PSD shaping is straightforward engineering.
That judgment was correct. Keep the same posture: both fixes are
physical/numerical, no decisions punt to the user.

If 004b's Whisper gate STILL fails after the PSD + injection-point
fixes, that IS escalation-worthy — that would mean the smoke pipeline
itself is producing fundamentally non-speech audio that no amount of
correct frication can rescue, and would point at body-model retrain
(pretraining/002) being a P0-blocker rather than a P3 task. Ping the
user with the specific Whisper transcript output if that happens.
