# correctness/004 — M7 per-section gated frication noise (CORE)

**Phase:** correctness    **Status:** pending    **Effort:** L (3+ sessions; split candidate)
**User-input:** **escalation-only** — Whisper gate replaces the human ear-test. Ping the user only on a genuine ambiguity Whisper can't resolve.
**Blocks:** P0 close
**Blocked by:** correctness/003
**How to dispatch:** fresh Claude Code session

---

## Context

> **M7 is core speech functionality, NOT polish.** Frication is
> ~30% of the perceptual identity signal; sibilants, dental
> geometry, aspiration, and burst releases all live here. Failing
> M7 = failing voice cloning regardless of how good vowel
> resonances are.

`../../../ARCHITECTURE.md` §0, §1.4, §2.3 (frication physically
derived not gated), §10 (validation discipline). Reference papers:
Narayanan & Alwan 2000.
`tractus_engine_advanced/SCOPE.md` §3.A (per-section gated noise
design from Chen & Sitzmann 2025).

The body model emits constriction geometry + pressure;
the frication module **derives** noise gain from physical
conditions, NOT free per-frame gates:
```
frication_gain = f_phys(constriction_area,
                         pressure_drop,
                         place,
                         voicing,
                         dental_obstacle_proxy)
```
A small learned `frication_color` residual is allowed for
calibration only — energy-capped, ablation-gated.

## Deliverable

1. Implement the physical frication module per the §2.3 contract.
   Replace any existing `frication_gates[T, ~40]` free-output path
   with the derived form.
2. Per-section noise sources placed at every supraglottal section
   (Chen-Sitzmann style). Smooth gating, dipole structure as a
   per-section soft upstream/downstream weight pair (no
   obstacle-id branching).
3. Articulator-class gain (sibilant `/s/` vs velar `/x/` vs lip
   `/f/`) emerges from the physical model + soft articulator-class
   weights coming from the body model output, not a switch.
4. Filter noise rejection sampling (replaces the deleted upstream's
   rejection-sampling routine; algorithm reference only) with a
   reparameterized `tanh(z) · k` of unit Gaussian — deterministic
   given `noise_key`.
5. Reflect the new contract in tests (the
   `frication-physically-derived` test from infra/001 should now
   pass instead of `xfail`).

## Acceptance gates (automated where possible; one user-review step)

- ✅ Zero constriction area → zero frication gain (mathematical
  invariant; tested on synthetic constriction-area sweep).
- ✅ Frication gain monotone in constriction (within physically
  plausible operating range): tested on a constriction sweep at
  fixed pressure.
- ✅ Disabling `frication_color` residual changes loss in
  **5–20%** Goldilocks range (not unused; not over-bound).
- ✅ Gradient FD agreement on the frication module per-input
  channel.
- ✅ Spectral parity for fricated segments at soft-mode level: log-mel
  distance ≤ 0.5 dB on a 10-utterance fricative-rich corpus vs
  saved fixture audio. **Escape clause**: if log-mel exceeds 0.5 dB
  but `runbooks/source_tract_sep.py` reports frication-band energy
  within 3 dB AND the user-review step approves, the new physical
  derivation is the truth. Saved fixtures predate the new
  formulation; some divergence is expected and correct.
- ✅ `runbooks/source_tract_sep.py` reports frication-band energy
  within 3 dB on each of /s/, /ʃ/, /f/, /θ/, /z/, /v/, plus
  aspiration and burst releases. The fricative-rich corpus is a
  declared dependency on smoke/001's template-gesture set; if
  smoke/001 hasn't shipped a fricative-rich subset, this task adds
  one.
- ✅ **Whisper gate replaces the human ear-test.** Render a curated
  fricative-rich sentence set ("she sells seashells", "fifty-five
  fife players", "the thrush thrived", etc. — pick ~10 sentences
  that span /s/, /ʃ/, /f/, /θ/, /z/, /v/, /h/-aspiration, and
  burst releases) through the smoke pipeline + new frication
  module. Run `runbooks/whisper_gate.sh` on the rendered audio;
  pass criterion is fricative-character substitution rate **at
  least 50% lower** post-frication-module than pre-frication-module
  on the same sentence set (i.e., Whisper confuses /s/-for-/k/ etc.
  half as often or better). This is the canonical pass/fail; no
  human review required.
- ⏸️ **Escalation-only user review**: Ping the user ONLY if Whisper
  produces a genuinely ambiguous result (e.g., a phone class is
  routinely substituted by Whisper but `source_tract_sep` says the
  frication-band energy is correct — that's a question about
  Whisper's reliability on isolated phones rather than the model
  itself). Do NOT ping for routine ship-or-rework decisions; the
  Whisper gate is binary.

## Anti-deliverable / scope guard

- Do NOT use soft-argmin to detect constrictions (`SCOPE.md` §3.A
  is explicit on this — won't produce crisp fricatives).
- Do NOT add a free `frication_gates` channel anywhere. The
  identifiability ledger forbids it (`ARCHITECTURE.md` §3).
- Do NOT use stochastic noise without the reparameterization;
  gradients must be deterministic given `noise_key`.

## Notes

This is the largest correctness-phase task. Likely splits into
sub-tasks if needed:
- correctness/004a: physical-frication module + ablation tests +
  Whisper gate on the fricative-rich sentence set.
- correctness/004b (ONLY if 004a's Whisper gate flags a genuine
  ambiguity): spectrogram visualization of the ambiguous case +
  user ping for ear-resolution.

Worker should consider checkpointing progress; if 1 session can't
land all of it, file follow-ups. **004b is opportunistic, not
mandatory.** Most fricative work should close on the automated
Whisper + source_tract_sep gates without involving a human.
