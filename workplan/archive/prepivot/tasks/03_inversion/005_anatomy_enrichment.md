# inversion/005 — Anatomy enrichment 9 → 30–60 scalars

**Phase:** inversion    **Status:** pending    **Effort:** M
**User-input:** review (basis-shape sign-off at end)
**Blocks:** P2 close
**Blocked by:** inversion/004
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §0 risk #4, §2.5. v0 9-scalar basis is
the smoke version. v1 needs 30–60 scalars to capture speaker
identity:
- Global anatomy (8–16): length, ratio, lip, velum, glottis
  RT/RL/RF0, hyoid, larynx height.
- Source/tissue (8–16): mucosal damping, tension, fold thickness
  profile, glottal pulse asymmetry.
- Side-cavity / frication (8–16): sinus volumes, fossa volume,
  dental geometry proxies, sibilant constriction shape.

(Area residual — rank 16–32 per-section perturbations, orthonormal,
constant component projected out — is a v1.1 opt-in extension; see
anti-deliverable below. Not part of the v1 default basis.)

This task expands `tractus_engine_advanced/notes/anatomy_basis.md`
design from 9 → 30–60 and re-validates inversion/004 with the
richer basis.

## Deliverable

1. Pick the new basis composition (target 30–60 total). Document
   in `notes/anatomy_basis.md` with a per-scalar table of (name,
   units, hard-min, hard-max, prior σ). The σ rule — ±1σ ≈ 1/3 of
   hard range (deliverable 3) — applies per scalar from this table.
2. Implement `apply_anatomy_basis_v1` (pure dict-in/dict-out, no
   JAX in the body) extending the 9-scalar v0 implementation.
3. Identifiability mitigations: hard clamps, soft Gaussian prior
   toward the default-speaker baseline, sigma per param chosen so ±1σ ≈ 1/3 of hard range.
4. Re-run inversion/004's experiment with v1 basis. Verify the
   richer basis improves held-out reconstruction further.
5. Sanity-check identifiability on multi-speaker data: fit
   anatomy_v1 on 5 different procedurally-perturbed "speakers"; assert
   fitted anatomies are distinguishable (cluster) and not
   collapsed onto each other.
6. Run `runbooks/anticheat_unreachable.py` again — richer basis
   should NOT be more cheat-prone.

## Acceptance gates (automated where possible; one user-review)

- ✅ inversion/004's gates still pass with v1 basis on the SAME
  v1-generated corpus (re-run the v0 baseline on the v1 corpus, not
  on the original v0 corpus). Held-out improvement now ≥ **2 dB**
  log-mel (not just 1 dB) because we have more degrees of freedom.
- ✅ 5 perturbed speakers cluster identifiably in basis space:
  fitted anatomies span ≥ 50% of their per-axis variance across the
  5 speakers (i.e., the fit isn't collapsing speakers onto a shared
  subspace).
- ✅ Anti-cheat refusal rate stays ≥ 80%.
- ✅ Per-scalar prior σ tunings documented.
- ⏸️ **User review**: validator session signs off on the basis
  composition. This is a durable choice — the basis shape locks
  the per-speaker checkpoint format.

## Anti-deliverable / scope guard

- Do NOT make the basis larger than ~60 scalars. Above that the
  identifiability story gets harder and we're approaching the
  free-residual failure mode.
- Do NOT enable the rank-r area residual at v1 by default — keep
  it as an opt-in extension.
- Do NOT change the 9-scalar v0 for backward compatibility — old
  fits should continue to load.

## Notes

This task informs whether v0 anatomy is the actual ceiling or
whether we needed v1 all along. If the gates pass with both v0 and
v1, document the gap (v1 - v0 dB improvement) for future reference.
If v0 already saturates, v1 is over-parameterized for current
fitter strength; if v0 is the bottleneck, v1 lifts the ceiling.
