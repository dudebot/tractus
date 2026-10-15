# research/001 — Procedural body model exploration (the (a) fork)

**Phase:** research    **Status:** pending    **Effort:** L
**User-input:** **decision** (multi-month investment; only if (b) plateaus)
**Blocked by:** P4 closed
**Blocks:** —
**How to dispatch:** fresh Claude Code session preceded by user decision

---

## Context

`../../../ARCHITECTURE.md` §2.3 fork (a) vs (b). v0 ships learned
body model (b); (a) procedural physics is the v1+ research path.
Hand-coded jaw kinematics, tongue spline mechanics, lip protrusion,
mass-spring soft tissue. Strong identifiability by construction.
High implementation effort (multi-month).

This task fires **only after** cloning/* shows where (b) plateaus.
If (b) is good enough, (a) is unnecessary and a multi-month
distraction.

## Pre-task decision

Validator session presents:
- cloning/005 listener-A/B and identity-cosine results.
- inversion/006 dimensionality-discovery report (does (b) saturate
  at small D, or does it benefit from more?).
- Specific failure modes (b) shows on cross-speaker eval that (a)
  would plausibly fix.

User decides: pursue (a), continue refining (b), or pivot to
research/002 (anatomy beyond 60) instead.

## Deliverable (if pursued)

1. Survey published procedural vocal-body models: Maeda 1990,
   Story 2005 articulatory-synthesis review,
   Pink Trombone, recent biomechanical tongue/jaw work.
2. Pick the minimal procedural layer to add: typically jaw
   kinematics + tongue spline + lip protrusion mechanics. Mass-
   spring tissue is a v1+ research item beyond even this.
3. Implement in pure JAX, differentiable.
4. Train interface: feed the same GestureEvent input; produce the
   same TEA renderer interface. Procedural layer replaces (b)'s
   neural mapping for whichever subset it covers; (b) handles the
   rest.
5. A/B vs (b)-only on cloning/002's 5-speaker test set.

## Acceptance gates (if pursued)

- ✅ Procedural layer differentiable; `jax.grad` finite over
  10K random in-envelope inputs.
- ✅ A/B against (b)-only: (a) wins on at least one of [identity,
  WER, per-axis source/tract separation] without losing on others.
- ✅ Implementation effort documented; sets baseline for further
  procedural work.

## Anti-deliverable / scope guard

- Do NOT pursue this without the pre-task decision. Multi-month
  investment.
- Do NOT replace (b) entirely. Hybrid (procedural for joints +
  learned for soft tissue) is fine; full procedural is a v2+ goal.

## Notes

Research-tier task. Outcomes here drive whether we publish a
procedural variant or stick with the small-learned-stack story.
If (a) is adopted, ARCH §2.3 fork status MUST be updated in the
same commit — the architecture currently records v0 as fork (b).
