# inversion/006 — Body-state dimensionality discovery

**Phase:** inversion    **Status:** pending    **Effort:** L
**User-input:** **decision** (pick `D_body_effective`)
**Blocks:** P3 starts (pretraining/002, /003)
**Blocked by:** inversion/004, inversion/005
**How to dispatch:** fresh Claude Code session for analysis;
user-decision step at end

---

## Context

`../../../ARCHITECTURE.md` §0 (D_body_start vs D_body_effective),
§2.3, §10 lesson 14. The user's "~132 parameters" instinct is the
**overcomplete starting envelope**, not a finished number. P2.6
prunes to the smallest body-state dimensionality that preserves
intelligibility, identity, prosody, frication, and physical
plausibility.

Two analyses + a sweep:
- **PCA** on plausible body-state trajectories: which directions
  actually move?
- **Acoustic-Jacobian SVD** (active subspace): which directions
  does TEA actually hear?
- **Sweep** over `D_body ∈ {24, 32, 48, 64, 96, 132}` (per ARCH
  §1206-1211; do not add D=16): train at each, gate on
  intelligibility + identity + anti-cheat.

Together: PCA tells you what moves; acoustic-Jacobian SVD tells you
what matters acoustically. The minimum useful `D_body` is the
intersection.

`runbooks/dimensionality_report.py` (from infra/003) is the
analysis tool.

## Deliverable

1. Collect body-state trajectory corpus from the inversion/005
   fits + smoke/001 templates + procedural-perturbation data. Aim for ≥ 10
   K body-state frames spanning realistic operating envelope.
2. Run `runbooks/dimensionality_report.py` on the corpus.
3. Run the sweep: train a body-model variant at each
   `D_body ∈ {24, 32, 48, 64, 96, 132}`. Use the same training
   recipe (saved fixtures + procedural perturbations). Compute
   budget: ≤ 8 GPU-hours per variant; if a variant doesn't converge
   at budget, document and include the partial result.
4. Per-D evaluation:
   - inversion/004 gates pass / fail
   - Held-out reconstruction log-mel
   - Anti-cheat refusal rate
   - `runbooks/source_tract_sep.py` per-axis aggregate
   - Speaker-distinguishability cluster score (from inversion/005's
     5-speaker test set)
5. Compose a one-page report
   `outputs/dimensionality_discovery/report.md` with:
   - PCA cumulative variance plot
   - Acoustic-Jacobian SVD plot
   - Sweep table (D × all gates)
   - Recommended minimum D
6. **User decision step**: validator session presents the report;
   user picks `D_body_effective`. Decision recorded in
   `report.md` + `INDEX.md`.

## Acceptance gates (automated for the analysis; user-input for the choice)

- ✅ PCA cumulative variance reported at standard D values.
- ✅ Acoustic-Jacobian SVD reported with rank-condition numbers.
- ✅ Sweep table covers all 6 D values; each variant trained to
  convergence (held-out log-mel Δ over last 10% of training <
  0.1 dB) or documented as not-converged with reason and partial
  metrics included.
- ✅ For each D, all inversion/004 gates run; pass/fail recorded.
- ✅ Recommended minimum D justified in the report.
- ⏸️ **User decision**: pick `D_body_effective`. Likely answer
  30–80 per analysis predictions; final value chosen by user from
  the candidate list the analysis provides.

## Anti-deliverable / scope guard

- Do NOT pick D_body_effective without running the sweep. The
  analysis informs the choice; it doesn't replace it.
- Do NOT pick D > 132. The starting envelope is the upper bound.
- Do NOT skip the anti-cheat check at small D. Smaller D may
  reduce expressiveness but should not enable cheating.

## Notes

This is the **largest single task** in inversion/*. Worth ~3
sessions of focused work: corpus collection (1), sweep training
(1–2), analysis + report (1).

Worker produces the full report, then this task's status flips to
`pending-decision` in INDEX. The validator (creator) session has the
design context and shepherds the user choice efficiently from there.
