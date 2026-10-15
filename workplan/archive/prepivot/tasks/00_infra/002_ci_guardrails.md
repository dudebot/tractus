# infra/002 — CI guardrails

**Phase:** infra    **Status:** pending    **Effort:** M
**User-input:** none
**Blocks:** (lands at P0 close)
**Blocked by:** —
**How to dispatch:** fresh Claude Code session

---

## Context

`ARCHITECTURE.md` §16 enumerates architectural invariants that must
become CI checks as each module reaches its phase. This task lands
the CI infrastructure once the test suite (infra/001) is trustworthy.

## Deliverable

GitHub Actions workflow (or local `make ci` / `nox` equivalent) that
runs on every PR + push to master:

1. **Surrogate-call ban.** Lint that fails if any user-facing CLI or
   inference-path file imports OR calls the surrogate module (per
   ARCH §1663-1664, §1435-1438). The grep target is the current
   package + CLI allowlist (e.g., `tractus_controller/cli.py`,
   `tractus_controller/inference*` once it exists), not a hardcoded
   `inference/` directory. Both the import edge AND any
   `surrogate.synthesize(`-style call site must be zero.
2. **Cheat-channel ablation gate.** Pytest job that runs a tiny
   deterministic CI fixture (NOT full inversion/* fits) with
   `wall_dynamic_residual=0` and `frication_color=0` and checks
   loss-change is in the 5–20% Goldilocks range. < 5% → channel was
   unused; > 20% → channel was overpowered. Either result fails CI.
   A separate `@pytest.mark.full_ablation` slice runs the full
   inversion-driven version off the PR critical path.
3. **Frication-color physical-support gate.** Pytest job confirming
   `frication_color` cannot synthesize frication when constriction +
   pressure are absent (per ARCH §1684-1685). Skipped until M7 lands
   in correctness/004.
4. **Non-circularity rule.** Lint that body-model training scripts
   never import `synthesize_dynamic_tube` as their primary loss
   target. Specifically: `grep -rE "loss.*=.*synthesize_dynamic_tube|
   audio_loss.*body_model_output"` over the current package set
   must return 0.
5. **Consent rule.** CLI manifest validator returns non-zero on
   manifests missing `consent_verified: true`. Pytest job that
   constructs a manifest without the flag and asserts the validator
   rejects it.
6. **Doc-coherence regression lint.** Greps that catch reintroduction
   of stale architecture framing (per ARCH §1699-1703): no
   `four-module` / `four module` outside historical-marked sections;
   no `jax_enable_x64=True` or `dtype=jnp.float64` defaults in
   production paths; no resurrected `ges_to_tract` references;
   README pointer to ARCHITECTURE.md remains intact.
7. **Reference-pack hygiene.** Lint that every URL in
    `ARCHITECTURE.md` §15 has a corresponding entry in
    `docs/references/pages/reference_links.md` OR a local PDF.

## Acceptance gates (automated)

- ✅ `make ci` passes on a fresh checkout post-correctness/* and
  with the new test suite from infra/001.
- ✅ Each guardrail has a deliberately-violating test fixture under
  `tests/ci_fixtures/` that shows the guardrail catches the
  violation. (Negative-case tests for the negative-case detector.)
- ✅ The workflow file is parseable on GitHub Actions; a dry-run
  pass via `act` (if available) or local script equivalent
  succeeds.
- ✅ `pre-commit` hook installed for cheap lint-only guardrails
  (surrogate-call ban, non-circularity, consent, doc-coherence,
  reference-pack hygiene) so they fire locally before push.

## Anti-deliverable / scope guard

- Do NOT implement the modules being guarded (surrogate, residual,
  channel model). Those are pretraining/* and cloning/* tasks. Where
  a guardrail needs the module to exist (e.g., the residual
  energy-cap runtime check), the guardrail is registered with
  `pytest.skip(reason="awaits <task-id>")` until the module ships —
  do NOT add stub modules just to wire the lint.
- Do NOT make CI gating-perfect on architectural changes that
  haven't yet shipped — `xfail` or `skip` keeps the workflow green
  now and tightens as modules land.

## Dispatch notes

Worker should know GitHub Actions YAML + pre-commit hook conventions.
The 8 guardrails decompose into ~8 small lint scripts; can be
parallelized within the session.
