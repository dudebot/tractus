# CI guardrail scripts

These scripts back the architectural invariants in `ARCHITECTURE.md` §16
and the CI gates listed in `workplan/tasks/00_infra/002_ci_guardrails.md`.

Each script exits non-zero on violation. Run individually or via
`scripts/ci/lint_all.sh` (which is what `make ci` and the GitHub Actions
workflow invoke).

| Script | Guards |
|---|---|
| `lint_surrogate_ban.py` | inference path imports/calls of training surrogate (ARCH §1663-1664, §1435-1438) |
| `lint_non_circularity.py` | body-model training scripts using `synthesize_dynamic_tube` as primary loss (ARCH §1656, non-circularity rule) |
| `lint_doc_coherence.py` | resurrection of `four-module`, fp64-defaults, `ges_to_tract`, README pointer (ARCH §1699-1703) |
| `lint_reference_pack.py` | every URL in ARCH §15 has a corresponding entry in `docs/references/pages/reference_links.md` or local PDF |
| `lint_consent.py` | manifest validator rejects `consent_verified != true` (delegated to `tractus_shared.manifest_natural`; this script is a smoke wrapper) |

Common conventions:

- `--strict` (default) → non-zero exit on violation.
- `--report` → print findings, exit 0 (used in dev / pre-merge audits).
- All scripts accept an optional path-allowlist argument so future tasks
  can extend the surface area without rewriting.

The runtime portions of the gates (e.g., the cheat-channel Goldilocks
range, frication physical-support, channel ablation) live as pytest
tests gated by `pytest.skip(reason="awaits <task-id>")` in the
`tests/ci/` slice; they tighten as modules ship.

`tests/ci_fixtures/` contains **deliberately-violating fixtures** that
each lint script must catch. The `tests/ci/test_lint_guardrails.py` meta
suite proves each lint catches its fixture (negative-case test for the
negative-case detector).
