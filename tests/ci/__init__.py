"""CI guardrail tests.

Two kinds of tests live here:

1. **Meta-tests** (``test_lint_guardrails.py``) prove each lint script
   catches its deliberately-violating fixture under ``tests/ci_fixtures/``.
   These run on every PR.
2. **Runtime gate placeholders** (``test_runtime_gates.py``) for guards
   that need a module to ship before they have anything to assert
   (residual energy cap, channel ablation, frication physical-support,
   cheat-channel Goldilocks). They use ``pytest.skip(reason='awaits ...')``
   so the workflow stays green and tightens as modules land.
"""
