"""DELIBERATELY VIOLATES surrogate-call ban (infra/002 fixture).

Imported by tests/ci/test_lint_guardrails.py to verify that
scripts/ci/lint_surrogate_ban.py catches a real violation. This file
is exempt from the lint by default (lives under tests/ci_fixtures/);
the meta-test re-includes the fixture path explicitly.
"""

# This is the kind of import the lint must catch on any inference path.
from tractus_controller.surrogate import Surrogate  # noqa: F401  (intentional violation)


def render(audio):
    surrogate = Surrogate()
    # And this is the kind of call site it must catch.
    return surrogate.synthesize(audio)
