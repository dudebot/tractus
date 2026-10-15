"""Deliberately-violating fixtures for CI guardrail meta-tests.

Each file here intentionally trips a specific lint script. They are
isolated under ``tests/ci_fixtures/`` so the lint scripts ignore them by
default; the ``tests/ci/test_lint_guardrails.py`` meta-suite invokes
each lint with this directory explicitly included via ``--root``.

Touching these files: each fixture's name encodes the lint it triggers.
DO NOT make these files innocent — that defeats the negative-case test
for the negative-case detector (acceptance gate #2 of infra/002).
"""
