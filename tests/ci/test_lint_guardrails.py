"""Meta-tests: every lint script must catch its deliberately-violating fixture.

Acceptance gate #2 of infra/002: "Each guardrail has a deliberately-
violating test fixture under tests/ci_fixtures/ that shows the
guardrail catches the violation. (Negative-case tests for the
negative-case detector.)"

We invoke each lint as a subprocess (matching how CI runs it) with a
``--root`` pointing at the fixture tree and ``--include-tests`` so the
``tests/`` exclusion is lifted.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = REPO_ROOT / "scripts" / "ci"
FIXTURES = REPO_ROOT / "tests" / "ci_fixtures"


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPTS / script), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )


def test_lint_surrogate_ban_catches_fixture() -> None:
    """The surrogate-ban lint allowlist is path-pattern based; we point it
    at the fixture file via ``--root tests/ci_fixtures/inference_alike`` so
    the fixture sits where an inference path would. We do that by adding a
    symlinkless copy on the fly."""
    # Easiest: directly invoke the check function with a custom allowlist.
    sys.path.insert(0, str(SCRIPTS))
    try:
        import lint_surrogate_ban as ls  # type: ignore[import-not-found]

        # Temporarily widen the inference allowlist to include the fixture.
        original = ls.INFERENCE_ALLOWLIST
        ls.INFERENCE_ALLOWLIST = original + ("tests/ci_fixtures/violates_surrogate_ban.py",)
        try:
            rc = ls.check(roots=(REPO_ROOT,), excludes=(), strict=True)
        finally:
            ls.INFERENCE_ALLOWLIST = original
        assert rc != 0, "surrogate-ban lint failed to flag fixture"
    finally:
        sys.path.remove(str(SCRIPTS))


def test_lint_surrogate_ban_passes_clean_repo() -> None:
    """No surrogate references on the real inference path today."""
    cp = _run("lint_surrogate_ban.py")
    assert cp.returncode == 0, f"surrogate-ban lint flagged clean repo:\n{cp.stdout}\n{cp.stderr}"


def test_lint_non_circularity_catches_fixture() -> None:
    cp = _run(
        "lint_non_circularity.py",
        "--root", "tests/ci_fixtures",
        "--include-tests",
    )
    assert cp.returncode != 0, (
        f"non-circularity lint failed to flag fixture:\n{cp.stdout}\n{cp.stderr}"
    )
    assert "violates_non_circularity" in cp.stdout


def test_lint_non_circularity_passes_clean_repo() -> None:
    cp = _run("lint_non_circularity.py")
    assert cp.returncode == 0, (
        f"non-circularity lint flagged clean repo:\n{cp.stdout}\n{cp.stderr}"
    )


def test_lint_doc_coherence_catches_fixture() -> None:
    cp = _run(
        "lint_doc_coherence.py",
        "--root", "tests/ci_fixtures",
        "--include-tests",
    )
    assert cp.returncode != 0, (
        f"doc-coherence lint failed to flag fixture:\n{cp.stdout}\n{cp.stderr}"
    )
    # Must hit at least the four-module + fp64 + ges_to_tract patterns.
    out = cp.stdout
    assert "four-module" in out.lower() or "five module" in out.lower() or "5 modules" in out
    assert "fp64" in out.lower() or "float64" in out
    assert "ges_to_tract" in out


def test_lint_doc_coherence_passes_clean_repo() -> None:
    cp = _run("lint_doc_coherence.py")
    assert cp.returncode == 0, (
        f"doc-coherence lint flagged clean repo:\n{cp.stdout}\n{cp.stderr}"
    )


def test_lint_reference_pack_passes_today() -> None:
    cp = _run("lint_reference_pack.py")
    assert cp.returncode == 0, (
        f"reference-pack hygiene flagged real repo:\n{cp.stdout}\n{cp.stderr}"
    )


def test_lint_reference_pack_catches_missing_url(tmp_path) -> None:
    """Inject a synthetic URL into a copy of ARCHITECTURE.md §15 and verify
    the lint flags it. We do this by running with a temp ``ARCHITECTURE.md``
    via ``--root`` indirection (the script reads the canonical file), so
    we monkey-patch the helper directly via in-process invocation."""
    sys.path.insert(0, str(SCRIPTS))
    try:
        import lint_reference_pack as lr  # type: ignore[import-not-found]

        original = lr._arch_section_15_text  # noqa: SLF001
        synthetic = (
            "## 15. References\n\n"
            "- Some paper https://example.com/this-url-is-not-in-links\n\n"
            "## 16. Next\n"
        )
        lr._arch_section_15_text = lambda: synthetic  # noqa: SLF001
        try:
            rc = lr.check(roots=(), excludes=(), strict=True)
        finally:
            lr._arch_section_15_text = original  # noqa: SLF001
        assert rc != 0, "reference-pack lint failed to flag synthetic missing URL"
    finally:
        sys.path.remove(str(SCRIPTS))


@pytest.mark.skipif(
    os.environ.get("TRACTUS_SKIP_CONSENT_LINT") == "1",
    reason="tractus_shared not on path in this environment",
)
def test_lint_consent_passes_clean_repo() -> None:
    cp = _run("lint_consent.py")
    # If tractus_shared isn't installed, the script prints SKIP and exits 0.
    assert cp.returncode == 0, cp.stdout + cp.stderr


def test_lint_all_runner_executable() -> None:
    """The aggregator script must be executable and (today) pass."""
    runner = SCRIPTS / "lint_all.sh"
    assert runner.exists()
    cp = subprocess.run(
        ["bash", str(runner)],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        check=False,
    )
    assert cp.returncode == 0, (
        f"lint_all.sh failed on clean repo:\n{cp.stdout}\n{cp.stderr}"
    )
