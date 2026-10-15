"""Shared helpers for CI guardrail lint scripts.

Keeps individual scripts small and uniform. infra/004 (regression-prevention
lint) extends the same surface — add new check functions here and a thin
script per check, rather than rewriting any existing scripts.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]


# Production source roots. Tests, scripts/ci, fixtures, and the worktrees
# scratch tree are NEVER scanned for production-policy violations: their
# whole job is to violate things on purpose.
PRODUCTION_PY_ROOTS: tuple[Path, ...] = (
    REPO_ROOT / "tractus_engine_advanced" / "src",
    REPO_ROOT / "tractus_controller" / "tractus_controller",
    REPO_ROOT / "tractus_shared",
)

# Default exclusion globs evaluated against the repo-relative POSIX path.
DEFAULT_EXCLUDES: tuple[str, ...] = (
    "**/.git/**",
    "**/.claude/**",
    "**/build/**",
    "**/dist/**",
    "**/*.egg-info/**",
    "**/__pycache__/**",
    "**/.venv/**",
    "**/.tox/**",
    "**/node_modules/**",
    "tests/ci_fixtures/**",
    # Test files are excluded from production-policy scans by default; each
    # script can re-include test trees with --include if it needs to.
    "**/tests/**",
    # Lint scripts themselves.
    "scripts/ci/**",
)


@dataclass(frozen=True)
class Hit:
    path: Path
    line: int
    text: str

    def fmt(self) -> str:
        rel = self.path.relative_to(REPO_ROOT) if self.path.is_absolute() else self.path
        return f"{rel}:{self.line}: {self.text.rstrip()}"


def _matches_any(rel_posix: str, patterns: Iterable[str]) -> bool:
    from fnmatch import fnmatchcase

    return any(fnmatchcase(rel_posix, p) for p in patterns)


def iter_files(
    roots: Sequence[Path] = PRODUCTION_PY_ROOTS,
    suffixes: tuple[str, ...] = (".py",),
    excludes: tuple[str, ...] = DEFAULT_EXCLUDES,
    extra_includes: tuple[str, ...] = (),
) -> Iterator[Path]:
    """Yield every file under ``roots`` matching ``suffixes`` not in ``excludes``.

    ``extra_includes`` are repo-relative globs that override ``excludes``
    (so a script can re-include a normally-excluded subtree like ``tests/``).
    """

    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if not p.is_file() or p.suffix not in suffixes:
                continue
            if p in seen:
                continue
            seen.add(p)
            try:
                rel = p.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                rel = p.as_posix()
            if _matches_any(rel, excludes) and not _matches_any(rel, extra_includes):
                continue
            yield p


def grep(
    pattern: re.Pattern[str] | str,
    files: Iterable[Path],
    *,
    line_filter: re.Pattern[str] | None = None,
) -> List[Hit]:
    """Return every matching line as a :class:`Hit`.

    ``line_filter``, if given, is a POSITIVE filter — only lines that match it
    are scanned for ``pattern``. Used to scope a check to e.g. ``loss = ...``
    lines without rewriting the main pattern.
    """

    if isinstance(pattern, str):
        pattern = re.compile(pattern)
    hits: list[Hit] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if line_filter is not None and not line_filter.search(line):
                continue
            if pattern.search(line):
                hits.append(Hit(path=path, line=lineno, text=line))
    return hits


def report(hits: Sequence[Hit], message: str, strict: bool) -> int:
    """Print findings and return a CLI-appropriate exit code."""

    if not hits:
        print(f"OK: {message}")
        return 0
    print(f"FAIL: {message} ({len(hits)} hit{'s' if len(hits) != 1 else ''})")
    for h in hits:
        print(f"  {h.fmt()}")
    return 1 if strict else 0


def base_arg_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument(
        "--report",
        action="store_true",
        help="print findings but exit 0 (default: exit non-zero on hits)",
    )
    p.add_argument(
        "--root",
        action="append",
        default=None,
        help=(
            "Repo-relative path to scan in addition to the default production "
            "roots. May be given multiple times. Used by infra/004 etc. to "
            "extend the surface."
        ),
    )
    p.add_argument(
        "--include-tests",
        action="store_true",
        help="re-include the tests/ subtree in scanning",
    )
    return p


def resolve_roots(extra: list[str] | None) -> tuple[Path, ...]:
    if not extra:
        return PRODUCTION_PY_ROOTS
    extras = tuple((REPO_ROOT / r).resolve() for r in extra)
    return PRODUCTION_PY_ROOTS + extras


def make_excludes(include_tests: bool) -> tuple[str, ...]:
    if not include_tests:
        return DEFAULT_EXCLUDES
    # Drop both the regular tests/ subtree exclusion *and* the
    # ci_fixtures/ exclusion. The fixtures exist to violate; the
    # meta-test slice (tests/ci/test_lint_guardrails.py) flips this on
    # to verify each lint catches its fixture.
    return tuple(p for p in DEFAULT_EXCLUDES if "tests" not in p)


def main_for(check_fn) -> int:  # pragma: no cover - CLI plumbing
    """Used by tiny scripts that just delegate to a single check function."""

    parser = base_arg_parser(check_fn.__doc__ or "")
    args = parser.parse_args()
    roots = resolve_roots(args.root)
    excludes = make_excludes(args.include_tests)
    return check_fn(roots=roots, excludes=excludes, strict=not args.report)


def cli_run(check_fn) -> None:  # pragma: no cover - CLI plumbing
    sys.exit(main_for(check_fn))
