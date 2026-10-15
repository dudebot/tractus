#!/usr/bin/env python3
"""Surrogate-call ban (ARCH §1663-1664, §1435-1438).

Inference / user-facing CLI paths must NOT import or call the training
surrogate (``Module 5A`` in the architecture). The surrogate is a
training-time speedup; using it at inference would silently turn the
"physics-first" demo into a neural decoder.

What we lint:

1. Any ``import`` that resolves to ``tractus_*.surrogate*`` /
   ``tractus_*.module5a*`` from inside an inference-allowlisted file.
2. Any ``surrogate.synthesize(`` / ``module5a.synthesize(`` /
   ``Surrogate(*).render(`` -style call site in those same files.

Inference allowlist is INTENTIONALLY conservative. Today it is just
``tractus_controller/cli.py`` and ``tractus_controller/inference*``
(currently absent — the lint pre-registers the path so it's wired the
moment that module ships). The training surrogate itself, when it
lands, lives outside the allowlist and is unaffected.

The lint is content-based grep — it does NOT need the surrogate module
to exist to be useful. Once the module exists (pretraining/005), every
new file added to the allowlist is scanned automatically.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _common import (  # type: ignore[import-not-found]
    Hit,
    REPO_ROOT,
    base_arg_parser,
    grep,
    report,
    resolve_roots,
)


# Repo-relative globs of files that constitute the *inference* path. Anything
# matching is forbidden from importing or calling the surrogate.
INFERENCE_ALLOWLIST: tuple[str, ...] = (
    "tractus_controller/tractus_controller/cli.py",
    "tractus_controller/tractus_controller/inference/**/*.py",
    "tractus_controller/tractus_controller/inference.py",
    "tractus_controller/tractus_controller/sidecar.py",
    "tractus_controller/tractus_controller/copy_synth/**/*.py",
)


# Surrogate module name fragments. Comments inside source files (which are
# allowed to *talk about* the surrogate) are stripped before matching.
IMPORT_PAT = re.compile(
    r"""
    (?:^\s*from\s+([\w.]*surrogate[\w.]*|[\w.]*module5a[\w.]*)\s+import)
    | (?:^\s*import\s+([\w.]*surrogate[\w.]*|[\w.]*module5a[\w.]*))
    """,
    re.VERBOSE,
)

# Call sites: anything that looks like calling a surrogate to render audio.
CALL_PAT = re.compile(
    r"""
    \b(?:surrogate|module5a|Module5A|Surrogate)\b
    [\w.]*
    \s*\.\s*
    (?:synthesize|render|forward|generate|infer|decode|__call__)
    \s*\(
    """,
    re.VERBOSE,
)


def _strip_comment(line: str) -> str:
    """Strip a Python ``#`` comment without breaking on '#' inside strings.

    Conservative: anything that looks like a string literal start before the
    first ``#`` keeps the line intact. False negatives here are fine — this
    is a security check; we'd rather flag too much than too little. False
    POSITIVES would be the bug, and would only matter if a string literal
    contained ``surrogate.synthesize(``, which is itself suspicious.
    """

    in_str: str | None = None
    out: list[str] = []
    i = 0
    while i < len(line):
        ch = line[i]
        if in_str:
            out.append(ch)
            if ch == "\\" and i + 1 < len(line):
                out.append(line[i + 1])
                i += 2
                continue
            if ch == in_str:
                in_str = None
        elif ch in ("'", '"'):
            in_str = ch
            out.append(ch)
        elif ch == "#":
            break
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _filter_files(rel_globs: tuple[str, ...], roots: tuple[Path, ...]) -> list[Path]:
    from fnmatch import fnmatchcase

    found: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            try:
                rel = p.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                continue
            if any(fnmatchcase(rel, g) for g in rel_globs):
                found.append(p)
    return found


def check(roots: tuple[Path, ...], excludes: tuple[str, ...], strict: bool) -> int:
    """Surrogate must not appear on any inference-allowlisted path."""

    files = _filter_files(INFERENCE_ALLOWLIST, roots)
    hits: list[Hit] = []
    for path in files:
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = _strip_comment(raw)
            if IMPORT_PAT.search(line) or CALL_PAT.search(line):
                hits.append(Hit(path=path, line=lineno, text=raw))
    return report(
        hits,
        "surrogate-call ban (ARCH §1663-1664): inference path must not "
        "import or call the training surrogate",
        strict,
    )


def main() -> int:  # pragma: no cover - CLI plumbing
    parser = base_arg_parser(__doc__ or "")
    args = parser.parse_args()
    roots = resolve_roots(args.root)
    return check(roots=roots, excludes=(), strict=not args.report)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
