#!/usr/bin/env python3
"""Doc-coherence regression lint (ARCH §1699-1703).

Catches stale architectural framing that has been intentionally purged
and must NOT be silently resurrected:

1. ``four-module`` / ``four module`` — superseded by the 5-module
   architecture in ARCH §2.
2. ``jax_enable_x64=True`` / ``dtype=jnp.float64`` defaults in
   production paths. Production is fp32 by policy (ARCH §7); fp64 is
   gated by ``tractus_engine_advanced.precision`` (env-var opt-in) +
   parity tests in ``tests/`` (subprocess-isolated).
3. README pointer to ``ARCHITECTURE.md`` remains intact.

Per the task spec, the policy module ``precision.py`` and the
``@pytest.mark.parity`` / ``@pytest.mark.fp64_oracle`` test slices are
EXEMPT — they are the policy + the fixture sites, not violations.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _common import (  # type: ignore[import-not-found]
    Hit,
    REPO_ROOT,
    base_arg_parser,
    iter_files,
    report,
    resolve_roots,
)


# (regex, message) pairs. All scoped to "production" trees.
#
# Scope note on fp64 patterns: per the task spec (item 9) the lint targets
# ``jnp.float64`` defaults — i.e., the TEA *JAX forward graph*. Numpy host-side
# helpers (sidecar metadata, IIR design coefficients, gate resampling, etc.)
# are legitimate fp64 — they are not in the JAX forward path and CPU fp64 is
# free. ARCH §7 only constrains the TEA forward precision; the precision
# policy module owns ``jax_enable_x64``.
LEGACY_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bfour[-\s]module\b", re.IGNORECASE), "stale 'four-module' framing (ARCH §2 is 5 modules)"),
    (re.compile(r"\bjax_enable_x64\s*=\s*True\b"), "fp64 default outside precision policy (ARCH §7)"),
    (re.compile(r"\bdtype\s*=\s*jnp\.float64\b"), "fp64 default in JAX forward graph (ARCH §7)"),
    (re.compile(r"\bges_to_tract\b"), "legacy ges_to_tract residue"),
)


# Files that EITHER define the policy or are intentional fp64 fixture
# sites. Excluded from fp64-default scanning, but other patterns still apply.
PRECISION_POLICY_EXEMPT: tuple[str, ...] = (
    "tractus_engine_advanced/src/tractus_engine_advanced/precision.py",
)


def _exempt_for_pattern(rel: str, pat: re.Pattern[str]) -> bool:
    """fp64-pattern lines under precision.py are policy, not violations."""
    is_fp64_pattern = "float64" in pat.pattern or "x64" in pat.pattern
    return is_fp64_pattern and rel in PRECISION_POLICY_EXEMPT


def _scan_python(roots: tuple[Path, ...], excludes: tuple[str, ...]) -> list[Hit]:
    hits: list[Hit] = []
    for path in iter_files(roots=roots, suffixes=(".py",), excludes=excludes):
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("#", 1)[0]
            for pat, msg in LEGACY_PATTERNS:
                if _exempt_for_pattern(rel, pat):
                    continue
                if pat.search(line):
                    hits.append(Hit(path=path, line=lineno, text=f"[{msg}] {raw}"))
    return hits


def _check_readme_pointer() -> list[Hit]:
    """README.md must still link to ARCHITECTURE.md as the source of truth."""
    readme = REPO_ROOT / "README.md"
    if not readme.exists():
        return [Hit(path=readme, line=0, text="README.md missing")]
    text = readme.read_text(encoding="utf-8", errors="replace")
    if "ARCHITECTURE.md" not in text:
        return [Hit(path=readme, line=0, text="README.md no longer references ARCHITECTURE.md")]
    return []


def check(roots: tuple[Path, ...], excludes: tuple[str, ...], strict: bool) -> int:
    """Doc-coherence regression lint."""

    hits = _scan_python(roots=roots, excludes=excludes)
    hits.extend(_check_readme_pointer())
    return report(
        hits,
        "doc-coherence regression lint (ARCH §1699-1703)",
        strict,
    )


def main() -> int:  # pragma: no cover
    parser = base_arg_parser(__doc__ or "")
    args = parser.parse_args()
    roots = resolve_roots(args.root)
    return check(roots=roots, excludes=(), strict=not args.report)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
