#!/usr/bin/env python3
"""Non-circularity rule (ARCH §445-456, §1656).

Body-model training scripts must NOT use ``synthesize_dynamic_tube``
(or any TEA-synth-output) as their primary loss target. Doing so closes
a circular loop where the body model is trained to fit the simulator
that's supposed to be teaching it — the resulting "physics-first"
system is just a neural decoder with extra steps.

Per the task spec: ``grep -rE "loss.*=.*synthesize_dynamic_tube|
audio_loss.*body_model_output"`` over the current package set must
return zero.

Scope:

- Production source under the body-model training surfaces. Today this
  means everything under ``tractus_controller/`` and
  ``tractus_engine_advanced/src/`` *except* the TEA pipeline itself
  (which obviously calls ``synthesize_dynamic_tube`` — that's its job).
- Lint exempts the TEA pipeline file by path, NOT by name match. Any
  *other* file that puts ``synthesize_dynamic_tube`` on the right-hand
  side of a ``loss =`` / ``audio_loss =`` is a violation.
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


# Files that legitimately *implement* the synthesis primitives. They define
# them; they don't use them as a training target. Excluded from the lint.
IMPLEMENTATION_EXEMPT: tuple[str, ...] = (
    "tractus_engine_advanced/src/tractus_engine_advanced/pipeline.py",
    "tractus_engine_advanced/src/tractus_engine_advanced/__init__.py",
    "tractus_controller/tractus_controller/copy_synth/tea_backend.py",
)

# Pattern from the task spec.
PRIMARY_LOSS_PAT = re.compile(
    r"(?:\b\w*loss\w*\s*=.*\bsynthesize_dynamic_tube\b)"
    r"|(?:\baudio_loss\b.*\bbody_model_output\b)",
)


def check(roots: tuple[Path, ...], excludes: tuple[str, ...], strict: bool) -> int:
    """Body-model training must not close the TEA-synth circular loop."""

    hits: list[Hit] = []
    for path in iter_files(roots=roots, excludes=excludes):
        try:
            rel = path.relative_to(REPO_ROOT).as_posix()
        except ValueError:
            rel = path.as_posix()
        if rel in IMPLEMENTATION_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for lineno, raw in enumerate(text.splitlines(), start=1):
            line = raw.split("#", 1)[0]
            if PRIMARY_LOSS_PAT.search(line):
                hits.append(Hit(path=path, line=lineno, text=raw))
    return report(
        hits,
        "non-circularity rule (ARCH §445-456): body-model training must "
        "not use synthesize_dynamic_tube / body_model_output as primary loss",
        strict,
    )


def main() -> int:  # pragma: no cover
    parser = base_arg_parser(__doc__ or "")
    args = parser.parse_args()
    roots = resolve_roots(args.root)
    return check(roots=roots, excludes=(), strict=not args.report)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
