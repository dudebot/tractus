#!/usr/bin/env python3
"""Reference-pack hygiene (ARCH §1706-1711).

Every URL cited in ``ARCHITECTURE.md`` §15 must have either:

a. A corresponding entry in ``docs/references/pages/reference_links.md``, or
b. A locally mirrored PDF under ``docs/references/`` whose filename
   contains an obvious slug from the URL (e.g., the arXiv ID).

Caveat: ARCHITECTURE.md cites several textbook references with no URL
(Stevens 1998, Fant 1960, etc.). Those are explicitly skipped.

This is a structural lint — bit-rot of references is the whole risk;
the script does not network-fetch (offline-friendly).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

from _common import (  # type: ignore[import-not-found]
    Hit,
    REPO_ROOT,
    base_arg_parser,
    report,
)


URL_RE = re.compile(r"https?://[^\s)>\]\"']+", re.IGNORECASE)


def _arch_section_15_text() -> str:
    arch = (REPO_ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8", errors="replace")
    # Crude but stable: take everything from "## 15. References" up to the
    # next "## " heading (which is "## 16.").
    m = re.search(r"^## 15\. References.*?(?=^## )", arch, re.DOTALL | re.MULTILINE)
    return m.group(0) if m else ""


def _references_links_text() -> str:
    p = REPO_ROOT / "docs" / "references" / "pages" / "reference_links.md"
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8", errors="replace")


def _local_pdf_slugs() -> set[str]:
    """Collect filename-stems of local mirrored PDFs (lowercase, no extension)."""
    refs = REPO_ROOT / "docs" / "references"
    out: set[str] = set()
    if not refs.exists():
        return out
    for p in refs.rglob("*.pdf"):
        out.add(p.stem.lower())
    return out


def _arxiv_id_from(url: str) -> str | None:
    m = re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", url, re.IGNORECASE)
    return m.group(1) if m else None


def check(roots, excludes, strict: bool) -> int:
    """Reference-pack hygiene."""

    section_15 = _arch_section_15_text()
    if not section_15:
        return report(
            [Hit(path=REPO_ROOT / "ARCHITECTURE.md", line=0, text="§15 not found")],
            "reference-pack hygiene",
            strict,
        )

    urls = sorted(set(URL_RE.findall(section_15)))
    links_text = _references_links_text()
    pdf_slugs = _local_pdf_slugs()

    hits: list[Hit] = []
    for url in urls:
        # Trim trailing punctuation that often glues onto markdown URLs.
        clean = url.rstrip(".,);:")
        if clean in links_text:
            continue
        # Allow either a textually-equal URL OR a known short form (arXiv id
        # appearing in the links file, which lists abs-form URLs).
        arxiv = _arxiv_id_from(clean)
        if arxiv and arxiv in links_text:
            continue
        if arxiv and any(arxiv in s for s in pdf_slugs):
            continue
        # Last-resort fuzzy: domain+path-tail in links_text.
        m = re.search(r"https?://([^/]+)(/.*)?$", clean)
        if m:
            host = m.group(1)
            tail = (m.group(2) or "").rstrip("/").rsplit("/", 1)[-1]
            if host and tail and host in links_text and tail in links_text:
                continue
        hits.append(
            Hit(
                path=REPO_ROOT / "ARCHITECTURE.md",
                line=0,
                text=f"§15 URL not in reference_links.md or local PDF: {clean}",
            )
        )

    return report(
        hits,
        "reference-pack hygiene (ARCH §1706-1711)",
        strict,
    )


def main() -> int:  # pragma: no cover
    parser = base_arg_parser(__doc__ or "")
    args = parser.parse_args()
    return check(roots=(), excludes=(), strict=not args.report)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
