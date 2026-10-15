"""JSON output helpers — schema-validated stdout reports."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def emit(report: dict, summary: str, schema_path: Path | None = None) -> None:
    """Print JSON to stdout, summary to stderr, optionally validating
    against a JSON schema file. Validation failures are themselves
    written to stderr but do not corrupt the JSON on stdout — callers
    decide whether to bubble that into a non-zero exit."""
    if schema_path is not None and schema_path.exists():
        try:
            import jsonschema  # type: ignore

            schema = json.loads(schema_path.read_text())
            jsonschema.validate(report, schema)
        except Exception as exc:  # pragma: no cover - reported, not fatal
            print(f"WARN: schema validation failed: {exc}", file=sys.stderr)
    print(json.dumps(report, indent=2))
    print(summary, file=sys.stderr)
