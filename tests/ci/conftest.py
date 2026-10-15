"""pytest configuration for the CI-guardrail meta-test slice.

The lint scripts under ``scripts/ci/`` are invoked as standalone
processes (no ``__init__.py``, sibling-relative imports). To make them
importable for in-process testing we put ``scripts/ci`` on
``sys.path``; tests still prefer ``subprocess.run`` for end-to-end
parity with how CI actually invokes them.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_CI = REPO_ROOT / "scripts" / "ci"
if str(SCRIPTS_CI) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_CI))
