"""Project path conventions. Single source of truth for layout decisions.

Layout (workspace monorepo):

    tractus/                         <-- workspace_root()
    ├── tractus_controller/          <-- package_root() — TC's package dir
    │   ├── tractus_controller/      <-- this file's grandparent
    │   ├── tests/
    │   └── outputs/                 <-- TC-local outputs
    ├── tractus_engine_advanced/
    └── tractus_shared/

The `TRACTUS_WORKSPACE_ROOT` env var overrides workspace detection — useful
when running tests from an installed wheel where the source tree isn't
where you'd expect.
"""

from __future__ import annotations

import os
from pathlib import Path


def package_root() -> Path:
    """The tractus_controller package's own directory.

    `paths.py` lives at `<package_root>/tractus_controller/paths.py`, so two
    levels up = the package's outer directory (where `pyproject.toml` and
    `tests/` live).
    """
    return Path(__file__).resolve().parent.parent


def workspace_root() -> Path:
    """The monorepo workspace root.

    Override via `TRACTUS_WORKSPACE_ROOT` env var. Default: the parent of
    the package_root, which is `tractus/` in the standard checkout.
    """
    override = os.environ.get("TRACTUS_WORKSPACE_ROOT")
    if override:
        return Path(override).resolve()
    return package_root().parent


def fixtures_dir() -> Path:
    """TC-OWN test fixtures (own .ges files, daisy_phonemes.json, etc)."""
    return package_root() / "tests" / "fixtures"


def outputs_dir() -> Path:
    return package_root() / "outputs"
