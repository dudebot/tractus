"""Story-1996-derived anchor articulations — held-out sanity gate.

Loads `outputs/fixtures/story_1996_anchors.npz` (built by
`scripts/build_story_anchors.py`) and yields per-shape area + length
arrays at TEA's 40-section supraglottal resolution.

**Role per the 2026-05-09 course correction.** Story anchors are
**held-out vowel sanity gates only**, NOT primary supervision
targets. Render `/a, i, u/` through the body model + TEA, measure
F1/F2/F3 via `scripts/transfer_function_oral_only.py`, gate at
±20 % of Story values. Area-MSE-against-Story as a primary training
loss is rejected — see `ARCHITECTURE.md` §3.2 + `workplan/handoff/
2026-05-09_course_correction.md`.

Each yielded tuple is
``(name, area_cm2_tea, length_cm_tea)`` where:

- ``name`` is the IPA-ish key (e.g. ``"/i/"``);
- ``area_cm2_tea`` is a (40,) float32 area-function;
- ``length_cm_tea`` is a (40,) float32 per-section length (constant
  per shape, equals VTL/40).

Only shapes with ``shape_status == "ingested"`` are yielded — the
"missing" placeholders (``/ɝ/``, ``/m/``, ``/n/``, ``/ŋ/``, ``/b/``,
``/d/``, ``/g/``) are skipped at the loader level so callers don't
have to special-case NaN. See the NPZ ``provenance`` field for why
those are missing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np


# Default fixture location, relative to repo root.
_DEFAULT_NPZ = Path("tractus_controller/outputs/fixtures/story_1996_anchors.npz")


def _resolve_default_path() -> Path:
    """Walk up from this file to find the repo root, then resolve."""
    here = Path(__file__).resolve()
    # tractus_controller/tractus_controller/body_model/anchors.py
    # → repo root is parents[3].
    repo_root = here.parents[3]
    return repo_root / _DEFAULT_NPZ


def iter_story_anchors(
    npz_path: Path | str | None = None,
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(name, area_cm2_tea, length_cm_tea)`` for each ingested
    Story anchor.

    Parameters
    ----------
    npz_path : Path | str | None
        Path to the anchor NPZ. Defaults to the repo-root fixture
        location.

    Yields
    ------
    (name, area, length) : tuple of (str, ndarray (40,), ndarray (40,))
        Float32 arrays. Total tract length per shape = ``length.sum()``.
    """
    path = Path(npz_path) if npz_path is not None else _resolve_default_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Story anchor fixture not found at {path}. "
            f"Build it with `python scripts/build_story_anchors.py`."
        )
    data = np.load(path, allow_pickle=False)
    names = data["shape_names"]
    statuses = data["shape_status"]
    areas = data["area_cm2_tea"]
    lengths = data["length_cm_tea"]
    for i, status in enumerate(statuses):
        if str(status) != "ingested":
            continue
        yield (
            str(names[i]),
            np.asarray(areas[i], dtype=np.float32),
            np.asarray(lengths[i], dtype=np.float32),
        )
