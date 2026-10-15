"""Module-1 (Linguistic / Prosody Planner) — text → segments + prosody.

ARCHITECTURE.md §2.1. Speaker-agnostic; sees no audio loss; size cap
≤ 10 M params. v0 ships an ARPAbet-tokenised transformer encoder–decoder
with five heads (duration, F0 envelope, loudness, phrase-break, alignment).

Affect is a first-class input (D_affect = 8). Unlabelled samples → zero
vector ("neutral"). The model is shared across speakers; F0 *range* (not
contour) is the only speaker-conditional input that ever reaches it.

This package is orthogonal to TEA (no audio gradients ever cross this
seam). Renderer/audio code lives in ``tractus_engine_advanced``.
"""

from __future__ import annotations

# Tokens are pure-Python and always-importable.
from .tokens import (
    AFFECT_DIM,
    PAD_ID,
    PHONE_VOCAB,
    PhoneTokenizer,
    pad_sequence,
)

# Model + losses depend on torch, which is an optional extra
# (see tractus_controller/pyproject.toml `[project.optional-dependencies] torch`).
# Lazy-load on first attribute access so `import tractus_controller.planner`
# works on a default `uv sync` without the torch extra installed.
_TORCH_LAZY = {
    "PlannerConfig": ("tractus_controller.planner.model", "PlannerConfig"),
    "PlannerModel": ("tractus_controller.planner.model", "PlannerModel"),
    "planner_param_count": ("tractus_controller.planner.model", "planner_param_count"),
    "PlannerLossWeights": ("tractus_controller.planner.losses", "PlannerLossWeights"),
    "planner_loss": ("tractus_controller.planner.losses", "planner_loss"),
}


def __getattr__(name: str):  # PEP 562
    if name in _TORCH_LAZY:
        import importlib
        mod_name, attr = _TORCH_LAZY[name]
        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise ImportError(
                f"tractus_controller.planner.{name} requires torch (an optional "
                "extra). Install with: uv sync --extra torch"
            ) from e
        return getattr(mod, attr)
    raise AttributeError(f"module 'tractus_controller.planner' has no attribute {name!r}")


__all__ = [
    "AFFECT_DIM",
    "PAD_ID",
    "PHONE_VOCAB",
    "PhoneTokenizer",
    "pad_sequence",
    "PlannerConfig",
    "PlannerModel",
    "planner_param_count",
    "PlannerLossWeights",
    "planner_loss",
]
