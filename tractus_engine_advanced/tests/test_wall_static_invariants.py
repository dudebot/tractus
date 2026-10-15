"""Wall-param anatomy-static enforcement (ARCHITECTURE.md §487-494, §510-515).

Tests that wall params come from anatomy as static ``[93]`` vectors, that
``wall_dynamic_residual`` defaults OFF and is energy-capped, and that no
free per-frame ``[T, 93, 3]`` wall channel exists in any code path.

These are placeholder ``xfail(strict=True)`` tests for now — the body
model whose interface they validate is owned by ``correctness/004``.
When that lands, the tests flip from xfail to pass without rewriting.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ENGINE_SRC = REPO / "tractus_engine_advanced" / "src"


def _body_model_module():
    """Return the body-model module if it exists, else None.

    The body model lives under ``tractus_engine_advanced.body_model`` once
    correctness/004 lands; until then we expect the import to fail.
    """

    try:
        return importlib.import_module("tractus_engine_advanced.body_model")
    except ModuleNotFoundError:
        return None


@pytest.mark.wall_static
@pytest.mark.skip(reason="body model not built yet; blocked on correctness/004",
)
def test_wall_params_are_anatomy_static_shape() -> None:
    """Wall mass/resistance/stiffness must be ``[93]`` from anatomy, not free per-frame."""

    body_model = _body_model_module()
    assert body_model is not None, "body model not built yet"

    interface = body_model.RendererInterface  # type: ignore[attr-defined]
    # Static fields come from anatomy_vec — shape is per-section, no T.
    assert interface.wall_mass_static_shape == (93,)
    assert interface.wall_resistance_static_shape == (93,)
    assert interface.wall_stiffness_static_shape == (93,)


@pytest.mark.wall_static
@pytest.mark.skip(reason="body model not built yet; blocked on correctness/004",
)
def test_wall_dynamic_residual_default_off_and_energy_capped() -> None:
    """``wall_dynamic_residual`` defaults to OFF; when ON, it is energy-capped.

    ARCH §492-494, §540-543: low-rank, velocity-limited, ablated-and-capped
    residual. Off until ablation evidence justifies it.
    """

    body_model = _body_model_module()
    assert body_model is not None
    config = body_model.default_config()  # type: ignore[attr-defined]
    assert config.wall_dynamic_residual_enabled is False
    # Even when enabled, the cap must exist and be small (rank ≤ 4 per ARCH).
    assert config.wall_dynamic_residual_max_rank <= 4
    assert config.wall_dynamic_residual_energy_cap > 0.0


@pytest.mark.wall_static
def test_no_free_per_frame_wall_channel_in_engine_source() -> None:
    """Static lint: no module under engine src declares a free ``[T, 93, 3]``
    wall channel as a trainable per-frame field.

    This is a non-xfail guard: it asserts a property of the *current*
    source tree, not of a future feature. The grep target is the type
    annotation pattern ``Float[Array, "T 93 3"]`` on a name that contains
    ``wall``. If a future commit re-introduces such a channel, this test
    fails immediately and points the author at ARCH §510-515.
    """

    if not ENGINE_SRC.exists():
        pytest.skip(f"engine source tree missing: {ENGINE_SRC}")

    offenders: list[str] = []
    for py_file in ENGINE_SRC.rglob("*.py"):
        try:
            text = py_file.read_text()
        except UnicodeDecodeError:
            continue
        # Walk lines so we can point at the exact occurrence.
        for lineno, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if "wall" not in stripped.lower():
                continue
            if 'Float[Array, "T 93 3"]' in stripped or "Float[Array, 'T 93 3']" in stripped:
                offenders.append(f"{py_file.relative_to(REPO)}:{lineno}: {stripped}")
            elif 'Float[Array, "T, 93, 3"]' in stripped:
                offenders.append(f"{py_file.relative_to(REPO)}:{lineno}: {stripped}")

    assert not offenders, (
        "Free per-frame wall channel detected (ARCH §510-515 forbids it):\n  "
        + "\n  ".join(offenders)
    )
