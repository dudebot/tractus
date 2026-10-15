"""Runtime CI guardrail tests."""

from __future__ import annotations

import importlib

import pytest


# ---------------------------------------------------------------------------
# Cheat-channel ablation gate
# ---------------------------------------------------------------------------


def _frication_module():
    try:
        return importlib.import_module("tractus_engine_advanced.frication")
    except ModuleNotFoundError:
        return None


def _residual_module():
    try:
        return importlib.import_module("tractus_engine_advanced.wall_residual")
    except ModuleNotFoundError:
        return None


@pytest.mark.cheat_channel
def test_cheat_channel_goldilocks_tiny_fixture() -> None:
    """ARCH §540-543, §1682-1685.

    Ablating ``wall_dynamic_residual = 0`` AND ``frication_color = 0``
    on a *tiny* CI fixture (T <= 20 frames; explicitly NOT the full
    inversion-driven version) must change loss in [5%, 20%].

    < 5% → channel was unused (delete it).
    > 20% → channel was overpowered (re-regularize).

    The full inversion-driven version lives behind
    ``@pytest.mark.full_ablation`` (off the PR critical path; see the
    pytest marker registered in pyproject.toml below).
    """

    frication = _frication_module()
    residual = _residual_module()
    if frication is None or residual is None:
        pytest.skip(
            "awaits correctness/004 (frication) + cloning's wall-residual "
            "module; tiny CI fixture lights up when both are present"
        )

    # Once both modules ship:
    # 1. Build a tiny deterministic frame sequence (T=10..20) with a known
    #    fricative phone in it.
    # 2. Compute audio_loss(target, with_channels) and audio_loss(target,
    #    with_channels=0). Use the small-T fixture so we don't depend on
    #    correctness/007's chunked-scan landing.
    # 3. assert 0.05 <= |delta_loss / base_loss| <= 0.20.
    raise AssertionError("placeholder — flip on once modules ship")


# ---------------------------------------------------------------------------
# Frication-color physical-support gate
# ---------------------------------------------------------------------------


@pytest.mark.frication_physical
def test_frication_color_cannot_synthesize_without_physics() -> None:
    """ARCH §1684-1685: ``frication_color`` cannot synthesize frication
    where ``(constriction_area, pressure_drop)`` doesn't support it.

    Identical contract to
    tractus_engine_advanced/tests/test_frication_physical.py
    test_color_residual_cannot_create_frication_without_physics, but
    surfaced under tests/ci/ so the CI guardrail slice owns it. The
    engine-side test stays as a unit test; this is the *gate*.
    """

    frication = _frication_module()
    if frication is None:
        pytest.skip("awaits correctness/004: frication module not yet built")

    import numpy as np

    constriction = np.zeros(93, dtype=np.float32)
    pressure = np.zeros(93, dtype=np.float32)
    color = np.full(93, 1.0, dtype=np.float32)
    gain = frication.gain_with_color(  # type: ignore[attr-defined]
        constriction_area_cm2=constriction,
        pressure_drop_dPa=pressure,
        place_index=np.zeros(93, dtype=np.int32),
        voicing=np.zeros(93, dtype=np.float32),
        obstacle_proxy=np.zeros(93, dtype=np.float32),
        color=color,
    )
    assert np.max(np.abs(gain)) < 1e-6, (
        f"frication_color synthesized {np.max(np.abs(gain)):.3g} energy "
        "without physical support"
    )
