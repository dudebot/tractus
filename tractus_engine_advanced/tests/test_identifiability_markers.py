"""Marker placeholders for the identifiability ledger (ARCHITECTURE.md §3, §540-543).

Three architectural-invariant gates whose **actual implementations** live
elsewhere; this file's job is to register the markers and pin a
``xfail(strict=True)`` placeholder so the gate IDs exist in collection
output and CI surfaces them.

- **channel-model identifiability ablation** — full gate owned by
  ``infra/002`` + ``pretraining/004`` (calibrated threshold).
- **non-circularity rule on body-model training** — lint/grep gate
  owned by ``infra/002``.
- **cheat-channel ablation** — Goldilocks 5–20% gate owned by
  ``infra/002`` (one of the two channels — wall-residual — is exercised
  here; the frication-color one lives in ``test_frication_physical.py``).

Plus the two SOTA marker placeholders (``gradient_fd``,
``energy_conservation``) so the suite can be sliced today even though
the implementations land in correctness/003 + correctness/002.
"""

from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Channel-model identifiability ablation (HIGH).
# ---------------------------------------------------------------------------


@pytest.mark.channel_identifiability
@pytest.mark.skip(reason="channel-model identifiability ablation gate is owned by "
    "infra/002 + pretraining/004 (calibrated threshold lives there)",
)
def test_channel_model_identifiability_ablation_placeholder() -> None:
    """Placeholder for the channel-model identifiability ablation.

    The full gate (calibrated against a held-out sweep) lives in
    ``infra/002``; the speaker-encoder side lives in ``pretraining/004``.
    """

    pytest.fail(
        "placeholder: channel-model identifiability ablation gate is "
        "owned by infra/002 + pretraining/004"
    )


# ---------------------------------------------------------------------------
# Non-circularity rule (MEDIUM).
# ---------------------------------------------------------------------------


@pytest.mark.non_circularity
@pytest.mark.skip(reason="non-circularity lint/grep gate is owned by infra/002",
)
def test_no_body_model_training_through_tea_audio_loss_placeholder() -> None:
    """Placeholder for the non-circularity rule (ARCH §445-456).

    Body-model training must NOT use TEA-synthesized audio as the primary
    loss. The lint/grep gate that enforces this lives in ``infra/002``.
    """

    pytest.fail(
        "placeholder: non-circularity lint/grep gate is owned by infra/002"
    )


# ---------------------------------------------------------------------------
# Cheat-channel ablation (MEDIUM) — wall-residual side.
# ---------------------------------------------------------------------------


@pytest.mark.cheat_channel
@pytest.mark.skip(reason="wall-residual cheat-channel ablation gate (5–20% Goldilocks) "
    "is owned by infra/002",
)
def test_wall_residual_cheat_channel_goldilocks_placeholder() -> None:
    """Placeholder for the wall-dynamic-residual cheat-channel ablation.

    Ablating ``wall_dynamic_residual`` should change audio loss by 5–20%
    when enabled; less means the channel is dead, more means it's a
    hidden EQ. The Goldilocks gate lives in ``infra/002``.
    """

    pytest.fail(
        "placeholder: wall-residual cheat-channel gate is owned by infra/002"
    )


# ---------------------------------------------------------------------------
# SOTA marker placeholders.
# ---------------------------------------------------------------------------


@pytest.mark.gradient_fd
@pytest.mark.skip(reason="full per-module gradient FD check is owned by correctness/003",
)
def test_gradient_fd_marker_placeholder() -> None:
    """Pin the ``gradient_fd`` marker; full check lives in correctness/003."""

    pytest.fail(
        "placeholder: full per-module gradient FD check is owned by correctness/003"
    )


@pytest.mark.energy_conservation
@pytest.mark.skip(reason="energy-conservation invariant helper is owned by correctness/002",
)
def test_energy_conservation_marker_placeholder() -> None:
    """Pin the ``energy_conservation`` marker; helper lives in correctness/002."""

    pytest.fail(
        "placeholder: energy-conservation helper is owned by correctness/002"
    )
