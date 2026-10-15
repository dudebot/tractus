"""Smoke pipeline tests (smoke/001 front-half).

Coverage:
- ARCH §315 GestureEvent dataclass: construction + serialization.
- text -> phones G2P
- pipeline.render_text_to_audio on a single short utterance
- WAV + sidecar writers produce well-formed artifacts
- determinism: same seed -> bit-equal audio across two invocations

The 20-utterance regression render is exercised by
``scripts/smoke_render_corpus.py``; running it here would dominate
test wall-time (~3 min) and is not the right fit for unit tests.

The Whisper-WER intelligibility gate (smoke/002) is intentionally
NOT run here — that's gated on P0 close.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _have_espeak() -> bool:
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _have_natasha_fixture() -> bool:
    """Check whether the natasha_shape neutral baseline is reachable.

    Smoke needs ``outputs/fixtures/natasha_shape.npz`` (gitignored).
    Tests skip cleanly if it's missing.
    """
    from tractus_controller.smoke import templates

    try:
        templates._natasha_path()
        return True
    except FileNotFoundError:
        return False


def _have_r1_checkpoint() -> bool:
    from tractus_controller.adapters.r1_inference import DEFAULT_CHECKPOINT

    return Path(DEFAULT_CHECKPOINT).exists()


def _x64_leaked() -> bool:
    """Detect cross-test JAX x64 pollution.

    Some tests (parity / planner) enable ``jax_enable_x64`` without
    subprocess isolation, which makes the engine pipeline crash on
    fp32 arrays. The smoke pipeline expects the fp32 default
    advertised in CLAUDE.md / correctness/001. When the leak has
    already happened, skip rather than fail — the leak is its own
    open issue, separately tracked by ``test_fp32_default.py``.
    """
    try:
        import jax
        return bool(jax.config.x64_enabled)
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------------------------
# GestureEvent dataclass.
# ---------------------------------------------------------------------------


def test_gesture_event_construction_arch_315() -> None:
    """ARCH §315 schema. Field names are channel_id / onset_s /
    target_value / duration_s / release_s / strength / ramp_shape /
    prosody_tags / affect_tags. (The placeholder test_gesture_event.py
    file uses an older proposed schema and is skipped.)
    """
    from tractus_controller.gestures import GestureEvent

    g = GestureEvent(
        channel_id="lip_aperture",
        onset_s=0.10,
        target_value=0.5,
        duration_s=0.15,
        release_s=0.02,
        strength=0.8,
        ramp_shape="hold_release",
        prosody_tags=("primary_stress",),
        affect_tags=(),
    )
    assert g.channel_id == "lip_aperture"
    assert g.offset_s == pytest.approx(0.25)
    assert g.end_s() == pytest.approx(0.27)


def test_gesture_event_validation() -> None:
    from tractus_controller.gestures import GestureEvent

    with pytest.raises(ValueError):
        GestureEvent(channel_id="x", onset_s=0.0, target_value=0.0,
                     duration_s=0.0)
    with pytest.raises(ValueError):
        GestureEvent(channel_id="x", onset_s=0.0, target_value=0.0,
                     duration_s=0.1, strength=1.5)
    with pytest.raises(ValueError):
        GestureEvent(channel_id="x", onset_s=0.0, target_value=0.0,
                     duration_s=0.1, ramp_shape="cubic-bezier")  # type: ignore[arg-type]


def test_gesture_event_round_trip() -> None:
    from tractus_controller.gestures import GestureEvent

    g = GestureEvent(
        channel_id="F0", onset_s=0.0, target_value=110.0, duration_s=0.1,
        prosody_tags=("primary_stress", "phone:EY"),
    )
    payload = g.to_dict()
    restored = GestureEvent.from_dict(payload)
    assert restored == g


# ---------------------------------------------------------------------------
# Phone templates.
# ---------------------------------------------------------------------------


def test_phone_templates_cover_arpabet_subset() -> None:
    """Every ARPAbet phone in DAISY_SUBSET has a template."""
    from tractus_controller.phonemes.arpabet import DAISY_SUBSET
    from tractus_controller.smoke.templates import PHONE_TEMPLATES

    missing = sorted(set(DAISY_SUBSET) - set(PHONE_TEMPLATES))
    assert not missing, f"phones missing templates: {missing}"


def test_template_for_resolves_stress_digits() -> None:
    from tractus_controller.smoke.templates import template_for

    a = template_for("EY1")
    b = template_for("EY")
    assert a is b


# ---------------------------------------------------------------------------
# G2P.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _have_espeak(), reason="espeak-ng not installed")
def test_text_to_phones_daisy() -> None:
    from tractus_controller.smoke.pipeline import text_to_phones

    tokens = text_to_phones("Daisy")
    assert tokens[0] == "D"
    assert "Z" in tokens
    assert "EY1" in tokens or "EY0" in tokens or "EY2" in tokens


# ---------------------------------------------------------------------------
# Render + sidecar.
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (_have_espeak() and _have_natasha_fixture() and _have_r1_checkpoint()),
    reason="espeak-ng / natasha_shape / R1 checkpoint required for end-to-end smoke",
)
def test_render_short_utterance_finite_and_in_range(tmp_path: Path) -> None:
    """Render a short utterance; verify finite audio + sane RMS."""
    if _x64_leaked():
        pytest.skip(
            "cross-test jax_enable_x64 leak (separate issue, see "
            "test_fp32_default)"
        )
    from tractus_controller.smoke.pipeline import (
        SmokeConfig,
        render_text_to_audio,
        write_sidecar,
        write_wav,
    )

    cfg = SmokeConfig(sample_rate=22050, deterministic_seed=0)
    r = render_text_to_audio("hi there", cfg)

    assert np.isfinite(r.audio).all(), "non-finite samples"
    assert not np.isnan(r.audio).any()
    assert not np.isinf(r.audio).any()
    rms = float(np.sqrt(np.mean(r.audio.astype(np.float64) ** 2)))
    assert 0.005 <= rms <= 0.5, f"RMS {rms} outside the 0.005..0.5 sanity band"

    wav = write_wav(tmp_path / "hi.wav", r.audio, r.sample_rate)
    sidecar = write_sidecar(wav, r)
    assert wav.exists()
    assert sidecar.exists()

    payload = json.loads(sidecar.read_text())
    smoke = payload["extra"]["smoke"]
    for key in (
        "git_hash", "config", "input_text", "arpabet_tokens",
        "durations_s", "gesture_list", "anatomy", "consent_verified",
        "deterministic_seed",
    ):
        assert key in smoke, f"sidecar missing field {key}"
    assert smoke["consent_verified"] is True
    assert smoke["input_text"] == "hi there"


@pytest.mark.skipif(
    not (_have_espeak() and _have_natasha_fixture() and _have_r1_checkpoint()),
    reason="espeak-ng / natasha_shape / R1 checkpoint required for determinism check",
)
def test_render_is_deterministic_under_fixed_seed() -> None:
    """Same seed -> bit-equal audio across two invocations."""
    if _x64_leaked():
        pytest.skip(
            "cross-test jax_enable_x64 leak (separate issue, see "
            "test_fp32_default)"
        )
    from tractus_controller.smoke.pipeline import (
        SmokeConfig,
        render_text_to_audio,
    )

    cfg = SmokeConfig(sample_rate=22050, deterministic_seed=42)
    a = render_text_to_audio("the cat", cfg)
    b = render_text_to_audio("the cat", cfg)
    assert a.audio.shape == b.audio.shape
    assert np.array_equal(a.audio, b.audio), "deterministic-seed audio drift"
