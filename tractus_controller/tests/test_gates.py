"""Tests for the multi-gate copy-synth evaluation panel.

Heavy items (Whisper inference) are skipped unless explicitly enabled via the
``RUN_WHISPER_TESTS=1`` environment variable — they take >10s on CPU and are
not appropriate for the default unit-test loop. The pure logic (text norm,
WER/CER, trajectory metrics, phone boundary alignment, CLI plumbing) all run
in the default loop with ``--skip-whisper``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_TC = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# Text metrics
# ---------------------------------------------------------------------------


def test_word_jaccard_basic():
    from tractus_controller.copy_synth.gates import word_jaccard

    assert word_jaccard("hello world", "hello world") == pytest.approx(1.0)
    assert word_jaccard("hello world", "goodbye moon") == pytest.approx(0.0)
    # punctuation/case-insensitive
    assert word_jaccard("Hello, World!", "hello world") == pytest.approx(1.0)


def test_wer_zero_for_identical():
    from tractus_controller.copy_synth.gates import word_error_rate

    assert word_error_rate("you are loved", "you are loved") == pytest.approx(0.0)


def test_wer_one_for_disjoint():
    from tractus_controller.copy_synth.gates import word_error_rate

    # All 3 ref words substituted → WER = 3/3 = 1.0
    assert word_error_rate("you are loved", "totally different phrase") == pytest.approx(1.0)


def test_cer_partial_overlap():
    from tractus_controller.copy_synth.gates import character_error_rate

    cer = character_error_rate("you are loved", "you are loved indeed")
    assert 0.0 < cer < 1.0


# ---------------------------------------------------------------------------
# Trajectory plausibility
# ---------------------------------------------------------------------------


def test_normalized_diffs_zero_for_constant_knots():
    from tractus_controller.copy_synth.gates import normalized_diffs

    K, C = 5, 3
    knots = np.full((K, C), 2.0)
    bounds = np.tile(np.array([0.0, 10.0]), (C, 1))
    times = np.linspace(0, 1, K)
    vel, acc, jrk = normalized_diffs(knots, bounds, times)
    assert vel.shape == (K - 1, C)
    assert acc.shape == (K - 2, C)
    assert jrk.shape == (K - 3, C)
    assert np.all(vel == 0.0)
    assert np.all(acc == 0.0)
    assert np.all(jrk == 0.0)


def test_per_channel_drift_zero_when_at_init():
    from tractus_controller.copy_synth.gates import per_channel_drift

    init = np.array([[1.0, 2.0], [3.0, 4.0]])
    knots = init.copy()
    bounds = np.array([[0.0, 10.0], [0.0, 10.0]])
    drift = per_channel_drift(knots, init, bounds)
    assert drift.shape == (2,)
    assert np.allclose(drift, 0.0)


def test_per_channel_drift_normalized_to_width():
    from tractus_controller.copy_synth.gates import per_channel_drift

    init = np.zeros((2, 1))
    knots = np.full((2, 1), 5.0)
    bounds = np.array([[0.0, 10.0]])
    drift = per_channel_drift(knots, init, bounds)
    assert drift[0] == pytest.approx(0.5)


def test_phone_boundary_alignment_perfect_when_synced():
    from tractus_controller.copy_synth.gates import phone_boundary_alignment

    # Build a 1-D trajectory where channel idx 0 has clean sign-changes at 100ms
    # and 200ms (frame indices 5 and 10 at 49 Hz).
    knot_rate = 49.0
    n = 30
    traj = np.zeros((n, 30))
    # Velocity sign-changes at frame 5 and frame 10.
    times = np.arange(n) / knot_rate
    traj[:, 0] = np.where(times < 5 / knot_rate, times, 2 * 5 / knot_rate - times)
    # Make a second sign change near frame 10.
    traj[10:, 0] = np.linspace(traj[10, 0], traj[10, 0] + 0.1, n - 10)

    phones = [
        {"phoneme": "AA", "t_start": 0.0, "t_end": 5 / knot_rate},
        {"phoneme": "BB", "t_start": 5 / knot_rate, "t_end": 10 / knot_rate},
        {"phoneme": "CC", "t_start": 10 / knot_rate, "t_end": (n - 1) / knot_rate},
    ]
    score = phone_boundary_alignment(
        traj, knot_rate, phones, free_idxs=(0,), tol_ms=30.0
    )
    assert score > 0.0


def test_phone_boundary_alignment_empty_when_no_phones():
    from tractus_controller.copy_synth.gates import phone_boundary_alignment

    traj = np.zeros((10, 30))
    assert phone_boundary_alignment(traj, 49.0, [], free_idxs=(0,)) == 0.0


# ---------------------------------------------------------------------------
# evaluate_fit on real fixture (skipping Whisper)
# ---------------------------------------------------------------------------


FIT_NPZ = REPO_TC / "outputs/fits/natasha_2_s15.npz"
FIT_WAV = REPO_TC / "outputs/fits/natasha_2_s15_fit.wav"
TARGET_WAV = REPO_TC / "outputs/natural_audio/natasha/chunks/natasha_2_s15.wav"
ALIGNMENT = REPO_TC / "outputs/fits/natasha_2_s15_alignment.json"
LAB = REPO_TC / "outputs/natural_audio/natasha/chunks/natasha_2_s15.lab"

V3_FIT_NPZ = REPO_TC / "outputs/fits_v3/natasha_2_s15.npz"
V3_FIT_WAV = REPO_TC / "outputs/fits_v3/natasha_2_s15_fit.wav"


_FIXTURES_PRESENT = all(
    p.exists() for p in (FIT_NPZ, FIT_WAV, TARGET_WAV, ALIGNMENT, LAB)
)


@pytest.mark.skipif(not _FIXTURES_PRESENT, reason="natasha_2_s15 fixtures not present")
def test_evaluate_fit_returns_finite_metrics():
    from tractus_controller.copy_synth.gates import evaluate_fit

    text = LAB.read_text().strip()
    report = evaluate_fit(
        FIT_NPZ, FIT_WAV, TARGET_WAV, text, ALIGNMENT, skip_whisper=True
    )
    # All four metric blocks must be non-NaN and finite.
    for d in (
        report.max_velocity_per_channel,
        report.max_acceleration_per_channel,
        report.max_jerk_per_channel,
        report.drift_per_channel,
    ):
        assert d, "expected non-empty per-channel dict"
        for v in d.values():
            assert np.isfinite(v), f"non-finite metric: {v!r}"
    assert np.isfinite(report.phone_boundary_alignment_score)
    # Verdict booleans always valid.
    assert isinstance(report.speech_recognized, bool)
    assert isinstance(report.physically_plausible, bool)
    assert isinstance(report.overall_pass, bool)


@pytest.mark.skipif(not _FIXTURES_PRESENT, reason="natasha_2_s15 fixtures not present")
def test_evaluate_fit_flags_v1_fit_as_failed():
    """The current natasha_2_s15 fit is known to be physically implausible
    (drift on F0, velocity excursions). The trajectory gates alone should fail it."""
    from tractus_controller.copy_synth.gates import evaluate_fit

    text = LAB.read_text().strip()
    report = evaluate_fit(
        FIT_NPZ, FIT_WAV, TARGET_WAV, text, ALIGNMENT, skip_whisper=True
    )
    assert report.physically_plausible is False, report.flags
    assert report.flags, "expected at least one flag for known-bad fit"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _FIXTURES_PRESENT, reason="natasha_2_s15 fixtures not present")
def test_cli_emits_valid_json_with_skip_whisper():
    """CLI must exit 0 and emit a parseable JSON GateReport."""
    text = LAB.read_text().strip()
    cmd = [
        sys.executable,
        "-m",
        "tractus_controller.copy_synth.gates",
        str(FIT_NPZ),
        str(FIT_WAV),
        str(TARGET_WAV),
        "--text",
        text,
        "--alignment",
        str(ALIGNMENT),
        "--skip-whisper",
        "--json",
    ]
    proc = subprocess.run(
        cmd, cwd=str(REPO_TC), capture_output=True, text=True, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    data = json.loads(proc.stdout)
    # Required top-level keys.
    for k in (
        "whisper_jaccard",
        "whisper_wer",
        "whisper_cer",
        "max_velocity_per_channel",
        "max_acceleration_per_channel",
        "max_jerk_per_channel",
        "drift_per_channel",
        "phone_boundary_alignment_score",
        "speech_recognized",
        "physically_plausible",
        "overall_pass",
        "flags",
    ):
        assert k in data, f"missing key: {k}"


# ---------------------------------------------------------------------------
# Optional Whisper round-trip (slow; gated on env)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not _FIXTURES_PRESENT or os.environ.get("RUN_WHISPER_TESTS") != "1",
    reason="set RUN_WHISPER_TESTS=1 to run Whisper integration test",
)
def test_evaluate_fit_with_whisper():
    from tractus_controller.copy_synth.gates import evaluate_fit

    text = LAB.read_text().strip()
    report = evaluate_fit(FIT_NPZ, FIT_WAV, TARGET_WAV, text, ALIGNMENT)
    assert report.whisper_target_text  # Whisper read the natural audio
    assert 0.0 <= report.whisper_jaccard <= 1.0
    assert report.whisper_wer >= 0.0
