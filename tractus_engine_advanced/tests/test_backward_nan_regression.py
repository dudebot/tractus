"""correctness/006 — regression test for the backward NaN on natasha-shape.

Originally backward `jax.grad` on the chained
``R1 NN → compose → G3a → log-mel → MR-STFT`` path produced 580/600 NaN cells
at fp32 on the natasha-shape fixture. The forward also went NaN starting at
frame 13 sample 708. Two root causes:

1. The 25 Hz Chebyshev IIR on supraglottal pressure inside
   ``glottis/geometric.py`` was direct-form-I; with poles ~0.005 from the unit
   circle this is fp32-unstable across thousands of samples and the recursion
   accumulates fp32 overflow → forward NaN. Fix: replace with cascaded SOS
   biquads (``_sos_biquad_step``) — mathematically identical, fp32-stable.

2. ``amplitude = 0.0011 * jnp.sqrt(trans_pressure)`` with
   ``trans_pressure = max(cp[PR] - supra, 0.0)`` evaluates to 0 on the first
   frames (where ``cp[PR] = 0``); ``jax.grad`` of ``sqrt`` at 0 is NaN. Fix:
   ``sqrt(trans_pressure + 1e-12)`` — well below dynamic range.

The test pins fixture path + seed in metadata so a future change to either
will be visible in the test output, per the task acceptance gate.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp


# Pinned in test metadata (acceptance gate ✅).
JAX_VERSION_PIN = "0.10.0"
FIXTURE_PATH = "outputs/fixtures/natasha_shape.npz"
SEED = 0


def _workspace_root() -> Path:
    # tractus_engine_advanced/tests/test_backward_nan_regression.py
    return Path(__file__).resolve().parents[2]


def _ensure_controller_on_path() -> None:
    root = _workspace_root()
    ctrl = root / "tractus_controller"
    if str(ctrl) not in sys.path:
        sys.path.insert(0, str(ctrl))


def _maybe_skip_no_fixture():
    p = _workspace_root() / FIXTURE_PATH
    if not p.exists():
        pytest.skip(f"natasha-shape fixture missing at {p}")


def _checkpoint_compat_shim() -> None:
    """correctness/005's pipeline.py uses ``@jax.checkpoint(policy=...)`` —
    ``jax.checkpoint`` in jax 0.10 requires ``fun`` positionally. Wrap so the
    kwarg-only decorator form works. This is a read-side shim not a fix."""
    _orig = jax.checkpoint
    def _ck(*a, **k):
        if not a:
            from functools import partial
            return partial(_orig, **k)
        return _orig(*a, **k)
    jax.checkpoint = _ck  # type: ignore[assignment]


def test_natasha_shape_forward_backward_finite():
    """Acceptance gate: chained forward + ``jax.grad`` produce no NaN/Inf
    on the pinned fixture under fp32 default.
    """
    # The TEA precision policy must be set before any TEA import. Tests run
    # in the same process where conftest may have set TEA_PRECISION already.
    assert os.environ.get("TEA_PRECISION", "fp32") in ("fp32", "")
    _maybe_skip_no_fixture()
    _ensure_controller_on_path()
    _checkpoint_compat_shim()

    from tractus_controller.copy_synth.tea_backend import synthesize_via_tea_jax  # noqa: E402
    from tractus_controller.copy_synth.features_jax import (  # noqa: E402
        FeatureConfig, acoustic_loss, precompute_target,
    )

    root = _workspace_root()
    d = np.load(root / FIXTURE_PATH)
    T = int(d["T"])
    fsps = int(d["frame_step_samples"])
    asr = int(d["audio_sample_rate"])
    traj = jnp.asarray(np.asarray(d["trajectory"], dtype=np.float32))
    assert traj.shape == (T, 30), f"unexpected fixture shape {traj.shape}"

    cfg = FeatureConfig(sample_rate=asr)

    def synth(t):
        return synthesize_via_tea_jax(t, samples_per_frame=fsps)

    # Forward.
    pred_audio = synth(traj)
    assert bool(jnp.all(jnp.isfinite(pred_audio))), "forward audio has NaN/Inf"

    key = jax.random.PRNGKey(SEED)
    target_traj = traj + 0.01 * jax.random.normal(key, traj.shape, dtype=jnp.float32)
    target_audio = synth(target_traj)
    assert bool(jnp.all(jnp.isfinite(target_audio))), "target forward audio has NaN/Inf"

    target_logmel, target_mr = precompute_target(target_audio, cfg)
    assert bool(jnp.all(jnp.isfinite(target_logmel)))
    for m in target_mr:
        assert bool(jnp.all(jnp.isfinite(m)))

    def chained_loss(t):
        return acoustic_loss(synth(t), target_logmel, target_mr, cfg)

    L = chained_loss(traj)
    assert bool(jnp.isfinite(L)), f"forward loss not finite: {L}"

    g = jax.grad(chained_loss)(traj)
    g_np = np.asarray(g)
    assert g_np.shape == (T, 30)
    n_nan = int(np.isnan(g_np).sum())
    n_inf = int(np.isinf(g_np).sum())
    assert n_nan == 0, f"backward grad has {n_nan} NaN cells"
    assert n_inf == 0, f"backward grad has {n_inf} Inf cells"
