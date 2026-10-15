"""Parity + smoke tests for the JAX port of copy_synth.features.

Acceptance per Wave 2b dispatch (T2.4):
  1. Pure-JAX functions, signature-compatible with Torch ones.
  2. Parity: 100 random pairs at the project sample rate, with
     |jax_loss - torch_loss| <= 1e-5 * max(1.0, |torch_loss|).
  3. Grad-smoke: jax.grad(loss_fn)(target, fit) is finite, non-NaN.
  4. Self-test: identical inputs => loss = 0 (or <= 1e-9).

PARITY MODULE: every test here requires fp64 (jax_enable_x64=True).
The default slice (``pytest -q``) excludes parity tests; conftest
further blocks them outside the subprocess runner. The x64 flip is
deferred to a fixture so module import is fp32-clean even during
collection — without that, mere collection of this file polluted
``tests/test_fp32_default.py`` (cross-test global-config leak).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import jax
import jax.numpy as jnp


# Mark every test in this module as parity. The conftest's
# `pytest_collection_modifyitems` adds an unconditional skip outside
# the parity subprocess runner — so in the default slice the x64
# flip below never executes.
pytestmark = [pytest.mark.parity]


@pytest.fixture(autouse=True)
def _enable_x64_for_parity():
    """Enable jax_enable_x64 for parity-test bodies only.

    Module-level ``jax.config.update("jax_enable_x64", True)`` was the
    cross-test leak: pytest imports every collected file at startup, so
    any unrelated run (default slice, smoke tests, fp32_default) saw
    x64=True and broke. Deferring to a fixture keeps the flip scoped
    to actual test execution. The flag is process-global by JAX
    design; the conftest still requires the parity subprocess runner
    to keep the flip from polluting other tests in the same run.
    """
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pair(rng: np.random.Generator, sr: int):
    """Return a (target, pred) pair of float32 audio arrays."""
    duration_s = float(rng.uniform(0.3, 1.5))
    n = int(duration_s * sr)
    amp = float(rng.uniform(0.05, 0.5))
    target = (rng.standard_normal(n) * amp).astype(np.float32)
    perturb = float(rng.uniform(0.01, 0.2))
    pred = target + (rng.standard_normal(n) * perturb).astype(np.float32)
    return target, pred


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_signature_compatibility():
    """Same public surface as the Torch module."""
    from tractus_controller.copy_synth import features as f_torch
    from tractus_controller.copy_synth import features_jax as f_jax

    for name in ("FeatureConfig", "log_mel", "mr_stft_magnitudes",
                 "acoustic_loss", "precompute_target"):
        assert hasattr(f_jax, name), f"features_jax missing {name}"
        assert hasattr(f_torch, name), f"features missing {name}"

    # Same default config keys.
    cfg_t = f_torch.FeatureConfig()
    cfg_j = f_jax.FeatureConfig()
    for field in ("sample_rate", "n_mels", "win_ms", "hop_ms", "f_min",
                  "f_max", "mr_stft_n_ffts", "mr_stft_win_factors",
                  "mr_stft_hop_factors"):
        assert getattr(cfg_t, field) == getattr(cfg_j, field), field


def test_self_loss_is_zero():
    """Identical inputs => loss == 0 (or <= 1e-9)."""
    from tractus_controller.copy_synth import features_jax as f_jax

    rng = np.random.default_rng(42)
    cfg = f_jax.FeatureConfig()
    audio = (rng.standard_normal(int(0.7 * cfg.sample_rate)) * 0.2).astype(np.float64)
    audio_j = jnp.asarray(audio, dtype=jnp.float64)
    tgt_mel, tgt_mr = f_jax.precompute_target(audio_j, cfg)
    loss = float(f_jax.acoustic_loss(audio_j, tgt_mel, tgt_mr, cfg))
    assert loss <= 1e-9, f"self-loss not zero: {loss}"


def test_parity_with_torch_100_pairs():
    """Per-pair |jax_loss - torch_loss| <= 1e-5 * max(1, |torch_loss|).

    Both backends run on the same float32 audio so there's no dtype-mismatch
    handicap. The parity loop expects every pair to satisfy the tolerance.
    """
    pytest.importorskip("torch")

    from tractus_controller.copy_synth import features as f_torch
    from tractus_controller.copy_synth import features_jax as f_jax

    cfg_t = f_torch.FeatureConfig()
    cfg_j = f_jax.FeatureConfig()
    sr = cfg_t.sample_rate

    rng = np.random.default_rng(0)
    n_trials = 100
    diffs = []
    rels = []
    for _ in range(n_trials):
        target, pred = _make_pair(rng, sr)
        t_pre = f_torch.precompute_target(target, cfg_t)
        j_pre = f_jax.precompute_target(jnp.asarray(target, dtype=jnp.float32), cfg_j)
        torch_loss = f_torch.acoustic_loss(pred, t_pre[0], t_pre[1], cfg_t)
        jax_loss = float(
            f_jax.acoustic_loss(
                jnp.asarray(pred, dtype=jnp.float32), j_pre[0], j_pre[1], cfg_j,
            )
        )
        d = abs(torch_loss - jax_loss)
        rel = d / max(1.0, abs(torch_loss))
        diffs.append(d)
        rels.append(rel)

    diffs = np.array(diffs)
    rels = np.array(rels)
    assert rels.max() <= 1e-5, (
        f"parity violated: max abs diff = {diffs.max():.3e}, "
        f"max rel diff = {rels.max():.3e}"
    )


def test_grad_is_finite():
    """jax.grad(loss)(pred) returns a finite, non-NaN gradient."""
    from tractus_controller.copy_synth import features_jax as f_jax

    rng = np.random.default_rng(7)
    cfg = f_jax.FeatureConfig()
    sr = cfg.sample_rate
    n = int(0.5 * sr)
    target = (rng.standard_normal(n) * 0.2).astype(np.float64)
    pred = (target + rng.standard_normal(n) * 0.05).astype(np.float64)
    target_j = jnp.asarray(target, dtype=jnp.float64)
    pred_j = jnp.asarray(pred, dtype=jnp.float64)

    tgt_mel, tgt_mr = f_jax.precompute_target(target_j, cfg)

    def loss_fn(p):
        return f_jax.acoustic_loss(p, tgt_mel, tgt_mr, cfg)

    grad = jax.grad(loss_fn)(pred_j)
    grad_np = np.asarray(grad)
    assert grad_np.shape == pred_j.shape
    assert np.all(np.isfinite(grad_np)), "gradient contains non-finite entries"
    # And it should be non-trivially non-zero (we shifted pred away from target).
    assert np.abs(grad_np).max() > 0.0, "gradient is identically zero"


def test_log_mel_returns_array_not_python_float():
    """The whole point of the port: don't strip the gradient channel."""
    from tractus_controller.copy_synth import features_jax as f_jax

    cfg = f_jax.FeatureConfig()
    audio = jnp.asarray(np.zeros(2048, dtype=np.float32))
    out = f_jax.log_mel(audio, cfg)
    assert isinstance(out, jax.Array)
    assert out.ndim == 2
