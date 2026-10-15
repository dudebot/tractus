"""JAX port of :mod:`copy_synth.features` (log-mel + multi-resolution STFT).

Mirrors the Torch / torchaudio implementation in
:mod:`tractus_controller.copy_synth.features` bit-for-bit (within float64
roundoff). Used by the gradient fitter so a JAX-side TEA forward can carry
gradients through the loss (T2.5 in the Wave 2b dispatch).

The Torch reference uses:

- ``torch.hann_window`` with default ``periodic=True``
  (i.e. ``w[n] = 0.5 * (1 - cos(2*pi*n / N))`` for n in 0..N-1).
- ``torch.stft`` with default ``center=True``, ``pad_mode='reflect'``,
  ``normalized=False``, ``onesided=True``. When ``win_length < n_fft`` the
  window is **center-padded with zeros** to length ``n_fft``.
- ``torchaudio.transforms.MelSpectrogram`` with ``power=2``, ``norm=None``,
  ``mel_scale='htk'``, freq grid ``linspace(0, sample_rate // 2, n_fft//2+1)``
  (note integer division on ``sample_rate // 2``).
- ``log(mel + 1e-6)``.

This module produces the same dataclass ``FeatureConfig`` keys as the Torch
module (re-exported here for convenience). All public functions return JAX
arrays — not Python floats — so callers can ``jax.grad`` through them.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    n_mels: int = 80
    win_ms: float = 25.0
    hop_ms: float = 10.0
    f_min: float = 50.0
    f_max: float | None = None  # default: sr / 2
    mr_stft_n_ffts: tuple[int, ...] = (512, 1024, 2048)
    mr_stft_win_factors: tuple[float, ...] = (1.0, 1.0, 1.0)  # win = factor * n_fft
    mr_stft_hop_factors: tuple[float, ...] = (0.25, 0.25, 0.25)


# ---------------------------------------------------------------------------
# Low-level kernels
# ---------------------------------------------------------------------------


def _hann_window_periodic(n: int, dtype) -> jnp.ndarray:
    """Periodic Hann (matches ``torch.hann_window(n)`` with default args)."""
    if n == 1:
        return jnp.ones((1,), dtype=dtype)
    k = jnp.arange(n, dtype=dtype)
    return 0.5 * (1.0 - jnp.cos(2.0 * jnp.pi * k / n))


def _center_pad_window(window: jnp.ndarray, n_fft: int) -> jnp.ndarray:
    """Center-pad ``window`` with zeros to length ``n_fft`` (matches torch.stft).

    When ``win_length < n_fft``, ``torch.stft`` pads the window symmetrically;
    if the gap is odd, the extra zero goes on the right
    (``pad_left = (n_fft - win) // 2``).
    """
    win = int(window.shape[0])
    if win == n_fft:
        return window
    pad_left = (n_fft - win) // 2
    pad_right = n_fft - win - pad_left
    return jnp.concatenate(
        [
            jnp.zeros((pad_left,), dtype=window.dtype),
            window,
            jnp.zeros((pad_right,), dtype=window.dtype),
        ],
        axis=0,
    )


def _stft_torch_compat(
    x: jnp.ndarray,
    *,
    n_fft: int,
    hop_length: int,
    window: jnp.ndarray,
) -> jnp.ndarray:
    """Mimic ``torch.stft(..., center=True, pad_mode='reflect', return_complex=True)``.

    Returns a complex array of shape ``(n_fft//2 + 1, n_frames)``.

    ``window`` should already be length ``n_fft`` (caller is responsible for
    center-padding short windows).
    """
    assert window.shape[0] == n_fft

    # Reflect-pad input by n_fft//2 on each side (center=True default).
    pad = n_fft // 2
    x_padded = jnp.pad(x, (pad, pad), mode="reflect")
    n_padded = x_padded.shape[0]

    # Frame.
    n_frames = 1 + (n_padded - n_fft) // hop_length
    # Build (n_frames, n_fft) view.
    starts = jnp.arange(n_frames) * hop_length
    idx = starts[:, None] + jnp.arange(n_fft)[None, :]
    frames = x_padded[idx]  # (n_frames, n_fft)

    # Apply window.
    frames = frames * window[None, :]

    # rfft along axis=-1; transpose so freq-axis is first like torch.stft.
    spec = jnp.fft.rfft(frames, n=n_fft, axis=-1)  # (n_frames, n_fft//2+1)
    return spec.T  # (n_fft//2+1, n_frames)


# ---------------------------------------------------------------------------
# Mel filterbank (HTK, no slaney normalization — matches torchaudio default)
# ---------------------------------------------------------------------------


def _hz_to_mel_htk_np(freq: float) -> float:
    """Pure-Python HTK Hz->mel; used only on static config values so it
    doesn't introduce traced ops inside :func:`jax.jit`."""
    import math
    return 2595.0 * math.log10(1.0 + freq / 700.0)


def _mel_to_hz_htk(mels):
    return 700.0 * (jnp.power(10.0, mels / 2595.0) - 1.0)


def _mel_filterbank(
    n_freqs: int,
    f_min: float,
    f_max: float,
    n_mels: int,
    sample_rate: int,
    dtype,
) -> jnp.ndarray:
    """Triangular HTK mel filterbank, matching ``torchaudio.functional.melscale_fbanks``.

    Returns an array of shape ``(n_freqs, n_mels)``.
    """
    # Note: torchaudio uses sample_rate // 2 (integer division) for the upper
    # endpoint of all_freqs. Reproduce that exactly.
    sr_half = sample_rate // 2
    all_freqs = jnp.linspace(0.0, float(sr_half), n_freqs, dtype=dtype)

    # mel endpoints come from static (Python) config — keep them concrete so
    # _mel_filterbank can run inside jax.jit.
    m_min = _hz_to_mel_htk_np(float(f_min))
    m_max = _hz_to_mel_htk_np(float(f_max))
    m_pts = jnp.linspace(m_min, m_max, n_mels + 2, dtype=dtype)
    f_pts = _mel_to_hz_htk(m_pts)

    # Triangular filters (from torchaudio._create_triangular_filterbank).
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1,)
    slopes = f_pts[None, :] - all_freqs[:, None]  # (n_freqs, n_mels + 2)
    down_slopes = -slopes[:, :-2] / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = jnp.maximum(jnp.zeros((), dtype=dtype), jnp.minimum(down_slopes, up_slopes))
    return fb


# ---------------------------------------------------------------------------
# Audio normalization
# ---------------------------------------------------------------------------


def _ensure_mono(audio) -> jnp.ndarray:
    """Coerce ``audio`` to a 1-D JAX array, mean-folding stereo channels."""
    a = jnp.asarray(audio)
    if a.ndim == 2:
        a = a.mean(axis=-1)
    if a.ndim != 1:
        raise ValueError(f"audio must be 1-D after mono fold; got {tuple(a.shape)}")
    return a


# ---------------------------------------------------------------------------
# Public API (signature-compatible with copy_synth.features)
# ---------------------------------------------------------------------------


def log_mel(audio, cfg: FeatureConfig) -> jnp.ndarray:
    """Return log-mel spectrogram (n_mels, T_frames).

    Uses MelSpectrogram with power=2, then takes log of (mel + 1e-6).
    """
    wav = _ensure_mono(audio)
    dtype = wav.dtype
    n_fft = 1
    while n_fft < int(cfg.sample_rate * cfg.win_ms / 1000.0):
        n_fft *= 2
    win = max(1, int(round(cfg.sample_rate * cfg.win_ms / 1000.0)))
    hop = max(1, int(round(cfg.sample_rate * cfg.hop_ms / 1000.0)))
    f_max = cfg.f_max if cfg.f_max is not None else float(cfg.sample_rate) / 2.0

    window = _hann_window_periodic(win, dtype)
    window = _center_pad_window(window, n_fft)
    spec = _stft_torch_compat(wav, n_fft=n_fft, hop_length=hop, window=window)
    mag2 = jnp.abs(spec) ** 2  # (n_freqs, n_frames)

    fb = _mel_filterbank(
        n_freqs=n_fft // 2 + 1,
        f_min=cfg.f_min,
        f_max=f_max,
        n_mels=cfg.n_mels,
        sample_rate=cfg.sample_rate,
        dtype=dtype,
    )
    # torchaudio: (..., time, freq) @ (freq, n_mels) -> (..., time, n_mels) -> transpose
    # Equivalent to: fb.T @ mag2 -> (n_mels, time).
    mel = fb.T @ mag2
    return jnp.log(mel + 1e-6)


def mr_stft_magnitudes(audio, cfg: FeatureConfig) -> list[jnp.ndarray]:
    """List of magnitude STFTs at the configured resolutions."""
    wav = _ensure_mono(audio)
    dtype = wav.dtype
    out = []
    for n_fft, win_factor, hop_factor in zip(
        cfg.mr_stft_n_ffts, cfg.mr_stft_win_factors, cfg.mr_stft_hop_factors
    ):
        win = max(1, int(round(n_fft * win_factor)))
        hop = max(1, int(round(n_fft * hop_factor)))
        window = _hann_window_periodic(win, dtype)
        window = _center_pad_window(window, n_fft)
        spec = _stft_torch_compat(wav, n_fft=n_fft, hop_length=hop, window=window)
        out.append(jnp.abs(spec))
    return out


def acoustic_loss(
    pred_audio,
    target_log_mel: jnp.ndarray,
    target_mr_stfts: list[jnp.ndarray],
    cfg: FeatureConfig,
    *,
    mel_weight: float = 1.0,
    stft_weight: float = 1.0,
) -> jnp.ndarray:
    """Sum of weighted log-mel L1 + mean MR-STFT L1.

    Returns a 0-D JAX array (NOT a Python float). This is the load-bearing
    departure from the Torch version: keeping the array form lets
    ``jax.grad`` flow through the loss.
    """
    pred_mel = log_mel(pred_audio, cfg)
    pred_mr = mr_stft_magnitudes(pred_audio, cfg)

    n = min(int(pred_mel.shape[-1]), int(target_log_mel.shape[-1]))
    mel_l1 = jnp.mean(jnp.abs(pred_mel[..., :n] - target_log_mel[..., :n]))

    stft_terms = []
    for p, t in zip(pred_mr, target_mr_stfts):
        m = min(int(p.shape[-1]), int(t.shape[-1]))
        stft_terms.append(jnp.mean(jnp.abs(p[..., :m] - t[..., :m])))
    stft_l1 = jnp.stack(stft_terms).mean()

    return mel_weight * mel_l1 + stft_weight * stft_l1


def precompute_target(audio, cfg: FeatureConfig):
    """Pre-extract (log_mel, [mr_stft_magnitudes...]) for the target audio."""
    return log_mel(audio, cfg), mr_stft_magnitudes(audio, cfg)
