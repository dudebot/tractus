"""Acoustic feature extraction for the copy-synth loss.

Implements log-mel spectrograms and multi-resolution STFT magnitudes via
torchaudio. The output format is shared with the controller training
loop so extractors can be reused there.

We deliberately avoid librosa to keep the dep set small — torchaudio is
already pinned via the controller training stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torchaudio


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


def _ensure_mono_torch(audio: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(audio, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))
    else:
        t = audio.to(torch.float32)
    if t.ndim == 2:
        t = t.mean(dim=-1)
    if t.ndim != 1:
        raise ValueError(f"audio must be 1-D after mono fold; got {tuple(t.shape)}")
    return t


def log_mel(audio: np.ndarray | torch.Tensor, cfg: FeatureConfig) -> torch.Tensor:
    """Return log-mel spectrogram (n_mels, T_frames).

    Uses MelSpectrogram with power=2, then takes log of (mel + 1e-6).
    """
    wav = _ensure_mono_torch(audio)
    n_fft = 1
    while n_fft < int(cfg.sample_rate * cfg.win_ms / 1000.0):
        n_fft *= 2
    win = max(1, int(round(cfg.sample_rate * cfg.win_ms / 1000.0)))
    hop = max(1, int(round(cfg.sample_rate * cfg.hop_ms / 1000.0)))
    f_max = cfg.f_max if cfg.f_max is not None else float(cfg.sample_rate) / 2.0
    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_fft=n_fft,
        win_length=win,
        hop_length=hop,
        n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=f_max,
        power=2.0,
    )
    spec = mel(wav)
    return torch.log(spec + 1e-6)


def mr_stft_magnitudes(
    audio: np.ndarray | torch.Tensor, cfg: FeatureConfig
) -> list[torch.Tensor]:
    """List of magnitude STFTs at the configured resolutions."""
    wav = _ensure_mono_torch(audio)
    out = []
    for n_fft, win_factor, hop_factor in zip(
        cfg.mr_stft_n_ffts, cfg.mr_stft_win_factors, cfg.mr_stft_hop_factors
    ):
        win = max(1, int(round(n_fft * win_factor)))
        hop = max(1, int(round(n_fft * hop_factor)))
        spec = torch.stft(
            wav,
            n_fft=n_fft,
            hop_length=hop,
            win_length=win,
            window=torch.hann_window(win),
            return_complex=True,
        )
        out.append(spec.abs())
    return out


def acoustic_loss(
    pred_audio: np.ndarray | torch.Tensor,
    target_log_mel: torch.Tensor,
    target_mr_stfts: list[torch.Tensor],
    cfg: FeatureConfig,
    *,
    mel_weight: float = 1.0,
    stft_weight: float = 1.0,
) -> float:
    """Sum of weighted log-mel L1 + mean MR-STFT L1.

    Time-axis lengths between target and prediction may differ by a frame or
    two due to integer-rounding when audio length differs slightly. Truncate
    to the common length.
    """
    pred_mel = log_mel(pred_audio, cfg)
    pred_mr = mr_stft_magnitudes(pred_audio, cfg)

    n = min(pred_mel.shape[-1], target_log_mel.shape[-1])
    mel_l1 = torch.mean(torch.abs(pred_mel[..., :n] - target_log_mel[..., :n]))

    stft_terms = []
    for p, t in zip(pred_mr, target_mr_stfts):
        m = min(p.shape[-1], t.shape[-1])
        stft_terms.append(torch.mean(torch.abs(p[..., :m] - t[..., :m])))
    stft_l1 = torch.stack(stft_terms).mean()

    loss = mel_weight * mel_l1 + stft_weight * stft_l1
    return float(loss.item())


def precompute_target(audio: np.ndarray | torch.Tensor, cfg: FeatureConfig):
    """Pre-extract (log_mel, [mr_stft_magnitudes...]) for the target audio."""
    return log_mel(audio, cfg), mr_stft_magnitudes(audio, cfg)
