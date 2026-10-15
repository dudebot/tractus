"""Planner training losses (no audio gradients, ever).

Five terms per task spec:

1. Duration regression — Huber on log-duration, masked.
2. F0 regression — MSE on log-Hz, voiced-mask weighted.
3. Loudness regression — MSE on dBFS, frame-masked.
4. Phrase-break classification — BCEWithLogits, phone-masked.
5. Alignment — soft enforcement that summed predicted durations match
   total target duration per utterance (Huber on the difference).

Default weights are uniform-ish; surface them as a dataclass so the
training script can override.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import PlannerOutput


@dataclass
class PlannerLossWeights:
    duration: float = 1.0
    f0: float = 1.0
    loudness: float = 0.5
    phrase_break: float = 0.5
    voiced_bce: float = 0.5
    alignment: float = 0.25


@dataclass
class PlannerLossTargets:
    log_dur_target: torch.Tensor  # [B, L]
    phone_mask: torch.Tensor  # [B, L] bool, True = real phone
    phrase_break_target: torch.Tensor  # [B, L] float 0/1
    log_f0_target: torch.Tensor  # [B, T_frame]
    voiced_target: torch.Tensor  # [B, T_frame] float 0/1
    loud_db_target: torch.Tensor  # [B, T_frame]


def _huber(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Masked Huber (smooth-L1 with delta=1) loss; mean over mask=True."""
    if mask.sum() == 0:
        return pred.new_zeros(())
    loss = F.smooth_l1_loss(pred, target, reduction="none", beta=1.0)
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def _mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return pred.new_zeros(())
    loss = (pred - target) ** 2
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def _bce(logit: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum() == 0:
        return logit.new_zeros(())
    loss = F.binary_cross_entropy_with_logits(logit, target.float(), reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum().clamp(min=1.0)


def planner_loss(
    out: PlannerOutput,
    tgt: PlannerLossTargets,
    weights: PlannerLossWeights | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Total weighted loss + a dict of unweighted per-term scalars (for logging)."""
    w = weights or PlannerLossWeights()
    pm = tgt.phone_mask
    fm = out.frame_mask & (tgt.log_f0_target.shape == out.log_f0_pred.shape)
    # If shapes differ (predicted frame count != target), trim to common length.
    L_dur = min(out.log_dur_pred.shape[1], tgt.log_dur_target.shape[1])
    log_dur_pred = out.log_dur_pred[:, :L_dur]
    log_dur_tgt = tgt.log_dur_target[:, :L_dur]
    pm_dur = tgt.phone_mask[:, :L_dur]
    phrase_pred = out.phrase_break_logit[:, :L_dur]
    phrase_tgt = tgt.phrase_break_target[:, :L_dur]
    T = min(out.log_f0_pred.shape[1], tgt.log_f0_target.shape[1])
    log_f0_pred = out.log_f0_pred[:, :T]
    log_f0_tgt = tgt.log_f0_target[:, :T]
    voiced_pred = out.voiced_logit[:, :T]
    voiced_tgt = tgt.voiced_target[:, :T]
    loud_pred = out.loud_db_pred[:, :T]
    loud_tgt = tgt.loud_db_target[:, :T]
    frame_mask = out.frame_mask[:, :T]
    voiced_frame_mask = frame_mask & (voiced_tgt > 0.5)

    l_dur = _huber(log_dur_pred, log_dur_tgt, pm_dur)
    l_f0 = _mse(log_f0_pred, log_f0_tgt, voiced_frame_mask)
    l_loud = _mse(loud_pred, loud_tgt, frame_mask)
    l_phrase = _bce(phrase_pred, phrase_tgt, pm_dur)
    l_voiced = _bce(voiced_pred, voiced_tgt, frame_mask)
    # alignment: predicted total duration (in frames) ≈ target total
    pred_total = (torch.exp(log_dur_pred) - 1e-3).clamp(min=0.0).sum(dim=1) * 100.0
    tgt_total = (torch.exp(log_dur_tgt) - 1e-3).clamp(min=0.0).sum(dim=1) * 100.0
    l_align = F.smooth_l1_loss(pred_total, tgt_total, reduction="mean", beta=10.0)

    total = (
        w.duration * l_dur
        + w.f0 * l_f0
        + w.loudness * l_loud
        + w.phrase_break * l_phrase
        + w.voiced_bce * l_voiced
        + w.alignment * l_align
    )
    return total, {
        "loss/duration": l_dur.detach(),
        "loss/f0": l_f0.detach(),
        "loss/loudness": l_loud.detach(),
        "loss/phrase_break": l_phrase.detach(),
        "loss/voiced": l_voiced.detach(),
        "loss/alignment": l_align.detach(),
        "loss/total": total.detach(),
    }
