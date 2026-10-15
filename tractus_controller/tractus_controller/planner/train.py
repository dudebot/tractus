"""Planner training loop.

Single-GPU only (consumer hardware: 16 GB GPU) per workplan
constraint. No DDP, no FSDP — keeping the surface small so worker
sessions can hand the loop off without re-arguing infra.

CLI: see ``scripts/train_planner.py`` for a thin wrapper that wires the
loop to a manifest path or a synthetic smoke run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR

from .data import CollatedBatch, Sample, collate
from .losses import (
    PlannerLossTargets,
    PlannerLossWeights,
    planner_loss,
)
from .model import PlannerConfig, PlannerModel

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    batch_size: int = 16
    n_epochs: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 200
    grad_clip: float = 1.0
    log_every: int = 25
    eval_every: int | None = None  # epochs
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    loss_weights: PlannerLossWeights = field(default_factory=PlannerLossWeights)


def _batches_from_samples(
    samples: list[Sample], batch_size: int, shuffle: bool, seed: int
) -> list[CollatedBatch]:
    idx = list(range(len(samples)))
    if shuffle:
        rng = torch.Generator().manual_seed(seed)
        idx = torch.randperm(len(samples), generator=rng).tolist()
    out: list[CollatedBatch] = []
    for i in range(0, len(idx), batch_size):
        chunk = [samples[j] for j in idx[i : i + batch_size]]
        out.append(collate(chunk))
    return out


def _build_targets(batch: CollatedBatch) -> PlannerLossTargets:
    return PlannerLossTargets(
        log_dur_target=batch.log_dur_target,
        phone_mask=batch.phone_mask,
        phrase_break_target=batch.phrase_break_target,
        log_f0_target=batch.log_f0_target,
        voiced_target=batch.voiced_target,
        loud_db_target=batch.loud_db_target,
    )


def train(
    model: PlannerModel,
    train_samples: list[Sample],
    cfg: TrainConfig,
    val_samples: list[Sample] | None = None,
) -> dict[str, list[float]]:
    """Single-GPU training loop. Returns a dict of per-step loss histories."""
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model.to(device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_steps_per_epoch = max(1, (len(train_samples) + cfg.batch_size - 1) // cfg.batch_size)
    total_steps = max(1, cfg.n_epochs * n_steps_per_epoch)
    warmup = min(cfg.warmup_steps, max(1, total_steps // 4))
    scheduler = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=1e-3, end_factor=1.0, total_iters=warmup),
            LinearLR(opt, start_factor=1.0, end_factor=0.1, total_iters=max(1, total_steps - warmup)),
        ],
        milestones=[warmup],
    )
    history: dict[str, list[float]] = {"loss/total": []}
    step = 0
    for epoch in range(cfg.n_epochs):
        batches = _batches_from_samples(
            train_samples, cfg.batch_size, shuffle=True, seed=cfg.seed + epoch
        )
        model.train()
        epoch_t0 = time.perf_counter()
        for batch in batches:
            batch = batch.to(device)
            out = model(
                batch.token_ids,
                batch.stress,
                batch.affect,
                batch.speaker_f0_range_hz,
                durations_frames=batch.durations_frames,
            )
            loss, parts = planner_loss(
                out, _build_targets(batch), weights=cfg.loss_weights
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            scheduler.step()
            history["loss/total"].append(float(parts["loss/total"].item()))
            for k, v in parts.items():
                history.setdefault(k, []).append(float(v.item()))
            step += 1
            if step % cfg.log_every == 0:
                log.info(
                    "epoch=%d step=%d loss=%.4f dur=%.4f f0=%.4f loud=%.4f pb=%.4f",
                    epoch, step,
                    parts["loss/total"].item(), parts["loss/duration"].item(),
                    parts["loss/f0"].item(), parts["loss/loudness"].item(),
                    parts["loss/phrase_break"].item(),
                )
        log.info("epoch %d done in %.1fs", epoch, time.perf_counter() - epoch_t0)
    return history
