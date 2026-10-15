"""Held-out evaluation against the task acceptance gates.

Computes:

- ``phone_dur_rmse_ms``: per-phone duration RMSE in milliseconds (gate ≤ 25 ms).
- ``f0_cents_rmse``: F0 RMSE on voiced frames in cents (gate ≤ 50 cents).
- ``loud_rmse_db``: loudness RMSE in dBFS (no gate; logged).
- ``phrase_break_f1``: per-phone phrase-break F1 (gate ≥ 0.85).
- ``param_count``: trainable param count (gate ≤ 10 M).
- ``inference_ms_5s``: median of 10 inference passes for a synthetic
  ~5-second utterance (gate ≤ 100 ms on consumer GPU).

Returned as a plain dict so the train script and tests can both inspect it.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from .data import CollatedBatch, collate, synthetic_sample
from .losses import PlannerLossTargets
from .model import PlannerModel


def evaluate(
    model: PlannerModel,
    batches: list[CollatedBatch],
    device: torch.device | str,
) -> dict[str, float]:
    """Run gate metrics on a list of pre-collated held-out batches."""
    model.eval()
    dur_sse = 0.0
    dur_n = 0
    f0_cent_sse = 0.0
    f0_n = 0
    loud_sse = 0.0
    loud_n = 0
    pb_tp = pb_fp = pb_fn = 0
    cents_per_log = 1200.0 / math.log(2.0)
    with torch.no_grad():
        for batch in batches:
            batch = batch.to(device)
            out = model(
                batch.token_ids,
                batch.stress,
                batch.affect,
                batch.speaker_f0_range_hz,
                durations_frames=batch.durations_frames,
            )
            # ---- phone duration RMSE (in seconds, then × 1000 ms) -----
            log_dur_eps = model.cfg.log_dur_eps
            pred_dur_s = (torch.exp(out.log_dur_pred) - log_dur_eps).clamp(min=0.0)
            tgt_dur_s = (torch.exp(batch.log_dur_target) - log_dur_eps).clamp(min=0.0)
            mask = batch.phone_mask
            if mask.any():
                d = ((pred_dur_s - tgt_dur_s) * mask).flatten()
                m = mask.flatten()
                dur_sse += float((d**2).sum().item())
                dur_n += int(m.sum().item())
            # ---- F0 cents RMSE on voiced -------------------------------
            T = min(out.log_f0_pred.shape[1], batch.log_f0_target.shape[1])
            voiced_mask = (
                batch.frame_mask[:, :T] & (batch.voiced_target[:, :T] > 0.5)
            )
            if voiced_mask.any():
                diff = (out.log_f0_pred[:, :T] - batch.log_f0_target[:, :T]) * cents_per_log
                f0_cent_sse += float(((diff**2) * voiced_mask).sum().item())
                f0_n += int(voiced_mask.sum().item())
            # ---- loudness MSE -----------------------------------------
            fm = batch.frame_mask[:, :T]
            if fm.any():
                d = (out.loud_db_pred[:, :T] - batch.loud_db_target[:, :T]) * fm
                loud_sse += float((d**2).sum().item())
                loud_n += int(fm.sum().item())
            # ---- phrase-break F1 -------------------------------------
            pb_pred = (torch.sigmoid(out.phrase_break_logit) > 0.5) & mask
            pb_tgt = (batch.phrase_break_target > 0.5) & mask
            pb_tp += int((pb_pred & pb_tgt).sum().item())
            pb_fp += int((pb_pred & (~pb_tgt) & mask).sum().item())
            pb_fn += int(((~pb_pred) & pb_tgt & mask).sum().item())
    dur_rmse_ms = math.sqrt(dur_sse / max(dur_n, 1)) * 1000.0
    f0_cents_rmse = math.sqrt(f0_cent_sse / max(f0_n, 1))
    loud_rmse_db = math.sqrt(loud_sse / max(loud_n, 1))
    if pb_tp == 0 and (pb_fp + pb_fn) == 0:
        pb_f1 = 1.0  # vacuously perfect on a zero-positive split
    else:
        prec = pb_tp / max(pb_tp + pb_fp, 1)
        rec = pb_tp / max(pb_tp + pb_fn, 1)
        pb_f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {
        "phone_dur_rmse_ms": dur_rmse_ms,
        "f0_cents_rmse": f0_cents_rmse,
        "loud_rmse_db": loud_rmse_db,
        "phrase_break_f1": pb_f1,
    }


def measure_inference_ms_5s(
    model: PlannerModel, device: torch.device | str, n_runs: int = 10
) -> float:
    """Median wall-clock for a single ~5-second utterance forward.

    Builds one synthetic 5-second utterance (~50 phones at average phone
    rate). Returns the *median* of ``n_runs`` to dodge JIT / cudnn warmup.
    """
    import random

    model.eval()
    sample = synthetic_sample(random.Random(0), n_phones=50)
    batch = collate([sample]).to(device)
    times: list[float] = []
    with torch.no_grad():
        # warmup
        for _ in range(3):
            _ = model(
                batch.token_ids,
                batch.stress,
                batch.affect,
                batch.speaker_f0_range_hz,
                durations_frames=batch.durations_frames,
            )
        if device != "cpu" and (torch.cuda.is_available() and "cuda" in str(device)):
            torch.cuda.synchronize()
        for _ in range(n_runs):
            t0 = time.perf_counter()
            _ = model(
                batch.token_ids,
                batch.stress,
                batch.affect,
                batch.speaker_f0_range_hz,
                durations_frames=batch.durations_frames,
            )
            if device != "cpu" and (torch.cuda.is_available() and "cuda" in str(device)):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return times[len(times) // 2]
