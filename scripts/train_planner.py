#!/usr/bin/env python
"""Drive ``tractus_controller.planner.train``.

Two modes:

- ``--manifest <path.jsonl>`` for real corpus training. Each JSONL row
  must follow ``CorpusEntry`` (wav, textgrid, transcript, speaker_id,
  optional f0_range_hz/affect).
- ``--synthetic`` for a smoke run that exercises the full loss path
  without any external data. Used by CI and by the next worker session
  to validate plumbing before pulling LibriTTS-R / VCTK.

Eval gates (per workplan/tasks/04_pretraining/001_planner_pretrain.md):
phone-dur RMSE ≤ 25 ms, F0 cents-RMSE ≤ 50, phrase-break F1 ≥ 0.85,
≤ 10 M params, ≤ 100 ms inference for a 5 s utterance.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch

from tractus_controller.planner.data import (
    collate,
    load_manifest,
    load_sample_from_entry,
    synthetic_dataset,
)
from tractus_controller.planner.eval import evaluate, measure_inference_ms_5s
from tractus_controller.planner.model import PlannerConfig, PlannerModel, planner_param_count
from tractus_controller.planner.tokens import PhoneTokenizer
from tractus_controller.planner.train import TrainConfig, train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_planner")


def _split(items, val_frac=0.1, seed=0):
    import random
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n_val = max(1, int(len(items) * val_frac))
    return items[n_val:], items[:n_val]


def _load_real_samples(manifest: Path) -> list:
    tok = PhoneTokenizer()
    entries = load_manifest(str(manifest))
    samples = []
    for e in entries:
        try:
            samples.append(load_sample_from_entry(e, tok))
        except Exception as ex:
            log.warning("dropped %s: %s", e.wav, ex)
    if not samples:
        raise SystemExit(f"no usable samples in {manifest}")
    return samples


def _gate_report(metrics: dict, params: int, infer_ms: float) -> dict:
    gates = {
        "phone_dur_rmse_ms": (metrics["phone_dur_rmse_ms"], 25.0, "<="),
        "f0_cents_rmse": (metrics["f0_cents_rmse"], 50.0, "<="),
        "phrase_break_f1": (metrics["phrase_break_f1"], 0.85, ">="),
        "param_count": (params, 10_000_000, "<="),
        "inference_ms_5s": (infer_ms, 100.0, "<="),
    }
    out: dict = {"metrics": metrics, "params": params, "infer_ms_5s": infer_ms}
    pass_all = True
    for k, (v, thr, op) in gates.items():
        ok = (v <= thr) if op == "<=" else (v >= thr)
        out.setdefault("gates", {})[k] = {"value": v, "threshold": thr, "op": op, "pass": ok}
        pass_all = pass_all and ok
    out["pass"] = pass_all
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", type=Path, help="JSONL of CorpusEntry rows.")
    src.add_argument("--synthetic", action="store_true", help="Smoke synth path.")
    p.add_argument("--n-synth-train", type=int, default=512)
    p.add_argument("--n-synth-val", type=int, default=64)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", type=Path, default=None, help="Save trained weights here.")
    p.add_argument("--report", type=Path, default=None, help="Write gate report JSON here.")
    p.add_argument("--cfg", type=str, default=None, help="JSON of PlannerConfig overrides.")
    args = p.parse_args(argv)

    cfg = PlannerConfig()
    if args.cfg:
        for k, v in json.loads(args.cfg).items():
            setattr(cfg, k, v)
    model = PlannerModel(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("planner param count: %d (cap 10_000_000)", n_params)
    if n_params > 10_000_000:
        raise SystemExit(f"PARAM CAP EXCEEDED: {n_params} > 10M")

    if args.synthetic:
        train_samples = synthetic_dataset(args.n_synth_train, seed=0)
        val_samples = synthetic_dataset(args.n_synth_val, seed=10_000)
    else:
        all_samples = _load_real_samples(args.manifest)
        train_samples, val_samples = _split(all_samples, val_frac=0.1, seed=0)
    log.info("train=%d  val=%d", len(train_samples), len(val_samples))

    tcfg = TrainConfig(
        batch_size=args.batch_size, n_epochs=args.epochs, lr=args.lr, device=args.device
    )
    history = train(model, train_samples, tcfg, val_samples=val_samples)
    log.info("final training loss: %.4f", history["loss/total"][-1])

    val_batches = []
    bs = args.batch_size
    for i in range(0, len(val_samples), bs):
        val_batches.append(collate(val_samples[i : i + bs]))
    metrics = evaluate(model, val_batches, args.device)
    infer_ms = measure_inference_ms_5s(model, args.device)
    report = _gate_report(metrics, n_params, infer_ms)
    log.info("eval: %s", json.dumps(report, indent=2))

    if args.ckpt:
        args.ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__}, args.ckpt)
        log.info("checkpoint → %s", args.ckpt)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2))
        log.info("gate report → %s", args.report)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
