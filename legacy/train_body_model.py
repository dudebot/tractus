#!/usr/bin/env python3
"""Train the body model on the synthetic gesture-sweep corpus.

pretraining/002 session 2.

Inputs
------

The corpus produced by :mod:`scripts.gen_body_corpus`. Each shard NPZ
holds a batch of samples with arrays:

   gesture          : (B, T, 20)   float32   model input
   anatomy          : (B, 64)      float32   model input
   target_per_frame : (B, T, 385)  float32   gesture-supervision target
                                             (PER_FRAME_FREE_FIELDS, in order)
   target_static    : (B, 279)     float32   static-supervision target
                                             (STATIC_FIELDS, in order)
   slice_kind       : (B,)         <U11      slice this sample came from
   unreachable      : (B,)         bool      anti-cheat flag

Loss
----

   total = MSE_gesture + λ_smooth · smoothness + λ_static · static
        + λ_refusal · refusal_for_unreachable

* MSE_gesture: per-channel-std-normalized MSE on the per-frame free
  fields. Each of the 385 columns is divided by the training-set
  std for that column before the squared error is taken; this keeps
  large-magnitude fields (e.g. subglottal pressure ~ 8000 dPa)
  from dominating small-magnitude fields (e.g. tube_length ~ 1 cm).
  Channels with std < 1e-4 are dropped from the loss (they're
  effectively constant in the corpus and would explode the
  normalized residual).

* smoothness: L2 norm of the first temporal difference of the
  per-frame output, averaged over (T-1, F). Continuous-time units —
  the diff is computed on the post-50Hz-interp grid (which is what
  the model emits). Picked L2 over L1 because the model emits
  smooth-by-default per-frame outputs and L2 keeps the gradient
  well-behaved near the bottom of the well; the v0 budget doesn't
  warrant an L1 ablation.

* static: per-channel-std-normalized MSE on the static (wall_*)
  fields. These are decoded purely from anatomy and the corpus
  always uses the same default tube → in v0 the static target is
  constant across the corpus and the model just learns to memorize
  it. The static loss exists so the static head doesn't drift.

* refusal: anti-cheat. For samples where ``unreachable=True``, we
  swap the supervision target to a fixed "neutral / refusal"
  posture (the same baseline the model's :func:`_add_neutral_bias`
  centers on). Reachable samples still see their real target.
  This lets a SINGLE loss term carry both the gesture-fit signal
  and the anti-cheat signal — which is what the model would learn
  anyway if you chained them: "for unreachable inputs, ignore the
  fictional target and emit the rest posture."

Forbidden
---------

This script MUST NOT import ``synthesize_dynamic_tube`` or any other
TEA pipeline forward. The non-circularity lint
(``scripts/ci/lint_non_circularity.py``) guards this; the test
``tests/body_model/test_train_no_tea_import.py`` adds an extra
ASTwalk over this file.

Usage
-----

   python scripts/train_body_model.py \\
       --corpus outputs/body_corpus \\
       --out outputs/body_model_ckpt \\
       --epochs 10 \\
       --batch 64 \\
       --lr 3e-4
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tractus_controller"))


import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import optax  # noqa: E402

from tractus_controller.body_model import (  # noqa: E402
    BodyModel,
    BodyModelConfig,
)
from tractus_controller.body_model.channels import (  # noqa: E402
    PER_FRAME_FREE_FIELDS,
    RENDERER_INTERFACE_FIELDS,
    STATIC_FIELDS,
    per_frame_free_dim,
    static_field_dim,
)
from tractus_controller.body_model.model import (  # noqa: E402
    forward as body_model_forward,
    init_params,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("train_body_model")


# ---------------------------------------------------------------------------
# Refusal / neutral baseline
# ---------------------------------------------------------------------------

# These are the exact bias values used by `_add_neutral_bias` in the
# body model — the model already sits near these without any learning.
# When an unreachable sample is fed, we want the model to emit something
# close to this baseline regardless of the gesture trajectory.
#
# Layout matches PER_FRAME_FREE_FIELDS column ranges.
_F = 385  # per_frame_free_dim()


def _build_refusal_target() -> np.ndarray:
    """Return a (F=385,) per-frame neutral target. Broadcast over T."""
    # Layout: 0:93 length, 93:186 area, 186:197 glottis, 197 nasal,
    #          198 sub, 199:292 fric_gain, 292:385 fric_color.
    out = np.zeros(_F, dtype=np.float32)
    out[0:93] = 1.0     # tube_length_cm baseline
    out[93:186] = 1.0   # tube_area_cm2 baseline
    # glottis_state baseline: F0=110, PR=8000, others zero/0.5
    g_bias = np.array(
        [110.0, 8000.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        dtype=np.float32,
    )
    out[186:197] = g_bias
    out[197] = 0.0      # nasal_coupling baseline
    out[198] = 8000.0   # subglottal_press baseline
    out[199:292] = 0.0  # frication_gain baseline
    out[292:385] = 0.0  # frication_color baseline
    return out


REFUSAL_PER_FRAME: np.ndarray = _build_refusal_target()
assert REFUSAL_PER_FRAME.shape == (_F,)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class CorpusManifest:
    n_samples: int
    n_shards: int
    T_per_sample: int
    d_gesture: int
    d_anatomy: int
    per_frame_dim: int
    static_dim: int


def load_manifest(corpus_dir: Path) -> CorpusManifest:
    j = json.loads((corpus_dir / "manifest.json").read_text())
    pf_dim = sum(int(np.prod(RENDERER_INTERFACE_FIELDS[f]))
                 for f in PER_FRAME_FREE_FIELDS)
    st_dim = sum(int(np.prod(RENDERER_INTERFACE_FIELDS[f]))
                 for f in STATIC_FIELDS)
    return CorpusManifest(
        n_samples=j["n_samples"],
        n_shards=j["n_shards"],
        T_per_sample=j["T_per_sample"],
        d_gesture=j["d_gesture"],
        d_anatomy=j["d_anatomy"],
        per_frame_dim=pf_dim,
        static_dim=st_dim,
    )


def list_shards(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.glob("shard_*.npz"))


def load_shard(path: Path) -> dict:
    """Load one shard into a dict of numpy arrays."""
    d = np.load(path, allow_pickle=True)
    return {
        "gesture": d["gesture"].astype(np.float32),
        "anatomy": d["anatomy"].astype(np.float32),
        "target_per_frame": d["target_per_frame"].astype(np.float32),
        "target_static": d["target_static"].astype(np.float32),
        "unreachable": d["unreachable"].astype(np.bool_),
    }


# ---------------------------------------------------------------------------
# Per-channel std normalization
# ---------------------------------------------------------------------------


def compute_per_channel_stats(
    train_shards: list[Path],
    sample_n_shards: int = 16,
) -> dict:
    """Compute per-column statistics for inputs (gesture, anatomy) and
    outputs (per_frame target, static target).

    The output stats are used for **loss normalization**: each channel's
    contribution to the loss is divided by its training-set std so that
    large-magnitude fields (subglottal pressure ~ 8000 dPa) don't drown
    out small-magnitude ones (tube_length ~ 1 cm).

    The input stats are used for **input normalization**: gesture
    columns range from -7 (jaw_open) to 20000 (lung_pressure). Without
    Z-scoring, the Glorot-initialized network produces saturated decoder
    outputs that the soft-clamp's gradient cannot recover from.
    """
    rng = np.random.default_rng(0)
    pick = rng.choice(len(train_shards), size=min(sample_n_shards, len(train_shards)),
                      replace=False)
    pf_chunks = []
    st_chunks = []
    g_chunks = []
    a_chunks = []
    for i in pick:
        d = load_shard(train_shards[i])
        pf_chunks.append(d["target_per_frame"].reshape(-1, d["target_per_frame"].shape[-1]))
        st_chunks.append(d["target_static"])
        g_chunks.append(d["gesture"].reshape(-1, d["gesture"].shape[-1]))
        a_chunks.append(d["anatomy"])
    pf = np.concatenate(pf_chunks, axis=0)
    st = np.concatenate(st_chunks, axis=0)
    g = np.concatenate(g_chunks, axis=0)
    a = np.concatenate(a_chunks, axis=0)
    pf_std = pf.std(axis=0)
    st_std = st.std(axis=0)
    pf_mean = pf.mean(axis=0)
    st_mean = st.mean(axis=0)
    g_std = g.std(axis=0)
    g_mean = g.mean(axis=0)
    a_std = a.std(axis=0)
    a_mean = a.mean(axis=0)
    return {
        "per_frame_std": pf_std.astype(np.float32),
        "per_frame_mean": pf_mean.astype(np.float32),
        "static_std": st_std.astype(np.float32),
        "static_mean": st_mean.astype(np.float32),
        "gesture_std": g_std.astype(np.float32),
        "gesture_mean": g_mean.astype(np.float32),
        "anatomy_std": a_std.astype(np.float32),
        "anatomy_mean": a_mean.astype(np.float32),
    }


def normalize_gesture(gesture: np.ndarray | jnp.ndarray, stats: dict
                      ) -> jnp.ndarray:
    """Z-score gesture inputs using the training stats."""
    g_mean = jnp.asarray(stats["gesture_mean"])
    g_std_safe = jnp.maximum(jnp.asarray(stats["gesture_std"]), 1e-6)
    return (jnp.asarray(gesture) - g_mean) / g_std_safe


def normalize_anatomy(anatomy: np.ndarray | jnp.ndarray, stats: dict
                      ) -> jnp.ndarray:
    """Z-score anatomy inputs using the training stats."""
    a_mean = jnp.asarray(stats["anatomy_mean"])
    a_std_safe = jnp.maximum(jnp.asarray(stats["anatomy_std"]), 1e-6)
    return (jnp.asarray(anatomy) - a_mean) / a_std_safe


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def make_loss_fn(
    cfg: BodyModelConfig,
    pf_std_safe: jnp.ndarray,        # (F,) — set to 1 where std<eps
    pf_mask: jnp.ndarray,             # (F,) — 1 if std>=eps, else 0
    st_std_safe: jnp.ndarray,
    st_mask: jnp.ndarray,
    refusal_per_frame: jnp.ndarray,   # (F,)
    lambda_smooth: float = 1e-3,
    lambda_static: float = 0.1,
    lambda_refusal: float = 1.0,
):
    """Build a JIT-able loss function closing over normalization stats.

    The gesture_b / anatomy_b inputs to ``total_loss`` are expected to
    be already-Z-scored by the caller (see :func:`normalize_gesture` /
    :func:`normalize_anatomy`). The stats are stored in the checkpoint
    so eval can apply the same transform.
    """

    pf_std_safe_recip = 1.0 / pf_std_safe
    st_std_safe_recip = 1.0 / st_std_safe

    def per_sample_loss(params, gesture, anatomy, target_pf, target_st,
                        unreachable):
        # gesture: (T, 20) NORMALIZED; anatomy: (64,) NORMALIZED;
        # target_pf: (T, F); target_st: (Fs,); unreachable: bool scalar.
        out = body_model_forward(params, gesture, anatomy, cfg,
                                 use_hard_clamp=False)
        pred_pf = out.to_flat_per_frame()  # (T, F)
        pred_st = jnp.concatenate([
            out.wall_mass_static,
            out.wall_resistance_static,
            out.wall_stiffness_static,
        ], axis=-1)  # (Fs,)

        # Anti-cheat: replace target with refusal baseline when unreachable.
        T = pred_pf.shape[-2]
        refusal_tiled = jnp.broadcast_to(refusal_per_frame, (T, _F))
        target_pf_eff = jnp.where(unreachable, refusal_tiled, target_pf)

        # Per-frame loss: std-normalized + masked
        diff = (pred_pf - target_pf_eff) * pf_std_safe_recip * pf_mask
        l_pf_per_sample_mean = (diff ** 2).mean()

        # Smoothness on prediction (post-50Hz grid, time axis -2):
        smooth = jnp.diff(pred_pf, axis=-2) * pf_std_safe_recip * pf_mask
        l_smooth = (smooth ** 2).mean()

        # Static loss
        diff_st = (pred_st - target_st) * st_std_safe_recip * st_mask
        l_st = (diff_st ** 2).mean()

        # Component for tracking: refusal MSE on unreachable samples
        # (refusal pull on reachable ones is zero by definition since
        # target is unchanged; we report it conditionally in eval).
        return {
            "loss_per_frame": l_pf_per_sample_mean,
            "loss_smooth": l_smooth,
            "loss_static": l_st,
        }

    @jax.jit
    def total_loss(params, gesture_b, anatomy_b, target_pf_b, target_st_b,
                   unreachable_b):
        # Vmap over the leading batch axis.
        per = jax.vmap(per_sample_loss, in_axes=(None, 0, 0, 0, 0, 0))(
            params, gesture_b, anatomy_b, target_pf_b, target_st_b,
            unreachable_b,
        )
        # Mean over the batch
        l_pf = per["loss_per_frame"].mean()
        l_smooth = per["loss_smooth"].mean()
        l_static = per["loss_static"].mean()

        # Anti-cheat term: emphasize the unreachable subset's per-frame
        # error (above and beyond the per-frame loss already reweighting
        # them). Avoids the ~3% unreachable subset getting drowned out.
        # We add an *extra* refusal-only term keyed on the boolean mask.
        unreachable_f = unreachable_b.astype(jnp.float32)
        n_unr = unreachable_f.sum()
        # `per["loss_per_frame"]` was already computed with the right
        # target swap, so it's exactly the refusal MSE on those samples.
        # Use the original per-sample values (vmap output) keyed by mask.
        refusal_term = (per["loss_per_frame"] * unreachable_f).sum() / jnp.maximum(n_unr, 1.0)

        total = (
            l_pf
            + lambda_smooth * l_smooth
            + lambda_static * l_static
            + lambda_refusal * refusal_term
        )
        return total, {
            "total": total,
            "per_frame": l_pf,
            "smooth": l_smooth,
            "static": l_static,
            "refusal": refusal_term,
            "n_unreachable": n_unr,
        }

    return total_loss


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def make_train_step(loss_fn, optimizer):

    @jax.jit
    def train_step(params, opt_state, gesture_b, anatomy_b, target_pf_b,
                   target_st_b, unreachable_b):
        (total, components), grads = jax.value_and_grad(
            loss_fn, has_aux=True
        )(params, gesture_b, anatomy_b, target_pf_b, target_st_b,
          unreachable_b)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, components

    return train_step


# ---------------------------------------------------------------------------
# Train/val split + iteration
# ---------------------------------------------------------------------------


def split_shards(shards: list[Path], val_frac: float = 0.1, seed: int = 0
                 ) -> tuple[list[Path], list[Path]]:
    """Split shards into (train, val). If there's only one shard, both
    splits get the same shard (the within-shard sample shuffle then
    decides what's train vs val for any given iter); the alternative
    of 1/0 train/val makes smoke runs impossible. Real corpora have many
    shards so this collapse only triggers in tests."""
    rng = np.random.default_rng(seed)
    n = len(shards)
    if n <= 1:
        return list(shards), list(shards)
    n_val = max(1, int(round(n * val_frac)))
    n_val = min(n_val, n - 1)  # leave at least 1 shard for train
    perm = rng.permutation(n)
    val_idx = set(perm[:n_val].tolist())
    train_paths = [shards[i] for i in range(n) if i not in val_idx]
    val_paths = [shards[i] for i in range(n) if i in val_idx]
    return train_paths, val_paths


def iter_batches(shard_paths: list[Path], batch_size: int,
                 rng: np.random.Generator, shuffle: bool = True) -> Iterable[dict]:
    """Yield batches of `batch_size` rows by streaming shard-at-a-time.

    Within a shard we permute, then yield slices. Shards are visited in
    random order each epoch.
    """
    order = list(range(len(shard_paths)))
    if shuffle:
        rng.shuffle(order)
    leftover = None
    for s_i in order:
        d = load_shard(shard_paths[s_i])
        n = d["gesture"].shape[0]
        idx = rng.permutation(n) if shuffle else np.arange(n)
        # Glue any leftover from previous shard
        if leftover is not None:
            need = batch_size - leftover["gesture"].shape[0]
            take = idx[:need]
            idx = idx[need:]
            batch = {
                k: np.concatenate([leftover[k], d[k][take]], axis=0)
                for k in leftover
            }
            leftover = None
            yield batch
        # Whole batches
        i = 0
        while i + batch_size <= len(idx):
            sel = idx[i: i + batch_size]
            batch = {k: d[k][sel] for k in d}
            yield batch
            i += batch_size
        if i < len(idx):
            tail = idx[i:]
            leftover = {k: d[k][tail] for k in d}
    # Drop final leftover; not worth padding


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(out_dir: Path, params, cfg: BodyModelConfig,
                    stats: dict, meta: dict, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "params": jax.tree_util.tree_map(np.asarray, params),
        "cfg": asdict(cfg),
        "stats": stats,
        "meta": meta,
    }
    path = out_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    log.info("checkpoint saved: %s (%.1f MB)", path,
             path.stat().st_size / 1e6)


def load_checkpoint(path: Path) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path,
                    default=REPO_ROOT / "outputs" / "body_corpus")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs" / "body_model_ckpt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--lambda-smooth", type=float, default=1e-3)
    ap.add_argument("--lambda-static", type=float, default=0.1)
    ap.add_argument("--lambda-refusal", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ckpt-every", type=int, default=1,
                    help="Save checkpoint every N epochs.")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-train-batches-per-epoch", type=int, default=None,
                    help="Cap batches per epoch (smoke / debug).")
    ap.add_argument("--max-val-batches", type=int, default=None,
                    help="Cap val batches.")
    args = ap.parse_args()

    log.info("loading manifest from %s", args.corpus)
    manifest = load_manifest(args.corpus)
    log.info("manifest: %s", manifest)

    shards = list_shards(args.corpus)
    log.info("found %d shards", len(shards))
    if not shards:
        log.error("no shards found under %s — run gen_body_corpus.py first",
                  args.corpus)
        sys.exit(2)

    train_shards, val_shards = split_shards(shards, args.val_frac, args.seed)
    log.info("split: %d train / %d val shards", len(train_shards), len(val_shards))

    log.info("computing per-channel normalization stats from train split")
    stats = compute_per_channel_stats(train_shards)
    pf_std = stats["per_frame_std"]
    st_std = stats["static_std"]
    pf_mean = stats["per_frame_mean"]
    st_mean = stats["static_mean"]
    # Active mask: drop channels that are effectively constant in the
    # corpus (std ≪ |mean|). Use 1e-3 absolute floor as the "is this
    # channel doing anything" threshold; that catches both flat-zero
    # and flat-nonzero (where mean dominates) channels.
    pf_mask = (pf_std >= 1e-3).astype(np.float32)
    st_mask = (st_std >= 1e-3).astype(np.float32)
    # Clamp the divisor away from tiny values so a near-constant channel
    # with std ~1e-4 doesn't blow the loss by 1e8 if init pred is off by
    # 1.0. Use the channel-magnitude-aware floor: max(std, 1% of |mean|).
    pf_floor = np.maximum(0.01 * np.abs(pf_mean), 1e-3)
    st_floor = np.maximum(0.01 * np.abs(st_mean), 1e-3)
    pf_std_safe = np.maximum(pf_std, pf_floor)
    st_std_safe = np.maximum(st_std, st_floor)
    log.info("per-frame std: %d/%d channels active (mask threshold 1e-3)",
             int(pf_mask.sum()), pf_mask.size)
    log.info("static std:    %d/%d channels active (mask threshold 1e-3)",
             int(st_mask.sum()), st_mask.size)

    # Build model + optimizer
    log.info("building model")
    bm = BodyModel.from_default(seed=args.seed)
    cfg = bm.cfg
    params = bm.params
    n_params = bm.num_params()
    log.info("model has %d params (%.2f M)", n_params, n_params / 1e6)
    if n_params > 5_000_000:
        log.error("model exceeds 5M param cap — refusing to train")
        sys.exit(3)

    optimizer = optax.adamw(learning_rate=args.lr, weight_decay=1e-4)
    opt_state = optimizer.init(params)

    pf_std_safe_j = jnp.asarray(pf_std_safe)
    pf_mask_j = jnp.asarray(pf_mask)
    st_std_safe_j = jnp.asarray(st_std_safe)
    st_mask_j = jnp.asarray(st_mask)
    refusal_j = jnp.asarray(REFUSAL_PER_FRAME)

    loss_fn = make_loss_fn(
        cfg, pf_std_safe_j, pf_mask_j, st_std_safe_j, st_mask_j,
        refusal_j,
        lambda_smooth=args.lambda_smooth,
        lambda_static=args.lambda_static,
        lambda_refusal=args.lambda_refusal,
    )
    train_step = make_train_step(loss_fn, optimizer)

    # Loss curve log
    args.out.mkdir(parents=True, exist_ok=True)
    curve_path = args.out / "loss_curve.csv"
    with open(curve_path, "w") as f:
        f.write("step,epoch,phase,total,per_frame,smooth,static,refusal,n_unreachable\n")

    rng = np.random.default_rng(args.seed)

    global_step = 0
    t_train0 = time.time()
    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_losses = []
        n_batches = 0
        last_log = time.time()
        for batch in iter_batches(train_shards, args.batch, rng, shuffle=True):
            g_norm = normalize_gesture(batch["gesture"], stats)
            a_norm = normalize_anatomy(batch["anatomy"], stats)
            params, opt_state, comps = train_step(
                params, opt_state,
                g_norm,
                a_norm,
                jnp.asarray(batch["target_per_frame"]),
                jnp.asarray(batch["target_static"]),
                jnp.asarray(batch["unreachable"]),
            )
            comps_h = {k: float(v) for k, v in comps.items()}
            train_losses.append(comps_h)
            with open(curve_path, "a") as f:
                f.write(
                    f"{global_step},{epoch},train,"
                    f"{comps_h['total']:.6f},"
                    f"{comps_h['per_frame']:.6f},"
                    f"{comps_h['smooth']:.6f},"
                    f"{comps_h['static']:.6f},"
                    f"{comps_h['refusal']:.6f},"
                    f"{int(comps_h['n_unreachable'])}\n"
                )
            global_step += 1
            n_batches += 1
            now = time.time()
            if now - last_log > 10.0:
                log.info("epoch %d batch %d step %d total=%.4f pf=%.4f sm=%.4f st=%.4f ref=%.4f",
                         epoch, n_batches, global_step,
                         comps_h["total"], comps_h["per_frame"],
                         comps_h["smooth"], comps_h["static"],
                         comps_h["refusal"])
                last_log = now
            if (args.max_train_batches_per_epoch is not None
                    and n_batches >= args.max_train_batches_per_epoch):
                break

        # Validation
        val_losses = []
        n_val_batches = 0
        for vbatch in iter_batches(val_shards, args.batch, rng, shuffle=False):
            g_norm = normalize_gesture(vbatch["gesture"], stats)
            a_norm = normalize_anatomy(vbatch["anatomy"], stats)
            _, vcomps = loss_fn(
                params,
                g_norm,
                a_norm,
                jnp.asarray(vbatch["target_per_frame"]),
                jnp.asarray(vbatch["target_static"]),
                jnp.asarray(vbatch["unreachable"]),
            )
            vcomps_h = {k: float(v) for k, v in vcomps.items()}
            val_losses.append(vcomps_h)
            with open(curve_path, "a") as f:
                f.write(
                    f"{global_step},{epoch},val,"
                    f"{vcomps_h['total']:.6f},"
                    f"{vcomps_h['per_frame']:.6f},"
                    f"{vcomps_h['smooth']:.6f},"
                    f"{vcomps_h['static']:.6f},"
                    f"{vcomps_h['refusal']:.6f},"
                    f"{int(vcomps_h['n_unreachable'])}\n"
                )
            n_val_batches += 1
            if (args.max_val_batches is not None
                    and n_val_batches >= args.max_val_batches):
                break

        def _avg(rs, k):
            if not rs:
                return float("nan")
            return float(np.mean([r[k] for r in rs]))

        elapsed = time.time() - t_epoch
        log.info(
            "epoch %d done in %.1fs:  TRAIN total=%.4f pf=%.4f sm=%.4f st=%.4f ref=%.4f"
            "  |  VAL total=%.4f pf=%.4f sm=%.4f st=%.4f ref=%.4f",
            epoch, elapsed,
            _avg(train_losses, "total"), _avg(train_losses, "per_frame"),
            _avg(train_losses, "smooth"), _avg(train_losses, "static"),
            _avg(train_losses, "refusal"),
            _avg(val_losses, "total"), _avg(val_losses, "per_frame"),
            _avg(val_losses, "smooth"), _avg(val_losses, "static"),
            _avg(val_losses, "refusal"),
        )

        if (epoch + 1) % args.ckpt_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(
                args.out, params, cfg, stats,
                {"epoch": epoch, "step": global_step,
                 "elapsed_s": time.time() - t_train0,
                 "args": vars(args) | {"corpus": str(args.corpus),
                                       "out": str(args.out)}},
                f"epoch_{epoch:03d}",
            )

    save_checkpoint(
        args.out, params, cfg, stats,
        {"epoch": args.epochs - 1, "step": global_step,
         "elapsed_s": time.time() - t_train0,
         "args": vars(args) | {"corpus": str(args.corpus),
                               "out": str(args.out)}},
        "final",
    )
    log.info("training done. total wall-clock: %.1fs", time.time() - t_train0)


if __name__ == "__main__":
    main()
