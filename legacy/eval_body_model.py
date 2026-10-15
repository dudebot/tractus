#!/usr/bin/env python3
"""Evaluate a body-model checkpoint against pretraining/002 acceptance gates.

Three gates run here (gate D = inversion/004 is session 3, deferred):

* Gate A — gesture-target RMSE on a held-out 10% of the corpus,
  channel-wise. Pass = RMSE/std ≤ 0.10 on each active channel
  (channels with std<1e-4 are excluded — they're constant in the
  corpus and the ratio is undefined).

* Gate B — reachability. Generate 10K random in-envelope inputs,
  run the model with hard-clamp, assert no clamp violation in the
  output. Pass = zero violations.

* Gate C — anti-cheat refusal. Feed unreachable samples through
  the model and measure how close the output sits to the refusal
  baseline (the per-frame neutral). Pass = ≥ 80% refusal rate, where
  refusal is defined as: per-sample normalized L2 distance from the
  refusal baseline below a threshold (default 0.5 in std-units).

Exit code 0 iff all three gates pass.

Usage::

    python scripts/eval_body_model.py \\
        --ckpt outputs/body_model_ckpt/final.pkl \\
        --corpus outputs/body_corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tractus_controller"))


import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tractus_controller.body_model import (  # noqa: E402
    BodyModel,
    BodyModelConfig,
    D_ANATOMY_PLACEHOLDER,
    N_GESTURE_CHANNELS,
)
from tractus_controller.body_model.channels import (  # noqa: E402
    PER_FRAME_FREE_FIELDS,
    RENDERER_INTERFACE_FIELDS,
    STATIC_FIELDS,
)
from tractus_controller.body_model.model import (  # noqa: E402
    forward as body_model_forward,
)
from tractus_controller.body_model.reachability import (  # noqa: E402
    DEFAULT_REACHABILITY,
    FRICATION_COLOR_CAP,
)


# Reuse the refusal target + normalization helpers from training
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from train_body_model import (  # noqa: E402
    REFUSAL_PER_FRAME, list_shards, split_shards,
    normalize_gesture, normalize_anatomy,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("eval_body_model")


# ---------------------------------------------------------------------------
# Per-frame layout helpers
# ---------------------------------------------------------------------------


def per_frame_field_ranges() -> list[tuple[str, int, int, tuple[int, ...]]]:
    """Return [(name, start, width, shape_suffix), ...] over the flat
    per-frame target axis."""
    out = []
    cursor = 0
    for name in PER_FRAME_FREE_FIELDS:
        suffix = RENDERER_INTERFACE_FIELDS[name]
        width = int(np.prod(suffix)) if suffix else 1
        out.append((name, cursor, width, suffix))
        cursor += width
    return out


def static_field_ranges() -> list[tuple[str, int, int, tuple[int, ...]]]:
    out = []
    cursor = 0
    for name in STATIC_FIELDS:
        suffix = RENDERER_INTERFACE_FIELDS[name]
        width = int(np.prod(suffix)) if suffix else 1
        out.append((name, cursor, width, suffix))
        cursor += width
    return out


# ---------------------------------------------------------------------------
# Load model from checkpoint
# ---------------------------------------------------------------------------


def load_body_model(ckpt_path: Path) -> tuple[BodyModel, dict]:
    with open(ckpt_path, "rb") as f:
        payload = pickle.load(f)
    cfg = BodyModelConfig(**payload["cfg"])
    params = jax.tree_util.tree_map(jnp.asarray, payload["params"])
    bm = BodyModel(params=params, cfg=cfg)
    return bm, payload


# ---------------------------------------------------------------------------
# Gate A — held-out RMSE
# ---------------------------------------------------------------------------


@jax.jit
def _batched_predict(params, cfg_static, gesture_b, anatomy_b):
    # cfg_static is a static config dict; we recompose. Static-arg pattern.
    fn = lambda g, a: body_model_forward(
        params, g, a, cfg_static, use_hard_clamp=False,
    ).to_flat_per_frame()
    return jax.vmap(fn)(gesture_b, anatomy_b)


def _flat_static_pred(out_static_dict_jax) -> jnp.ndarray:
    return jnp.concatenate([
        out_static_dict_jax.wall_mass_static,
        out_static_dict_jax.wall_resistance_static,
        out_static_dict_jax.wall_stiffness_static,
    ], axis=-1)


def _predict_with_static(params, cfg, gesture_b, anatomy_b):
    """Returns (per_frame_pred (B, T, F), static_pred (B, Fs))."""
    @jax.jit
    def fn(g, a):
        out = body_model_forward(params, g, a, cfg, use_hard_clamp=False)
        pf = out.to_flat_per_frame()
        st = jnp.concatenate([
            out.wall_mass_static,
            out.wall_resistance_static,
            out.wall_stiffness_static,
        ], axis=-1)
        return pf, st
    return jax.vmap(fn)(gesture_b, anatomy_b)


def gate_a_rmse(bm: BodyModel, val_shards: list[Path], stats: dict,
                batch: int = 128) -> dict:
    """Compute per-channel RMSE on the held-out val set."""
    log.info("[Gate A] running held-out RMSE on %d val shards", len(val_shards))

    pf_field_ranges = per_frame_field_ranges()
    st_field_ranges = static_field_ranges()

    # Streaming accumulators
    pf_se_sum = None    # (F,)
    pf_t_sum = None     # (F,) running sum for std calc on target
    pf_t_sq_sum = None
    pf_n = 0

    st_se_sum = None    # (Fs,)
    st_t_sum = None
    st_t_sq_sum = None
    st_n = 0

    rng = np.random.default_rng(0)
    for sp in val_shards:
        d = np.load(sp, allow_pickle=True)
        gesture = d["gesture"].astype(np.float32)
        anatomy = d["anatomy"].astype(np.float32)
        target_pf = d["target_per_frame"].astype(np.float32)
        target_st = d["target_static"].astype(np.float32)
        unreachable = d["unreachable"].astype(np.bool_)

        # Skip unreachable samples — gate A is on reachable held-out only
        keep = ~unreachable
        if not keep.any():
            continue
        gesture = gesture[keep]
        anatomy = anatomy[keep]
        target_pf = target_pf[keep]
        target_st = target_st[keep]
        n = gesture.shape[0]

        # Process in batches
        for i in range(0, n, batch):
            sl = slice(i, min(i + batch, n))
            g_j = normalize_gesture(gesture[sl], stats)
            a_j = normalize_anatomy(anatomy[sl], stats)
            pred_pf, pred_st = _predict_with_static(bm.params, bm.cfg, g_j, a_j)
            pred_pf = np.asarray(pred_pf)  # (b, T, F)
            pred_st = np.asarray(pred_st)  # (b, Fs)

            err_pf = (pred_pf - target_pf[sl]) ** 2  # (b, T, F)
            err_st = (pred_st - target_st[sl]) ** 2  # (b, Fs)

            # Sum over batch + time → (F,)
            sum_pf = err_pf.sum(axis=(0, 1))
            n_pf = err_pf.shape[0] * err_pf.shape[1]
            tg_pf = target_pf[sl].reshape(-1, target_pf.shape[-1])
            t_sum_pf = tg_pf.sum(axis=0)
            t_sq_sum_pf = (tg_pf ** 2).sum(axis=0)

            sum_st = err_st.sum(axis=0)
            n_st = err_st.shape[0]
            t_sum_st = target_st[sl].sum(axis=0)
            t_sq_sum_st = (target_st[sl] ** 2).sum(axis=0)

            if pf_se_sum is None:
                pf_se_sum = sum_pf
                pf_t_sum = t_sum_pf
                pf_t_sq_sum = t_sq_sum_pf
                st_se_sum = sum_st
                st_t_sum = t_sum_st
                st_t_sq_sum = t_sq_sum_st
            else:
                pf_se_sum = pf_se_sum + sum_pf
                pf_t_sum = pf_t_sum + t_sum_pf
                pf_t_sq_sum = pf_t_sq_sum + t_sq_sum_pf
                st_se_sum = st_se_sum + sum_st
                st_t_sum = st_t_sum + t_sum_st
                st_t_sq_sum = st_t_sq_sum + t_sq_sum_st
            pf_n += n_pf
            st_n += n_st

    # Per-channel RMSE
    pf_rmse = np.sqrt(pf_se_sum / pf_n)
    pf_mean = pf_t_sum / pf_n
    pf_var = pf_t_sq_sum / pf_n - pf_mean ** 2
    pf_std = np.sqrt(np.maximum(pf_var, 0.0))

    st_rmse = np.sqrt(st_se_sum / st_n)
    st_mean = st_t_sum / st_n
    st_var = st_t_sq_sum / st_n - st_mean ** 2
    st_std = np.sqrt(np.maximum(st_var, 0.0))

    # Field-level summary: mean of RMSE/std for active channels in each field.
    field_summary = []
    overall_pass = True
    for name, start, width, suffix in pf_field_ranges:
        rmse_block = pf_rmse[start: start + width]
        std_block = pf_std[start: start + width]
        active = std_block >= 1e-4
        if active.any():
            ratio = rmse_block[active] / std_block[active]
            max_ratio = float(ratio.max())
            mean_ratio = float(ratio.mean())
        else:
            max_ratio = float("nan")
            mean_ratio = float("nan")
        passed = active.any() and max_ratio <= 0.10
        if not passed and active.any():
            overall_pass = False
        field_summary.append({
            "field": name,
            "n_channels": int(width),
            "n_active": int(active.sum()),
            "max_rmse_over_std": max_ratio,
            "mean_rmse_over_std": mean_ratio,
            "passed_gate_A": passed if active.any() else None,
        })

    static_summary = []
    for name, start, width, suffix in st_field_ranges:
        rmse_block = st_rmse[start: start + width]
        std_block = st_std[start: start + width]
        active = std_block >= 1e-4
        if active.any():
            ratio = rmse_block[active] / std_block[active]
            max_ratio = float(ratio.max())
            mean_ratio = float(ratio.mean())
        else:
            max_ratio = float("nan")
            mean_ratio = float("nan")
        # Static fields are mostly constant; they're NOT in the spec gate
        # criterion, so we don't fail the overall gate on them — we just
        # report.
        static_summary.append({
            "field": name,
            "n_channels": int(width),
            "n_active": int(active.sum()),
            "max_rmse_over_std": max_ratio,
            "mean_rmse_over_std": mean_ratio,
        })

    return {
        "passed": overall_pass,
        "per_frame_fields": field_summary,
        "static_fields": static_summary,
        "n_per_frame_samples": int(pf_n),
        "n_static_samples": int(st_n),
    }


# ---------------------------------------------------------------------------
# Gate B — reachability constraints never violated
# ---------------------------------------------------------------------------


def gate_b_reachability(bm: BodyModel, stats: dict, n_samples: int = 10_000,
                        batch: int = 128, T: int = 50,
                        seed: int = 0) -> dict:
    """Generate random in-envelope inputs, run the model with hard clamp,
    assert no output is outside the physical envelope.

    Hard-clamp produces post-clamp values; we double-check they sit inside
    the bounds to a numerical tolerance. The point of this gate is to
    confirm the *clamp wiring* is doing its job (any future regression
    that bypasses the clamp would be caught here)."""
    log.info("[Gate B] reachability scan: n=%d", n_samples)

    bounds = DEFAULT_REACHABILITY
    rng = np.random.default_rng(seed)

    violations: dict[str, int] = {}
    examples: dict[str, dict] = {}
    n_total = 0
    n_batches = (n_samples + batch - 1) // batch

    @jax.jit
    def predict_hardclamp(params, gesture_b, anatomy_b):
        fn = lambda g, a: body_model_forward(
            params, g, a, bm.cfg, use_hard_clamp=True,
        )
        return jax.vmap(fn)(gesture_b, anatomy_b)

    g_mean = stats["gesture_mean"]
    g_std = np.maximum(stats["gesture_std"], 1e-6)
    a_mean = stats["anatomy_mean"]
    a_std = np.maximum(stats["anatomy_std"], 1e-6)

    for bi in range(n_batches):
        b = min(batch, n_samples - n_total)
        # Sample raw inputs from the training distribution (mean ± std),
        # then Z-score them with the same stats the model was trained
        # against. This keeps the test "in-envelope" while exercising the
        # same input transform as eval-time inference.
        gesture_raw = (g_mean + g_std *
                       rng.normal(0.0, 1.0, size=(b, T, N_GESTURE_CHANNELS))
                       ).astype(np.float32)
        anatomy_raw = (a_mean + a_std *
                       rng.normal(0.0, 1.0, size=(b, D_ANATOMY_PLACEHOLDER))
                       ).astype(np.float32)
        gesture = normalize_gesture(gesture_raw, stats)
        anatomy = normalize_anatomy(anatomy_raw, stats)

        out = predict_hardclamp(bm.params, gesture, anatomy)
        # Pull host-side
        check_fields = {
            "tube_length_cm": (np.asarray(out.tube_length_cm),
                               bounds.min_length_cm, bounds.max_length_cm),
            "tube_area_cm2": (np.asarray(out.tube_area_cm2),
                              bounds.min_area_cm2, bounds.max_area_cm2),
            "F0": (np.asarray(out.glottis_state[..., 0]),
                   bounds.min_f0_hz, bounds.max_f0_hz),
            "lung_pressure": (np.asarray(out.glottis_state[..., 1]),
                              bounds.min_lung_pa, bounds.max_lung_pa),
            "nasal_coupling": (np.asarray(out.nasal_coupling),
                               bounds.min_nasal, bounds.max_nasal),
            "subglottal_press": (np.asarray(out.subglottal_press),
                                 bounds.min_lung_pa, bounds.max_lung_pa),
            "frication_gain": (np.asarray(out.frication_gain), 0.0, np.inf),
            "frication_color": (np.asarray(out.frication_color),
                                -bounds.frication_color_cap,
                                bounds.frication_color_cap),
        }
        tol = 1e-5
        for fname, (arr, lo, hi) in check_fields.items():
            v_low = (arr < lo - tol).sum()
            v_high = (arr > hi + tol).sum() if np.isfinite(hi) else 0
            v_nan = (~np.isfinite(arr)).sum()
            v = int(v_low + v_high + v_nan)
            if v > 0:
                violations[fname] = violations.get(fname, 0) + v
                if fname not in examples:
                    flat_low = arr.flatten()
                    examples[fname] = {
                        "min": float(flat_low.min()),
                        "max": float(flat_low.max()),
                        "n_lo": int(v_low), "n_hi": int(v_high),
                        "n_nan": int(v_nan),
                    }
        n_total += b

    passed = len(violations) == 0
    return {
        "passed": passed,
        "n_samples_run": n_total,
        "violations": violations,
        "examples": examples,
    }


# ---------------------------------------------------------------------------
# Gate C — anti-cheat refusal
# ---------------------------------------------------------------------------


def gate_c_refusal(bm: BodyModel, val_shards: list[Path],
                   train_stats: dict, stats: dict,
                   threshold: float = 0.5,
                   batch: int = 128) -> dict:
    """Feed unreachable samples through the model and measure refusal.

    A sample is "refused" when its per-frame output is close to the
    refusal baseline (REFUSAL_PER_FRAME), measured as a per-sample
    normalized L2 distance ≤ ``threshold`` (in per-channel-std units).

    The threshold is in per-channel-std units — this matches the
    training loss normalization, so threshold=0.5 means "the model's
    mean per-channel deviation from the refusal baseline is at most
    half a standard deviation, in normalized space."
    """
    log.info("[Gate C] anti-cheat refusal scan")

    pf_std = train_stats["per_frame_std"]
    pf_std_safe = np.where(pf_std >= 1e-4, pf_std, 1.0)
    pf_mask = (pf_std >= 1e-4).astype(np.float32)
    refusal = REFUSAL_PER_FRAME  # (F,)

    distances: list[float] = []
    for sp in val_shards:
        d = np.load(sp, allow_pickle=True)
        unreachable = d["unreachable"].astype(np.bool_)
        if not unreachable.any():
            continue
        gesture = d["gesture"][unreachable].astype(np.float32)
        anatomy = d["anatomy"][unreachable].astype(np.float32)
        n = gesture.shape[0]
        for i in range(0, n, batch):
            sl = slice(i, min(i + batch, n))
            g_n = normalize_gesture(gesture[sl], stats)
            a_n = normalize_anatomy(anatomy[sl], stats)
            pred_pf, _ = _predict_with_static(bm.params, bm.cfg, g_n, a_n)
            pred_pf = np.asarray(pred_pf)  # (b, T, F)
            # Per-sample normalized L2 distance from refusal baseline:
            diff = (pred_pf - refusal[None, None, :]) / pf_std_safe[None, None, :]
            diff = diff * pf_mask[None, None, :]
            # Mean over (T, F) → (b,)
            per_sample = np.sqrt((diff ** 2).mean(axis=(1, 2)))
            distances.extend(per_sample.tolist())

    if not distances:
        log.warning("no unreachable samples found in val shards — gate inconclusive")
        return {
            "passed": None,
            "n_unreachable": 0,
            "refusal_rate": float("nan"),
            "threshold": threshold,
        }

    distances = np.asarray(distances)
    refused = (distances <= threshold).astype(np.float32)
    refusal_rate = float(refused.mean())
    passed = refusal_rate >= 0.80

    return {
        "passed": passed,
        "n_unreachable": int(len(distances)),
        "refusal_rate": refusal_rate,
        "threshold": threshold,
        "distance_p50": float(np.percentile(distances, 50)),
        "distance_p90": float(np.percentile(distances, 90)),
        "distance_p99": float(np.percentile(distances, 99)),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--corpus", type=Path,
                    default=REPO_ROOT / "outputs" / "body_corpus")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-reach-samples", type=int, default=10_000)
    ap.add_argument("--refusal-threshold", type=float, default=0.5)
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs" / "body_model_eval.json")
    args = ap.parse_args()

    log.info("loading checkpoint %s", args.ckpt)
    bm, payload = load_body_model(args.ckpt)
    # Stats from training: includes input means/stds (for normalization)
    # and target means/stds (for loss-aware diagnostics).
    stats = {k: np.asarray(v) for k, v in payload["stats"].items()}
    train_stats = stats  # alias for clarity in gate-C call below
    log.info("model loaded: %d params", bm.num_params())

    log.info("listing corpus shards")
    shards = list_shards(args.corpus)
    train_shards, val_shards = split_shards(shards, args.val_frac, args.seed)
    log.info("split: %d train / %d val shards (using val for eval)",
             len(train_shards), len(val_shards))

    t0 = time.time()
    a = gate_a_rmse(bm, val_shards, stats)
    t_a = time.time() - t0
    log.info("[Gate A] passed=%s (%.1fs)", a["passed"], t_a)

    t0 = time.time()
    b = gate_b_reachability(bm, stats, n_samples=args.n_reach_samples)
    t_b = time.time() - t0
    log.info("[Gate B] passed=%s (%.1fs)", b["passed"], t_b)

    t0 = time.time()
    c = gate_c_refusal(bm, val_shards, train_stats, stats,
                       threshold=args.refusal_threshold)
    t_c = time.time() - t0
    log.info("[Gate C] passed=%s (%.1fs)", c["passed"], t_c)

    summary = {
        "ckpt": str(args.ckpt),
        "corpus": str(args.corpus),
        "gate_A_rmse_held_out": a,
        "gate_B_reachability": b,
        "gate_C_refusal": c,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))
    log.info("eval report: %s", args.out)

    # ---- Print summary table
    print()
    print("=" * 72)
    print(f"{'Gate':<16} {'Pass':<8} {'Detail'}")
    print("-" * 72)
    print(f"{'A: held-out RMSE':<16} {str(a['passed']):<8} "
          f"max RMSE/std per field below:")
    for f in a["per_frame_fields"]:
        active_msg = (f"max={f['max_rmse_over_std']:.3f} "
                      f"mean={f['mean_rmse_over_std']:.3f}"
                      if f["n_active"] > 0 else "n/a (all constant)")
        marker = " " if (f["passed_gate_A"] is True or f["passed_gate_A"] is None) else "FAIL"
        print(f"  per-frame.{f['field']:<22} {active_msg}  {marker}")
    print(f"{'B: reachability':<16} {str(b['passed']):<8} "
          f"violations={b['violations'] or 'none'}; "
          f"n_run={b['n_samples_run']}")
    print(f"{'C: refusal':<16} {str(c['passed']):<8} "
          f"rate={c.get('refusal_rate', float('nan')):.3f} "
          f"(threshold={c['threshold']}, n_unreach={c.get('n_unreachable', 0)})")
    print("=" * 72)

    all_pass = (a["passed"] is True and b["passed"] is True and
                c["passed"] is True)
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
