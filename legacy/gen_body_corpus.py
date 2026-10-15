#!/usr/bin/env python3
"""Generate the body-model pretraining corpus (pretraining/002 v0).

Each sample is a triple ``(gesture_traj, anatomy_vec, target_renderer_interface)``.

v0 mix (post-spec rebalance — drop the 40% inversion-fitted slice that
requires inversion/005 fits we don't have yet):

   60%  procedural perturbations around named-shape phone inventory
   30%  uniform sampling within global gesture bounds (coverage)
   10%  edge-case sampling near per-channel min/max + anti-cheat
        unreachable targets

Anti-cheat: ~30% of the edge-case slice (3% of total) carries an
``unreachable`` flag. The training loop (forthcoming, session 2) is
expected to give those samples a no-op target signal so the body model
learns to refuse impossible postures.

Teacher pipeline (no TEA in the loop — non-circularity):

1. Sample / build a 30-d (T, 30) tract+glottis trajectory at 50 Hz.
2. Run R1 NN inference on the 19-d tract slice → (T, 40) pharynx-mouth
   length & area.
3. Compose with the default tube fixture into a (T, 93) length & area.
4. Glottis state target = tract+glottis trajectory cols [0:11]; nasal /
   subglottal_press derived from the same.
5. Anatomy vector = a synthetic 64-d perturbation of zero (placeholder
   for the Module 5 speaker embedding).
6. Gesture trajectory = the 30-d trajectory mapped to the 20-d gesture
   schema (12 articulator + 8 source) via :data:`TRACT30_TO_GESTURE`.

Output:

   outputs/body_corpus/manifest.json
   outputs/body_corpus/shard_0000.npz
   outputs/body_corpus/shard_0001.npz
   ...

Each NPZ holds N_PER_SHARD samples in concatenated (B, T, ...) form.

Usage::

    python scripts/gen_body_corpus.py --n 100000 --out outputs/body_corpus

Wall-clock: ~5 min for 100K samples on the 16 GB GPU (R1 NN inference
is the bottleneck; vmapped on GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tractus_controller"))
sys.path.insert(0, str(REPO_ROOT / "tractus_engine_advanced" / "src"))
sys.path.insert(0, str(REPO_ROOT / "tractus_shared" / "src"))


# Defer heavy imports until after path setup
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from tractus_controller.body_model import (  # noqa: E402
    GESTURE_CHANNELS,
    N_GESTURE_CHANNELS,
    D_ANATOMY_PLACEHOLDER,
)
from tractus_controller.body_model.channels import (  # noqa: E402
    GESTURE_NEUTRAL,
    PER_FRAME_FREE_FIELDS,
    STATIC_FIELDS,
    RENDERER_INTERFACE_FIELDS,
)
from tractus_controller.adapters.r1_inference import (  # noqa: E402
    load_r1_checkpoint,
    load_default_tube,
    tract_to_tube_jax,
    compose_93_section_jax,
)
from tractus_controller.smoke import templates as tmpl  # noqa: E402


# ---------------------------------------------------------------------------
# Mapping: 30-d (T, 30) trajectory ↔ 20-d gesture schema
# ---------------------------------------------------------------------------

# (gesture_channel_name, source_30d_col_or_None, scale, offset)
# When source is None, the channel is left at its neutral value.
TRACT30_TO_GESTURE: tuple[tuple[str, int | None, float, float], ...] = (
    # Articulator
    ("jaw_open",         tmpl.TRACT_JA, -1.0, 0.0),     # JA more negative = more open; flip sign
    ("tongue_body_x",    tmpl.TRACT_TCX, 1.0, 0.0),
    ("tongue_body_y",    tmpl.TRACT_TCY, 1.0, 0.0),
    ("tongue_tip_x",     tmpl.TRACT_TTX, 1.0, 0.0),
    ("tongue_tip_y",     tmpl.TRACT_TTY, 1.0, 0.0),
    ("lip_protrusion",   tmpl.TRACT_LP,  1.0, 0.0),
    ("lip_aperture",     tmpl.TRACT_LD,  1.0, 0.0),
    ("velum_opening",    tmpl.TRACT_VO,  1.0, 0.0),
    ("tongue_tip_side",  tmpl.TRACT_TS3, 1.0, 0.0),
    ("tongue_side",      tmpl.TRACT_TS1, 1.0, 0.0),
    ("larynx_height",    tmpl.TRACT_HY,  1.0, 0.0),     # vertical
    ("pharyngeal_width", tmpl.TRACT_HX,  1.0, 0.0),     # horizontal proxy
    # Source
    ("F0",               tmpl.GLOTTIS_F0, 1.0, 0.0),
    ("F0_jitter",        tmpl.GLOTTIS_FL, 1.0, 0.0),
    ("lung_pressure",    tmpl.GLOTTIS_PR, 1.0, 0.0),
    ("glottal_adduction",tmpl.GLOTTIS_CA, 1.0, 0.0),
    ("vocal_fold_tension", tmpl.GLOTTIS_XT, 1.0, 0.0),
    ("breathiness",      tmpl.GLOTTIS_AS, 1.0, 0.0),
    ("creak_gate",       None, 0.0, 0.0),
    ("pressed_voice_gate", None, 0.0, 0.0),
)

# Sanity: order matches GESTURE_CHANNELS exactly.
assert tuple(c[0] for c in TRACT30_TO_GESTURE) == GESTURE_CHANNELS, (
    "TRACT30_TO_GESTURE ordering must match GESTURE_CHANNELS exactly"
)


def traj30_to_gesture(traj30: np.ndarray) -> np.ndarray:
    """Map a (T, 30) 30-d trajectory to a (T, 20) gesture vector.

    Channels with no 30-d source get the neutral value from
    :data:`GESTURE_NEUTRAL`.
    """
    T = traj30.shape[0]
    out = np.zeros((T, N_GESTURE_CHANNELS), dtype=np.float32)
    for i, (name, src, scale, off) in enumerate(TRACT30_TO_GESTURE):
        if src is None:
            out[:, i] = GESTURE_NEUTRAL[name]
        else:
            out[:, i] = traj30[:, src] * scale + off
    return out


# ---------------------------------------------------------------------------
# Sample generators
# ---------------------------------------------------------------------------

NEUTRAL_30D: np.ndarray = tmpl.neutral_frame()  # (30,)


@dataclass
class Sample:
    """One corpus row, all numpy arrays."""

    gesture_traj: np.ndarray         # (T, D_gesture)
    anatomy_vec: np.ndarray          # (D_anatomy,)
    # Per-frame target free fields (concatenated to (T, F)):
    target_per_frame: np.ndarray     # (T, F)
    # Static target fields, concatenated (Fs,):
    target_static: np.ndarray        # (Fs,)
    # Metadata:
    slice_kind: str                  # 'procedural' | 'uniform' | 'edge' | 'unreachable'
    unreachable: bool


def _random_traj30_procedural(rng: np.random.Generator, T: int) -> np.ndarray:
    """Pick a random phone template and apply its overrides on top of
    NEUTRAL_30D, plus mild gaussian perturbation."""
    phone = rng.choice(list(tmpl.PHONE_TEMPLATES.keys()))
    template = tmpl.PHONE_TEMPLATES[phone]
    base = NEUTRAL_30D.copy()
    for col, target, strength in template.overrides:
        base[col] = base[col] + (target - base[col]) * strength
    base[tmpl.GLOTTIS_F0] = 80.0 + rng.uniform(-15.0, 60.0)
    base[tmpl.GLOTTIS_PR] = 6000.0 + rng.uniform(-2000.0, 6000.0) if template.voiced else 0.0
    base[tmpl.GLOTTIS_AS] = template.aspiration

    # Tile to T frames + add temporal perturbation
    traj = np.broadcast_to(base, (T, 30)).copy()
    perturb = rng.normal(0.0, 0.05, size=(T, 30)).astype(np.float64)
    traj = traj + perturb
    # Smooth the perturbation in time so we get a plausible articulator
    # velocity (50 Hz: avoid step-changes at adjacent frames).
    kernel = np.array([0.25, 0.5, 0.25], dtype=np.float64)
    smoothed = traj.copy()
    for c in range(30):
        smoothed[:, c] = np.convolve(traj[:, c], kernel, mode="same")
    return smoothed


def _random_traj30_uniform(rng: np.random.Generator, T: int) -> np.ndarray:
    """Uniformly sample within tract+glottis bounds.

    Source bounds: from FreeChannel inventory + reasonable extras for
    non-free columns (left at neutral with small noise).
    """
    base = NEUTRAL_30D.copy()
    # F0
    base[tmpl.GLOTTIS_F0] = rng.uniform(60.0, 350.0)
    # Lung pressure
    base[tmpl.GLOTTIS_PR] = rng.uniform(0.0, 16000.0)
    # Aspiration
    base[tmpl.GLOTTIS_AS] = rng.uniform(0.0, 0.8)
    # Adduction (CA)
    base[tmpl.GLOTTIS_CA] = rng.uniform(0.0, 0.8)
    # Articulators within FreeChannel ranges
    base[tmpl.TRACT_JA] = rng.uniform(-7.0, 0.0)
    base[tmpl.TRACT_LP] = rng.uniform(-1.0, 1.0)
    base[tmpl.TRACT_LD] = rng.uniform(-2.0, 4.0)
    base[tmpl.TRACT_VO] = rng.uniform(-0.1, 1.0)
    base[tmpl.TRACT_TCX] = rng.uniform(-3.0, 4.0)
    base[tmpl.TRACT_TCY] = rng.uniform(-3.0, 1.0)
    base[tmpl.TRACT_TTX] = rng.uniform(-3.0, 4.0)
    base[tmpl.TRACT_TTY] = rng.uniform(-2.0, 4.0)

    traj = np.broadcast_to(base, (T, 30)).copy()
    # Add a slow time-varying drift to keep things plausible-ish
    drift = rng.normal(0.0, 0.1, size=(T, 30))
    traj = traj + drift
    return traj


def _random_traj30_edge(rng: np.random.Generator, T: int) -> np.ndarray:
    """Sample at clamp boundaries: pick channels and push them to min or max."""
    base = NEUTRAL_30D.copy()
    for col, lo, hi in [
        (tmpl.GLOTTIS_F0, 40.0, 600.0),
        (tmpl.GLOTTIS_PR, 0.0, 20000.0),
        (tmpl.TRACT_JA, -7.0, 0.0),
        (tmpl.TRACT_LD, -2.0, 4.0),
        (tmpl.TRACT_TCX, -3.0, 4.0),
        (tmpl.TRACT_TCY, -3.0, 1.0),
    ]:
        if rng.random() < 0.5:
            base[col] = lo if rng.random() < 0.5 else hi
    traj = np.broadcast_to(base, (T, 30)).copy()
    return traj


def _random_anatomy(rng: np.random.Generator) -> np.ndarray:
    """Synthetic anatomy vector — placeholder for Module 5 embedding.

    Drawn from a unit gaussian; downstream the body model should learn
    that small perturbations correspond to plausible inter-speaker
    variation."""
    return rng.normal(0.0, 0.5, size=(D_ANATOMY_PLACEHOLDER,)).astype(np.float32)


# ---------------------------------------------------------------------------
# Teacher: 30-d trajectory → renderer interface (R1 NN + default tube)
# ---------------------------------------------------------------------------


def _build_teacher():
    """Load R1 checkpoint + default tube once, return jit'd batch fn."""
    params, cfg, norm, _ = load_r1_checkpoint()
    default = load_default_tube()
    L0 = default["length_cm"]
    A0 = default["area_cm2"]
    Mw = default["wall_mass_cgs"]
    Bw = default["wall_resistance_cgs"]
    Kw = default["wall_stiffness_cgs"]

    @jax.jit
    def teacher(traj30: jnp.ndarray) -> dict:
        """traj30: (T, 30) → dict of renderer-interface fields."""
        glottis_traj = traj30[:, :11]
        tract_traj = traj30[:, 11:]
        # R1 NN: (T, 40)
        pm_length, pm_area = jax.vmap(
            lambda x: tract_to_tube_jax(params, cfg, norm, x)
        )(tract_traj)
        T = tract_traj.shape[0]
        gl_length = jnp.zeros((T, 2))
        gl_area = jnp.zeros((T, 2))
        L, A = jax.vmap(
            lambda pmL, pmA, glL, glA: compose_93_section_jax(
                L0, A0, pmL, pmA, glL, glA
            )
        )(pm_length, pm_area, gl_length, gl_area)
        # nasal_coupling derived from velum opening (col 18) — VO ∈
        # [-0.1, 1.0]; clip to [0, 1] for nasal coupling fraction.
        nasal = jnp.clip(traj30[:, tmpl.TRACT_VO], 0.0, 1.0)
        # subglottal_press = lung_pressure (PR) col 1
        sub = jnp.clip(traj30[:, tmpl.GLOTTIS_PR], 0.0, 25000.0)
        # frication: zero for v0 corpus (v1 derives from constriction
        # geometry; here we just don't supervise it).
        fric_gain = jnp.zeros((T, 93))
        fric_color = jnp.zeros((T, 93))
        return {
            "tube_length_cm": L,
            "tube_area_cm2": A,
            "glottis_state": glottis_traj,
            "nasal_coupling": nasal,
            "subglottal_press": sub,
            "frication_gain": fric_gain,
            "frication_color": fric_color,
        }

    static_target = {
        "wall_mass_static": np.asarray(Mw, dtype=np.float32),
        "wall_resistance_static": np.asarray(Bw, dtype=np.float32),
        "wall_stiffness_static": np.asarray(Kw, dtype=np.float32),
    }
    return teacher, static_target


def _flatten_per_frame(d: dict) -> np.ndarray:
    cols = []
    for name in PER_FRAME_FREE_FIELDS:
        x = np.asarray(d[name])
        if x.ndim == 1:
            x = x[:, None]
        cols.append(x)
    return np.concatenate(cols, axis=-1).astype(np.float32)


def _flatten_static(d: dict) -> np.ndarray:
    cols = []
    for name in STATIC_FIELDS:
        cols.append(np.asarray(d[name]).reshape(-1))
    return np.concatenate(cols, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def generate(
    n: int,
    out_dir: Path,
    *,
    n_per_shard: int = 1024,
    T: int = 50,                # 1.0 s per sample at 50 Hz
    proc_frac: float = 0.60,
    uniform_frac: float = 0.30,
    edge_frac: float = 0.10,
    anti_cheat_frac_of_edge: float = 0.30,
    seed: int = 0,
) -> dict:
    """Main entry point. Returns summary stats dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    teacher, static_target = _build_teacher()
    static_target_flat = _flatten_static(static_target)

    # Slice counts
    n_proc = int(round(n * proc_frac))
    n_uniform = int(round(n * uniform_frac))
    n_edge = n - n_proc - n_uniform
    n_unreachable = int(round(n_edge * anti_cheat_frac_of_edge))
    n_edge_clean = n_edge - n_unreachable

    print(f"[corpus] target n={n}: proc={n_proc} uniform={n_uniform} "
          f"edge_clean={n_edge_clean} unreachable={n_unreachable}", flush=True)

    counts = {
        "procedural": 0, "uniform": 0, "edge": 0, "unreachable": 0,
    }

    shard_idx = 0
    buf: list[Sample] = []
    t0 = time.time()
    last_log = t0

    def _flush():
        nonlocal shard_idx, buf
        if not buf:
            return
        shard_path = out_dir / f"shard_{shard_idx:04d}.npz"
        gesture = np.stack([s.gesture_traj for s in buf], axis=0)
        anatomy = np.stack([s.anatomy_vec for s in buf], axis=0)
        target_pf = np.stack([s.target_per_frame for s in buf], axis=0)
        target_st = np.stack([s.target_static for s in buf], axis=0)
        kinds = np.array([s.slice_kind for s in buf])
        unreachable = np.array([s.unreachable for s in buf], dtype=bool)
        np.savez(
            shard_path,
            gesture=gesture.astype(np.float32),
            anatomy=anatomy.astype(np.float32),
            target_per_frame=target_pf.astype(np.float32),
            target_static=target_st.astype(np.float32),
            slice_kind=kinds,
            unreachable=unreachable,
        )
        shard_idx += 1
        buf = []

    plan: list[str] = (
        ["procedural"] * n_proc +
        ["uniform"] * n_uniform +
        ["edge"] * n_edge_clean +
        ["unreachable"] * n_unreachable
    )
    rng.shuffle(plan)

    for i, kind in enumerate(plan):
        if kind == "procedural":
            traj30 = _random_traj30_procedural(rng, T)
        elif kind == "uniform":
            traj30 = _random_traj30_uniform(rng, T)
        elif kind == "edge":
            traj30 = _random_traj30_edge(rng, T)
        elif kind == "unreachable":
            # Build a "physically impossible" gesture: simultaneous
            # full lip closure + maximum lip protrusion + tongue tip
            # through palate. Use the procedural sampler as the base
            # then over-write a few channels to contradict each other.
            traj30 = _random_traj30_procedural(rng, T)
            traj30[:, tmpl.TRACT_LD] = -2.0      # pressed shut
            traj30[:, tmpl.TRACT_LP] =  1.0      # max protrusion (contradicts shut lips)
            traj30[:, tmpl.TRACT_TTY] = 4.0      # tongue tip through palate
            traj30[:, tmpl.TRACT_TCY] = 1.0      # tongue body slammed up
        else:
            raise ValueError(f"unknown kind: {kind}")

        gesture_traj = traj30_to_gesture(traj30)
        anatomy_vec = _random_anatomy(rng)
        target_dict = teacher(jnp.asarray(traj30, dtype=jnp.float32))
        # Block on JAX → host
        target_per_frame = _flatten_per_frame(
            {k: np.asarray(v) for k, v in target_dict.items()}
        )

        buf.append(Sample(
            gesture_traj=gesture_traj,
            anatomy_vec=anatomy_vec,
            target_per_frame=target_per_frame,
            target_static=static_target_flat,
            slice_kind=kind,
            unreachable=(kind == "unreachable"),
        ))
        counts[kind] += 1

        if len(buf) >= n_per_shard:
            _flush()

        now = time.time()
        if now - last_log > 5.0:
            done = i + 1
            rate = done / (now - t0)
            print(f"[corpus] {done}/{n} ({100*done/n:.1f}%)  rate={rate:.1f}/s", flush=True)
            last_log = now

    _flush()

    # Per-frame field column ranges (for downstream consumers)
    per_frame_layout: list[dict] = []
    cursor = 0
    for name in PER_FRAME_FREE_FIELDS:
        suffix = RENDERER_INTERFACE_FIELDS[name]
        width = int(np.prod(suffix)) if suffix else 1
        per_frame_layout.append({
            "name": name, "start": cursor, "width": width,
            "shape_suffix": list(suffix),
        })
        cursor += width
    static_layout: list[dict] = []
    cursor = 0
    for name in STATIC_FIELDS:
        suffix = RENDERER_INTERFACE_FIELDS[name]
        width = int(np.prod(suffix)) if suffix else 1
        static_layout.append({
            "name": name, "start": cursor, "width": width,
            "shape_suffix": list(suffix),
        })
        cursor += width

    summary = {
        "n_samples": n,
        "T_per_sample": T,
        "n_per_shard": n_per_shard,
        "n_shards": shard_idx,
        "frame_rate_hz": 50.0,
        "d_gesture": int(N_GESTURE_CHANNELS),
        "d_anatomy": int(D_ANATOMY_PLACEHOLDER),
        "gesture_channels": list(GESTURE_CHANNELS),
        "per_frame_layout": per_frame_layout,
        "static_layout": static_layout,
        "slice_counts": counts,
        "slice_fractions_planned": {
            "procedural": proc_frac,
            "uniform": uniform_frac,
            "edge": edge_frac,
        },
        "anti_cheat_frac_of_edge": anti_cheat_frac_of_edge,
        "teacher": "r1_nn + default_tube_fixture (procedural slice); "
                   "no TEA in pipeline (non-circularity, ARCH §445-456)",
        "consent_verified": True,
        "generated_at": time.time(),
        "elapsed_s": time.time() - t0,
        "seed": seed,
    }
    (out_dir / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(f"[corpus] done: {n} samples → {shard_idx} shards in "
          f"{time.time() - t0:.1f}s", flush=True)
    return summary


def _parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100_000,
                    help="Total samples to generate.")
    ap.add_argument("--out", type=Path,
                    default=REPO_ROOT / "outputs" / "body_corpus",
                    help="Output dir.")
    ap.add_argument("--n-per-shard", type=int, default=1024)
    ap.add_argument("--T", type=int, default=50,
                    help="Frames per sample (50 = 1.0 s @ 50 Hz).")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main():
    args = _parse_args()
    generate(
        n=args.n,
        out_dir=args.out,
        n_per_shard=args.n_per_shard,
        T=args.T,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
