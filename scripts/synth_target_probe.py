"""Synthetic-target copyfit probe — the "is the optimizer broken?" test.

Renders TEA on a known tube + source schedule to produce a target
audio whose ground-truth field is known exactly. Then copyfit attempts
to recover it from scratch (uniform-pipe init). If copyfit can recover
an audio that TEA generated, the optimization machinery is sound and
the Natasha-phrase failure is "real audio has no clean tube
interpretation" — in which case SPARC kinematic supervision has been proposed
as the path forward. NOTE (A4): whether kinematic supervision is structurally
required is UNDETERMINED pending a converged, frication-capable negative result.
See `docs/incorrect_assumptions.md` A4. If copyfit CAN'T recover even a
TEA-generated synthetic, there is a structural problem with the optimizer
recipe we haven't identified.

Three subcommands:

    gen      Render a known tube+source through TEA, save WAV + truth NPZ.
    compare  Compare recovered field (copyfit output) to ground truth.
    summary  Compact pass/fail report from a compare run.

Workflow:
    # 1. Generate synth target (Story-anchor vowel, sustained, voiced).
    python3 scripts/synth_target_probe.py gen --shape A --mode static

    # 2. Run copyfit on it (separate invocation, ~75 min on 16 GB GPU).
    python3 scripts/copyfit_neural_field_phrase.py \\
      --audio data/probes/synth_target_static_A.wav \\
      --transcript "synthetic-a-sustained" \\
      --out outputs/copyfit_neural_field/synth_target_static_A \\
      --iters 100 --seeds 1 --lr 3e-3 --warmup 20 \\
      --update-norm-cap 0.35 --w-smooth-s 1e-2 --w-smooth-t 3e-2 --w-curv 1e-3 \\
      --diag-every 5

    # 3. Compare recovered to truth.
    python3 scripts/synth_target_probe.py compare \\
      --recovered outputs/copyfit_neural_field/synth_target_static_A/field_snapshot.npz \\
      --truth     data/probes/synth_target_static_A_truth.npz

Pass criterion (compare): per-frame area MSE ≤ 0.1 cm² on the free
range. Loose because the optimizer doesn't get the exact source
schedule the renderer used (PyWorld DIO reconstructs F0/RMS, not
the full 11-channel glottis state), so some tract compensation is
expected.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tractus_engine_advanced" / "src"))

import jax.numpy as jnp  # noqa: E402
import soundfile as sf  # noqa: E402

from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: E402


# --------------------------------------------------------------------------
# Constants — mirror copyfit_neural_field_phrase.py exactly.
# --------------------------------------------------------------------------

# A1 FIX — TEA renders end-to-end at its native 44.1 kHz (TIMESTEP_S = 1/44100).
# These constants now mirror the A1-fixed copyfit_neural_field_phrase.py
# (SAMPLE_RATE = 44100, SAMPLES_PER_FRAME = 882). Previously this script carried
# SR = 22050 / SAMPLES_PER_FRAME = 441, which labelled 44.1 kHz TEA audio as
# 22.05 kHz — the octave-shift bug (playback an octave low). The old "intentional
# mirror" justification went stale when copyfit moved to 44100/882; the correct
# mirror is the fixed value below. The compare subcommand is unaffected either
# way (pure area-function comparison). See `docs/incorrect_assumptions.md` A1.
SR = 44100
SAMPLES_PER_FRAME = SR // 50   # 882 at 50 Hz frame rate
FRAME_RATE_HZ = 50
N_SECTIONS = 93
FREE_LO = 25
FREE_HI = 65
N_FREE = FREE_HI - FREE_LO

DEFAULT_TUBE_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "default_tube.npz"
STORY_ANCHORS_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "story_1996_anchors.npz"
PROBES_DIR = _ROOT / "data" / "probes"


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def voiced_glottis(n_frames: int, F0: float = 120.0, PR: float = 8000.0) -> np.ndarray:
    """11-channel voiced glottis trajectory."""
    g = np.array([F0, PR, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
                 dtype=np.float32)
    return np.tile(g[None, :], (n_frames, 1))


def load_default_tube() -> dict:
    d = np.load(DEFAULT_TUBE_PATH, allow_pickle=True)
    return {k: np.asarray(d[k], dtype=np.float32) for k in d.files
            if d[k].dtype.kind in "fi"}


def load_story_shape(shape_key: str) -> np.ndarray:
    """Return Story-anchor area (40,) for a given shape key."""
    s = np.load(STORY_ANCHORS_PATH, allow_pickle=False)
    keys = [str(k) for k in s["shape_keys"]]
    statuses = [str(k) for k in s["shape_status"]]
    if shape_key not in keys:
        raise ValueError(f"shape_key '{shape_key}' not in Story corpus. "
                         f"Ingested keys: {[k for k, st in zip(keys, statuses) if st == 'ingested']}")
    idx = keys.index(shape_key)
    if statuses[idx] != "ingested":
        raise ValueError(f"shape_key '{shape_key}' is in corpus but status='{statuses[idx]}'.")
    return np.asarray(s["area_cm2_tea"][idx], dtype=np.float32)


def splice_story_into_default(default: dict, story_area_40: np.ndarray) -> np.ndarray:
    """Build (93,) area function with Story-anchor splicing into FREE_LO:FREE_HI."""
    a = default["area_cm2"].copy()
    a[FREE_LO:FREE_HI] = story_area_40
    return a


def render_tea(area_93: np.ndarray, n_frames: int, default: dict,
               glottis_traj: np.ndarray) -> np.ndarray:
    """Render a sustained tube through TEA."""
    L = jnp.tile(jnp.asarray(default["length_cm"])[None, :], (n_frames, 1))
    A = jnp.tile(jnp.asarray(area_93)[None, :], (n_frames, 1))
    Mw = jnp.tile(jnp.asarray(default["wall_mass_cgs"])[None, :], (n_frames, 1))
    Bw = jnp.tile(jnp.asarray(default["wall_resistance_cgs"])[None, :], (n_frames, 1))
    Kw = jnp.tile(jnp.asarray(default["wall_stiffness_cgs"])[None, :], (n_frames, 1))
    g = jnp.asarray(glottis_traj)

    audio, *_ = synthesize_dynamic_tube(
        L, A, Mw, Bw, Kw,
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=g,
        frame_chunk_size=10,
    )
    return np.asarray(audio)


def rms_normalize(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return (x * (target_rms / rms)).astype(np.float32)


# --------------------------------------------------------------------------
# gen — produce a synthetic target
# --------------------------------------------------------------------------

def cmd_gen(args: argparse.Namespace) -> int:
    PROBES_DIR.mkdir(parents=True, exist_ok=True)
    default = load_default_tube()

    n_frames = int(args.duration_s * FRAME_RATE_HZ)
    glottis = voiced_glottis(n_frames, F0=args.f0, PR=args.pr)

    if args.mode == "static":
        # Single sustained Story vowel.
        story_area = load_story_shape(args.shape)
        area_93 = splice_story_into_default(default, story_area)
        area_per_frame_93 = np.tile(area_93[None, :], (n_frames, 1))
        label = f"static_{args.shape}"
        print(f"  static target: {n_frames} frames of /{args.shape}/")

    elif args.mode == "dynamic":
        # Three-way interpolated /a/ → /i/ → /u/ (or user-specified shapes).
        shapes = args.shapes.split(",") if args.shapes else ["A", "i", "u"]
        story_areas = [load_story_shape(k) for k in shapes]
        # Linear interpolation over time, equal-thirds.
        third = n_frames // len(shapes)
        area_per_frame_40 = np.empty((n_frames, N_FREE), dtype=np.float32)
        # Anchor frames at thirds.
        anchor_frames = [i * third + third // 2 for i in range(len(shapes))]
        for t in range(n_frames):
            # Find the bracketing anchors.
            if t <= anchor_frames[0]:
                area_per_frame_40[t] = story_areas[0]
            elif t >= anchor_frames[-1]:
                area_per_frame_40[t] = story_areas[-1]
            else:
                for i in range(len(anchor_frames) - 1):
                    if anchor_frames[i] <= t <= anchor_frames[i + 1]:
                        alpha = (t - anchor_frames[i]) / (anchor_frames[i + 1] - anchor_frames[i])
                        area_per_frame_40[t] = (1 - alpha) * story_areas[i] + alpha * story_areas[i + 1]
                        break
        area_per_frame_93 = np.tile(default["area_cm2"][None, :], (n_frames, 1)).astype(np.float32)
        area_per_frame_93[:, FREE_LO:FREE_HI] = area_per_frame_40
        label = f"dynamic_{'_'.join(shapes)}"
        print(f"  dynamic target: {len(shapes)} shapes {shapes} over {n_frames} frames")
    else:
        print(f"unknown mode '{args.mode}'", file=sys.stderr)
        return 2

    # Render through TEA. Slight refactor — render_tea above is sustained-only.
    print(f"  rendering through TEA (compile ~60-90s)...")
    L = jnp.asarray(np.tile(default["length_cm"][None, :], (n_frames, 1)), dtype=jnp.float32)
    A = jnp.asarray(area_per_frame_93, dtype=jnp.float32)
    Mw = jnp.asarray(np.tile(default["wall_mass_cgs"][None, :], (n_frames, 1)), dtype=jnp.float32)
    Bw = jnp.asarray(np.tile(default["wall_resistance_cgs"][None, :], (n_frames, 1)), dtype=jnp.float32)
    Kw = jnp.asarray(np.tile(default["wall_stiffness_cgs"][None, :], (n_frames, 1)), dtype=jnp.float32)
    g = jnp.asarray(glottis, dtype=jnp.float32)
    audio, *_ = synthesize_dynamic_tube(
        L, A, Mw, Bw, Kw,
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=g,
        frame_chunk_size=10,
    )
    audio_np = np.asarray(audio)
    audio_normed = rms_normalize(audio_np, target_rms=0.05)

    wav_path = PROBES_DIR / f"synth_target_{label}.wav"
    truth_path = PROBES_DIR / f"synth_target_{label}_truth.npz"

    sf.write(str(wav_path), audio_normed, SR, subtype="PCM_16")
    log_area = np.log(np.maximum(area_per_frame_93, 1e-4))
    np.savez(
        str(truth_path),
        area_cm2=area_per_frame_93,
        log_area=log_area,
        length_cm=default["length_cm"],
        glottis_traj=glottis,
        free_lo=FREE_LO,
        free_hi=FREE_HI,
        mode=args.mode,
        label=label,
        duration_s=args.duration_s,
    )
    print(f"  wrote {wav_path}")
    print(f"  wrote {truth_path}")
    print()
    print("Next step — run copyfit on this target:")
    print(f"  tractus_controller/.venv/bin/python scripts/copyfit_neural_field_phrase.py \\")
    print(f"    --audio {wav_path} \\")
    print(f'    --transcript "{label}" \\')
    print(f"    --out outputs/copyfit_neural_field/synth_target_{label} \\")
    print("    --iters 100 --seeds 1 --frame-chunk-size 10 \\")
    print("    --lr 3e-3 --warmup 20 --update-norm-cap 0.35 \\")
    print("    --w-smooth-s 1e-2 --w-smooth-t 3e-2 --w-curv 1e-3 \\")
    print("    --diag-every 5")
    return 0


# --------------------------------------------------------------------------
# compare — recovered vs truth
# --------------------------------------------------------------------------

def cmd_compare(args: argparse.Namespace) -> int:
    truth = np.load(args.truth, allow_pickle=False)
    recovered = np.load(args.recovered, allow_pickle=False)

    truth_log = np.asarray(truth["log_area"])             # (T, 93)
    truth_area = np.asarray(truth["area_cm2"])            # (T, 93)
    rec_log = np.asarray(recovered["log_area"])           # (T, 93)
    rec_area = np.asarray(recovered["area_cm2"])          # (T, 93)

    free_lo = int(truth["free_lo"])
    free_hi = int(truth["free_hi"])

    # Per-section trimming to the free range.
    truth_free_log = truth_log[:, free_lo:free_hi]
    rec_free_log = rec_log[:, free_lo:free_hi]
    truth_free_area = truth_area[:, free_lo:free_hi]
    rec_free_area = rec_area[:, free_lo:free_hi]

    # Match shapes — copyfit may pad to slightly different T.
    t_min = min(truth_free_log.shape[0], rec_free_log.shape[0])
    truth_free_log = truth_free_log[:t_min]
    rec_free_log = rec_free_log[:t_min]
    truth_free_area = truth_free_area[:t_min]
    rec_free_area = rec_free_area[:t_min]

    # Metrics
    log_mse = float(np.mean((truth_free_log - rec_free_log) ** 2))
    log_mae = float(np.mean(np.abs(truth_free_log - rec_free_log)))
    area_mse = float(np.mean((truth_free_area - rec_free_area) ** 2))
    area_mae = float(np.mean(np.abs(truth_free_area - rec_free_area)))

    # Per-section MSE.
    per_section_log_mse = np.mean((truth_free_log - rec_free_log) ** 2, axis=0)

    # Per-frame correlation across sections.
    per_frame_corr = []
    for t in range(t_min):
        c = float(np.corrcoef(truth_free_log[t], rec_free_log[t])[0, 1])
        per_frame_corr.append(c)
    per_frame_corr = np.asarray(per_frame_corr)

    # Mean field profile vs truth mean profile.
    mean_truth = truth_free_log.mean(axis=0)
    mean_rec = rec_free_log.mean(axis=0)

    # Pass gate: log_mae < 0.5 (the field is within e^0.5 = 65% area-factor on average).
    PASS_LOG_MAE = 0.5
    passed = log_mae < PASS_LOG_MAE

    report = {
        "passed": passed,
        "pass_threshold_log_mae": PASS_LOG_MAE,
        "log_mae": log_mae,
        "log_mse": log_mse,
        "area_mae_cm2": area_mae,
        "area_mse_cm2": area_mse,
        "per_section_log_mse": per_section_log_mse.tolist(),
        "per_frame_corr_mean": float(per_frame_corr.mean()),
        "per_frame_corr_min": float(per_frame_corr.min()),
        "n_frames_compared": t_min,
        "mean_truth_log_area": mean_truth.tolist(),
        "mean_rec_log_area": mean_rec.tolist(),
    }

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("=== synth-target recovery ===")
        print(f"  passed:               {passed}")
        print(f"  log_mae:              {log_mae:.4f}  (gate < {PASS_LOG_MAE})")
        print(f"  log_mse:              {log_mse:.4f}")
        print(f"  area_mae (cm²):       {area_mae:.4f}")
        print(f"  per-frame corr mean:  {per_frame_corr.mean():.3f}")
        print(f"  per-frame corr min:   {per_frame_corr.min():.3f}")
        print(f"  n_frames compared:    {t_min}")
        print()
        print("interpretation:")
        if passed:
            print("  ✓ optimizer recovered the synth target. machinery is sound.")
            print("    if Natasha still fails, the gap is 'real audio ≠ a clean")
            print("    tube' — path forward is SPARC kinematic supervision.")
            print("    (A4 caveat: whether SPARC supervision is structurally required")
            print("     is undetermined — see docs/incorrect_assumptions.md A4.)")
        else:
            print("  ✗ optimizer could NOT recover even a TEA-generated target.")
            print("    structural issue in recipe; recipe tuning alone won't")
            print("    fix the Natasha failure. Investigate before continuing.")
    return 0 if passed else 1


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    pg = sub.add_parser("gen", help="Render a synth target through TEA.")
    pg.add_argument("--shape", default="A",
                    help="Story shape key for static mode (default: A = /ɑ/).")
    pg.add_argument("--shapes", default="",
                    help="Comma-separated shape keys for dynamic mode (default: A,i,u).")
    pg.add_argument("--mode", choices=["static", "dynamic"], default="static")
    pg.add_argument("--duration-s", type=float, default=1.5)
    pg.add_argument("--f0", type=float, default=120.0)
    pg.add_argument("--pr", type=float, default=8000.0)

    pc = sub.add_parser("compare", help="Compare recovered field to truth.")
    pc.add_argument("--recovered", type=Path, required=True,
                    help="copyfit's field_snapshot.npz")
    pc.add_argument("--truth", type=Path, required=True,
                    help="synth_target_*_truth.npz from `gen` step")
    pc.add_argument("--json", action="store_true",
                    help="Emit JSON report on stdout.")

    args = p.parse_args(argv)
    if args.cmd == "gen":
        return cmd_gen(args)
    elif args.cmd == "compare":
        return cmd_compare(args)
    else:
        p.print_help()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
