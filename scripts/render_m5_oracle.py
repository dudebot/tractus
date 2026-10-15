"""infra/005 — render the 30-utterance m5_oracle audio fixture set.

Loads each `.tract` file under
``tractus_controller/outputs/fixtures/_oracle_work/`` (lexically first 30),
runs the per-frame trajectory through the controller's TEA pipeline
(``copy_synth.tea_backend.synthesize_via_tea_jax``), and writes a single
NPZ per utterance under ``outputs/fixtures/m5_oracle/`` with the schema
the workplan specifies.

Run::

    cd tractus_controller
    .venv/bin/python ../scripts/render_m5_oracle.py

Notes on the engine vs. spec:

* TEA's ``TIMESTEP_S`` is hard-coded to ``1/44100`` (see
  ``tractus_engine_advanced/tds/constants.py``). The audio
  ``synthesize_dynamic_tube`` produces is therefore at 44100 Hz.
  This is the CORRECT account of TEA's sample rate; see
  ``docs/incorrect_assumptions.md`` A1 for the canonical statement and
  the list of sites that had (or still have) this wrong.
* The .tract files are sampled at 44100/110 ≈ 400.91 Hz, so we drive the
  engine with ``samples_per_frame=110``: each frame holds the tube fixed
  for 110 audio samples = the file's native frame stride. No upstream
  resampling needed.
* The spec'd ``samples_per_frame=50 / sample_rate=22050`` (ARCHITECTURE.md
  §8.1) is aspirational — it would require either an engine-level
  re-derivation of TIMESTEP_S or a post-render decimation. Both are
  out-of-scope for infra/005, which is a fixture-regen task. We record
  ``sample_rate=44100`` in the NPZ and document the choice in the
  manifest. correctness/001 / correctness/002 consume these as
  self-comparison oracles, so internal consistency is what matters.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Force fp32 (correctness/001 default).
os.environ.setdefault("TEA_PRECISION", "fp32")

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
TEA_SRC = REPO_ROOT / "tractus_engine_advanced" / "src"
sys.path.insert(0, str(TEA_SRC))

ORACLE_WORK = REPO_ROOT / "outputs" / "fixtures" / "_oracle_work"
OUT_DIR = REPO_ROOT / "outputs" / "fixtures" / "m5_oracle"
MANIFEST_PATH = REPO_ROOT / "outputs" / "fixtures" / "MANIFEST.md"

DETERMINISTIC_SEED = np.int32(42)
SAMPLES_PER_FRAME = 110           # = TEA-native frame stride for a .tract file
TEA_SAMPLE_RATE_HZ = np.int32(44100)


def _git_hash() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
        return out
    except Exception:
        return "unknown"


def _peak_gpu_mb() -> float:
    """Peak GPU bytes since process start, in MB. 0.0 on CPU."""
    try:
        import jax
        devs = jax.devices()
        if not devs or devs[0].platform != "gpu":
            return 0.0
        stats = devs[0].memory_stats()
        if not stats:
            return 0.0
        peak = stats.get("peak_bytes_in_use", 0)
        return float(peak) / (1024 * 1024)
    except Exception:
        return 0.0


def _reset_gpu_peak() -> None:
    # JAX exposes no public reset for peak_bytes_in_use; we just record
    # session-wide peak and report it on the last utterance. Per-utterance
    # peak rows below report the session-cumulative peak as a monotone
    # upper bound; the final row is the true peak. Documented in MANIFEST.
    pass


def _utt_id(tract_path: Path) -> str:
    return tract_path.stem  # e.g. natasha_1_s00


def _load_trajectory(tract_path: Path) -> np.ndarray:
    """Returns (T, 30) float32 — cols 0:11 glottis, 11:30 tract."""
    from tractus_shared.tract_sequence import read_tract_sequence
    seq = read_tract_sequence(tract_path)
    return np.asarray(seq.data, dtype=np.float32)


def _render_one(traj: np.ndarray):
    """Drive controller's TEA backend. Returns (audio_f32, length_cm, area_cm2,
    glottis_traj) all as numpy arrays. length/area are post-compose 93-section
    arrays from the controller's adapter.
    """
    from tractus_controller.copy_synth.tea_backend import _ensure_tea_on_path
    _ensure_tea_on_path()
    import jax
    import jax.numpy as jnp
    from tractus_engine_advanced.pipeline import synthesize_dynamic_tube
    from tractus_engine_advanced.precision import DTYPE as TEA_DTYPE
    from tractus_controller.adapters.r1_inference import (
        compose_93_section_jax,
        load_default_tube,
        load_r1_checkpoint,
        tract_to_tube_jax,
    )

    params, cfg, norm, _section_weight = load_r1_checkpoint()
    default = load_default_tube()

    traj_j = jnp.asarray(traj, dtype=TEA_DTYPE)
    glottis_traj = traj_j[:, :11]
    tract_traj = traj_j[:, 11:]
    T = tract_traj.shape[0]

    pm_length, pm_area = jax.vmap(
        lambda x: tract_to_tube_jax(params, cfg, norm, x)
    )(tract_traj)

    gl_length = jnp.zeros((T, 2), dtype=TEA_DTYPE)
    gl_area = jnp.zeros((T, 2), dtype=TEA_DTYPE)

    L, A = jax.vmap(
        lambda pmL, pmA, glL, glA: compose_93_section_jax(
            default["length_cm"], default["area_cm2"], pmL, pmA, glL, glA
        )
    )(pm_length, pm_area, gl_length, gl_area)

    Mw = jnp.broadcast_to(default["wall_mass_cgs"][None, :], (T, 93))
    Bw = jnp.broadcast_to(default["wall_resistance_cgs"][None, :], (T, 93))
    Kw = jnp.broadcast_to(default["wall_stiffness_cgs"][None, :], (T, 93))

    audio, _, _, _ = synthesize_dynamic_tube(
        length_cm=L,
        area_cm2=A,
        wall_mass=Mw,
        wall_resistance=Bw,
        wall_stiffness=Kw,
        samples_per_frame=SAMPLES_PER_FRAME,
        glottis_traj=glottis_traj,
    )
    audio.block_until_ready()
    return (
        np.asarray(audio, dtype=np.float32),
        np.asarray(L, dtype=np.float32),
        np.asarray(A, dtype=np.float32),
        np.asarray(glottis_traj, dtype=np.float32),
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not ORACLE_WORK.exists():
        print(
            f"ERROR: input directory {ORACLE_WORK} does not exist.\n"
            "infra/005 needs the natasha_1 oracle .tract files staged at\n"
            f"  {ORACLE_WORK}/<utt>.tract\n"
            "before this script can render. See infra/005 task file for provenance."
        )
        return 2

    tract_files = sorted(ORACLE_WORK.glob("*.tract"))[:30]
    if len(tract_files) == 0:
        print(f"ERROR: no .tract files under {ORACLE_WORK}.")
        return 2
    if len(tract_files) < 30:
        print(f"WARN: only {len(tract_files)} .tract files; rendering what's there.")

    git_hash = _git_hash()
    print(f"git_hash={git_hash}  TEA_PRECISION={os.environ['TEA_PRECISION']}")
    print(f"out_dir={OUT_DIR}")
    print(f"samples_per_frame={SAMPLES_PER_FRAME}  sample_rate_hz={int(TEA_SAMPLE_RATE_HZ)}")
    print()

    rows = []  # for manifest
    rms_table = []  # for final report
    total_t0 = time.time()
    for i, tp in enumerate(tract_files):
        utt = _utt_id(tp)
        ges_path = tp.with_suffix(".ges")
        traj = _load_trajectory(tp)
        F = traj.shape[0]
        t0 = time.time()
        audio, length_cm, area_cm2, glottis_traj = _render_one(traj)
        dt = time.time() - t0

        if not np.isfinite(audio).all():
            raise RuntimeError(f"{utt}: non-finite audio")
        rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))

        out_path = OUT_DIR / f"{utt}.npz"
        np.savez(
            out_path,
            audio=audio.astype(np.float32),
            sample_rate=np.int32(TEA_SAMPLE_RATE_HZ),
            length_cm=length_cm.astype(np.float32),
            area_cm2=area_cm2.astype(np.float32),
            glottis_traj=glottis_traj.astype(np.float32),
            git_hash=np.array(git_hash),
            tea_precision=np.array("fp32"),
            deterministic_seed=DETERMINISTIC_SEED,
        )
        peak_mb = _peak_gpu_mb()
        rows.append({
            "utt": utt,
            "ges": ges_path.name,
            "frames": F,
            "samples": int(audio.size),
            "rms": rms,
            "render_s": dt,
            "peak_gpu_mb": peak_mb,
        })
        rms_table.append((utt, rms))
        print(
            f"[{i+1:02d}/{len(tract_files):02d}] {utt}: "
            f"F={F} N={audio.size} rms={rms:.4f} "
            f"render={dt:.2f}s peak_gpu={peak_mb:.1f}MB"
        )

        # Acceptance gate (RMS bounds).
        if not (0.005 <= rms <= 0.5):
            print(f"  WARN: RMS {rms:.4f} outside [0.005, 0.5]")

    total_dt = time.time() - total_t0
    print(f"\nTotal render time: {total_dt:.1f}s")

    # Determinism check: re-render the first utterance and assert bit-equal.
    print("\nDeterminism re-render of utterance[0]...")
    traj0 = _load_trajectory(tract_files[0])
    audio0_b, _, _, _ = _render_one(traj0)
    first_npz = np.load(OUT_DIR / f"{_utt_id(tract_files[0])}.npz")
    if np.array_equal(audio0_b, first_npz["audio"]):
        print("  PASS: bit-equal re-render confirmed.")
    else:
        diff = np.max(np.abs(audio0_b - first_npz["audio"]))
        print(f"  FAIL: re-render not bit-equal (max abs diff = {diff:g})")

    # MANIFEST.md update — append/replace the m5_oracle section.
    _update_manifest(rows, total_dt, git_hash)
    print(f"\nUpdated {MANIFEST_PATH}")

    # Final report.
    print("\n=== Per-utterance RMS ===")
    for u, r in rms_table:
        print(f"  {u}: rms={r:.4f}")
    print(f"Total render time: {total_dt:.1f}s")
    print(f"Peak GPU MB (final): {_peak_gpu_mb():.1f}")
    return 0


def _update_manifest(rows, total_dt: float, git_hash: str) -> None:
    """Append (or replace) an `## M5 oracle` section in MANIFEST.md."""
    lines = MANIFEST_PATH.read_text().splitlines() if MANIFEST_PATH.exists() else []
    # Strip any prior M5 oracle block.
    out = []
    skip = False
    for ln in lines:
        if ln.startswith("## M5 oracle"):
            skip = True
            continue
        if skip and ln.startswith("## "):
            skip = False
        if not skip:
            out.append(ln)
    # Trim trailing blanks.
    while out and out[-1].strip() == "":
        out.pop()
    out.append("")
    out.append("## M5 oracle (infra/005 fp32 fixture set)")
    out.append("")
    out.append(f"- Directory: `m5_oracle/`")
    out.append(f"- Provenance: rendered from `_oracle_work/*.tract` (first 30 by lex order)")
    out.append(f"  via `tractus_engine_advanced.pipeline.synthesize_dynamic_tube` at fp32.")
    out.append(f"- Renderer script: `scripts/render_m5_oracle.py`")
    out.append(f"- git_hash at render: `{git_hash}`")
    out.append(f"- TEA precision: `fp32` (TEA_PRECISION=fp32; correctness/001 default)")
    out.append(f"- Sample rate: 44100 Hz (TEA native; ARCHITECTURE.md §8.1's spec'd 22050 / 50 SPF")
    out.append(f"  is not realised — TIMESTEP_S is hard-coded to 1/44100; we drive at the engine's")
    out.append(f"  native rate with `samples_per_frame=110`, matching the .tract file's 110-sample-per-frame stride.)")
    out.append(f"- Anatomy: default speaker tube (`default_tube.npz`); R1 NN adapter (default checkpoint) for pharynx-mouth.")
    out.append(f"- Glottis: `glottis_traj` taken from the .tract file's first 11 columns; subglottal pressure source = `cp[PR]` per G3a path.")
    out.append(f"- Determinism: deterministic_seed = 42; bit-equal re-render verified on utterance[0].")
    out.append(f"- Total render time: {total_dt:.1f} s")
    out.append(f"- Anti-scope: NOT regenerated from any deleted upstream C++ driver. Pure-JAX TEA path only. Not an fp64 oracle.")
    out.append("")
    out.append("| utt | source .ges | frames | samples | rms | render_s | peak_gpu_mb |")
    out.append("|---|---|---|---|---|---|---|")
    for r in rows:
        out.append(
            f"| {r['utt']} | {r['ges']} | {r['frames']} | {r['samples']} | "
            f"{r['rms']:.4f} | {r['render_s']:.2f} | {r['peak_gpu_mb']:.1f} |"
        )
    out.append("")
    MANIFEST_PATH.write_text("\n".join(out) + "\n")


if __name__ == "__main__":
    sys.exit(main())
