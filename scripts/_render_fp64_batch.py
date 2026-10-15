"""Render the full correctness corpus at fp64 in a single subprocess.

Pre-set ``TEA_PRECISION=fp64`` before invoking. Reads each NPZ under
``outputs/fixtures/correctness_corpus/`` (in the manifest order) and
writes ``outputs/correctness_m5/_work/<utt>.fp64.npz`` with key ``audio``
as float64 samples.

Single-process amortizes JAX JIT compile cost (~1-2 min) across all 30
utterances, vs. paying it once per utterance from a parent driver.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

if os.environ.get("TEA_PRECISION", "").lower() not in ("fp64", "float64", "double", "f64"):
    print("ERROR: this script must be invoked with TEA_PRECISION=fp64", file=sys.stderr)
    sys.exit(2)

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tractus_engine_advanced" / "src"))

CORPUS_DIR = REPO_ROOT / "outputs" / "fixtures" / "correctness_corpus"
WORK_DIR = REPO_ROOT / "outputs" / "correctness_m5" / "_work"


def main(argv: list[str]) -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n-utterances", type=int, default=30)
    args = p.parse_args(argv)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CORPUS_DIR / "manifest.json").read_text())
    manifest = manifest[: args.n_utterances]

    import jax  # noqa: F401
    import jax.numpy as jnp
    from tractus_engine_advanced.pipeline import synthesize_dynamic_tube
    from tractus_engine_advanced.precision import IS_FP64
    if not IS_FP64:
        print("ERROR: precision policy didn't pick up fp64", file=sys.stderr)
        return 2

    print(f"=== fp64 corpus render — {len(manifest)} utterances ===")
    t0_total = time.time()
    for ix, m in enumerate(manifest):
        in_path = CORPUS_DIR / f"{m['utt_id']}.npz"
        out_path = WORK_DIR / f"{m['utt_id']}.fp64.npz"
        if out_path.exists():
            print(f"[{ix+1:02d}/{len(manifest)}] {m['utt_id']}: cached, skipping")
            continue
        d = np.load(in_path)
        t0 = time.time()
        audio, _, _, _ = synthesize_dynamic_tube(
            length_cm=jnp.asarray(d["length_cm"], dtype=jnp.float64),
            area_cm2=jnp.asarray(d["area_cm2"], dtype=jnp.float64),
            wall_mass=jnp.asarray(d["wall_mass"], dtype=jnp.float64),
            wall_resistance=jnp.asarray(d["wall_resistance"], dtype=jnp.float64),
            wall_stiffness=jnp.asarray(d["wall_stiffness"], dtype=jnp.float64),
            samples_per_frame=int(d["samples_per_frame"]),
            glottis_traj=jnp.asarray(d["glottis_traj"], dtype=jnp.float64),
        )
        audio.block_until_ready()
        np.savez(out_path, audio=np.asarray(audio, dtype=np.float64))
        print(f"[{ix+1:02d}/{len(manifest)}] {m['utt_id']}: {time.time() - t0:.2f}s")

    print(f"Total fp64 render time: {time.time() - t0_total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
