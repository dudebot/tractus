"""Inverse-fit demo: gradient-descend tube area function until rendered
spectrum matches a literature-formant target.

Per ``workplan/tasks/00_infra/006_target_sound_inverse_fit.md`` and the
user's "gradient descent for target sounds" framing. Settles whether
the simulator can produce strongly-differentiated vowel formants AT ALL,
independent of upstream layers (planner, controller, body model).

For each of /a/, /i/, /u/:
1. Build a target log-PSD with bell-curve peaks at literature F1/F2/F3.
2. Initialize log-area-cm2 from the default speaker tube (so step 0 is
   the baseline tube — same as ``scripts/phoneme_test.py:shape_*``).
3. Free parameter: log-area on supraglottal sections 25-64 (40 scalars).
   Trachea + glottis + walls + nasal frozen at default-speaker baseline.
4. Glottis trajectory frozen at default voiced (F0=120, PR=8000).
5. Render T=20 frames (~200 ms; well within correctness/008's b=1
   backward budget at ~few hundred MB).
6. Loss: ``spectral_template_loss_jnp`` against target.
7. Adam, ~150 steps, lr ~5e-2 on log-area.
8. Save before / after WAVs + JSON report.

Run::

    cd tractus_controller
    .venv/bin/python ../scripts/inverse_fit_demo.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TEA_PRECISION", "fp32")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tractus_controller"))
sys.path.insert(0, str(ROOT / "tractus_engine_advanced" / "src"))

import jax
import jax.numpy as jnp
import optax
import soundfile as sf

from tractus_engine_advanced.acoustic_gates import (
    VOWEL_FORMANTS_MALE,
    formant_gate,
    make_target_log_psd,
    spectral_template_loss_jnp,
)
from tractus_engine_advanced.pipeline import synthesize_dynamic_tube
from tractus_engine_advanced.tds.tube_topology import (
    FIRST_PHARYNX_SECTION,
    LAST_MOUTH_SECTION,
)


SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 441   # 10 ms / frame
F = 20                     # frames; 200 ms total render
LR = 5e-2
N_STEPS = 150
FFT_SIZE = 4096

OUT_DIR = ROOT / "outputs/inverse_fit_demo"

# Free supraglottal sections — pharynx + mouth, leaving trachea + glottis +
# nasal at default-speaker baseline.
FREE_LO = FIRST_PHARYNX_SECTION   # 25
FREE_HI = LAST_MOUTH_SECTION + 1  # 65 (exclusive)
N_FREE = FREE_HI - FREE_LO        # 40


# --------------------------------------------------------------------------
# Default speaker baseline.
# --------------------------------------------------------------------------

def load_default_speaker() -> dict:
    npz = np.load(ROOT / "tractus_controller/outputs/fixtures/default_tube.npz")
    return {
        "length_cm": np.asarray(npz["length_cm"], dtype=np.float32),
        "area_cm2": np.asarray(npz["area_cm2"], dtype=np.float32),
        "wall_mass_cgs": np.asarray(npz["wall_mass_cgs"], dtype=np.float32),
        "wall_resistance_cgs": np.asarray(npz["wall_resistance_cgs"], dtype=np.float32),
        "wall_stiffness_cgs": np.asarray(npz["wall_stiffness_cgs"], dtype=np.float32),
    }


def voiced_glottis_traj(F0_Hz: float = 120.0, PR_dPa: float = 8000.0) -> np.ndarray:
    g = np.array(
        [F0_Hz, PR_dPa, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
        dtype=np.float32,
    )
    return np.tile(g[None, :], (F, 1))


# --------------------------------------------------------------------------
# Render path: log_area_free (40,) -> audio
# --------------------------------------------------------------------------

def make_render_fn(default: dict, glottis_traj_np: np.ndarray):
    """Closure: takes (40,) log-area parameters, returns audio (N,)."""
    L_default = jnp.asarray(default["length_cm"])
    A_default = jnp.asarray(default["area_cm2"])
    Mw_default = jnp.asarray(default["wall_mass_cgs"])
    Bw_default = jnp.asarray(default["wall_resistance_cgs"])
    Kw_default = jnp.asarray(default["wall_stiffness_cgs"])
    L_tiled = jnp.tile(L_default[None, :], (F, 1))
    Mw_tiled = jnp.tile(Mw_default[None, :], (F, 1))
    Bw_tiled = jnp.tile(Bw_default[None, :], (F, 1))
    Kw_tiled = jnp.tile(Kw_default[None, :], (F, 1))
    g_traj = jnp.asarray(glottis_traj_np)

    def render(log_area_free):
        # Build full (93,) area function: free sections come from
        # exp(log_area_free); rest at default-speaker baseline.
        area_free = jnp.exp(log_area_free)
        A_full = A_default.at[FREE_LO:FREE_HI].set(area_free)
        A_tiled = jnp.tile(A_full[None, :], (F, 1))
        audio, *_ = synthesize_dynamic_tube(
            L_tiled, A_tiled, Mw_tiled, Bw_tiled, Kw_tiled,
            samples_per_frame=SAMPLES_PER_FRAME,
            glottis_traj=g_traj,
        )
        return audio

    return render


def make_loss_fn(render_fn, target_log_psd: np.ndarray):
    target_jnp = jnp.asarray(target_log_psd, dtype=jnp.float32)

    def loss(log_area_free):
        audio = render_fn(log_area_free)
        return spectral_template_loss_jnp(
            audio, target_jnp, SAMPLE_RATE,
            fft_size=FFT_SIZE, f_min_Hz=100.0, f_max_Hz=11000.0,
        )

    return loss


# --------------------------------------------------------------------------
# Per-vowel demo.
# --------------------------------------------------------------------------

def fit_one_vowel(name: str, F1: float, F2: float, F3: float, default: dict) -> dict:
    print(f"\n=== fitting /{name}/ — target F1={F1}, F2={F2}, F3={F3} ===")

    target = make_target_log_psd(
        SAMPLE_RATE,
        formant_peaks_Hz=(F1, F2, F3),
        formant_amps_dB=(20.0, 16.0, 10.0),
        formant_bandwidths_Hz=(80.0, 100.0, 120.0),
        spectral_tilt_dB_per_oct=-12.0,
        fft_size=FFT_SIZE,
    )

    # Init: default-speaker area on free sections.
    log_area_init = jnp.log(jnp.asarray(default["area_cm2"][FREE_LO:FREE_HI]))
    glottis = voiced_glottis_traj()

    render_fn = make_render_fn(default, glottis)
    loss_fn = make_loss_fn(render_fn, target)
    # JIT-compile the value-and-grad. Without this, each call re-traces
    # (T=20 × 441 = 8820-step backward through TEA), making per-step
    # cost prohibitive.
    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    # Also JIT the forward-only render for the before/after audio.
    render_fn_jit = jax.jit(render_fn)

    optimizer = optax.adam(LR)
    opt_state = optimizer.init(log_area_init)
    log_area = log_area_init

    # Warmup compile before the timed loop. First call traces + compiles
    # both forward and backward; subsequent calls are fast.
    print("  warming up jit (forward)…", flush=True)
    t_warm = time.time()
    audio_before = np.asarray(render_fn_jit(log_area))
    print(f"    forward compile+run: {time.time() - t_warm:.1f}s", flush=True)
    print("  warming up jit (grad)…", flush=True)
    t_warm = time.time()
    initial_loss, _ = grad_fn(log_area)
    initial_loss = float(initial_loss)
    print(f"    grad compile+run:    {time.time() - t_warm:.1f}s", flush=True)
    print(f"  initial loss: {initial_loss:.4f}", flush=True)

    losses = [initial_loss]
    t0 = time.time()
    for step in range(N_STEPS):
        loss_val, grad = grad_fn(log_area)
        updates, opt_state = optimizer.update(grad, opt_state, log_area)
        log_area = optax.apply_updates(log_area, updates)
        log_area = jnp.clip(log_area, jnp.log(0.05), jnp.log(8.0))
        loss_val_f = float(loss_val)
        losses.append(loss_val_f)
        if step % 10 == 0 or step == N_STEPS - 1:
            elapsed = time.time() - t0
            print(f"  step {step:3d}: loss={loss_val_f:.4f}  ({elapsed:.1f}s)", flush=True)

    final_loss = losses[-1]
    audio_after = np.asarray(render_fn_jit(log_area))

    # Convergence ratio.
    ratio = final_loss / max(initial_loss, 1e-12)
    converged = ratio <= 0.5
    print(f"  final loss: {final_loss:.4f}  (ratio {ratio:.3f}, "
          f"{'CONVERGED' if converged else 'STALLED'})")

    # Final formant measurement on rendered audio.
    fg = formant_gate(
        audio_after, SAMPLE_RATE,
        target_F1_Hz=F1, target_F2_Hz=F2, target_F3_Hz=F3,
    )
    print(f"  final formants: {fg.note}")

    # Save WAVs (peak-normalize for audibility).
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    def _save(path, audio):
        peak = float(np.max(np.abs(audio)))
        scale = (0.5 / peak) if peak > 0 else 1.0
        sf.write(str(path), audio * scale, SAMPLE_RATE)

    _save(OUT_DIR / f"{name}_before.wav", audio_before)
    _save(OUT_DIR / f"{name}_after.wav", audio_after)

    # Recovered area function.
    recovered_area = np.exp(np.asarray(log_area, dtype=np.float64))

    return {
        "vowel": name,
        "target": {"F1_Hz": F1, "F2_Hz": F2, "F3_Hz": F3},
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_ratio": ratio,
        "converged": bool(converged),
        "n_steps": N_STEPS,
        "lr": LR,
        "formant_gate_after": fg.to_dict(),
        "loss_history": losses,
        "recovered_area_cm2": recovered_area.tolist(),
        "free_sections": list(range(FREE_LO, FREE_HI)),
        "render_T_frames": F,
        "render_samples_per_frame": SAMPLES_PER_FRAME,
        "audio_before": str(OUT_DIR / f"{name}_before.wav"),
        "audio_after": str(OUT_DIR / f"{name}_after.wav"),
        "audio_initial_peak": float(np.max(np.abs(audio_before))),
        "audio_final_peak": float(np.max(np.abs(audio_after))),
    }


def main() -> int:
    default = load_default_speaker()
    print(f"Loaded default speaker tube. Free sections {FREE_LO}-{FREE_HI - 1} "
          f"({N_FREE} parameters).")
    print(f"Render: T={F} frames × {SAMPLES_PER_FRAME} samp/frame "
          f"= {F * SAMPLES_PER_FRAME} samples ({F * SAMPLES_PER_FRAME / SAMPLE_RATE:.2f} s)")
    print(f"Optimizer: Adam, lr={LR}, {N_STEPS} steps")

    results = {}
    for vowel in ("a", "i", "u"):
        F1, F2, F3 = VOWEL_FORMANTS_MALE[vowel]
        results[vowel] = fit_one_vowel(vowel, F1, F2, F3, default)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "report.json"
    report = {
        "results": results,
        "summary": {
            "n_converged": sum(1 for r in results.values() if r["converged"]),
            "n_total": len(results),
            "n_formant_gate_pass": sum(
                1 for r in results.values() if r["formant_gate_after"]["passed"]
            ),
        },
    }
    report_path.write_text(json.dumps(report, indent=2))

    print("\n=== summary ===")
    for v, r in results.items():
        status = "CONVERGED" if r["converged"] else "STALLED"
        gate = "PASS" if r["formant_gate_after"]["passed"] else "FAIL"
        print(f"  /{v}/: loss {r['initial_loss']:.3f} → {r['final_loss']:.3f} "
              f"({r['loss_ratio']:.2f}x) [{status}]  formant-gate [{gate}]")
    print(f"\nReport: {report_path.relative_to(ROOT)}")

    # Acceptance: 3/3 converged AND ≥ 2/3 formant-gate pass.
    n_conv = report["summary"]["n_converged"]
    n_gate = report["summary"]["n_formant_gate_pass"]
    if n_conv == 3 and n_gate >= 2:
        print("\nPASS: simulator can produce literature-formant shapes.")
        return 0
    print(f"\nFAIL: {n_conv}/3 converged, {n_gate}/3 formant-gate pass.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
