"""Task 009 — TEA-to-TEA discriminating experiment.

Falsify/confirm the "kinematic supervision is structurally required" (A4) claim
by inverting from acoustic loss to a *known* TEA ground-truth articulation — no
real-audio confounders.

Pipeline (per the spec, workplan/tasks/00_active/009_...md):

  1. GEN     Build 2 synthetic targets from known ground-truth tube trajectories,
             render through TEA @44.1 kHz. Save target WAV + ground-truth
             area-function / glottis NPZ.
               T1 vowel-heavy  : /a/->/i/->/u/ triangle, frication OFF.
               T2 consonant-heavy: vowel/constriction sequence, frication ON.
  2. INVERT  For each target, run copyfit inversion from random init, 3 seeds,
             --iters 300. copyfit writes per-run report.json + field_snapshot.npz.
  3. METRICS Two-axis recovery per (target, seed):
               acoustic    : final acoustic loss, MR-STFT match, plateaued flag.
               articulatory: log-area MAE over the free range (25..64),
                             tightest-constriction position error (sections),
                             tightest-constriction area error (cm^2).
  4. VERDICT Combine Gate-3 medians against the pre-registered thresholds
             (PROVEN / WEAK / FAILED) and write report.json.

CHECKPOINT / RESUME
  The 6 copyfit runs (2 targets x 3 seeds) dominate wall-clock (~30 GPU-h). This
  script is restartable: it tracks per-run status in
  ``task009_results/state.json`` and skips any run whose copyfit ``report.json``
  already exists. Progress (iteration is inside copyfit's own log; here we log
  run-level milestones, elapsed, and rolling metrics) is flushed to
  ``task009_results/state.json`` + ``task009_results/progress.log`` after every
  run and on a heartbeat. Re-invoking the same command resumes from the first
  incomplete run.

USAGE
  # full run (gen if needed, then invert all, then metrics+verdict):
  PYTHONPATH=tractus_controller:tractus_engine_advanced/src \\
    python scripts/tea_to_tea_discriminating.py run \\
      --results-dir task009_results --iters 300 --seeds 3

  # smoke (tiny): short audio + few iters, proves the loop end-to-end on GPU:
  ... run --smoke --results-dir task009_results/smoke --iters 20 --duration-s 0.3 --seeds 1
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "tractus_controller"))
sys.path.insert(0, str(_ROOT / "tractus_engine_advanced" / "src"))

import jax.numpy as jnp  # noqa: E402
import soundfile as sf  # noqa: E402

import tractus_engine_advanced  # noqa: F401,E402  (pins precision policy)
from tractus_engine_advanced.pipeline import synthesize_dynamic_tube  # noqa: E402
from tractus_engine_advanced.frication import Frication, PLACE_ALVEOLAR  # noqa: E402

# ---------------------------------------------------------------------------
# Constants — mirror copyfit_neural_field_phrase.py (A1-fixed, 44.1 kHz native).
# ---------------------------------------------------------------------------
SAMPLE_RATE = 44100
SAMPLES_PER_FRAME = 882
FRAME_RATE_HZ = 50
N_SECTIONS = 93
FREE_LO, FREE_HI = 25, 65
N_FREE = FREE_HI - FREE_LO           # 40

# Glottis 11-vector indices (glottis_geometric.py:31): F0, PR, XB, XT, CA, ...
G_F0, G_PR = 0, 1
PR_VOICED_DPA = 8000.0
PR_UNVOICED_DPA = 400.0              # low subglottal pressure during obstruents

COPYFIT = _ROOT / "scripts" / "copyfit_neural_field_phrase.py"
DEFAULT_TUBE_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "default_tube.npz"
ANCHORS_PATH = _ROOT / "tractus_controller" / "outputs" / "fixtures" / "story_1996_anchors.npz"

# Pre-registered verdict thresholds (spec:167-178).
V_PROVEN = dict(mae=0.3, pos=5, area=0.2)
V_FAILED = dict(mae=0.6, pos=10)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Journal:
    """Append-only progress log + atomic state.json writer (resume-safe)."""

    def __init__(self, results_dir: Path):
        self.dir = results_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.dir / "progress.log"
        self.state_path = self.dir / "state.json"

    def log(self, msg: str) -> None:
        line = f"[{_now()}] {msg}"
        print(line, flush=True)
        with self.log_path.open("a") as fh:
            fh.write(line + "\n")

    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {}

    def save_state(self, state: dict) -> None:
        state["updated_at"] = _now()
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2))
        tmp.replace(self.state_path)


def _load_tube() -> dict:
    d = np.load(DEFAULT_TUBE_PATH, allow_pickle=True)
    return {k: np.asarray(d[k], dtype=np.float32) for k in
            ("length_cm", "area_cm2", "wall_mass_cgs",
             "wall_resistance_cgs", "wall_stiffness_cgs")}


def _load_anchor(key: str) -> np.ndarray:
    s = np.load(ANCHORS_PATH, allow_pickle=False)
    keys = [str(k) for k in s["shape_keys"]]
    if key not in keys:
        raise ValueError(f"anchor '{key}' not in {keys}")
    return np.asarray(s["area_cm2_tea"][keys.index(key)], dtype=np.float32)


def _voiced_glottis(n_frames: int, f0: float = 120.0, pr: float = PR_VOICED_DPA) -> np.ndarray:
    g = np.array([f0, pr, 0.035, 0.040, 0.08, 1.22, 1.0, 0.05, 0.0, 25.0, -30.0],
                 dtype=np.float32)
    return np.tile(g[None, :], (n_frames, 1))


def _cosine_blend(a: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    w = 0.5 - 0.5 * np.cos(np.pi * alpha)          # smooth 0->1
    return (1.0 - w) * a + w * b


def _render(area_93: np.ndarray, glottis: np.ndarray, *,
            frication: bool,
            place_index: np.ndarray | None = None,
            voicing: np.ndarray | None = None) -> np.ndarray:
    """Render a (T,93) area trajectory through TEA -> audio (numpy)."""
    d = _load_tube()
    T = area_93.shape[0]
    L = jnp.asarray(np.tile(d["length_cm"][None, :], (T, 1)))
    A = jnp.asarray(area_93)
    Mw = jnp.asarray(np.tile(d["wall_mass_cgs"][None, :], (T, 1)))
    Bw = jnp.asarray(np.tile(d["wall_resistance_cgs"][None, :], (T, 1)))
    Kw = jnp.asarray(np.tile(d["wall_stiffness_cgs"][None, :], (T, 1)))
    g = jnp.asarray(glottis)
    if frication:
        import jax
        pi = (jnp.zeros((T, N_SECTIONS), dtype=jnp.int32)
              if place_index is None else jnp.asarray(place_index))
        vc = (jnp.zeros((T,), dtype=jnp.float32)
              if voicing is None else jnp.asarray(voicing))
        audio, *_ = synthesize_dynamic_tube(
            L, A, Mw, Bw, Kw, samples_per_frame=SAMPLES_PER_FRAME,
            glottis_traj=g, frame_chunk_size=10,
            frication=Frication.default(), place_index=pi, voicing=vc,
            noise_key=jax.random.key(0))
    else:
        audio, *_ = synthesize_dynamic_tube(
            L, A, Mw, Bw, Kw, samples_per_frame=SAMPLES_PER_FRAME,
            glottis_traj=g, frame_chunk_size=10)
    return np.asarray(audio)


def _rms_normalize(x: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    return (x * (target_rms / rms)).astype(np.float32)


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------
def build_target_T1(duration_s: float) -> tuple[np.ndarray, np.ndarray, dict]:
    """Vowel-heavy /a/->/i/->/u/ triangle, cosine-blended. Frication OFF."""
    n = int(round(duration_s * FRAME_RATE_HZ))
    d = _load_tube()
    shapes = [_load_anchor("A"), _load_anchor("i"), _load_anchor("u")]
    third = max(n // 3, 1)
    anchors = [third // 2, third + third // 2, 2 * third + third // 2]
    free = np.empty((n, N_FREE), dtype=np.float32)
    for t in range(n):
        if t <= anchors[0]:
            free[t] = shapes[0]
        elif t >= anchors[-1]:
            free[t] = shapes[-1]
        else:
            for i in range(len(anchors) - 1):
                if anchors[i] <= t <= anchors[i + 1]:
                    alpha = (t - anchors[i]) / max(anchors[i + 1] - anchors[i], 1)
                    free[t] = _cosine_blend(shapes[i], shapes[i + 1], alpha)
                    break
    area93 = np.tile(d["area_cm2"][None, :], (n, 1)).astype(np.float32)
    area93[:, FREE_LO:FREE_HI] = free
    glottis = _voiced_glottis(n, f0=120.0)
    meta = {"target": "T1", "n_frames": n, "frication": False,
            "description": "vowel /a/->/i/->/u/ triangle, cosine blend, F0=120"}
    return area93, glottis, meta


def build_target_T2(duration_s: float) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Consonant-heavy: vowel /a/ -> constriction@40 -> /i/ -> constriction@35
    -> /u/ -> constriction@30. Constrictions unvoiced (low PR). Frication ON.
    Returns (area93, glottis, meta, frication_inputs)."""
    n = int(round(duration_s * FRAME_RATE_HZ))
    d = _load_tube()
    a_A, a_i, a_u = _load_anchor("A"), _load_anchor("i"), _load_anchor("u")
    # 6 equal segments: V C V C V C.
    seg = max(n // 6, 1)
    # Constriction: tighten a small band around the given absolute section to a
    # tight area so the frication Re^2*area + dP gate fires. 0.12 cm^2 sits below
    # the turbulence threshold with high dP.
    CONSTR_AREA = 0.12
    constr_sections = {1: 40, 3: 35, 5: 30}   # segment_idx -> absolute section
    vowels = {0: a_A, 2: a_i, 4: a_u}
    free = np.empty((n, N_FREE), dtype=np.float32)
    glottis = _voiced_glottis(n, f0=120.0)
    place = np.zeros((n, N_SECTIONS), dtype=np.int32)
    voicing = np.ones((n,), dtype=np.float32)
    # base vowel per frame = nearest preceding vowel segment (schwa-blend at edges)
    for t in range(n):
        s_idx = min(t // seg, 5)
        if s_idx in vowels:
            free[t] = vowels[s_idx]
        else:
            # constriction segment — start from a neutral vowel and pinch.
            base = a_A if s_idx == 1 else (a_i if s_idx == 3 else a_u)
            f = base.copy()
            abs_sec = constr_sections[s_idx]
            lo = max(abs_sec - 1 - FREE_LO, 0)
            hi = min(abs_sec + 1 - FREE_LO + 1, N_FREE)
            f[lo:hi] = CONSTR_AREA
            free[t] = f
            # unvoiced: drop subglottal pressure so voicing weakens and the
            # obstruent frication source dominates.
            glottis[t, G_PR] = PR_UNVOICED_DPA
            voicing[t] = 0.0
            # tag the constricted band as a sibilant site; physics gate decides.
            place[t, FREE_LO:FREE_HI] = PLACE_ALVEOLAR
    area93 = np.tile(d["area_cm2"][None, :], (n, 1)).astype(np.float32)
    area93[:, FREE_LO:FREE_HI] = free
    # fricative event ground-truth timing (segment centers) for recall metric.
    fric_events = [{"segment": s, "section": constr_sections[s],
                    "t_center": int((s + 0.5) * seg)} for s in (1, 3, 5)]
    meta = {"target": "T2", "n_frames": n, "frication": True,
            "description": "V/C sequence, constrictions@40/35/30 unvoiced, F0=120",
            "constriction_area_cm2": CONSTR_AREA, "fricative_events": fric_events}
    fric_inputs = {"place_index": place, "voicing": voicing}
    return area93, glottis, meta, fric_inputs


def cmd_gen(args, jnl: Journal) -> dict:
    tdir = args.results_dir / "targets"
    tdir.mkdir(parents=True, exist_ok=True)
    out = {}
    # T1
    jnl.log("GEN T1 (vowel triangle)...")
    a1, g1, m1 = build_target_T1(args.duration_s)
    au1 = _rms_normalize(_render(a1, g1, frication=False))
    sf.write(str(tdir / "T1_target.wav"), au1, SAMPLE_RATE, subtype="PCM_16")
    np.savez(tdir / "T1_gt.npz", area_cm2=a1, glottis_traj=g1,
             log_area=np.log(np.maximum(a1, 1e-4)).astype(np.float32),
             **{k: np.asarray(v) for k, v in m1.items() if isinstance(v, (int, float, bool))})
    (tdir / "T1_meta.json").write_text(json.dumps(m1, indent=2, default=str))
    out["T1"] = {"wav": str(tdir / "T1_target.wav"), "gt": str(tdir / "T1_gt.npz"),
                 "n_frames": m1["n_frames"], "frication": False}
    jnl.log(f"  T1 wav rms={float(np.sqrt(np.mean(au1**2))):.4f} "
            f"finite={bool(np.all(np.isfinite(au1)))} n={au1.shape[0]}")
    # T2
    jnl.log("GEN T2 (consonant sequence)...")
    a2, g2, m2, fr2 = build_target_T2(args.duration_s)
    au2 = _rms_normalize(_render(a2, g2, frication=True,
                                 place_index=fr2["place_index"], voicing=fr2["voicing"]))
    sf.write(str(tdir / "T2_target.wav"), au2, SAMPLE_RATE, subtype="PCM_16")
    np.savez(tdir / "T2_gt.npz", area_cm2=a2, glottis_traj=g2,
             place_index=fr2["place_index"], voicing=fr2["voicing"],
             log_area=np.log(np.maximum(a2, 1e-4)).astype(np.float32))
    (tdir / "T2_meta.json").write_text(json.dumps(m2, indent=2, default=str))
    out["T2"] = {"wav": str(tdir / "T2_target.wav"), "gt": str(tdir / "T2_gt.npz"),
                 "n_frames": m2["n_frames"], "frication": True}
    jnl.log(f"  T2 wav rms={float(np.sqrt(np.mean(au2**2))):.4f} "
            f"finite={bool(np.all(np.isfinite(au2)))} n={au2.shape[0]}")
    return out


# ---------------------------------------------------------------------------
# Inversion (copyfit subprocess, resume-safe)
# ---------------------------------------------------------------------------
def _run_dir(results_dir: Path, target: str, seed: int) -> Path:
    return results_dir / "recovery" / f"{target}_seed{seed}"


def invert_one(args, jnl: Journal, target: str, seed: int,
               wav: str, frication: bool) -> dict:
    out_dir = _run_dir(args.results_dir, target, seed)
    report = out_dir / "report.json"
    if report.exists():
        jnl.log(f"SKIP {target} seed{seed} — report.json exists (resume).")
        return {"status": "done", "out_dir": str(out_dir), "resumed": True}
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(COPYFIT),
           "--audio", wav,
           "--transcript", "tea-to-tea-synthetic",
           "--out", str(out_dir),
           "--iters", str(args.iters),
           "--seeds", "1",
           "--seed-base", str(seed),
           "--diag-every", str(args.diag_every)]
    if frication:
        cmd.append("--frication")
    jnl.log(f"INVERT {target} seed{seed}: {' '.join(cmd)}")
    t0 = time.time()
    logf = out_dir / "copyfit.log"
    with logf.open("w") as fh:
        proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                              cwd=str(_ROOT), env=_child_env())
    dt = time.time() - t0
    ok = proc.returncode == 0 and report.exists()
    jnl.log(f"  {target} seed{seed} rc={proc.returncode} "
            f"elapsed={dt/60:.1f}min report={'ok' if report.exists() else 'MISSING'}")
    return {"status": "done" if ok else "failed", "out_dir": str(out_dir),
            "returncode": proc.returncode, "elapsed_s": dt}


def _child_env() -> dict:
    import os
    env = dict(os.environ)
    pp = f"{_ROOT / 'tractus_controller'}:{_ROOT / 'tractus_engine_advanced' / 'src'}"
    env["PYTHONPATH"] = pp + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Older CUDA GPUs + CUDA 13.2: the cuBLAS-LT autotuner fails with
    # "No supported configs found / DEVICE_TYPE_INVALID" on the small render-stage
    # matmuls (observed on the post-optimization render_jit). Disabling GEMM
    # autotuning forces the default cuBLAS path, which works. Also cap the BFC
    # allocator so the 8 GB card doesn't fragment on the compile-time probes.
    extra_xla = "--xla_gpu_autotune_level=0"
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " " + extra_xla).strip()
    env.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.9")
    return env


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _mr_stft_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cheap multi-resolution log-magnitude STFT L1 distance."""
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    total = 0.0
    for nfft in (1024, 2048, 4096):
        hop = nfft // 4
        def spec(x):
            frames = [x[i:i + nfft] for i in range(0, max(len(x) - nfft, 1), hop)]
            win = np.hanning(nfft)
            mags = [np.abs(np.fft.rfft(f * win)) for f in frames if len(f) == nfft]
            if not mags:
                return np.zeros((1, nfft // 2 + 1))
            return np.log1p(np.stack(mags))
        sa, sb = spec(a), spec(b)
        m = min(sa.shape[0], sb.shape[0])
        if m == 0:
            continue
        total += float(np.mean(np.abs(sa[:m] - sb[:m])))
    return total


def articulatory_metrics(gt_area: np.ndarray, rec_area: np.ndarray,
                         events: list | None) -> dict:
    """gt_area, rec_area: (T,93). Metrics restricted to the free range 25..64."""
    T = min(gt_area.shape[0], rec_area.shape[0])
    gt = gt_area[:T, FREE_LO:FREE_HI]
    rec = rec_area[:T, FREE_LO:FREE_HI]
    log_gt = np.log(np.maximum(gt, 1e-4))
    log_rec = np.log(np.maximum(rec, 1e-4))
    log_area_mae = float(np.mean(np.abs(log_gt - log_rec)))

    # tightest constriction per frame (min-area section index within free range).
    gt_pos = np.argmin(gt, axis=1)
    rec_pos = np.argmin(rec, axis=1)
    pos_err = float(np.median(np.abs(gt_pos - rec_pos)))
    gt_min = gt[np.arange(T), gt_pos]
    rec_min = rec[np.arange(T), rec_pos]
    max_area_err = float(np.median(np.abs(gt_min - rec_min)))

    # fricative-event position/timing recall (within 5 sections & 50 ms).
    event_recall = None
    if events:
        tol_frames = max(int(round(0.050 * FRAME_RATE_HZ)), 1)  # 50 ms
        hits = 0
        for ev in events:
            tc = ev["t_center"]
            sec = ev["section"] - FREE_LO
            lo = max(tc - tol_frames, 0)
            hi = min(tc + tol_frames + 1, T)
            # did the recovered field place a tight constriction near sec in-window?
            window_pos = rec_pos[lo:hi]
            if window_pos.size and np.min(np.abs(window_pos - sec)) <= 5:
                hits += 1
        event_recall = hits / len(events)

    return {"log_area_mae": log_area_mae,
            "position_error_sections": pos_err,
            "max_area_error_cm2": max_area_err,
            "fricative_event_recall": event_recall}


def cmd_metrics(args, jnl: Journal, targets: dict) -> dict:
    per_seed = {}
    for target, tinfo in targets.items():
        gt = np.load(tinfo["gt"])
        gt_area = np.asarray(gt["area_cm2"], dtype=np.float32)
        gt_wav, _ = sf.read(tinfo["wav"])
        meta = json.loads((args.results_dir / "targets" / f"{target}_meta.json").read_text())
        events = meta.get("fricative_events")
        rows = []
        for seed in range(args.seeds):
            out_dir = _run_dir(args.results_dir, target, seed)
            report_p = out_dir / "report.json"
            snap_p = out_dir / "field_snapshot.npz"
            if not (report_p.exists() and snap_p.exists()):
                jnl.log(f"  metrics: {target} seed{seed} missing artifacts, skipping")
                continue
            rep = json.loads(report_p.read_text())
            snap = np.load(snap_p)
            rec_area = np.asarray(snap["area_cm2"], dtype=np.float32)
            art = articulatory_metrics(gt_area, rec_area, events)
            recon_wav_p = out_dir / "recon.wav"
            stft = None
            if recon_wav_p.exists():
                rw, _ = sf.read(recon_wav_p)
                stft = _mr_stft_dist(np.asarray(gt_wav), np.asarray(rw))
            row = {"seed": seed,
                   "final_acoustic": rep.get("final_acoustic"),
                   "init_acoustic": rep.get("init_acoustic"),
                   "plateaued": rep.get("plateaued"),
                   "tail_descent_frac": rep.get("tail_descent_frac"),
                   "mr_stft_dist": stft,
                   **art}
            rows.append(row)
            jnl.log(f"  metrics {target} seed{seed}: "
                    f"mae={art['log_area_mae']:.3f} pos={art['position_error_sections']:.1f} "
                    f"area_err={art['max_area_error_cm2']:.3f} "
                    f"plateaued={rep.get('plateaued')}")
        per_seed[target] = rows
    return per_seed


def _median(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def verdict(per_seed: dict) -> dict:
    agg = {}
    for target, rows in per_seed.items():
        agg[target] = {
            "median_log_area_mae": _median([r["log_area_mae"] for r in rows]),
            "median_position_error": _median([r["position_error_sections"] for r in rows]),
            "median_max_area_error": _median([r["max_area_error_cm2"] for r in rows]),
            "n_seeds": len(rows),
            "all_plateaued": (all(r.get("plateaued") for r in rows) if rows else False),
        }

    def target_proven(a):
        return (a["median_log_area_mae"] is not None
                and a["median_log_area_mae"] < V_PROVEN["mae"]
                and a["median_position_error"] < V_PROVEN["pos"]
                and a["median_max_area_error"] < V_PROVEN["area"])

    def target_failed(a):
        return (a["median_log_area_mae"] is not None
                and a["median_log_area_mae"] > V_FAILED["mae"]
                and a["median_position_error"] > V_FAILED["pos"])

    proven = all(target_proven(a) for a in agg.values()) and len(agg) >= 1
    failed = all(target_failed(a) for a in agg.values()) and len(agg) >= 1
    proven_map = {t: target_proven(a) for t, a in agg.items()}

    if proven:
        v = "PROVEN_ACOUSTIC_SUFFICIENCY"
        interp = "A4 falsified — acoustic-only inversion recovered known articulation. SPARC optional."
    elif failed:
        v = "FAILED_ACOUSTIC_RECOVERY"
        interp = ("A4 strengthened — but FAILED cannot distinguish fundamental "
                  "non-uniqueness from a stuck discrete-bin optimizer. Check seed "
                  "consistency (consistent=manifold, divergent=optimizer) before "
                  "committing to SPARC.")
    elif any(proven_map.values()) and not all(proven_map.values()):
        v = "WEAK_ACOUSTIC_RECOVERY"
        interp = "Mixed — some targets recover, some don't. A4 narrowed (e.g. SPARC for fricatives only)."
    else:
        v = "UNDETERMINED"
        interp = ("Results land in the pre-registration gap band (0.3-0.6 MAE) or "
                  "artifacts incomplete. No verdict earned.")

    warn = [t for t, a in agg.items() if not a["all_plateaued"]]
    return {"verdict": v, "interpretation": interp, "per_target": agg,
            "plateau_warning_targets": warn}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def cmd_run(args) -> int:
    jnl = Journal(args.results_dir)
    state = jnl.load_state()
    jnl.log(f"=== Task 009 run start (smoke={args.smoke}) iters={args.iters} "
            f"seeds={args.seeds} duration_s={args.duration_s} ===")

    # 1. GEN (skip if targets already present).
    tdir = args.results_dir / "targets"
    if (tdir / "T1_target.wav").exists() and (tdir / "T2_target.wav").exists() \
            and "targets" in state:
        jnl.log("GEN: targets present, resuming.")
        targets = state["targets"]
    else:
        targets = cmd_gen(args, jnl)
        state["targets"] = targets
        jnl.save_state(state)

    # 2. INVERT (2 targets x N seeds, serial — GPU discipline).
    state.setdefault("runs", {})
    for target, tinfo in targets.items():
        for seed in range(args.seeds):
            key = f"{target}_seed{seed}"
            if state["runs"].get(key, {}).get("status") == "done" \
                    and (_run_dir(args.results_dir, target, seed) / "report.json").exists():
                jnl.log(f"RESUME: {key} already done.")
                continue
            res = invert_one(args, jnl, target, seed, tinfo["wav"], tinfo["frication"])
            state["runs"][key] = res
            jnl.save_state(state)     # heartbeat after every run

    # 3. METRICS + 4. VERDICT.
    jnl.log("METRICS...")
    per_seed = cmd_metrics(args, jnl, targets)
    state["per_seed_metrics"] = per_seed
    verd = verdict(per_seed)
    state["verdict"] = verd
    jnl.save_state(state)

    report = {"generated_at": _now(), "config": {
                  "iters": args.iters, "seeds": args.seeds,
                  "duration_s": args.duration_s, "smoke": args.smoke},
              "targets": targets, "per_seed_metrics": per_seed, **verd}
    (args.results_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    jnl.log(f"WROTE {args.results_dir / 'report.json'}")
    jnl.log(f"VERDICT: {verd['verdict']} — {verd['interpretation']}")
    jnl.log("=== Task 009 run complete ===")
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("run", "gen"):
        sp = sub.add_parser(name)
        sp.add_argument("--results-dir", type=Path, default=_ROOT / "task009_results")
        sp.add_argument("--iters", type=int, default=300)
        sp.add_argument("--seeds", type=int, default=3)
        sp.add_argument("--duration-s", type=float, default=2.4)
        sp.add_argument("--diag-every", type=int, default=25)
        sp.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "gen":
        jnl = Journal(args.results_dir)
        cmd_gen(args, jnl)
        return 0
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
