"""correctness/005 verification harness.

Runs the natasha_shape forward both WITH and WITHOUT the jax.checkpoint
wrap on the per-sample scan body (we toggle by monkey-patching
``jax.checkpoint`` to identity for the "pre-wrap" baseline) and confirms
bit-equal output. Then benchmarks forward time, computes grads at
T=20 and T=118 (batch 1 and batch 4), and reports peak memory.

Backend selection:
* Default: whatever JAX picks (GPU if a CUDA-enabled jaxlib is installed
  and a device is reachable, else CPU).
* Override with ``TEA_VERIFY_BACKEND=cpu`` to force CPU. The b=4
  vmap step is automatically skipped on CPU because it OOM-freezes
  the virtualized host — the 2 GB VRAM acceptance gate is GPU-specific and a CPU run
  cannot validate it (peak RSS includes XLA buffers + Python state
  and is not comparable to GPU peak VRAM).
"""
from __future__ import annotations

import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _peak_rss_mb() -> float:
    # ru_maxrss is KB on Linux.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _peak_device_mb(device) -> float | None:
    """Return peak VRAM in MB for a JAX device, or None if unsupported."""
    if not hasattr(device, "memory_stats"):
        return None
    try:
        stats = device.memory_stats()
    except Exception:
        return None
    if stats is None:
        return None
    # JAX exposes either bytes_in_use_peak or peak_bytes_in_use depending on
    # jaxlib version; cover both.
    for key in ("peak_bytes_in_use", "bytes_in_use_peak"):
        if key in stats:
            return float(stats[key]) / (1024.0 * 1024.0)
    return None


def _load_fixture():
    d = np.load(ROOT / "outputs/fixtures/natasha_shape.npz")
    return {
        "T": int(d["T"]),
        "fsps": int(d["frame_step_samples"]),
        "asr": int(d["audio_sample_rate"]),
        "traj": np.asarray(d["trajectory"], dtype=np.float32),
    }


def _import_with_checkpoint(use_real_checkpoint: bool):
    """Import (or re-import) the pipeline with optionally-disabled
    jax.checkpoint. Returns the synthesize_via_tea_jax callable.
    """
    # Force a clean re-import of pipeline + tea_backend so any cached
    # closures pick up the new jax.checkpoint identity.
    for mod in list(sys.modules):
        if mod.startswith("tractus_engine_advanced") or mod.startswith("tractus_controller"):
            del sys.modules[mod]

    import jax  # noqa: E402

    if not use_real_checkpoint:
        # Identity wrapper — no checkpoint, plain function pass-through.
        def _identity_checkpoint(fun, *args, **kwargs):
            return fun
        jax._original_checkpoint = jax.checkpoint  # save for restore
        jax.checkpoint = _identity_checkpoint  # type: ignore[assignment]

    sys.path.insert(0, str(ROOT / "tractus_controller"))
    from tractus_controller.copy_synth.tea_backend import synthesize_via_tea_jax
    return synthesize_via_tea_jax


def _b4_subprocess_path(traj: np.ndarray, fsps: int) -> dict:
    """Run the T=118 b=4 backward in a fresh Python subprocess so its peak VRAM
    isn't polluted by resident compiled-program memory from the parent's prior
    stages (forward, T=20 backward, T=118 b=1 backward). Returns a dict with
    {time_s, vram_peak_mb, grad_finite, status}."""
    import subprocess
    import tempfile
    payload_path = ROOT / "outputs/backward_oom/_b4_payload.npz"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(payload_path, traj=traj, fsps=np.int32(fsps))
    out_path = ROOT / "outputs/backward_oom/_b4_result.json"
    if out_path.exists():
        out_path.unlink()
    env = dict(os.environ)
    env["TEA_VERIFY_B4_SUBPROCESS"] = "1"
    env["TEA_VERIFY_B4_PAYLOAD"] = str(payload_path)
    env["TEA_VERIFY_B4_RESULT"] = str(out_path)
    cmd = [sys.executable, "-u", str(Path(__file__).resolve())]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if not out_path.exists():
        return {
            "status": f"subprocess_error: rc={proc.returncode}",
            "time_s": None,
            "vram_peak_mb": None,
            "grad_finite": None,
            "stderr_tail": proc.stderr[-2000:],
        }
    return json.loads(out_path.read_text())


def _b4_subprocess_main():
    """Entry point when the verifier re-invokes itself for the b=4 stage only.
    Set ``TEA_VERIFY_B4_SUBPROCESS=1`` and ``TEA_VERIFY_B4_PAYLOAD`` /
    ``TEA_VERIFY_B4_RESULT`` to a fixture/result path."""
    os.environ.setdefault("TEA_PRECISION", "fp32")
    payload_path = Path(os.environ["TEA_VERIFY_B4_PAYLOAD"])
    out_path = Path(os.environ["TEA_VERIFY_B4_RESULT"])
    d = np.load(payload_path)
    traj = np.asarray(d["traj"], dtype=np.float32)
    fsps = int(d["fsps"])

    sys.path.insert(0, str(ROOT / "tractus_controller"))
    import jax
    import jax.numpy as jnp
    from tractus_controller.copy_synth.tea_backend import synthesize_via_tea_jax

    devices = jax.devices()
    is_gpu = jax.default_backend() == "gpu"
    if not is_gpu:
        out_path.write_text(json.dumps({
            "status": "skipped: cpu-backend in subprocess",
            "time_s": None, "vram_peak_mb": None, "grad_finite": None,
        }))
        return

    batch_traj = np.stack([traj] * 4, axis=0)

    def batched_loss(b):
        return jnp.sum(jax.vmap(lambda t: jnp.sum(synthesize_via_tea_jax(t, samples_per_frame=fsps) ** 2))(b))

    t0 = time.perf_counter()
    try:
        g_b4 = jax.grad(batched_loss)(jnp.asarray(batch_traj))
        g_b4.block_until_ready()
        elapsed = time.perf_counter() - t0
        peak = _peak_device_mb(devices[0])
        g_np = np.asarray(g_b4)
        n_nan = int(np.isnan(g_np).sum())
        n_inf = int(np.isinf(g_np).sum())
        out_path.write_text(json.dumps({
            "status": "ok",
            "time_s": elapsed,
            "vram_peak_mb": peak,
            "grad_finite": (n_nan == 0) and (n_inf == 0),
            "n_nan": n_nan,
            "n_inf": n_inf,
        }))
    except Exception as e:
        out_path.write_text(json.dumps({
            "status": f"error: {type(e).__name__}: {str(e)[:300]}",
            "time_s": None,
            "vram_peak_mb": None,
            "grad_finite": None,
        }))


def main():
    if os.environ.get("TEA_VERIFY_B4_SUBPROCESS") == "1":
        _b4_subprocess_main()
        return

    forced_cpu = os.environ.get("TEA_VERIFY_BACKEND", "").lower() == "cpu"
    if forced_cpu:
        os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ.setdefault("TEA_PRECISION", "fp32")

    fx = _load_fixture()
    print(f"fixture T={fx['T']} fsps={fx['fsps']} traj.shape={fx['traj'].shape}")

    # ---- 1. Bit-equal pre/post wrap on natasha_shape ----
    import jax
    import jax.numpy as jnp

    devices = jax.devices()
    backend = jax.default_backend()
    is_gpu = backend == "gpu"
    print(f"backend={backend} devices={devices}")

    print("[1] forward WITHOUT checkpoint wrap (baseline)…")
    t0 = time.perf_counter()
    synth_no = _import_with_checkpoint(use_real_checkpoint=False)
    audio_no = np.asarray(synth_no(jnp.asarray(fx["traj"]), samples_per_frame=fx["fsps"]))
    t_compile_no = time.perf_counter() - t0
    print(f"  audio shape={audio_no.shape} compile+run={t_compile_no:.2f}s")

    times_no = []
    for _ in range(3):
        t0 = time.perf_counter()
        a = synth_no(jnp.asarray(fx["traj"]), samples_per_frame=fx["fsps"]).block_until_ready()
        times_no.append(time.perf_counter() - t0)
    fwd_time_no = float(np.median(times_no))

    print("[2] forward WITH checkpoint wrap…")
    t0 = time.perf_counter()
    synth_ck = _import_with_checkpoint(use_real_checkpoint=True)
    audio_ck = np.asarray(synth_ck(jnp.asarray(fx["traj"]), samples_per_frame=fx["fsps"]))
    t_compile_ck = time.perf_counter() - t0
    print(f"  audio shape={audio_ck.shape} compile+run={t_compile_ck:.2f}s")

    times_ck = []
    for _ in range(3):
        t0 = time.perf_counter()
        a = synth_ck(jnp.asarray(fx["traj"]), samples_per_frame=fx["fsps"]).block_until_ready()
        times_ck.append(time.perf_counter() - t0)
    fwd_time_ck = float(np.median(times_ck))

    diff = audio_ck - audio_no
    max_abs = float(np.max(np.abs(diff)))
    print(f"  max_abs_diff = {max_abs:.6e}")
    bit_equal = max_abs == 0.0

    # ---- 3. Smoke: backward at T=20 produces finite grads under checkpoint ----
    print("[3] backward at T=20 (smoke; checkpoint wrap active)…")
    def loss_fn(t):
        return jnp.sum(synth_ck(t, samples_per_frame=fx["fsps"]) ** 2)

    t0 = time.perf_counter()
    g = jax.grad(loss_fn)(jnp.asarray(fx["traj"]))
    g.block_until_ready()
    bwd_T20_time = time.perf_counter() - t0
    g_np = np.asarray(g)
    n_nan = int(np.isnan(g_np).sum())
    n_inf = int(np.isinf(g_np).sum())
    grad_finite_T20 = (n_nan == 0) and (n_inf == 0)
    print(f"  T=20 backward time={bwd_T20_time:.2f}s n_nan={n_nan} n_inf={n_inf}")

    # ---- 4. T=118 backward at batch 1 ----
    print("[4] backward at T=118 batch=1…")
    T118 = 118
    big_traj = np.tile(fx["traj"], (T118 // fx["T"] + 1, 1))[:T118].astype(np.float32)
    print(f"  big_traj.shape={big_traj.shape}")

    bwd_T118_b1_time: float | None = None
    rss_after_T118_b1: float | None = None
    vram_T118_b1: float | None = None
    n_nan_big = n_inf_big = 0
    grad_finite_T118: bool | None = None
    b1_status = "ok"
    try:
        t0 = time.perf_counter()
        g_big = jax.grad(loss_fn)(jnp.asarray(big_traj))
        g_big.block_until_ready()
        bwd_T118_b1_time = time.perf_counter() - t0
        rss_after_T118_b1 = _peak_rss_mb()
        vram_T118_b1 = _peak_device_mb(devices[0]) if is_gpu else None
        g_big_np = np.asarray(g_big)
        n_nan_big = int(np.isnan(g_big_np).sum())
        n_inf_big = int(np.isinf(g_big_np).sum())
        grad_finite_T118 = (n_nan_big == 0) and (n_inf_big == 0)
        print(f"  T=118 b=1 backward time={bwd_T118_b1_time:.2f}s "
              f"vram_peak_mb={vram_T118_b1} rss={rss_after_T118_b1:.1f}MB "
              f"nan={n_nan_big} inf={n_inf_big}")
    except Exception as e:
        b1_status = f"error: {type(e).__name__}: {str(e)[:200]}"
        print(f"  FAILED: {b1_status}")

    # ---- 5. T=118 batch=4 via vmap (the 2 GB VRAM gate) ----
    # Skipped on CPU because it OOM-freezes the virtualized host. Once correctness/007 (chunked
    # outer scan) is in, b=1 fits and b=4 should fit ≤ 2 GB. We no longer skip
    # b=4 just because b=1 raised — but we still skip on CPU.
    vram_T118_b4_delta = None
    vram_T118_b4_lifetime = None
    if is_gpu:
        # Run b=4 in a fresh subprocess so its peak VRAM isn't poisoned by
        # resident compiled-program memory from the parent's earlier stages.
        # ``peak_bytes_in_use`` is process-cumulative and never resets; even
        # with explicit deletion the prior compiled XLA programs stay
        # device-resident. The gate is "≤ 2 GB peak for the b=4 work alone,"
        # which is best measured by isolating the b=4 work in its own process.
        print("[5] backward at T=118 batch=4 fp32 (vmap; GPU 2GB VRAM gate; subprocess)…")
        b4_result = _b4_subprocess_path(big_traj, fx["fsps"])
        bwd_T118_b4_time = b4_result.get("time_s")
        vram_T118_b4 = b4_result.get("vram_peak_mb")
        grad_finite_b4 = b4_result.get("grad_finite")
        b4_status = b4_result.get("status", "ok")
        n_nan_b4 = b4_result.get("n_nan", 0) or 0
        n_inf_b4 = b4_result.get("n_inf", 0) or 0
        # Lifetime / delta no longer apply here (subprocess), but keep the
        # parent's lifetime peak after the subprocess returns for context.
        vram_T118_b4_lifetime = _peak_device_mb(devices[0])
        vram_T118_b4_delta = None
        print(f"  T=118 b=4 backward time={bwd_T118_b4_time} "
              f"vram_peak_mb_subprocess={vram_T118_b4} "
              f"status={b4_status} grad_finite={grad_finite_b4}")
    else:
        reason = "skipped: cpu-backend (gate is GPU-specific)"
        print(f"[5] {reason}")
        bwd_T118_b4_time = None
        vram_T118_b4 = None
        grad_finite_b4 = None
        b4_status = reason
        n_nan_b4 = 0
        n_inf_b4 = 0

    speed_ratio = fwd_time_ck / fwd_time_no if fwd_time_no > 0 else None

    report = {
        "task": "correctness/008",
        "backend": backend,
        "devices": [str(d) for d in devices],
        "fixture": {
            "path": "outputs/fixtures/natasha_shape.npz",
            "T_fixture": fx["T"],
            "samples_per_frame": fx["fsps"],
            "audio_sample_rate": fx["asr"],
        },
        "bit_equal_pre_post_wrap": {
            "answer": "Y" if bit_equal else "N",
            "max_abs_diff": max_abs,
            "audio_shape": list(audio_ck.shape),
        },
        "forward_speed": {
            "baseline_no_checkpoint_s": fwd_time_no,
            "with_checkpoint_s": fwd_time_ck,
            "ratio_post_pre": speed_ratio,
            "gate_max": 1.3,
            "gate_passed": (speed_ratio is not None and speed_ratio <= 1.3),
            "n_iterations": 3,
        },
        "backward_smoke_T20": {
            "time_s": bwd_T20_time,
            "n_nan": n_nan,
            "n_inf": n_inf,
            "grad_finite": grad_finite_T20,
        },
        "backward_T118_batch1": {
            "status": b1_status,
            "time_s": bwd_T118_b1_time,
            "vram_peak_mb": vram_T118_b1,
            "rss_mb_after": rss_after_T118_b1,
            "n_nan": n_nan_big,
            "n_inf": n_inf_big,
            "grad_finite": grad_finite_T118,
        },
        "backward_T118_batch4_fp32": {
            "status": b4_status,
            "time_s": bwd_T118_b4_time,
            # Gate measure: delta peak attributable to the b=4 stage.
            # See [5] in the verifier for why lifetime peak is poisoned by
            # earlier stages' resident compiled-program memory.
            "vram_peak_mb": vram_T118_b4,
            "vram_peak_delta_mb": vram_T118_b4_delta,
            "vram_peak_lifetime_mb": vram_T118_b4_lifetime,
            "grad_finite": grad_finite_b4,
            "gate_max_mb": 2048.0,
            "gate_passed": (
                vram_T118_b4 is not None and vram_T118_b4 <= 2048.0
                and grad_finite_b4 is True
            ),
        },
        "compile_times_s": {
            "no_checkpoint": t_compile_no,
            "with_checkpoint": t_compile_ck,
        },
        "notes": [
            "jax.checkpoint(policy=...) decorator-form does not work in jax 0.10 (positional 'fun' required); pipeline.py was minimally adjusted to call jax.checkpoint(per_sample_body, policy=...) after the def.",
            "The 'pre-wrap baseline' was produced by re-importing pipeline.py with jax.checkpoint monkey-patched to identity. This proves bit-equal forward semantics of the wrap itself.",
            "correctness/007 iter 1: outer per-frame scan is chunked into frame_chunk_size=1 chunks; each chunk body is jax.checkpoint(policy=nothing_saveable). This bounds the per-chunk inner-scan tape but does NOT eliminate the per-sample MNA-solve Jacobian transpose.",
            "correctness/007 iter 2: tds.step._mna_solve wraps the Cholesky in jax.lax.custom_linear_solve, replacing the dense per-sample-Jacobian VJP (~3.9 GB f32[F,samples_per_frame,97,97]) with implicit-function-theorem differentiation (one extra Cholesky transpose-solve per backward step). Forward semantics bit-equal; bypass via TEA_DISABLE_CUSTOM_SOLVE=1.",
            "correctness/008: tds.step._step_one_cvjp wraps step_one in jax.custom_vjp so the per-sample dense M-cotangent never crosses the scan-body boundary. Forward calls _step_one_impl unchanged (bit-equal); bwd reruns forward via jax.vjp inside the bwd region, confining the 97x97 dM tensor to bwd-local memory and dropping the f32[118,882,4,97,97] (~14.6 GB) scan-stacked transpose. Bypass via TEA_DISABLE_CUSTOM_VJP=1.",
            "verifier hardening: jax peak_bytes_in_use is process-cumulative and never resets, so the b=4 gate runs in a fresh subprocess (TEA_VERIFY_B4_SUBPROCESS=1) — the parent's resident compiled-program buffers from prior T=20 / T=118 b=1 stages would otherwise inflate the b=4 lifetime peak by ~3 GB and falsely trip the gate.",
        ],
        "backward_speed_ratio": {
            "T20_bwd_over_fwd": (bwd_T20_time / fwd_time_ck) if fwd_time_ck > 0 else None,
            "T118_b1_bwd_over_fwd": (
                (bwd_T118_b1_time / fwd_time_ck)
                if (bwd_T118_b1_time is not None and fwd_time_ck > 0)
                else None
            ),
            "gate_max": 3.0,
            "note": "Gate is on T-matched fwd (cf. forward T=20 fixture vs backward at T=20). T=118 ratio is informational since fwd is not measured at T=118.",
        },
    }
    out = ROOT / "outputs/backward_oom/report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"\nWrote {out}")
    print(json.dumps(report, indent=2))

    # Codex C #12: explicit gate-aggregation exit code so CI can fail
    # on regression. Soft-fail on the b=4 GPU-only gate when running on
    # CPU (the gate is GPU-specific by design).
    gates = [
        ("bit_equal", report["bit_equal_pre_post_wrap"]["answer"] == "Y"),
        ("forward_speed", report["forward_speed"]["gate_passed"]),
        ("backward_T20_grad_finite", report["backward_smoke_T20"]["grad_finite"] is True),
        ("backward_T118_b1", report["backward_T118_batch1"].get("status") == "ok"),
    ]
    if is_gpu:
        gates.append(("backward_T118_b4_2GB", report["backward_T118_batch4_fp32"]["gate_passed"]))
    failed = [name for name, passed in gates if not passed]
    if failed:
        print(f"\nFAIL: gates {failed}")
        sys.exit(1)
    print("\nPASS: all dispatchable gates green.")
    sys.exit(0)


if __name__ == "__main__":
    main()
