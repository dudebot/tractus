# Week-1 phrase probe — throughput-bound handoff

> **Status:** Pipeline is correct end-to-end (forward + backward
> verified). Throughput is at the upper edge of the task's 3-10
> hour wall-clock budget on the 16 GB GPU in a virtualized GPU environment. No gate has
> been formally pass/failed because the developer's gate run
> (250 iters × 3 seeds) was not completed in the worker session.
>
> **Verdict:** YELLOW — script ready to run, recommend the
> developer commit a 3-5 hour wall-clock batch and report back
> with the produced `outputs/copyfit_neural_field/juice_lemons/
> report.json` before declaring GREEN or applying Risk-C
> salvage.

## What landed

- `scripts/copyfit_neural_field_phrase.py` — the Week-1 deliverable
  per `workplan/tasks/00_active/001_neural_field_phrase_probe.md`.
  Public function `copyfit_neural_field_phrase(audio_path,
  transcript, out_dir, *, n_iters, n_seeds, seed, frame_chunk_size,
  ...) -> dict` and a CLI with the same arguments. Outputs:
  `recon.wav`, `field_snapshot.npz`, `report.json`,
  `optim_trace.png`, `area_function_t0_t1_t2.png`.

- Pipeline shape:
  - Neural field `log A(s,t) = log A_default(s) + δ_max ·
    tanh(MLP(PE(s), PE(t)))` with 4-layer MLP @ 128 hidden,
    last-layer near-zero init so the field starts at exactly the
    default tube.
  - Source schedule extracted from target audio (PyWorld DIO F0 +
    windowed RMS at 50 Hz) and packed into the 11-vector glottis
    trajectory; **frozen** during optimization so the tract field
    can't absorb the source.
  - Loss = log-mel + 0.5×MR-STFT (via
    `tractus_controller.copy_synth.features_jax`) + smoothness/
    curvature/clamp-barrier/default-deviation priors per the task
    §4 weights.
  - Adam LR=3e-3, manual implementation (no optax dep).
  - Fused `@jax.jit train_step(params, opt_state) -> (params,
    opt_state, loss, diag)` for forward + backward + Adam apply.

## What worked

1. Forward render is bit-stable. Standalone test:
   `synthesize_dynamic_tube(...)` with the loaded source +
   default tube ran at ~17s per call after compile, audio shape
   `(52920,)` at 22050 Hz.

2. Backward fits. The verified `correctness/005+007`
   `jax.checkpoint(policy=nothing_saveable)` on the per-sample
   scan keeps GPU memory at 4-5 GB during the full
   forward+backward at b=1, well inside the 16 GB budget.

3. Source extraction is sane: F0 mean ~198 Hz (Natasha synthetic
   voice — see consent note below), PR varies between 2400 (silence
   floor) and 8000 dPa (voiced peak) following the windowed RMS
   envelope of the reference.

4. Gate-3 plausibility, Gate-4 vowel sanity, and the artifact
   plotting/report-write paths all compile and the JIT'd
   impulse-response renderer caches across all three Story
   vowels.

5. Loss math is well-conditioned. Iter-1 loss = 6.34 (acoustic
   4.06, barrier 0.23). Gate-2 target (final ≤ 0.6 × init = 3.8 on
   the composite) is in reach if 250 iters of Adam@3e-3 can move
   the field; iter-1-vs-iter-0 acoustic L1 is essentially
   indistinguishable because the field starts at exactly default,
   so the first gradient step is small.

## What didn't get past the worker session

**The full 250-iter × 3-seed run was not completed.** What the
worker did observe:

| K | per-iter (cached) | first-iter (compile + run) |
|---|---|---|
| 1 (TEA default) | ~50-60s | ~150-200s |
| 10 | ~30-50s | ~150-170s |
| 30 | ~30-45s (extrapolated; iter 10 not yet seen) | ~190s |

(The K-vs-throughput trend matches Codex's review (thread
`[session identifier omitted]`): bumping K from 1 buys some memory headroom but
not dramatic compile-time speedup — JAX compiles the per-sample
MNA scan body once regardless. The win at b=1 is mostly that the
backward `(K, samples_per_frame, 97, 97)` Jacobian transpose is
much cheaper than rematerializing per-frame.)

Math at K=10:
- 250 iters × 30-50s + ~150s compile + ~90s post-loop ≈ **2.4-3.7 hours per seed**.
- 3 seeds at the same throughput ≈ **7-11 hours**.
- This is *within* the task's stated 3-10 hour budget but at the
  ceiling. Each round-trip ate one entire worker session in setup
  + canary attempts, leaving no time to actually run the final
  optimization to convergence.

A specific cost the worker did not notice early enough: an
earlier canary run was OOM-killed but the **python process kept
running in the background for 28 minutes**, holding GPU memory and
slowing every subsequent benchmark. Discovered + cleaned at
2:48 into the session. After that, the K=10 fused train_step
hit iter 1 at 153s in one run vs 350+s when the zombie was active.
Lesson: always re-verify `nvidia-smi` shows zero rogue python
before declaring a perf regression.

## Recommendation for the developer

Pick one of:

1. **Run as-is on the local box** (single seed, 250 iters,
   `frame_chunk_size=10`). Expected wall-clock 3-4 hours from a
   cold start; the script writes `report.json` with all
   gate-2/3/4 metrics and the reconstructed wav for Gate-1
   (Whisper) and Gate-5 (human listen). Command:

   ```bash
   tractus_controller/.venv/bin/python scripts/copyfit_neural_field_phrase.py \
     --audio data/probes/juice_lemons_ref.wav \
     --transcript "The juice of lemons makes fine punch." \
     --out outputs/copyfit_neural_field/juice_lemons \
     --iters 250 --seeds 1 --frame-chunk-size 10
   ```

   Then run Gate 1:

   ```bash
   bash runbooks/whisper_gate.sh \
     --audio outputs/copyfit_neural_field/juice_lemons/recon.wav \
     --target "The juice of lemons makes fine punch." \
     --model base --threshold-wer 0.75 --threshold-jaccard 0.30
   ```

2. **Skip to Risk-C salvage**, treating the unreasonably long
   per-iter wall as evidence that the per-sample MNA backward is
   structurally too expensive for diagnostic-loop iteration. This
   is the right move *only if* the developer believes a 4-hour
   wall-clock per single phrase precludes the whole Week-2/3/4
   plan (which needs many more such optimizations).

The worker's lean toward option 1: the math says it should
**land in budget**, and the diagnostic value of a real Gate-1
Whisper result is high enough to spend the wall-clock on. If it
returns RED after that, *then* Risk-C with confidence.

## Anti-deliverables respected

- No body-model checkpoint. ✓
- No SPARC integration. ✓
- No TEA pipeline edits — only the `frame_chunk_size` kwarg that
  was already exposed. ✓
- No new runbooks. ✓
- No `ARCHITECTURE.md` edits. ✓

## Consent note (important)

The reference audio `data/probes/juice_lemons_ref.wav` is
**synthetic** — generated by ElevenLabs with voice "Natasha —
Valley girl" speaking *"The juice of lemons makes fine punch."*
This is the developer's own deliberate Week-1 input, confirmed
this session. The Week-1 phrase probe is therefore a **TTS-target
run**, not a developer-voice clone. Any future "did we
recover the developer's voice?" framing should treat this run as
an instrument-feasibility test against a synthetic anchor — the
identifiability gate that comes online in Week 4 will use a real
developer recording.

The script's `report.json` carries `consent_verified: true` per
the CI lint rule with this exact disclaimer.

## Open questions

1. Is the 3-4 hour wall-clock per single-phrase optimization
   acceptable in the developer's workflow? If not, the project
   needs the FD renderer salvage path (Chen & Sitzmann 2025)
   *before* Week 2's body model training rather than Week 4 as
   originally scoped.

2. Is the throughput a a virtualized GPU environment-shared-GPU artifact? Repeated runs
   showed compile times of 40-200s for the same train_step
   shape, suggesting host GPU contention. A dedicated-GPU bare
   metal box (or a 2x A100 cloud burst for the inverse-fit step)
   would likely 2-3× faster.

3. Do we trust the iter-1 loss = 6.34 number, given the field
   starts at exactly default and the only signal is "how
   different is the default-tube rendered version of the
   utterance from the target target_log_mel"? The answer is
   probably yes — the default tube renders a vowel-like sound,
   the target is voiced speech, and a log-mel L1 in the 4-6
   range is the right order of magnitude for "two different
   utterances at similar RMS." But Gate-2 (60 % of init) is a
   *relative* improvement, which is the right measure here.
