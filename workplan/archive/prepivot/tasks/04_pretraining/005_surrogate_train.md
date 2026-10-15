# pretraining/005 — Train surrogate (Module 5A)

**Phase:** pretraining    **Status:** pending    **Effort:** L
**User-input:** review (architecture choice: HiFi-GAN-class vs neural-ODE)
**Blocked by:** pretraining/002 (need a stable body model to distill from)
**Blocks:** pretraining/004 (speaker encoder reconstruction loss).
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §2.6 (Module 5A — required at training,
banned at inference). The training surrogate is necessary because
backprop through TEA at full utterance length on 8 GB is not
credible (G4). The surrogate distills TEA's body-state→audio
mapping at lower fidelity but full-utterance throughput, enabling
controller and encoder training to flow through.

**CI guardrail (infra/002):** the surrogate must NEVER be importable
from inference paths. Lint enforces.

## Architecture choice (user review)

Two candidates, decision deferred until this task starts:
- **HiFi-GAN-class** waveform decoder conditioned on body-state.
  Fast, well-understood, lossy on physical structure. Likely
  ~5 M params.
- **Neural ODE / state-space surrogate** mimicking TEA's per-sample
  recursion at lower precision. Slower, more faithful. Likely
  ~10 M params.

DDSP-articulatory's 0.4 M-param result (Cho et al. 2024) suggests
even smaller is achievable. ~5 M is a defensible middle.

## Deliverable

1. Architecture decision (user review or worker proposes; cite
   tradeoff).
2. Distillation corpus: ~100 K utterances of (anatomy_vec,
   gesture_traj, body_state) → TEA-rendered audio, spanning
   N anatomies × M gesture trajectories with K utterances each
   (target N ≈ 50 anatomies sampled across the v1 basis space, M ≈
   2000 trajectories per anatomy = 100K total). The surrogate must
   accept anatomy_vec as a conditioning input (not just body_state)
   so it generalizes across speakers at training time. Generate
   offline; this is a one-time data-gen step (~hours wall-clock).
3. Train: 0.5 · log-mel + 0.5 · MR-STFT loss vs TEA outputs;
   held-out test split (held-out by anatomy AND by trajectory).
4. Eval:
   - Held-out log-mel L1 to TEA.
   - Per-axis source/tract separation between surrogate and TEA
     outputs.
   - Identity preservation: ECAPA-TDNN cosine between surrogate
     and TEA on synthetic-speaker outputs.
   - Forward speed: surrogate vs TEA on consumer GPU.
5. Inference-path lint test: explicit pytest that fails if a file
   under `inference/` (or equivalently, the user-facing CLI
   pipelines) imports the surrogate.

## Acceptance gates (automated)

- ✅ Held-out log-mel L1 ≤ **1.0 dB** to TEA on 90% of test
  utterances.
- ✅ Per-axis source/tract separation: surrogate-vs-TEA
  differences within `runbooks/source_tract_sep.py` thresholds.
- ✅ Identity preservation: cosine ≥ 0.90 between
  surrogate-rendered and TEA-rendered same-speaker output.
- ✅ Forward speed: ≥ **5×** TEA on the same GPU at the same
  utterance length.
- ✅ Inference-path lint passes.
- ✅ Model size ≤ 10 M params.
- ✅ Anatomy-generalization ablation: surrogate evaluated on a
  held-out anatomy (not seen during training) produces TEA-faithful
  audio (log-mel L1 ≤ 1.5 dB; degradation budget vs same-anatomy is
  +0.5 dB). Catches overfitting to seen anatomies.

## Anti-deliverable / scope guard

- Do NOT use the surrogate at inference under any circumstance.
  CI guardrail will fail; treat as a hard block.
- Do NOT train the surrogate to be a "better" renderer than TEA.
  It's a faster, lossy approximation. Its loss is **fidelity to
  TEA**, not waveform quality independent of TEA.
- Do NOT start training before pretraining/002 lands a stable
  body model. Training a surrogate against a moving body-model
  target is wasted effort.

## Notes

This is the bottleneck unblocking pretraining/003 and /004 at full
utterance. Effort is large (distillation corpus + training + eval +
lint), but the payoff is everything downstream.
