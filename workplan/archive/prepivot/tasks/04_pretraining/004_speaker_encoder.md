# pretraining/004 — Speaker encoder pretrain

**Phase:** pretraining    **Status:** pending    **Effort:** M
**User-input:** none
**Blocked by:** pretraining/002, pretraining/003, pretraining/005
**Blocks:** P3 close
**How to dispatch:** fresh Claude Code session

---

## Context

`../../../ARCHITECTURE.md` §2.5. Module 5 = speaker inference &
adaptation. From 30 s reference audio → speaker_embed (initial),
anatomy_vec (initial), source_defaults (initial), channel_model
(initial). The TEA inverse-refinement step (already implemented
in inversion/001) is the slow-but-accurate refinement on top.

**Channel model (Module 5b)** is non-negotiable: without it, the
anatomy fit absorbs mic/room/codec coloration and bakes it into
the cloned voice. Smallness prior + ablation gate.

## Deliverable

1. Encoder architecture: 5–15 M params. Audio → shared transformer
   trunk → 4 heads emitting anatomy_init, source_defaults_init,
   motor_style_embed, channel_model_init. (Shared trunk lets the
   speaker representation be learned once; per-target heads keep
   each output's loss term independent.)
2. Training: multi-speaker corpus (LibriTTS-R, VCTK).
   - Reconstruction-through-surrogate (5A) loss on full utterances
     (not TEA — too expensive at this stage). Weight: 0.7.
   - Anchor: short-window TEA reconstruction loss for
     anti-drift. Weight: 0.3.
   - Identity loss: pretrained ECAPA-TDNN cosine similarity
     between encoder's reconstructed audio and reference audio.
   - default-speaker-anchored anatomy prior with sigma per scalar.
   - Channel-model **smallness prior**: penalize channel
     magnitude when held-out reconstruction doesn't suffer.
3. Per-speaker output object: ≤ 50 KB total
   (anatomy + source_defaults + motor_style_embed +
   channel_model). Serializable as `.npz`.
4. Channel-model bypass: at synthesis time, channel is bypassed
   unless user explicitly requests "render in original recording
   conditions."

## Acceptance gates (automated)

- ✅ Held-out reconstruction: surrogate-stage log-mel L1 ≤
  **1.5 dB** on ≥ 90% of held-out test set, AND short-window TEA
  anchor log-mel L1 ≤ **1.0 dB** on the same set. (Total budget
  accounts separately for the surrogate's own ≤1.0 dB target from
  pretraining/005, leaving ≥0.5 dB headroom for the encoder.)
- ✅ Identity loss: ECAPA-TDNN cosine ≥ 0.80 between
  reconstructed-from-encoder audio and reference, averaged across
  test speakers.
- ✅ **Channel-model identifiability ablation** (CI guardrail
  from infra/002): zeroing channel must still produce voice that
  clusters as same speaker (cosine ≥ 0.85 of with-channel result —
  preliminary; tighten in cloning/* once cross-speaker eval lands
  and a calibrated threshold is in hand).
- ✅ Per-speaker output size ≤ 50 KB.
- ✅ Encoder size ≤ 15 M params.

## Anti-deliverable / scope guard

- Do NOT make the channel model bigger to chase reconstruction
  quality. The smallness prior is load-bearing for identity
  preservation.
- Do NOT skip the ablation gate — channel absorbing speaker
  identity is the most insidious failure mode and the literal
  reason this module exists.
- Do NOT train the encoder on full-utterance TEA backprop. G4
  out-of-scope. Surrogate (5A) is the throughput path; TEA
  short-window provides anchors.

## Notes

The user-fittable per-speaker object is the deliverable users
will see ("here's my voice as a 50 KB file"). Worth investing in a
clean serialization format.
