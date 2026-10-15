# research/005 — Cross-species phoneme inference (birds + cetaceans)

**Phase:** research    **Status:** pending    **Effort:** L (multi-month)
**User-input:** **decision** — multi-month investment; only fires after cloning-MVP ships
**Blocks:** —
**Blocked by:** P4 cloning closed (need a working human voice-cloning MVP first)
**How to dispatch:** decision tasks under `06_research/` are user-driven; orchestrator waits

---

## Context

The TEA stack has three layers that generalize very differently:

1. **Wave-equation propagation** (`tractus_engine_advanced/tds/mna.py`,
   1-D transmission-line solver) — generalizes near-fully across
   human-tract, bird-trachea, cetacean-nasal-complex, and any
   tube-shaped resonator. Same physics; only geometry + boundary
   conditions change.
2. **Source model** (`glottis/geometric.py`) — human-specific. Each
   non-human producer needs its own source physics module.
3. **Body / articulator model** (R1 NN tract → tube) — entirely
   human-specific.

Once human cloning ships (P4 closed), the source-resonator stack is
mature enough to start swapping out (1) source modules and (2) body
models for non-human vocal tracts. This task explores that.

The further-out research goal is **phoneme-inventory inference**:
given an arbitrary (source × resonator) physics model, derive what
distinctions it can make at what articulatory effort cost, and from
that derive plausible phoneme repertoires. Combined with optimal-
coding theory (Huffman / MDL / Zipf-optimal vocabularies), this
becomes **vocabulary engineering from first principles** — the path
from a sound-producing apparatus to a designed language.

## Deliverable (staged)

### Stage 1 — Bird syrinx source module

1. New `tractus_engine_advanced.glottis.syrinx` module mirroring the
   `glottis.geometric` interface (`make_anatomy`, `init_state`,
   `inc_time`, `calc_geometry`). Implements the two-bronchi
   self-oscillating membrane model from the Mindlin / Goller
   literature (see references below).
2. Two simultaneous parallel sources sharing a trachea — the trachea
   uses the existing TDS solver; only the source coupling at the
   bronchial junction is new.
3. JAX-portable, fp32 default, custom_vjp where the solve granularity
   warrants (the syrinx oscillator is small; default autodiff likely
   suffices).
4. Anatomy module with bird-typical parameters (zebra finch, canary,
   parrot — pick one for v0).
5. Acceptance: produce audible bird whistles + warbles that match the
   spectral envelope and modulation patterns of recorded reference
   tokens within reasonable bounds.

### Stage 2 — Cetacean source module

1. New `tractus_engine_advanced.glottis.cetacean` module — phonic-lips
   + spermaceti acoustic-lensing model for toothed whales (dolphin,
   sperm whale). Baleen-whale U-fold larynx + sac is a separate
   variant; pick one for v0.
2. Water-impedance coupling at the radiation boundary (current TEA
   assumes air; this is a real but bounded extension).
3. Acceptance: produce audible click trains (toothed) or moans
   (baleen) with formant structure consistent with the literature.

### Stage 3 — Phoneme-inventory inference layer

1. Information-theoretic layer on top of `inversion/006`'s
   dimensionality-discovery work. Given a fitted (source × resonator)
   model, compute the acoustic Jacobian's effective rank (already in
   tree for human) and the per-dimension articulatory effort cost.
2. From (rank, effort), enumerate distinguishable acoustic regions
   that meet a minimum SNR / discriminability threshold. Each region
   is a candidate phoneme.
3. Validate against known repertoires: human (~40 ARPAbet phonemes),
   zebra finch (~10 song syllables), bottlenose dolphin (debated;
   ~30-50 click types).
4. Acceptance: predicted inventory size matches measured ±2× for
   each species, AND the predicted regions overlap with measured
   formant / modulation-spectrum clusters.

### Stage 4 — Vocabulary engineering (optional, separable)

1. Given a phoneme inventory + per-phoneme effort cost, run optimal-
   coding analysis (Huffman / MDL) for hypothetical vocabulary sizes.
2. Compare predicted Zipf-shape against natural-language vocabularies.
3. Output: a "what would a language designed for THIS vocal tract
   look like?" report.

## Acceptance gates (automated, per stage)

- **Stage 1**: spectrogram cross-correlation ≥ 0.6 between rendered
  and reference bird songs on a held-out species split. Differentiable
  end-to-end (gradient FD passes per the M6 standard).
- **Stage 2**: same, for whale tokens. Water-impedance boundary tested
  against published radiation-pattern measurements.
- **Stage 3**: predicted inventory size within 2× of measured for all
  three species; regions overlap measured clusters with IoU ≥ 0.4.
- **Stage 4**: predicted vocabulary Zipf slope within ±0.2 of measured
  natural-language slopes (-1.0 ± 0.2 typical).

## Anti-deliverable / scope guard

- Do NOT take this on before P4 cloning closes. The human stack must
  be working as a v0 product first; cross-species is a research
  extension.
- Do NOT scope to "all species". Pick ONE bird + ONE cetacean for
  v0; expansion to additional species is a separate task.
- Do NOT attempt the Stage 4 vocab-engineering deliverable without
  Stage 3 producing reasonable inventory sizes — the layer above is
  meaningless without a defensible phoneme layer below.
- Do NOT attempt the Rocky / piezo / parallel-channel-direct-synthesis
  case as part of this task. That's a different abstraction (no
  resonator, parallel chord channels rather than serial phonemes).
  See "Rocky-class follow-up" below; if Stage 1 (bird syrinx) lands
  cleanly, that work would be the natural fork point for a separate
  research task.

## Rocky-class follow-up (deferred, separate task)

The Project-Hail-Mary case (5 piezoelectric transducers, no
resonator, parallel-frequency chord communication) is qualitatively
different from the bird/whale cases and does NOT drop into TEA's
source-resonator architecture. Piezos are trivially differentiable
(forced linear oscillators), but there's no tube to modulate; the
abstraction is direct waveform synthesis with parallel channels.

If Stage 1 lands successfully, the parallel-channel machinery from
the two-bronchi syrinx model becomes the foundation for an N-channel
direct-synthesis abstraction. At that point, file a separate
`research/00X_alien_chord_communication.md` task with that
abstraction at its core. Until then, this is conversation, not work.

## References / starting points

- Mindlin & Laje, *The Physics of Birdsong* (2005) — syrinx model.
- Goller & Larsen, "A new mechanism of sound generation in songbirds"
  (1997) — two-bronchi independence.
- Cranford et al., dolphin phonic-lips + spermaceti ray-acoustics
  literature.
- Reidenberg & Laitman, baleen-whale U-fold larynx anatomy.
- For the info-theory layer: Stevens, *Acoustic Phonetics* — covers
  the human-tract version of the rank-vs-effort argument that
  generalizes here.

## Notes

This is **the project's most interesting north-star research line**
once cloning ships. The transmission-line solver in `tds/mna.py` is
the load-bearing piece that makes all of this approachable; without
a working differentiable acoustic simulator, the cross-species work
would be intractable. The fact that we're building one for human use
gives this whole research direction a free first 60% of effort.

Estimated effort breakdown (post-P4):
- Stage 1 (syrinx): 2-3 months
- Stage 2 (cetacean): 2-3 months
- Stage 3 (info theory): 2 months
- Stage 4 (vocab engineering): 1 month, separable

Probability of success per stage (rough; from the architectural
review with Codex during the v0 push):
- Stage 1: ~55% for "audible bird-like sound matching reference"
- Stage 2: ~35% for "audible whale-like sound"; ~10% for matching a
  specific species recording
- Stage 3: ~35% for human; ~20% for non-human (training-data scarcity)
- Stage 4: ~60% conditional on Stage 3 producing usable inventories
