# pretraining/002a — Story 1996 vocal-tract area-function anchor ingestion

**Phase:** pretraining    **Status:** pending    **Effort:** S
**User-input:** none
**Blocked by:** —
**Blocks:** pretraining/002 (real anchor-grounded body model)
**How to dispatch:** fresh Claude Code session

---

## Context

`pretraining/002` requires real articulatory anchor data, not just R1
NN inference over hand-crafted templates. The cheapest credible
anchor source is Story et al. 1996 ("Vocal tract area functions from
magnetic resonance imaging," J. Acoust. Soc. Am. 100, 537):

- 18 published static area-function shapes from one adult male
  speaker: 12 vowels + 3 nasals + 3 plosives.
- 44-section MRI-derived cross sections.
- Public reference; numerical values widely reproduced in
  articulatory-synthesis literature.

This task ingests those shapes as TEA-compatible anchor articulations
and verifies each renders at literature formants. The output is the
anchor slice of the eventual `pretraining/002` corpus manifest.

## Deliverable

1. **Source the data.** Either digitize from the paper / a public
   reproduction (e.g., the supplementary tables in subsequent Story
   papers) OR depend on an existing tabulation (cite the source
   inline). Do NOT fabricate values — if no clean source is
   reachable, report blocker rather than guessing.

2. **`tractus_controller/outputs/fixtures/story_1996_anchors.npz`**
   (gitignored, will be regen-able). Schema:
   - `shape_names`: (18,) string array, e.g. `["/i/", "/I/", ..., "/m/", "/n/", "/N/", "/p/", "/t/", "/k/"]`
   - `area_cm2_44sec`: (18, 44) original 44-section area cross sections
   - `length_cm_44sec`: (18, 44) per-section lengths (or scalar total length per shape if uniform)
   - `area_cm2_tea`: (18, 40) resampled to TEA's 40 supraglottal sections (`FIRST_PHARYNX_SECTION` through `LAST_MOUTH_SECTION`)
   - `provenance`: dict with paper citation, digitization method, date

3. **Render gate.** New script `scripts/sanity_check_story_anchors.py`:
   - For each of the 18 shapes, plug `area_cm2_tea` into the default
     speaker tube's supraglottal slice, render through TEA with default
     voiced glottis, measure formants via `acoustic_gates.formant_gate`.
   - Print per-shape table: measured F1/F2/F3 vs literature targets
     (use Peterson & Barney for vowels; published nasal/plosive
     formant ranges otherwise).
   - **Acceptance**: at least 9 of 12 vowels within 15% on F1 AND F2
     (F3 looser tolerance acceptable). If fewer pass, the digitization
     or resampling has a bug; do not proceed.

4. **Anchor-corpus loader stub.** A small module under
   `tractus_controller/tractus_controller/body_model/anchors.py`
   exposing `iter_story_anchors() -> Iterator[(name, area_cm2_tea, length_cm_tea)]`
   for downstream corpus assembly.

## Acceptance gates (automated)

- ✅ `story_1996_anchors.npz` exists with the documented schema; `np.load`
  round-trips all keys.
- ✅ `python scripts/sanity_check_story_anchors.py` exits 0 with
  ≥ 9 of 12 vowels passing the formant tolerance.
- ✅ Provenance entry is non-empty and cites a primary source.

## Anti-deliverable / scope guard

- Do NOT fabricate or hand-tune area values. Either source them or
  report blocker.
- Do NOT extend to running consonants (CV transitions, coarticulation)
  — this task is static-shape ingest only.
- Do NOT train a body model here. That's pretraining/002.

## Notes

The 18 shapes are a calibration panel, not a full English body model.
They define WHERE the body model must agree with reality on canonical
articulations; the rest of the input space is filled by other corpus
sources (R1 fixtures, SPARC, procedural). Don't overweight to the
point of overfitting the panel.

Cross-references:
- `tractus_engine_advanced/src/tractus_engine_advanced/tds/tube_topology.py`
  for `FIRST_PHARYNX_SECTION` (25), `LAST_MOUTH_SECTION` (64).
- `tractus_engine_advanced/src/tractus_engine_advanced/acoustic_gates.py`
  for `formant_gate` and `VOWEL_FORMANTS_MALE`.
- `scripts/sanity_check_vowel_tube.py` for the existing
  hand-crafted-tube render pattern; lift its render path.
