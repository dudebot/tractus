# References — module → paper map

Per-module reading list. Mirror PDFs into the appropriate subfolder
manually as needed. Canonical URL set is in
`pages/reference_links.md`.

Folder layout:

```
docs/references/
├── core/          # tier 0: load-bearing physical / articulatory papers
├── foundations/   # gesture / motor-control theory
├── baselines/     # neural TTS/VC comparators
├── datasets/      # corpus documentation (not the data itself)
└── pages/
    └── reference_links.md   # canonical URLs for all of the above
```

## Module 1 — Linguistic / Prosody Planner

- F5-TTS — modern open zero-shot TTS baseline (arXiv:2410.06885)
- CosyVoice / CosyVoice 2 — streaming + multilingual baselines
- Kokoro — compact open TTS reference (82 M params)
- Saltzman & Munhall 1989 / Browman & Goldstein 1992 — gesture
  timing / articulatory phonology foundations

## Module 2 — Neuromotor Controller

- Saltzman & Munhall 1989 — Task Dynamic Model
- Browman & Goldstein 1992 — Articulatory Phonology
- DIVA / TorchDIVA — speech motor control framework
- Teaching Machines to Speak (arXiv:2510.05619) — RL articulatory
  control, syllable-level scope
- SPARC (arXiv:2406.12998) — articulatory feature extraction

## Module 3 — Biomechanical Body Model

- Stevens 1998 *Acoustic Phonetics* (textbook) — population-mean
  tract anatomy, formant tables, fricative source spectra
- Fant 1960 *Acoustic Theory of Speech Production* (textbook) —
  source-filter foundations, tract acoustics
- Maeda 1990 — low-dimensional vocal-tract shape representation
- SPARC — candidate external articulatory teacher (deferred)
- MOCHA-TIMIT / mngu0 / USC-TIMIT / USC Morphology MRI — validation
  candidates (and v1+ teacher candidates if licensing allows)
- Acoustic-to-articulatory inversion papers — body-model teacher
  candidates from real audio

## Module 4 — TEA Acoustic Renderer

- Stevens 1998 *Acoustic Phonetics* — formant tables, source
  spectra, radiation models
- Fant 1960 — source-filter theory, tract acoustics
- Narayanan & Alwan 2000 — distributed aerodynamic frication noise
- Chen & Sitzmann 2025 — differentiable simulator with analytic
  gradients

## Module 5 — Speaker Inference & Adaptation

- SPARC — articulatory + speaker identity decomposition
- RT-VC (arXiv:2506.10289) — real-time zero-shot voice conversion
- Acoustic-to-articulatory inversion papers — anatomy/source fitting
  precedent
- ECAPA-TDNN / WavLM — identity verification model candidates

## Surrogate (Module 5A) / Inference Residual (Module 5B)

- DDSP Articulatory Vocoder (arXiv:2409.02451) — **0.4 M-parameter**
  articulatory vocoder at quality comparable to a 9 M-parameter
  baseline. Direct precedent for surrogate sizing.
- RT-VC — real-time articulatory codec
- F5-TTS / CosyVoice / MegaTTS3 — neural-baseline comparators

## Differentiability / inverse fitting

- Chen & Sitzmann 2025 — differentiable simulator with analytic
  gradients
- VocalTrax — optimization-based articulatory synthesis (warning
  reference for quality risk)
- DAFx 2023 — gradient-descent vocal-tract area estimation

## Evaluation baselines

- Kokoro — small-model TTS naturalness baseline (B2)
- F5-TTS — modern zero-shot TTS (B3)
- CosyVoice / CosyVoice 2 — streaming/multilingual (B3)
- MegaTTS3 — strong zero-shot cloning (B3)
- ElevenLabs docs — matched-reference commercial comparison
