# Reference URLs

Canonical URL set for the references used in `ARCHITECTURE.md`.
Mirror per-paper as needed.

## Core (frontier articulatory + differentiable + small-model)

- SPARC paper: https://arxiv.org/abs/2406.12998
- SPARC repo: https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding
- SPARC demo: https://berkeley-speech-group.github.io/Audio-Articulatory-Coding/
- DDSP Articulatory Vocoder paper: https://arxiv.org/abs/2409.02451
- DDSP Articulatory repo: https://github.com/mctlab/DDSP-Articulatory-Synthesizer
- RT-VC paper: https://arxiv.org/abs/2506.10289
- RT-VC repo: https://github.com/Berkeley-Speech-Group/RT-VC
- Chen & Sitzmann 2025 (DBM Findings): https://openreview.net/forum?id=75AUKPeJme
- VocalTrax (OpenReview): https://openreview.net/forum?id=kIDxwvWznF
- VocalTrax repo: https://github.com/hunter-heidenreich/VocalTrax
- Teaching Machines to Speak: https://arxiv.org/abs/2510.05619
- DAFx 2023 area estimation by gradient descent (arXiv): https://arxiv.org/abs/2307.04702

## Frication / noise-source physics

- Narayanan & Alwan 2000 fricative noise (UCLA): https://www.seas.ucla.edu/spapl/paper/narayanan_sap00.pdf

## Acoustic-phonetic anatomy reference (textbooks; not auto-mirrored)

- Stevens 1998 *Acoustic Phonetics* (MIT Press, ISBN 978-0262692502)
- Fant 1960 *Acoustic Theory of Speech Production* (Mouton)
- Hillenbrand et al. 1995 American English vowel formants (JASA 97)
- Maeda 1990 ICSLP (HTML index): https://www.isca-archive.org/icslp_1990/maeda90b_icslp.html

## Foundations

- Saltzman & Munhall 1989 (Task Dynamic Model): https://sail.usc.edu/~lgoldste/ArtPhon/Papers/Saltzman%26Munhall.PDF
- Browman & Goldstein 1992 (Articulatory Phonology): https://users.umiacs.umd.edu/~nhf/courses/ling620/BrowmanGoldstein92.pdf
- TorchDIVA / DIVA framework (PMC): https://pmc.ncbi.nlm.nih.gov/articles/PMC3650855/
- TorchDIVA arXiv: https://arxiv.org/abs/2210.09334

## Baselines (Tier 2)

- Kokoro repo: https://github.com/hexgrad/kokoro
- Kokoro 82M model: https://huggingface.co/hexgrad/Kokoro-82M
- F5-TTS paper: https://arxiv.org/abs/2410.06885
- F5-TTS repo: https://github.com/SWivid/F5-TTS
- CosyVoice paper: https://arxiv.org/abs/2407.05407
- CosyVoice 2 paper: https://arxiv.org/abs/2412.10117
- CosyVoice repo: https://github.com/FunAudioLLM/CosyVoice
- MegaTTS3 paper: https://arxiv.org/abs/2502.18924
- MegaTTS3 repo: https://github.com/bytedance/MegaTTS3
- GLM-4-Voice repo: https://github.com/THUDM/GLM-4-Voice

## Datasets — direct (free, auto-fetched into `docs/references/datasets/`)

- MOCHA-TIMIT (free for non-commercial use, no agreement)
  - Project page: https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html
  - Direct data: https://data.cstr.ed.ac.uk/mocha/
- Hillenbrand 1995 vowel formant data (original WMU page is now 404; mirror)
  - GitHub mirror: https://github.com/santiagobarreda/hillenbrand_et_al_1995

## Datasets — gated (human-in-the-loop required)

- mngu0 — email Korin Richmond with name, organisation, country, intended
  use; he replies manually with a UserID:
  https://homepages.inf.ed.ac.uk/korin/site/research/mngu0/
- USC-TIMIT — sign SAIL release agreement:
  https://sail.usc.edu/span/usc-timit/
- USC Morphology MRI — sign SAIL release agreement:
  https://sail.usc.edu/span/morphdb/

## Acoustic-to-articulatory inversion (recent)

Verified 2026-05-04 — all four arXiv IDs resolve and titles match the topics
below; auto-fetched by `scripts/download_references.sh` into
`docs/references/core/`.

- "Acoustic-to-Articulatory Inversion of Clean Speech Using an MRI-Trained Model": https://arxiv.org/abs/2603.11845
- Azzouz, Vuissoz & Laprie, "Reconstruction of the Vocal Tract from Speech via Phonetic Representations Using MRI Data": https://arxiv.org/abs/2603.11847
- "Acoustic-to-articulatory Inversion of the Complete Vocal Tract from RT-MRI with Various Audio Embeddings and Dataset Sizes": https://arxiv.org/abs/2603.28723
- Yokota, Harakawa, Baba & Iwahashi, "Physics-Informed Neural Networks for Speech Production": https://arxiv.org/abs/2511.00428

## Product / evaluation references

- ElevenLabs voice cloning overview: https://elevenlabs.io/docs/eleven-creative/voices/voice-cloning
- ElevenLabs Instant Voice Cloning sample tips: https://help.elevenlabs.io/hc/en-us/articles/13416206830097-Are-there-any-tips-to-get-good-quality-cloned-voices
- ElevenLabs cloning sample-count guidance: https://help.elevenlabs.io/hc/en-us/articles/13434364550801-How-many-voice-samples-should-I-upload-for-Instant-Voice-Cloning
