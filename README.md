# Tractus — archived research

**Development stopped on 2026-10-04 UTC.** This repository is retained as a research record. There is no active roadmap, resumption issue, or planned training run. Historical plans and agent instructions describe abandoned work; this archive notice supersedes their dispatch instructions.

Tractus explored learning compact articulatory controls and rendering speech through a physical vocal-tract simulator. That premise is credible. It does **not** establish that the complete neural system needs fewer weights, less training, or less inference compute than a spectrogram-based system. Articulator geometry, tube areas, and muscle activations are different representations. Tractus never established its intended general TTS and voice-cloning result.

## Look here instead

| Interest | Starting point | What is actually available |
| --- | --- | --- |
| Learned controls driving a physical speech renderer | [TensorTract3 paper](https://www.essv.de/pdf/2026_112_119.pdf) and [separate VTL/neural audio examples](https://tensortract3.github.io/) | Strong prior art for intelligible physical **resynthesis**. Paper and demos verified; runnable TT3 code/weights not established. |
| An existing implementation to investigate | [TensorTract2](https://github.com/Altavo/tensortract2), [TensorTractLab integration](https://github.com/tensortract-dev/tensortractlab), [VocalTractLab backend](https://github.com/TUD-STKS/VocalTractLabBackend-dev) | Public code; TT2 advertises downloadable pretrained weights and VTL-compatible controls. TT2 has Apache 2.0 **plus Commons Clause** terms; integration/backend have GPL-3.0 licenses. Reproduction was not run here. |
| Compact articulatory coding and voice conversion | [SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) and [paper](https://arxiv.org/abs/2406.12998) | Public inference code and advertised checkpoints; neural waveform decoding rather than a physical tube renderer. Packaging says MIT, but no repository LICENSE file was found. |
| Differentiable physical acoustics and reconstruction | [Chen, Lee and Sitzmann](https://arxiv.org/html/2609.36737v1), [project demonstrations](https://people.csail.mit.edu/echen/vocal_recon/) | Frequency-domain simulation is relevant to Tractus's sequential solver cost. Paper/demos verified; a licensed runnable release was not established. |

These are research references, not claims of a ready replacement for arbitrary-text, reference-conditioned TTS. For the evidence, distinctions, and release checks, read the [final assessment](docs/archive-2026-10-04/assessment.md) and [source ledger](docs/archive-2026-10-04/ledger.md).

## Why keep this repository?

Keep it archived: the simulator, bug investigations, and failed or inconclusive experiments can prevent repeated work. They do not justify continued investment in this implementation. Buying more compute does not resolve an unvalidated training objective or missing end-to-end system.

The [earlier reset assessment](docs/handoffs/2026-10-04T003722Z/assessment.md), [historical architecture](ARCHITECTURE.md), and source packages remain for archaeology. Task 009's smoke run was not a converged negative result. No new model, audio-quality result, or training benchmark is claimed by this archival review.

Tractus source license: [MIT](LICENSE). Linked projects retain their own terms.
