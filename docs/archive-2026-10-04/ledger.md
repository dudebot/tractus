# Archive evidence ledger — 2026-10-04 UTC

Source checkout: `82925e420a6def9cbfd29805686c047c54a71eb4`, equal to fetched `origin/master` before archival edits. Previous implementation assessment: `e9b66ff46bb67389268f13a28bacc6e150b2208f`. The only pre-existing untracked files were this directory's upstream repository metadata and two license copies from the interrupted research review; those are retained.

## Primary-source checks

| Source | Specific check | Release conclusion |
| --- | --- | --- |
| [TT3 proceedings PDF](https://www.essv.de/pdf/2026_112_119.pdf) | §§2.1–2.2, Figure 1, Table 1, §4.1 | Separates physical and neural paths; architecture/results/limitations inspected. |
| [TT3 project source](https://github.com/TensorTract3/TensorTract3.github.io) | Root tree and `index.md` | Audio demo site; no implementation/checkpoints/license identified in this site repository. Does not prove no release exists elsewhere. |
| [TT2](https://github.com/Altavo/tensortract2) | `README.md`, root tree, `LICENSE` | Code and Google Drive weights link advertised; weights were not downloaded or executed. Apache 2.0 plus Commons Clause, not unrestricted Apache 2.0. |
| [TensorTractLab](https://github.com/tensortract-dev/tensortractlab) | Root tree and repository license metadata | Public integration code and GPL-3.0 license; not installed. |
| [VTL backend](https://github.com/TUD-STKS/VocalTractLabBackend-dev) | Root tree and `LICENSE` | Source/build/examples present; GPL-3.0. Not built or benchmarked. |
| [SPARC](https://github.com/Berkeley-Speech-Group/Speech-Articulatory-Coding) | README, recursive file tree, `setup.py` | Inference code and checkpoint loading instructions present. `setup.py` declares MIT; no LICENSE file found in tree. Training code remains a README TODO. Weights not downloaded/executed. |
| [Differentiable reconstruction paper](https://arxiv.org/html/2609.36737v1) and [project](https://people.csail.mit.edu/echen/vocal_recon/) | Paper and supplementary demo page | No licensed runnable implementation established from checked sources. |

Pinned upstream heads, default branches, and license metadata are in [upstream-repositories.json](upstream-repositories.json). The `arti-6` entry is an inherited exploratory lookup, not an endorsed alternative. License files are retained as checked evidence; they do not change Tractus's license.

## Repository action and validation

The README and final assessment replace the active-work entry point with the stop decision and references. Historical code, plans, and measurements are retained. No training or GPU experiment was run. Verification is documentation/link inspection and `git diff --check`; the prior reset assessment records existing code-check limitations, and no claim is made that this documentation change repairs them.

Authorized closeout: close every open issue as not planned, create no resumption issue, push the documentation to `master`, then archive the repository. The final externally verified repository state is reported in the closing issue comment and task handoff; this ledger records the intended transaction rather than claiming that a pre-archive commit can prove the later archive flag.
