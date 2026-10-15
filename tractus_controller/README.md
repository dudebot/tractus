# tractus_controller

The controller half of [tractus](../). Houses the learned modules
(phonemizer, motor policy, anatomy fit), the JAX gradient fitter, the
default speaker baseline, and the CLI.

**For project vision, architecture, and the current verified state of
this package, see the workspace-level
[`../ARCHITECTURE.md`](../ARCHITECTURE.md).** That document supersedes
this one on any conflict; this README only carries the local
quickstart.

## Naming

- Project / directory / Python package: `tractus_controller` (underscores).
- CLI tool: `tractus-controller` (hyphens; CLI tools allow them).

## Quickstart

```bash
cd tractus_controller/
python3 -m venv .venv
.venv/bin/pip install -U pip wheel
.venv/bin/pip install -e ".[dev]"
.venv/bin/pip install -e ../tractus_shared
.venv/bin/pip install torch torchaudio   # CUDA wheel; cu128 supports sm_75 + sm_120
```

```bash
.venv/bin/tractus-controller doctor --strict
```

The `doctor` subcommand checks os/python/libsndfile/espeak/phonemizer/
torch/cuda/gpu. It does NOT check for the legacy C++ articulator-synthesis dependency — that
was removed 2026-05-04.

## What works today

- `tractus-controller version`, `doctor`, `phonemize`,
  `prepare-manifest`, `train` (against a manifest of precomputed
  `.npy` knot caches).
- Pure-JAX synthesis path (R1 NN adapter + TEA G3a glottis):
  `copy_synth.tea_backend.synthesize_via_tea_jax(trajectory, ...)`.
- Adam fitter via `copy_synth.fitter_grad`.

## What's broken

- `tractus-controller synth` and `synth-template`: `NotImplementedError`.
  The phone→trajectory path needed two pieces of the C++ backend (R2
  and R3); they're stubbed pending replacement.
- Backward through full pipeline at production T (≈ 118 frames) OOMs
  on 16 GB GPU; NaN at small T. Both have known fixes (see
  `../ARCHITECTURE.md` §8).

## Layout

```
tractus_controller/
├── tractus_controller/
│   ├── adapters/      # R1 NN tract→tube emulator
│   ├── controller/    # legacy BiLSTM (under refactor)
│   ├── copy_synth/    # gradient fitter + features + parameterization
│   ├── phonemes/      # IPA / ARPAbet / SAMPA tables
│   ├── data/          # manifest helpers
│   ├── cli.py
│   ├── doctor.py
│   ├── paths.py
│   └── sidecar.py
├── scripts/
└── tests/
```

## Voice-cloning ethics

Every training manifest needs `consent_verified: true` per voice. No
public figures, no non-consenting voices. The `prepare-manifest`
helper enforces this; do not bypass.
