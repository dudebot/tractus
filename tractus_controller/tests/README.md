# tractus_controller — test taxonomy

Owner: `infra/001`. This file documents *what each test slice covers*
and *how to run it*. Markers are registered in
`pyproject.toml::[tool.pytest.ini_options]`.

## Quick reference

| Slice                                              | Command                                                                  |
|----------------------------------------------------|--------------------------------------------------------------------------|
| Default (fp32-default; safe in any process)        | `pytest -q`                                                              |
| Parity (fp64 opt-in; **subprocess-isolated**)      | `python -m tests._parity_subprocess_runner` (or `python tests/_parity_subprocess_runner.py`) |
| One-marker filter                                  | `pytest -q -m fp32_default`                                              |
| Collection sanity                                  | `pytest --collect-only`                                                  |

## Slice semantics

### Pure-linguistic + pure-math (default slice)

The kept-as-is tests live here:
`test_arpabet_sampa`, `test_phonemize`, `test_durations`,
`test_manifest`, `test_skeleton`, `test_gates`,
`copy_synth/test_features_jax`. None of these depend on a live oracle.

### `parity` and `fp64_oracle`

Same contract as the engine side: excluded from default `pytest -q`,
re-included only via the subprocess runner. Saved-fixture parity
tests must use the `fixture_oracle` marker and load from
`outputs/fixtures/` — the legacy upstream oracle is gone.

### `fp32_default`

Asserts `jax.config.x64_enabled is False` and
`jnp.array([1.0]).dtype == fp32` under default config. Catches a
regression where an import-time `jax.config.update("jax_enable_x64", True)`
sneaks into production code.

Note: `tractus_controller.copy_synth.tea_backend` calls
`jax.config.update("jax_enable_x64", True)` *inside* a function body
(not at module import time). If a refactor moves that to module top
level, this test fires.

### `espeak`, `torch`, `slow`

Optional-dependency / heavy-runtime gates, unchanged from prior
behavior. `conftest.py` auto-skips when the dep is missing.

### Architectural-invariant markers (placeholders)

| Marker                       | Coverage gap                                | Blocked by                              | Test file                       |
|------------------------------|---------------------------------------------|-----------------------------------------|----------------------------------|
| `gesture_event`              | composition / async overlap / dataclass     | `correctness/004`, `pretraining/003`    | `test_gesture_event.py`         |

The remaining architectural-invariant placeholders live on the engine
side (`tractus_engine_advanced/tests/`).

## Why subprocess isolation for parity

A single `jax.config.update("jax_enable_x64", True)` flips fp64 for
every later test in the same process. Hence: parity tests run in a
child process driven by `_parity_subprocess_runner.py`; the in-process
default loop skips them with a pointer to the runner.
