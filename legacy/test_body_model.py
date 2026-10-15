"""Sanity tests for the body model (pretraining/002 architecture step).

Cheap tests only — no training, no full corpus generation. Acceptance
gates 1, 2, 4, and the non-circularity CI hook are exercised here.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tractus_controller.body_model import (
    BodyModel,
    BodyModelConfig,
    D_ANATOMY_PLACEHOLDER,
    D_BODY_PLACEHOLDER,
    GESTURE_CHANNELS,
    N_GESTURE_CHANNELS,
    RendererInterface,
    apply_hard_clamp,
    compose_gesture_trajectory,
    gesture_channel_index,
)
from tractus_controller.body_model.channels import (
    PER_FRAME_FREE_FIELDS,
    STATIC_FIELDS,
    per_frame_free_dim,
    static_field_dim,
)
from tractus_controller.body_model.reachability import (
    DEFAULT_REACHABILITY,
    FRICATION_COLOR_CAP,
)
from tractus_controller.gestures import GestureEvent


# ---------------------------------------------------------------------------
# Architecture instantiates with documented param count (≤ 5 M)
# ---------------------------------------------------------------------------


def test_body_model_instantiates_under_5m_params():
    m = BodyModel.from_default()
    n = m.num_params()
    # Acceptance gate 4 of pretraining/002.
    assert n <= 5_000_000, f"body model has {n} params; spec cap is 5M"
    # Sanity floor — if this drops below 100K something is mis-wired.
    assert n >= 100_000, f"body model only has {n} params; expected >=100K"


def test_body_model_config_dims():
    cfg = BodyModelConfig()
    assert cfg.d_gesture == N_GESTURE_CHANNELS == 20
    assert cfg.d_body == D_BODY_PLACEHOLDER == 64
    assert cfg.d_anatomy == D_ANATOMY_PLACEHOLDER == 64


# ---------------------------------------------------------------------------
# Forward pass produces the right output shapes
# ---------------------------------------------------------------------------


def test_forward_output_shapes_single():
    m = BodyModel.from_default()
    T = 50
    g = jnp.zeros((T, N_GESTURE_CHANNELS))
    a = jnp.zeros((D_ANATOMY_PLACEHOLDER,))
    out = m(g, a)
    assert isinstance(out, RendererInterface)
    assert out.tube_length_cm.shape == (T, 93)
    assert out.tube_area_cm2.shape == (T, 93)
    assert out.glottis_state.shape == (T, 11)
    assert out.nasal_coupling.shape == (T,)
    assert out.subglottal_press.shape == (T,)
    assert out.frication_gain.shape == (T, 93)
    assert out.frication_color.shape == (T, 93)
    assert out.wall_mass_static.shape == (93,)
    assert out.wall_resistance_static.shape == (93,)
    assert out.wall_stiffness_static.shape == (93,)
    # Cheat-channel discipline: wall_dynamic_residual MUST be None by
    # default (ARCH §2.3 cheat-channel discipline; pretraining/002
    # anti-deliverable: do NOT add a free wall_dynamic_residual channel).
    assert out.wall_dynamic_residual is None


def test_forward_output_shapes_batched():
    m = BodyModel.from_default()
    T, B = 30, 4
    g = jnp.zeros((B, T, N_GESTURE_CHANNELS))
    a = jnp.zeros((B, D_ANATOMY_PLACEHOLDER))
    out = m.batched_forward(g, a)
    assert out.tube_length_cm.shape == (B, T, 93)
    assert out.glottis_state.shape == (B, T, 11)
    assert out.wall_mass_static.shape == (B, 93)


def test_per_frame_free_dim_matches_decoder_output():
    """per_frame_free_dim() = sum of free per-frame field widths."""
    expected = 93 + 93 + 11 + 1 + 1 + 93 + 93
    assert per_frame_free_dim() == expected
    # Renderer interface flat shape matches:
    m = BodyModel.from_default()
    T = 16
    g = jnp.zeros((T, N_GESTURE_CHANNELS))
    a = jnp.zeros((D_ANATOMY_PLACEHOLDER,))
    out = m(g, a)
    assert out.to_flat_per_frame().shape == (T, expected)


def test_static_field_dim_matches_static_decoder_output():
    expected = 93 + 93 + 93
    assert static_field_dim() == expected


# ---------------------------------------------------------------------------
# Reachability clamps fire when given out-of-envelope input
# ---------------------------------------------------------------------------


def test_hard_clamp_pulls_extreme_values_into_envelope():
    """Apply hard clamp directly to wildly out-of-range inputs and
    confirm the output is inside the bounds."""
    T = 8
    out = apply_hard_clamp(
        tube_length_cm=jnp.full((T, 93), 1000.0),
        tube_area_cm2=jnp.full((T, 93), -50.0),       # negative → clamped to MIN_AREA
        glottis_state=jnp.full((T, 11), 1e6),         # gigantic
        nasal_coupling=jnp.full((T,), -5.0),           # negative
        subglottal_press=jnp.full((T,), -100.0),      # negative
        frication_gain=jnp.full((T, 93), -20.0),       # negative
        frication_color=jnp.full((T, 93), 5.0),        # past cap
    )
    assert jnp.all(out["tube_length_cm"] <= DEFAULT_REACHABILITY.max_length_cm + 1e-6)
    assert jnp.all(out["tube_length_cm"] >= DEFAULT_REACHABILITY.min_length_cm - 1e-6)
    assert jnp.all(out["tube_area_cm2"] >= DEFAULT_REACHABILITY.min_area_cm2 - 1e-6)
    # Glottis F0 (col 0) clamped to MAX_F0
    assert jnp.all(out["glottis_state"][:, 0] <= DEFAULT_REACHABILITY.max_f0_hz + 1e-6)
    # Glottis PR (col 1) clamped to MAX_LUNG_PA
    assert jnp.all(out["glottis_state"][:, 1] <= DEFAULT_REACHABILITY.max_lung_pa + 1e-6)
    assert jnp.all(out["nasal_coupling"] >= 0.0)
    assert jnp.all(out["subglottal_press"] >= 0.0)
    assert jnp.all(out["frication_gain"] >= 0.0)
    # Frication color capped at FRICATION_COLOR_CAP
    assert jnp.all(jnp.abs(out["frication_color"]) <= FRICATION_COLOR_CAP + 1e-6)


def test_full_forward_with_extreme_input_stays_in_envelope():
    """Even with adversarially-large activations bubbling through the
    network, the soft-clamped output must stay finite and in range."""
    m = BodyModel.from_default()
    T = 16
    # Adversarial input: very large gestures + anatomy
    g = jnp.full((T, N_GESTURE_CHANNELS), 1000.0)
    a = jnp.full((D_ANATOMY_PLACEHOLDER,), 1000.0)
    out = m(g, a, use_hard_clamp=True)
    # All outputs finite
    for name in PER_FRAME_FREE_FIELDS:
        v = getattr(out, name)
        assert jnp.all(jnp.isfinite(v)), f"{name} has non-finite values"
    # Tube area in [min, max]
    assert jnp.all(out.tube_area_cm2 >= DEFAULT_REACHABILITY.min_area_cm2 - 1e-6)
    assert jnp.all(out.tube_area_cm2 <= DEFAULT_REACHABILITY.max_area_cm2 + 1e-6)
    # F0 in [min, max]
    assert jnp.all(out.glottis_state[..., 0] >= DEFAULT_REACHABILITY.min_f0_hz - 1e-6)
    assert jnp.all(out.glottis_state[..., 0] <= DEFAULT_REACHABILITY.max_f0_hz + 1e-6)


# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------


def test_compose_gesture_trajectory_basic():
    events = [
        GestureEvent("F0", onset_s=0.0, target_value=200.0, duration_s=0.5,
                     release_s=0.0, strength=1.0, ramp_shape="hold_release"),
        GestureEvent("lip_aperture", onset_s=0.1, target_value=0.0,
                     duration_s=0.2, release_s=0.05, strength=1.0),
    ]
    traj = compose_gesture_trajectory(events, duration_s=0.6)
    # 50 Hz × 0.6 s = 30 frames
    assert traj.shape == (30, N_GESTURE_CHANNELS)
    # F0 column reaches the 200 Hz target
    f0_col = gesture_channel_index("F0")
    assert traj[:, f0_col].max() == pytest.approx(200.0, abs=1e-3)
    # Channels with no events stay at neutral (jaw_open → 0.0)
    jaw_col = gesture_channel_index("jaw_open")
    assert np.all(traj[:, jaw_col] == 0.0)


def test_compose_gesture_unknown_channel_is_dropped():
    events = [
        GestureEvent("not_a_real_channel", onset_s=0.0, target_value=99.0,
                     duration_s=0.5, release_s=0.0, strength=1.0),
    ]
    # Should not raise; just produce a neutral baseline.
    traj = compose_gesture_trajectory(events, duration_s=0.2)
    assert traj.shape == (10, N_GESTURE_CHANNELS)


# ---------------------------------------------------------------------------
# Non-circularity (CRITICAL — pretraining/002 acceptance gate)
# ---------------------------------------------------------------------------


def test_non_circularity_no_tea_pipeline_import_in_body_model():
    """The body_model package MUST NOT import synthesize_dynamic_tube
    (or anything from tractus_engine_advanced.pipeline). That would
    close the circular training loop the spec forbids.

    Implementation: AST-walk every .py file in the body_model package
    and reject any import that pulls in pipeline / the synthesis
    primitive. Docstrings and comments that mention these by name are
    fine — only actual imports and attribute accesses are violations.
    """
    import ast
    import tractus_controller.body_model as bm_pkg
    pkg_dir = Path(bm_pkg.__file__).parent

    offenders: list[tuple[Path, int, str]] = []

    class Visitor(ast.NodeVisitor):
        def __init__(self, path: Path):
            self.path = path

        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                if alias.name == "tractus_engine_advanced.pipeline" or \
                   alias.name.endswith(".pipeline") and "tractus_engine_advanced" in alias.name:
                    offenders.append((self.path, node.lineno, f"import {alias.name}"))

        def visit_ImportFrom(self, node: ast.ImportFrom):
            mod = node.module or ""
            if mod == "tractus_engine_advanced.pipeline" or mod.startswith("tractus_engine_advanced.pipeline."):
                offenders.append((self.path, node.lineno, f"from {mod} import ..."))
            for alias in node.names:
                if alias.name == "synthesize_dynamic_tube":
                    offenders.append((self.path, node.lineno, f"from {mod} import synthesize_dynamic_tube"))

        def visit_Attribute(self, node: ast.Attribute):
            if node.attr == "synthesize_dynamic_tube":
                offenders.append((self.path, node.lineno, "attr access .synthesize_dynamic_tube"))
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if node.id == "synthesize_dynamic_tube":
                offenders.append((self.path, node.lineno, "name reference synthesize_dynamic_tube"))

    for path in pkg_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        Visitor(path).visit(tree)

    assert not offenders, (
        "Non-circularity violation: body_model package must not "
        "reference synthesize_dynamic_tube / pipeline. Offenders:\n"
        + "\n".join(f"  {p}:{ln}: {raw}" for p, ln, raw in offenders)
    )


def test_non_circularity_body_model_does_not_pull_pipeline_at_runtime():
    """Importing body_model must not transitively load
    tractus_engine_advanced.pipeline."""
    import sys
    # If pipeline was loaded by some other test, this only checks the
    # body_model itself doesn't depend on it. Safer: re-import body_model
    # in a clean(ish) way and check direct module dependencies.
    bm = importlib.import_module("tractus_controller.body_model")
    # Walk the package modules and check none of their declared imports
    # mention pipeline. Conservative — just walks the module registry.
    bm_modules = [
        name for name in sys.modules
        if name == "tractus_controller.body_model"
        or name.startswith("tractus_controller.body_model.")
    ]
    bad = []
    for name in bm_modules:
        mod = sys.modules[name]
        if mod is None:
            continue
        # Check for pipeline references in the module's globals
        for attr_name, attr in vars(mod).items():
            if attr_name.startswith("_"):
                continue
            mod_name = getattr(attr, "__module__", "") or ""
            if "tractus_engine_advanced.pipeline" in mod_name:
                bad.append((name, attr_name, mod_name))
    assert not bad, f"body_model loaded pipeline symbols: {bad}"
