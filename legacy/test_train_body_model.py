"""Smoke + non-circularity tests for scripts/train_body_model.py.

Exercise the training loop end-to-end on a tiny generated subset, and
confirm no module under :mod:`tractus_controller.body_model` or the
training script itself imports ``synthesize_dynamic_tube`` /
``tractus_engine_advanced.pipeline``.
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
TRAIN_SCRIPT = SCRIPTS_DIR / "train_body_model.py"
EVAL_SCRIPT = SCRIPTS_DIR / "eval_body_model.py"
GEN_SCRIPT = SCRIPTS_DIR / "gen_body_corpus.py"


@pytest.fixture(scope="module")
def tiny_corpus(tmp_path_factory) -> Path:
    """Generate a 256-sample corpus to a temp dir, used by smoke train."""
    out = tmp_path_factory.mktemp("body_corpus_tiny")
    # Keep the corpus tiny but still split into >1 shard for realism.
    cmd = [
        sys.executable, str(GEN_SCRIPT),
        "--n", "256",
        "--out", str(out),
        "--n-per-shard", "128",
        "--T", "30",
        "--seed", "0",
    ]
    env = os.environ.copy()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True,
                         cwd=str(REPO_ROOT), timeout=600)
    if res.returncode != 0:
        pytest.skip(f"corpus gen failed (likely missing fixtures): {res.stderr[-1000:]}")
    assert (out / "manifest.json").exists()
    return out


def test_train_body_model_smoke_runs(tiny_corpus, tmp_path):
    """Run the training script on the tiny corpus for a single epoch.
    Should produce a checkpoint."""
    ckpt_dir = tmp_path / "ckpt"
    cmd = [
        sys.executable, str(TRAIN_SCRIPT),
        "--corpus", str(tiny_corpus),
        "--out", str(ckpt_dir),
        "--epochs", "1",
        "--batch", "16",
        "--lr", "3e-4",
        "--seed", "0",
        "--max-train-batches-per-epoch", "5",
        "--max-val-batches", "1",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True,
                         cwd=str(REPO_ROOT), timeout=600)
    assert res.returncode == 0, (
        f"train_body_model.py smoke run failed:\n"
        f"STDOUT:\n{res.stdout[-2000:]}\n"
        f"STDERR:\n{res.stderr[-2000:]}"
    )
    assert (ckpt_dir / "final.pkl").exists()
    assert (ckpt_dir / "loss_curve.csv").exists()
    # Loss curve has at least header + 5 train rows + maybe 1 val row.
    lines = (ckpt_dir / "loss_curve.csv").read_text().splitlines()
    assert len(lines) >= 6


# ---------------------------------------------------------------------------
# Non-circularity guard on the training script + scripts/eval_body_model
# ---------------------------------------------------------------------------


def _ast_check_no_pipeline_or_synth(path: Path) -> list[str]:
    """Walk the AST; return a list of human-readable offenders."""
    offenders: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))

    class V(ast.NodeVisitor):
        def visit_Import(self, node: ast.Import):
            for alias in node.names:
                if "tractus_engine_advanced.pipeline" in (alias.name or ""):
                    offenders.append(f"line {node.lineno}: import {alias.name}")

        def visit_ImportFrom(self, node: ast.ImportFrom):
            mod = node.module or ""
            if "tractus_engine_advanced.pipeline" in mod:
                offenders.append(f"line {node.lineno}: from {mod} import ...")
            for alias in node.names:
                if alias.name == "synthesize_dynamic_tube":
                    offenders.append(
                        f"line {node.lineno}: from {mod} import synthesize_dynamic_tube"
                    )

        def visit_Attribute(self, node: ast.Attribute):
            if node.attr == "synthesize_dynamic_tube":
                offenders.append(
                    f"line {node.lineno}: attr access .synthesize_dynamic_tube"
                )
            self.generic_visit(node)

        def visit_Name(self, node: ast.Name):
            if node.id == "synthesize_dynamic_tube":
                offenders.append(f"line {node.lineno}: name synthesize_dynamic_tube")

    V().visit(tree)
    return offenders


def test_train_body_model_no_synth_import():
    offenders = _ast_check_no_pipeline_or_synth(TRAIN_SCRIPT)
    assert not offenders, (
        "scripts/train_body_model.py imports / references "
        "synthesize_dynamic_tube. The non-circularity rule (ARCH §445-456) "
        "forbids putting TEA audio in the body-model loss path. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_eval_body_model_no_synth_import():
    offenders = _ast_check_no_pipeline_or_synth(EVAL_SCRIPT)
    assert not offenders, (
        "scripts/eval_body_model.py imports / references "
        "synthesize_dynamic_tube. The non-circularity rule (ARCH §445-456) "
        "forbids putting TEA audio in the body-model eval path. Offenders:\n  "
        + "\n  ".join(offenders)
    )


def test_body_model_package_no_synth_import():
    """Re-cover (sibling test exists in test_body_model.py): package
    files under body_model/ must not import the synthesis primitive."""
    bm_dir = REPO_ROOT / "tractus_controller" / "tractus_controller" / "body_model"
    bad: list[str] = []
    for path in bm_dir.rglob("*.py"):
        offs = _ast_check_no_pipeline_or_synth(path)
        for o in offs:
            bad.append(f"{path.relative_to(REPO_ROOT)}: {o}")
    assert not bad, (
        "body_model package imports / references synthesize_dynamic_tube:\n  "
        + "\n  ".join(bad)
    )
