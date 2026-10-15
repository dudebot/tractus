"""Sanity tests + smoke training run for the Module-1 planner.

Tests are marked ``torch`` (auto-skipped if torch isn't installed; see
``conftest.py``). The smoke training run is small enough (≤ 30 s on
CPU) that we don't gate it behind a separate ``slow`` marker — if it
takes longer than that locally, something has regressed.

Coverage:

- Vocab + tokenizer round-trip.
- Param count ≤ 10 M.
- Forward shapes are right at training time (teacher-forced durations).
- Forward at inference time (predicted durations) returns non-empty
  frame output.
- Loss path is finite, total loss decreases over a 5-epoch synthetic
  smoke run.
- Affect input dimension is wired regardless of label availability
  (passing zeros yields the same shape as a non-zero affect vector).
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from tractus_controller.planner import (
    AFFECT_DIM,
    PHONE_VOCAB,
    PhoneTokenizer,
    PlannerConfig,
    PlannerLossWeights,
    PlannerModel,
    planner_loss,
    planner_param_count,
)
from tractus_controller.planner.data import collate, synthetic_dataset, synthetic_sample
from tractus_controller.planner.eval import evaluate, measure_inference_ms_5s
from tractus_controller.planner.losses import PlannerLossTargets
from tractus_controller.planner.train import TrainConfig, train


pytestmark = pytest.mark.torch


# ---- vocab / tokenizer ----------------------------------------------------


def test_vocab_size_and_specials():
    """Vocab includes 6 specials + 39 ARPAbet phones = 45 tokens."""
    assert len(PHONE_VOCAB) == 6 + 39
    for s in ("<pad>", "<bos>", "<eos>", "<unk>", "<sil>", "<phrase_break>"):
        assert s in PHONE_VOCAB


def test_tokenizer_encodes_with_stress():
    tok = PhoneTokenizer()
    ids, stress = tok.encode(["D", "EY1", "Z", "IY0"])
    # BOS + 4 phones + EOS = 6 ids
    assert len(ids) == 6
    assert stress == [0, 0, 1, 0, 0, 0]


def test_tokenizer_unknown_phone_maps_to_unk():
    tok = PhoneTokenizer(add_bos=False, add_eos=False)
    ids, _ = tok.encode(["NOTAPHONE"])
    assert ids == [PHONE_VOCAB.index("<unk>")]


# ---- model size + shapes --------------------------------------------------


def test_param_count_under_10m():
    n = planner_param_count()
    assert n <= 10_000_000, f"planner has {n} params; cap is 10M"
    # Make sure we're not WAY under (i.e. accidentally tiny):
    assert n >= 1_000_000, f"planner has only {n} params; suspicious"


def test_forward_shapes_with_teacher_durations():
    cfg = PlannerConfig()
    model = PlannerModel(cfg)
    samples = synthetic_dataset(4, seed=1)
    batch = collate(samples)
    out = model(
        batch.token_ids,
        batch.stress,
        batch.affect,
        batch.speaker_f0_range_hz,
        durations_frames=batch.durations_frames,
    )
    B, L = batch.token_ids.shape
    assert out.log_dur_pred.shape == (B, L)
    assert out.phrase_break_logit.shape == (B, L)
    assert out.frame_mask.shape[0] == B
    assert out.log_f0_pred.shape == out.frame_mask.shape
    assert out.voiced_logit.shape == out.frame_mask.shape
    assert out.loud_db_pred.shape == out.frame_mask.shape


def test_forward_inference_uses_predicted_durations():
    cfg = PlannerConfig()
    model = PlannerModel(cfg).eval()
    samples = synthetic_dataset(2, seed=2)
    batch = collate(samples)
    with torch.no_grad():
        out = model(
            batch.token_ids,
            batch.stress,
            batch.affect,
            batch.speaker_f0_range_hz,
            durations_frames=None,  # inference — derive from predictions
        )
    assert out.log_f0_pred.numel() > 0
    assert out.frame_mask.any()


# ---- affect wiring --------------------------------------------------------


def test_affect_input_dimension_wired():
    """Passing zero-affect (unlabeled corpus) and nonzero-affect both work,
    same output shape — confirms the input dim is wired regardless of label
    availability (task spec deliverable #4).
    """
    cfg = PlannerConfig()
    model = PlannerModel(cfg).eval()
    samples = synthetic_dataset(2, seed=3)
    batch = collate(samples)
    zero_aff = torch.zeros_like(batch.affect)
    rand_aff = torch.randn_like(batch.affect)
    with torch.no_grad():
        out_zero = model(
            batch.token_ids, batch.stress, zero_aff,
            batch.speaker_f0_range_hz, durations_frames=batch.durations_frames,
        )
        out_rand = model(
            batch.token_ids, batch.stress, rand_aff,
            batch.speaker_f0_range_hz, durations_frames=batch.durations_frames,
        )
    assert out_zero.log_f0_pred.shape == out_rand.log_f0_pred.shape
    # And they should differ (affect actually influences output):
    diff = (out_zero.log_f0_pred - out_rand.log_f0_pred).abs().sum().item()
    assert diff > 0.0, "affect input had no effect on output"


def test_affect_dim_is_eight_per_task_spec():
    """Task spec pins D_affect = 8."""
    assert AFFECT_DIM == 8


# ---- loss path is finite + decreasing on smoke run ------------------------


def test_loss_is_finite_on_first_step():
    cfg = PlannerConfig()
    model = PlannerModel(cfg)
    samples = synthetic_dataset(8, seed=4)
    batch = collate(samples)
    out = model(
        batch.token_ids, batch.stress, batch.affect,
        batch.speaker_f0_range_hz, durations_frames=batch.durations_frames,
    )
    targets = PlannerLossTargets(
        log_dur_target=batch.log_dur_target,
        phone_mask=batch.phone_mask,
        phrase_break_target=batch.phrase_break_target,
        log_f0_target=batch.log_f0_target,
        voiced_target=batch.voiced_target,
        loud_db_target=batch.loud_db_target,
    )
    loss, parts = planner_loss(out, targets)
    assert torch.isfinite(loss).item(), f"loss not finite: {loss}"
    for k, v in parts.items():
        assert torch.isfinite(v).item(), f"{k} not finite: {v}"


def test_smoke_train_loss_decreases_on_cpu():
    """Tiny synthetic run: loss must drop ≥ 20% from step 0 to last step.

    Forces device='cpu' so this is portable across machines without GPUs.
    """
    cfg = PlannerConfig(
        d_model=64, n_heads=2, ffn_dim=128, n_encoder_layers=2, n_decoder_layers=2
    )
    model = PlannerModel(cfg)
    samples = synthetic_dataset(32, seed=5)
    tcfg = TrainConfig(
        batch_size=8, n_epochs=3, lr=5e-3, device="cpu", warmup_steps=4, log_every=999
    )
    history = train(model, samples, tcfg)
    losses = history["loss/total"]
    assert len(losses) >= 4
    early = sum(losses[:2]) / 2
    late = sum(losses[-2:]) / 2
    assert late < 0.8 * early, f"loss did not decrease enough: {early:.3f} → {late:.3f}"


def test_eval_returns_all_gate_metrics():
    """``evaluate`` returns the four headline gate metrics plus loudness."""
    cfg = PlannerConfig(
        d_model=64, n_heads=2, ffn_dim=128, n_encoder_layers=2, n_decoder_layers=2
    )
    model = PlannerModel(cfg)
    samples = synthetic_dataset(16, seed=6)
    batch = collate(samples)
    metrics = evaluate(model, [batch], "cpu")
    for k in ("phone_dur_rmse_ms", "f0_cents_rmse", "loud_rmse_db", "phrase_break_f1"):
        assert k in metrics
        assert math.isfinite(metrics[k])
